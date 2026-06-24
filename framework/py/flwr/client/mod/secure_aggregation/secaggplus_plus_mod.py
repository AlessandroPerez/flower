# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Modifier for the SecAggPlusPlus protocol."""


from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from logging import DEBUG, WARNING
from typing import Any, cast

import numpy as np
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.app.typing import ConfigRecordValues
from flwr.clientapp.typing import ClientAppCallable
from flwr.common import NDArrays, Parameters, ndarray_to_bytes, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.pq_pke import (
    decrypt as pke_decrypt,
)
from flwr.common.secure_aggregation.crypto.pq_pke import (
    encrypt as pke_encrypt,
)
from flwr.common.secure_aggregation.crypto.pq_pke import generate_keypair
from flwr.common.secure_aggregation.crypto.shamir import create_shares
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    parameters_addition,
    parameters_mod,
    parameters_multiply,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    Key,
    Stage,
)
from flwr.common.secure_aggregation.secaggplus_utils import (
    derive_pairwise_key,
    pseudo_rand_gen,
    share_keys_plaintext_concat_plus,
    share_keys_plaintext_separate_plus,
)
from flwr.compat.common import recorddict_compat as compat


@dataclass
# pylint: disable-next=too-many-instance-attributes
class SecAggPlusPlusState:
    """State of the SecAggPlusPlus protocol."""

    current_stage: str = Stage.UNMASK

    nid: int = 0
    sample_num: int = 0
    share_num: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    target_range: int = 0
    mod_range: int = 0
    max_weight: float = 0.0

    # Shape of the (quantized) model parameters, needed to expand masks at
    # unmask time for the e_S correction.
    dimensions_list: list[tuple[int, ...]] = field(default_factory=list)

    # PQ PKE key pair
    sk: bytes = b""
    pk: bytes = b""

    # Master seeds for pairwise keys and self mask
    b_seed: bytes = b""
    u_seed: bytes = b""

    # Neighbour public keys: nid -> pk
    public_keys_dict: dict[int, bytes] = field(default_factory=dict)
    # Shares of b_seed intended for each neighbour
    b_seed_shares: dict[int, bytes] = field(default_factory=dict)
    # Shares of u_seed intended for each neighbour
    u_seed_shares: dict[int, bytes] = field(default_factory=dict)
    # Pairwise keys this client derived for each neighbour
    derived_keys: dict[int, bytes] = field(default_factory=dict)
    # Pairwise keys received from neighbours
    received_keys: dict[int, bytes] = field(default_factory=dict)
    # b_seed shares received from neighbours (share of neighbour's b_seed)
    received_b_shares: dict[int, bytes] = field(default_factory=dict)
    # u_seed shares received from neighbours
    received_u_shares: dict[int, bytes] = field(default_factory=dict)

    def __init__(self, **kwargs: ConfigRecordValues) -> None:
        for k, v in kwargs.items():
            if k.endswith(":V"):
                continue
            new_v: Any = v
            if k.endswith(":K"):
                k = k[:-2]
                keys = cast(list[int], v)
                values = cast(list[bytes], kwargs[f"{k}:V"])
                new_v = dict(zip(keys, values, strict=True))
            elif k == "dimensions_list" and isinstance(v, bytes):
                new_v = pickle.loads(v)
            self.__setattr__(k, new_v)

    def to_dict(self) -> dict[str, ConfigRecordValues]:
        """Convert the state to a dictionary."""
        ret = vars(self).copy()
        for k in list(ret.keys()):
            if isinstance(ret[k], dict):
                v = cast(dict[int, bytes], ret.pop(k))
                ret[f"{k}:K"] = list(v.keys())
                ret[f"{k}:V"] = list(v.values())
        # ConfigRecord does not accept list[tuple[int, ...]], so pickle it.
        if ret.get("dimensions_list"):
            ret["dimensions_list"] = pickle.dumps(
                cast(list[tuple[int, ...]], ret["dimensions_list"])
            )
        return ret


def secaggplus_plus_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Handle incoming message and return results for SecAggPlusPlus."""
    if msg.metadata.message_type != MessageType.TRAIN:
        return call_next(msg, ctxt)

    if RECORD_KEY_STATE not in ctxt.state.config_records:
        ctxt.state.config_records[RECORD_KEY_STATE] = ConfigRecord({})
    state_dict = ctxt.state.config_records[RECORD_KEY_STATE]
    state = SecAggPlusPlusState(**state_dict)

    configs = msg.content.config_records[RECORD_KEY_CONFIGS]

    check_stage(state.current_stage, configs)
    state.current_stage = cast(str, configs.pop(Key.STAGE))
    check_configs(state.current_stage, configs)

    out_content = RecordDict()
    if state.current_stage == Stage.SETUP:
        state.nid = msg.metadata.dst_node_id
        res = _setup(state, configs)
    elif state.current_stage == Stage.SHARE_KEYS:
        res = _share_keys(state, configs)
    elif state.current_stage == Stage.COLLECT_MASKED_VECTORS:
        out_msg = call_next(msg, ctxt)
        out_content = out_msg.content
        fitres = compat.recorddict_to_fitres(out_content, keep_input=True)
        res = _collect_masked_vectors(
            state, configs, fitres.num_examples, fitres.parameters
        )
        for arr_record in out_content.array_records.values():
            arr_record.clear()
    elif state.current_stage == Stage.UNMASK:
        res = _unmask(state, configs)
    else:
        raise ValueError(f"Unknown SecAggPlusPlus stage: {state.current_stage}")

    ctxt.state.config_records[RECORD_KEY_STATE] = ConfigRecord(state.to_dict())

    out_content.config_records[RECORD_KEY_CONFIGS] = ConfigRecord(res, False)
    return Message(out_content, reply_to=msg)


def check_stage(current_stage: str, configs: ConfigRecord) -> None:
    """Check the validity of the next stage."""
    if Key.STAGE not in configs:
        raise KeyError(
            f"The required key '{Key.STAGE}' is missing from the ConfigRecord."
        )

    next_stage = configs[Key.STAGE]
    if not isinstance(next_stage, str):
        raise TypeError(
            f"The value for the key '{Key.STAGE}' must be of type {str}, "
            f"but got {type(next_stage)} instead."
        )

    if next_stage == Stage.SETUP:
        if current_stage != Stage.UNMASK:
            log(WARNING, "Restart from the setup stage")
    else:
        stages = Stage.all()
        expected_next_stage = stages[(stages.index(current_stage) + 1) % len(stages)]
        if next_stage != expected_next_stage:
            raise ValueError(
                "Abort secure aggregation: "
                f"expect {expected_next_stage} stage, but receive {next_stage} stage"
            )


def check_configs(stage: str, configs: ConfigRecord) -> None:
    """Check the validity of the configs."""
    if stage == Stage.SETUP:
        key_type_pairs = [
            (Key.SAMPLE_NUMBER, int),
            (Key.SHARE_NUMBER, int),
            (Key.THRESHOLD, int),
            (Key.CLIPPING_RANGE, float),
            (Key.TARGET_RANGE, int),
            (Key.MOD_RANGE, int),
            (Key.MAX_WEIGHT, float),
        ]
        for key, expected_type in key_type_pairs:
            if key not in configs:
                raise KeyError(
                    f"Stage {Stage.SETUP}: the required key '{key}' is "
                    "missing from the ConfigRecord."
                )
            if (
                type(configs[key]) is not expected_type
            ):  # pylint: disable=unidiomatic-typecheck
                raise TypeError(
                    f"Stage {Stage.SETUP}: The value for the key '{key}' "
                    f"must be of type {expected_type}, "
                    f"but got {type(configs[key])} instead."
                )
    elif stage == Stage.SHARE_KEYS:
        for key, value in configs.items():
            if key == Key.STAGE:
                continue
            if not isinstance(value, bytes):
                raise TypeError(
                    f"Stage {Stage.SHARE_KEYS}: "
                    f"the value for the key '{key}' must be bytes."
                )
    elif stage == Stage.COLLECT_MASKED_VECTORS:
        key_type_pairs = [
            (Key.SOURCE_LIST, int),
            (Key.CIPHERTEXT_LIST, bytes),
        ]
        for key, expected_type in key_type_pairs:
            if key not in configs:
                raise KeyError(
                    f"Stage {Stage.COLLECT_MASKED_VECTORS}: "
                    f"the required key '{key}' is missing."
                )
            if not isinstance(configs[key], list) or any(
                elm
                for elm in cast(list[Any], configs[key])
                if type(elm)
                is not expected_type  # pylint: disable=unidiomatic-typecheck
            ):
                raise TypeError(
                    f"Stage {Stage.COLLECT_MASKED_VECTORS}: "
                    f"the value for the key '{key}' "
                    f"must be of type List[{expected_type.__name__}]"
                )
    elif stage == Stage.UNMASK:
        key_type_pairs = [
            (Key.ACTIVE_NODE_ID_LIST, int),
            (Key.DEAD_NODE_ID_LIST, int),
        ]
        for key, expected_type in key_type_pairs:
            if key not in configs:
                raise KeyError(
                    f"Stage {Stage.UNMASK}: the required key '{key}' is missing."
                )
            if not isinstance(configs[key], list) or any(
                type(elm) is not expected_type  # pylint: disable=unidiomatic-typecheck
                for elm in cast(list[Any], configs[key])
            ):
                raise TypeError(
                    f"Stage {Stage.UNMASK}: the value for the key '{key}' "
                    f"must be of type List[{expected_type.__name__}]"
                )
    else:
        raise ValueError(f"Unknown secagg stage: {stage}")


def _setup(
    state: SecAggPlusPlusState, configs: ConfigRecord
) -> dict[str, ConfigRecordValues]:
    """Handle the setup stage."""
    state.sample_num = cast(int, configs[Key.SAMPLE_NUMBER])
    state.share_num = cast(int, configs[Key.SHARE_NUMBER])
    state.threshold = cast(int, configs[Key.THRESHOLD])
    state.clipping_range = cast(float, configs[Key.CLIPPING_RANGE])
    state.target_range = cast(int, configs[Key.TARGET_RANGE])
    state.mod_range = cast(int, configs[Key.MOD_RANGE])
    state.max_weight = cast(float, configs[Key.MAX_WEIGHT])

    state.public_keys_dict = {}
    state.b_seed_shares = {}
    state.u_seed_shares = {}
    state.derived_keys = {}
    state.received_keys = {}
    state.received_b_shares = {}
    state.received_u_shares = {}

    state.b_seed = os.urandom(32)
    state.u_seed = os.urandom(32)

    state.pk, state.sk = generate_keypair()

    log(DEBUG, "Node %d: SecAggPlusPlus stage 0 completes.", state.nid)
    return {Key.PUBLIC_KEY_PAIRWISE: state.pk}


def _share_keys(
    state: SecAggPlusPlusState, configs: ConfigRecord
) -> dict[str, ConfigRecordValues]:
    """Handle the share-keys stage."""
    key_dict: dict[int, bytes] = {
        int(sid): cast(bytes, pk) for sid, pk in configs.items() if sid != Key.STAGE
    }
    state.public_keys_dict = key_dict
    log(DEBUG, "Node %d: SecAggPlusPlus stage 1 starts...", state.nid)

    if len(state.public_keys_dict) < state.threshold:
        raise ValueError("Available neighbours number smaller than threshold")

    own_pk = state.public_keys_dict[state.nid]
    if own_pk != state.pk:
        raise ValueError("Own public key is displayed in dict incorrectly")

    # Create Shamir shares of the master seeds.
    b_seed_shares = create_shares(state.b_seed, state.threshold, state.share_num)
    u_seed_shares = create_shares(state.u_seed, state.threshold, state.share_num)

    srcs, dsts, ciphertexts = [], [], []

    # public_keys_dict ordering must be deterministic; configs is ordered.
    for idx, (nid, pk) in enumerate(state.public_keys_dict.items()):
        if nid == state.nid:
            # Keep our own shares locally (not used for masking, only for
            # reconstruction robustness).
            state.b_seed_shares[state.nid] = b_seed_shares[idx]
            state.u_seed_shares[state.nid] = u_seed_shares[idx]
            continue

        k = derive_pairwise_key(state.b_seed, nid)
        state.derived_keys[nid] = k
        state.b_seed_shares[nid] = b_seed_shares[idx]
        state.u_seed_shares[nid] = u_seed_shares[idx]

        plaintext = share_keys_plaintext_concat_plus(
            state.nid, nid, k, b_seed_shares[idx], u_seed_shares[idx]
        )
        ciphertext = pke_encrypt(pk, plaintext)

        srcs.append(state.nid)
        dsts.append(nid)
        ciphertexts.append(ciphertext)

    log(DEBUG, "Node %d: SecAggPlusPlus stage 1 completes.", state.nid)
    return {
        Key.SOURCE_LIST: srcs,
        Key.DESTINATION_LIST: dsts,
        Key.CIPHERTEXT_LIST: ciphertexts,
    }


def _collect_masked_vectors(
    state: SecAggPlusPlusState,
    configs: ConfigRecord,
    num_examples: int,
    updated_parameters: Parameters,
) -> dict[str, ConfigRecordValues]:
    """Handle the collect-masked-vectors stage."""
    log(DEBUG, "Node %d: SecAggPlusPlus stage 2 starts...", state.nid)

    srcs = cast(list[int], configs[Key.SOURCE_LIST])
    ciphertexts = cast(list[bytes], configs[Key.CIPHERTEXT_LIST])

    if len(ciphertexts) + 1 < state.threshold:
        raise ValueError("Not enough available neighbour clients.")

    available_clients: list[int] = []

    for src, ciphertext in zip(srcs, ciphertexts, strict=True):
        if src == state.nid:
            # The server should not forward a self-message; skip if it does.
            continue
        plaintext = pke_decrypt(state.sk, ciphertext)
        actual_src, dst, k, b_share, u_share = share_keys_plaintext_separate_plus(
            plaintext
        )
        available_clients.append(src)
        if actual_src != src:
            raise ValueError(
                f"Node {state.nid}: received payload from {actual_src} "
                f"instead of {src}."
            )
        if dst != state.nid:
            raise ValueError(
                f"Node {state.nid}: received payload for Node {dst} "
                f"from Node {src}."
            )
        state.received_keys[src] = k
        state.received_b_shares[src] = b_share
        state.received_u_shares[src] = u_share

    # Train / compute masked parameters.
    ratio = num_examples / state.max_weight
    q_ratio = round(ratio * state.target_range)
    dq_ratio = q_ratio / state.target_range

    parameters = parameters_to_ndarrays(updated_parameters)
    parameters = parameters_multiply(parameters, dq_ratio)
    quantized_parameters = quantize(
        parameters, state.clipping_range, state.target_range
    )
    quantized_parameters = factor_combine(q_ratio, quantized_parameters)

    # The leading array is the quantization factor; masks only apply to params.
    factor = quantized_parameters[0]
    param_arrays = quantized_parameters[1:]

    dimensions_list: list[tuple[int, ...]] = [a.shape for a in param_arrays]
    state.dimensions_list = dimensions_list

    def add_mask(target: NDArrays, key: bytes) -> NDArrays:
        mask = pseudo_rand_gen(key, state.mod_range, dimensions_list)
        return parameters_addition(target, mask)

    # Add received pairwise masks.
    received_mask_sum: NDArrays = [np.zeros_like(arr) for arr in param_arrays]
    for node_id in available_clients:
        received_mask_sum = add_mask(received_mask_sum, state.received_keys[node_id])

    # Compute the sum of derived pairwise masks.
    derived_mask_sum: NDArrays = [np.zeros_like(arr) for arr in param_arrays]
    for node_id in state.derived_keys:
        derived_mask_sum = add_mask(derived_mask_sum, state.derived_keys[node_id])

    # Add self mask.
    self_mask = pseudo_rand_gen(state.u_seed, state.mod_range, dimensions_list)

    # c_S = w_S + sum of received masks + u_S
    c_params = parameters_addition(param_arrays, received_mask_sum)
    c_params = parameters_addition(c_params, self_mask)
    c_params = parameters_mod(c_params, state.mod_range)
    c_s = [factor] + c_params

    # sum_S = sum of derived masks.  The factor slot is left as zero so that
    # the quantization factors from each c_S still add up in the aggregate.
    sum_params = parameters_mod(derived_mask_sum, state.mod_range)
    sum_s = [np.zeros_like(factor)] + sum_params

    log(DEBUG, "Node %d: SecAggPlusPlus stage 2 completes.", state.nid)
    return {
        Key.MASKED_PARAMETERS: [ndarray_to_bytes(arr) for arr in c_s],
        Key.SUM_DERIVED_KEYS: [ndarray_to_bytes(arr) for arr in sum_s],
    }


def _unmask(
    state: SecAggPlusPlusState, configs: ConfigRecord
) -> dict[str, ConfigRecordValues]:
    """Handle the unmask stage."""
    log(DEBUG, "Node %d: SecAggPlusPlus stage 3 starts...", state.nid)

    active_nids = cast(list[int], configs[Key.ACTIVE_NODE_ID_LIST])
    dead_nids = cast(list[int], configs[Key.DEAD_NODE_ID_LIST])

    if len(active_nids) < state.threshold:
        raise ValueError("Available neighbours number smaller than threshold")

    # The server knows which clients sent a masked vector (active_nids) and
    # which dropped before that stage (dead_nids).  The client trusts these
    # lists: active clients need u_seed shares, dead clients need b_seed shares.

    # Build the ordered list of clients whose shares we upload:
    # - active clients first (u_seed shares, so the server can reconstruct
    #   every self mask from threshold shares rather than receiving it directly)
    # - dead clients second (b_seed shares, for pairwise-mask recovery)
    share_nids = active_nids + dead_nids
    shares: list[bytes] = []
    for nid in share_nids:
        if nid in active_nids:
            # Our own u_seed share is stored in u_seed_shares; shares for other
            # active clients were received during masked-vector collection.
            if nid == state.nid:
                shares.append(state.u_seed_shares[nid])
            else:
                shares.append(state.received_u_shares[nid])
        else:
            shares.append(state.received_b_shares[nid])

    # Compute e_S = sum of masks derived by this client for dead neighbours.
    # These cancel the -k_{S->D} terms that remain in sum_S.
    e_s: NDArrays = [np.zeros(shape, dtype=np.int64) for shape in state.dimensions_list]
    for nid in dead_nids:
        if nid in state.derived_keys:
            mask = pseudo_rand_gen(
                state.derived_keys[nid], state.mod_range, state.dimensions_list
            )
            e_s = parameters_addition(e_s, mask)
    e_s = parameters_mod(e_s, state.mod_range)

    log(DEBUG, "Node %d: SecAggPlusPlus stage 3 completes.", state.nid)
    return {
        Key.NODE_ID_LIST: share_nids,
        Key.SHARE_LIST: shares,
        Key.E_VALUE: [ndarray_to_bytes(arr) for arr in e_s],
    }
