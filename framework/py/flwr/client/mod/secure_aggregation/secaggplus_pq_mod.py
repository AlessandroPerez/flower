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
"""Post-quantum modifier for the SecAgg+ protocol."""


import os
from dataclasses import dataclass, field
from logging import DEBUG, WARNING
from typing import Any, cast

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.app.typing import ConfigRecordValues
from flwr.clientapp.typing import ClientAppCallable
from flwr.common import Parameters, ndarray_to_bytes, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.secure_aggregation.crypto.pq_kem import (
    decapsulate_secret,
    derive_pairwise_secret,
    derive_share_encryption_secret,
    encapsulate,
    generate_key_pair,
)
from flwr.common.secure_aggregation.crypto.shamir import create_shares
from flwr.common.secure_aggregation.crypto.symmetric_encryption import decrypt, encrypt
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    parameters_addition,
    parameters_mod,
    parameters_multiply,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    Key,
    Stage,
)
from flwr.common.secure_aggregation.secaggplus_utils import (
    pseudo_rand_gen,
    share_keys_plaintext_concat_ps,
    share_keys_plaintext_concat_rd,
    share_keys_plaintext_separate_ps,
    share_keys_plaintext_separate_rd,
)
from flwr.compat.common import recorddict_compat as compat


@dataclass
# pylint: disable-next=too-many-instance-attributes
class SecAggPlusPQState:
    """State of the post-quantum SecAgg+ protocol."""

    current_stage: str = Stage.UNMASK

    nid: int = 0
    sample_num: int = 0
    share_num: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    target_range: int = 0
    mod_range: int = 0
    max_weight: float = 0.0

    # X-Wing key pairs (pairwise masks and share encryption)
    sk_pairwise: bytes = b""
    pk_pairwise: bytes = b""
    sk_share_enc: bytes = b""
    pk_share_enc: bytes = b""

    # If True, use a single X-Wing key pair and one encapsulation per neighbour.
    # The pairwise seed and the share-encryption key are then derived from the
    # same shared secret via HKDF with different context strings.
    single_kem: bool = False

    # Random seed for the private mask
    rd_seed: bytes = b""

    # Share of nid's rd_seed that this client holds
    rd_seed_share_dict: dict[int, bytes] = field(default_factory=dict)
    # Share of ss_pairwise(owner, self) that this client holds
    pairwise_secret_share_dict: dict[int, bytes] = field(default_factory=dict)
    # Neighbor public keys: nid -> (pk_pairwise, pk_share_enc)
    public_keys_dict: dict[int, tuple[bytes, bytes]] = field(default_factory=dict)
    # X-Wing shared secrets this client generated: dst -> (ss_pairwise, ss_share_enc)
    shared_secrets_sent: dict[int, tuple[bytes, bytes]] = field(default_factory=dict)
    # X-Wing shared secrets received from neighbors:
    # src -> (ss_pairwise, ss_share_enc)
    shared_secrets_received: dict[int, tuple[bytes, bytes]] = field(
        default_factory=dict
    )
    # Fernet keys for encrypting shares to neighbors
    share_enc_keys_sent: dict[int, bytes] = field(default_factory=dict)
    # Fernet keys for decrypting shares from neighbors
    share_enc_keys_received: dict[int, bytes] = field(default_factory=dict)

    def __init__(self, **kwargs: ConfigRecordValues) -> None:
        for k, v in kwargs.items():
            if k.endswith(":V"):
                continue
            new_v: Any = v
            if k.endswith(":K"):
                k = k[:-2]
                keys = cast(list[int], v)
                values = cast(list[bytes], kwargs[f"{k}:V"])
                if len(values) > len(keys) and k in {
                    "public_keys_dict",
                    "shared_secrets_sent",
                    "shared_secrets_received",
                }:
                    new_v = {
                        key: (values[2 * i], values[2 * i + 1])
                        for i, key in enumerate(keys)
                    }
                else:
                    new_v = dict(zip(keys, values, strict=True))
            self.__setattr__(k, new_v)

    def to_dict(self) -> dict[str, ConfigRecordValues]:
        """Convert the state to a dictionary."""
        ret = vars(self).copy()
        for k in list(ret.keys()):
            if isinstance(ret[k], dict):
                v = cast(dict[str, Any], ret.pop(k))
                ret[f"{k}:K"] = list(v.keys())
                if k in {
                    "public_keys_dict",
                    "shared_secrets_sent",
                    "shared_secrets_received",
                }:
                    v_list: list[bytes] = []
                    for b1_b2 in cast(list[tuple[bytes, bytes]], v.values()):
                        v_list.extend(b1_b2)
                    ret[f"{k}:V"] = v_list
                else:
                    ret[f"{k}:V"] = list(v.values())
        return ret


def secaggplus_pq_mod(
    msg: Message,
    ctxt: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Handle incoming message and return results for PQ SecAgg+."""
    if msg.metadata.message_type != MessageType.TRAIN:
        return call_next(msg, ctxt)

    if RECORD_KEY_STATE not in ctxt.state.config_records:
        ctxt.state.config_records[RECORD_KEY_STATE] = ConfigRecord({})
    state_dict = ctxt.state.config_records[RECORD_KEY_STATE]
    state = SecAggPlusPQState(**state_dict)

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
        raise ValueError(f"Unknown SecAgg+ PQ stage: {state.current_stage}")

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


def check_configs(  # pylint: disable=too-many-branches
    stage: str, configs: ConfigRecord
) -> None:
    """Check the validity of the configs."""
    if stage == Stage.SETUP:
        key_type_pairs = [
            (Key.SAMPLE_NUMBER, int),
            (Key.SHARE_NUMBER, int),
            (Key.THRESHOLD, int),
            (Key.CLIPPING_RANGE, float),
            (Key.TARGET_RANGE, int),
            (Key.MOD_RANGE, int),
        ]
        for key, expected_type in key_type_pairs:
            if key not in configs:
                raise KeyError(
                    f"Stage {Stage.SETUP}: the required key '{key}' is "
                    "missing from the ConfigRecord."
                )
            # pylint: disable-next=unidiomatic-typecheck
            if type(configs[key]) is not expected_type:
                raise TypeError(
                    f"Stage {Stage.SETUP}: The value for the key '{key}' "
                    f"must be of type {expected_type}, "
                    f"but got {type(configs[key])} instead."
                )
    elif stage == Stage.SHARE_KEYS:
        for key, value in configs.items():
            if key == Key.STAGE:
                continue
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], bytes)
                or not isinstance(value[1], bytes)
            ):
                raise TypeError(
                    f"Stage {Stage.SHARE_KEYS}: "
                    f"the value for the key '{key}' must be a list of two bytes."
                )
    elif stage == Stage.COLLECT_MASKED_VECTORS:
        key_type_pairs = [
            (Key.CIPHERTEXT_PAIRWISE_LIST, bytes),
            (Key.CIPHERTEXT_SHARE_ENC_LIST, bytes),
            (Key.SOURCE_LIST, int),
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
                # pylint: disable-next=unidiomatic-typecheck
                if type(elm) is not expected_type
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
                # pylint: disable-next=unidiomatic-typecheck
                type(elm) is not expected_type
                for elm in cast(list[Any], configs[key])
            ):
                raise TypeError(
                    f"Stage {Stage.UNMASK}: the value for the key '{key}' "
                    f"must be of type List[{expected_type.__name__}]"
                )
    else:
        raise ValueError(f"Unknown secagg stage: {stage}")


def _setup(
    state: SecAggPlusPQState, configs: ConfigRecord
) -> dict[str, ConfigRecordValues]:
    """Handle the setup stage."""
    state.sample_num = cast(int, configs[Key.SAMPLE_NUMBER])
    state.share_num = cast(int, configs[Key.SHARE_NUMBER])
    state.threshold = cast(int, configs[Key.THRESHOLD])
    state.clipping_range = cast(float, configs[Key.CLIPPING_RANGE])
    state.target_range = cast(int, configs[Key.TARGET_RANGE])
    state.mod_range = cast(int, configs[Key.MOD_RANGE])
    state.max_weight = cast(float, configs[Key.MAX_WEIGHT])

    state.rd_seed_share_dict = {}
    state.pairwise_secret_share_dict = {}
    state.public_keys_dict = {}
    state.shared_secrets_sent = {}
    state.shared_secrets_received = {}
    state.share_enc_keys_sent = {}
    state.share_enc_keys_received = {}

    state.single_kem = bool(configs.get(Key.SINGLE_KEM, False))

    state.rd_seed = os.urandom(32)

    if state.single_kem:
        state.pk_pairwise, state.sk_pairwise = generate_key_pair()
        state.pk_share_enc = state.pk_pairwise
        state.sk_share_enc = state.sk_pairwise
    else:
        state.pk_pairwise, state.sk_pairwise = generate_key_pair()
        state.pk_share_enc, state.sk_share_enc = generate_key_pair()

    log(DEBUG, "Node %d: PQ stage 0 completes. uploading public keys...", state.nid)
    return {
        Key.PUBLIC_KEY_PAIRWISE: state.pk_pairwise,
        Key.PUBLIC_KEY_SHARE_ENC: state.pk_share_enc,
    }


# pylint: disable-next=too-many-locals
def _share_keys(
    state: SecAggPlusPQState, configs: ConfigRecord
) -> dict[str, ConfigRecordValues]:
    """Handle the share-keys stage (stage 1).

    Clients receive their neighbours' public keys, encapsulate to every
    neighbour, and upload the encapsulations together with encrypted
    Shamir shares of ``rd_seed``.
    """
    named_bytes_tuples = cast(dict[str, tuple[bytes, bytes]], configs)
    key_dict = {
        int(sid): (pk_pw, pk_se) for sid, (pk_pw, pk_se) in named_bytes_tuples.items()
    }
    state.public_keys_dict = key_dict
    log(DEBUG, "Node %d: PQ stage 1 starts...", state.nid)

    if len(state.public_keys_dict) < state.threshold:
        raise ValueError("Available neighbours number smaller than threshold")

    pk_list: list[bytes] = []
    if state.single_kem:
        for pk_pw, _ in state.public_keys_dict.values():
            pk_list.append(pk_pw)
    else:
        for pk_pw, pk_se in state.public_keys_dict.values():
            pk_list.extend([pk_pw, pk_se])
    if len(set(pk_list)) != len(pk_list):
        raise ValueError("Some public keys are identical")

    own_pk_pw, own_pk_se = state.public_keys_dict[state.nid]
    if own_pk_pw != state.pk_pairwise or own_pk_se != state.pk_share_enc:
        raise ValueError("Own public keys are displayed in dict incorrectly")

    # Create Shamir shares of the private mask seed.
    rd_seed_shares = create_shares(state.rd_seed, state.threshold, state.share_num)

    srcs, dsts = [], []
    ct_pairwise_list, ct_share_enc_list, ciphertexts = [], [], []

    for idx, (nid, (pk_pw, pk_se)) in enumerate(state.public_keys_dict.items()):
        if nid == state.nid:
            # Keep our own share locally.
            state.rd_seed_share_dict[state.nid] = rd_seed_shares[idx]
            # We do not encapsulate to ourselves.
            continue

        if state.single_kem:
            # One encapsulation per neighbour; derive both keys from ss.
            ss, ct = encapsulate(pk_pw)
            state.shared_secrets_sent[nid] = (ss, ss)
            state.share_enc_keys_sent[nid] = derive_share_encryption_secret(ss)
            ct_pairwise_list.append(ct)
            ct_share_enc_list.append(ct)
        else:
            # Pairwise encapsulation: self -> nid
            ss_pairwise_sent, ct_pairwise = encapsulate(pk_pw)
            # Share-encryption encapsulation: self -> nid
            ss_share_enc_sent, ct_share_enc = encapsulate(pk_se)

            state.shared_secrets_sent[nid] = (ss_pairwise_sent, ss_share_enc_sent)
            state.share_enc_keys_sent[nid] = derive_share_encryption_secret(
                ss_share_enc_sent
            )

            ct_pairwise_list.append(ct_pairwise)
            ct_share_enc_list.append(ct_share_enc)

        # Encrypt the rd_seed share for this neighbour.
        plaintext = share_keys_plaintext_concat_rd(state.nid, nid, rd_seed_shares[idx])
        ciphertext = encrypt(state.share_enc_keys_sent[nid], plaintext)

        srcs.append(state.nid)
        dsts.append(nid)
        ciphertexts.append(ciphertext)

    log(DEBUG, "Node %d: PQ stage 1 completes. uploading encapsulations...", state.nid)
    return {
        Key.SOURCE_LIST: srcs,
        Key.DESTINATION_LIST: dsts,
        Key.CIPHERTEXT_PAIRWISE_LIST: ct_pairwise_list,
        Key.CIPHERTEXT_SHARE_ENC_LIST: ct_share_enc_list,
        Key.CIPHERTEXT_LIST: ciphertexts,
    }


# pylint: disable-next=too-many-locals,too-many-statements,too-many-branches
def _collect_masked_vectors(
    state: SecAggPlusPQState,
    configs: ConfigRecord,
    num_examples: int,
    updated_parameters: Parameters,
) -> dict[str, ConfigRecordValues]:
    """Handle the collect-masked-vectors stage (stage 2).

    Clients receive neighbours' encapsulations and encrypted ``rd_seed``
    shares, compute the pairwise secrets, create Shamir shares of those
    secrets, encrypt them to neighbours, train, and upload the masked
    parameters.
    """
    log(DEBUG, "Node %d: PQ stage 2 starts...", state.nid)

    ct_pairwise_list = cast(list[bytes], configs[Key.CIPHERTEXT_PAIRWISE_LIST])
    ct_share_enc_list = cast(list[bytes], configs[Key.CIPHERTEXT_SHARE_ENC_LIST])
    srcs = cast(list[int], configs[Key.SOURCE_LIST])
    ciphertexts = cast(list[bytes], configs[Key.CIPHERTEXT_LIST])

    if len(ct_pairwise_list) + 1 < state.threshold:
        raise ValueError("Not enough available neighbour clients.")

    available_clients: list[int] = []

    if state.single_kem:
        # In single-KEM mode the two ciphertext lists are identical; one
        # decapsulation gives the shared secret used for both purposes.
        for src, ct, ciphertext in zip(
            srcs, ct_pairwise_list, ciphertexts, strict=True
        ):
            ss = decapsulate_secret(ct, state.sk_pairwise)
            state.shared_secrets_received[src] = (ss, ss)
            state.share_enc_keys_received[src] = derive_share_encryption_secret(ss)

            plaintext = decrypt(state.share_enc_keys_received[src], ciphertext)
            actual_src, dst, rd_seed_share = share_keys_plaintext_separate_rd(plaintext)
            available_clients.append(src)
            if actual_src != src:
                raise ValueError(
                    f"Node {state.nid}: received rd_seed share from {actual_src} "
                    f"instead of {src}."
                )
            if dst != state.nid:
                raise ValueError(
                    f"Node {state.nid}: received rd_seed share for Node {dst} "
                    f"from Node {src}."
                )
            state.rd_seed_share_dict[src] = rd_seed_share
    else:
        # Decapsulate incoming encapsulations and decrypt rd_seed shares.
        for src, ct_pw, ct_se, ciphertext in zip(
            srcs, ct_pairwise_list, ct_share_enc_list, ciphertexts, strict=True
        ):
            ss_pairwise_received = decapsulate_secret(ct_pw, state.sk_pairwise)
            ss_share_enc_received = decapsulate_secret(ct_se, state.sk_share_enc)
            state.shared_secrets_received[src] = (
                ss_pairwise_received,
                ss_share_enc_received,
            )
            state.share_enc_keys_received[src] = derive_share_encryption_secret(
                ss_share_enc_received
            )

            plaintext = decrypt(state.share_enc_keys_received[src], ciphertext)
            actual_src, dst, rd_seed_share = share_keys_plaintext_separate_rd(plaintext)
            available_clients.append(src)
            if actual_src != src:
                raise ValueError(
                    f"Node {state.nid}: received rd_seed share from {actual_src} "
                    f"instead of {src}."
                )
            if dst != state.nid:
                raise ValueError(
                    f"Node {state.nid}: received rd_seed share for Node {dst} "
                    f"from Node {src}."
                )
            state.rd_seed_share_dict[src] = rd_seed_share

    # Compute pairwise secrets and create Shamir shares for each.
    pairwise_secret_shares: dict[int, list[tuple[int, bytes]]] = {
        nid: [] for nid in state.public_keys_dict if nid != state.nid
    }
    for nid in state.public_keys_dict:
        if nid == state.nid:
            continue
        ss_pairwise_sent = state.shared_secrets_sent[nid][0]
        ss_pairwise_received = state.shared_secrets_received[nid][0]
        pw_secret = derive_pairwise_secret(ss_pairwise_sent, ss_pairwise_received)
        shares = create_shares(pw_secret, state.threshold, state.share_num)
        # shares[idx] is the share intended for the idx-th neighbour
        # (including self), following public_keys_dict ordering.
        for idx, dst_nid in enumerate(state.public_keys_dict):
            if dst_nid == state.nid:
                # Keep our own share locally; it is never uploaded because
                # if we drop our neighbours provide the shares.
                continue
            pairwise_secret_shares[dst_nid].append((nid, shares[idx]))

    # Encrypt and upload pairwise-secret share bundles.
    out_srcs, out_dsts, out_ciphertexts = [], [], []
    for dst_nid, ps_shares in pairwise_secret_shares.items():
        plaintext = share_keys_plaintext_concat_ps(state.nid, dst_nid, ps_shares)
        ciphertext = encrypt(state.share_enc_keys_sent[dst_nid], plaintext)
        out_srcs.append(state.nid)
        out_dsts.append(dst_nid)
        out_ciphertexts.append(ciphertext)

    # Train / compute masked parameters.
    ratio = num_examples / state.max_weight
    if ratio > 1:
        log(
            WARNING,
            "Potential overflow warning: the provided weight (%s) exceeds the "
            "specified max_weight (%s). This may lead to overflow issues.",
            num_examples,
            state.max_weight,
        )
    q_ratio = round(ratio * state.target_range)
    dq_ratio = q_ratio / state.target_range

    parameters = parameters_to_ndarrays(updated_parameters)
    parameters = parameters_multiply(parameters, dq_ratio)
    quantized_parameters = quantize(
        parameters, state.clipping_range, state.target_range
    )
    quantized_parameters = factor_combine(q_ratio, quantized_parameters)

    dimensions_list: list[tuple[int, ...]] = [a.shape for a in quantized_parameters]

    # Add private mask.
    private_mask = pseudo_rand_gen(state.rd_seed, state.mod_range, dimensions_list)
    quantized_parameters = parameters_addition(quantized_parameters, private_mask)

    # Add pairwise masks.
    for node_id in available_clients:
        pw_secret = derive_pairwise_secret(
            state.shared_secrets_sent[node_id][0],
            state.shared_secrets_received[node_id][0],
        )
        pairwise_mask = pseudo_rand_gen(pw_secret, state.mod_range, dimensions_list)
        if state.nid > node_id:
            quantized_parameters = parameters_addition(
                quantized_parameters, pairwise_mask
            )
        else:
            quantized_parameters = parameters_subtraction(
                quantized_parameters, pairwise_mask
            )

    quantized_parameters = parameters_mod(quantized_parameters, state.mod_range)
    log(
        DEBUG,
        "Node %d: PQ stage 2 completes, uploading masked parameters...",
        state.nid,
    )
    return {
        Key.SOURCE_LIST: out_srcs,
        Key.DESTINATION_LIST: out_dsts,
        Key.CIPHERTEXT_LIST: out_ciphertexts,
        Key.MASKED_PARAMETERS: [ndarray_to_bytes(arr) for arr in quantized_parameters],
    }


def _unmask(
    state: SecAggPlusPQState, configs: ConfigRecord
) -> dict[str, ConfigRecordValues]:
    """Handle the unmask stage (stage 3)."""
    log(DEBUG, "Node %d: PQ stage 3 starts...", state.nid)

    active_nids = cast(list[int], configs[Key.ACTIVE_NODE_ID_LIST])
    dead_nids = cast(list[int], configs[Key.DEAD_NODE_ID_LIST])
    ciphertexts = cast(list[bytes], configs.get(Key.CIPHERTEXT_LIST, []))
    srcs = cast(list[int], configs.get(Key.SOURCE_LIST, []))

    if len(active_nids) < state.threshold:
        raise ValueError("Available neighbours number smaller than threshold")

    # Decrypt incoming pairwise-secret share bundles.
    for src, ciphertext in zip(srcs, ciphertexts, strict=True):
        plaintext = decrypt(state.share_enc_keys_received[src], ciphertext)
        actual_src, dst, _ = share_keys_plaintext_separate_ps(plaintext)
        if actual_src != src:
            raise ValueError(
                f"Node {state.nid}: received pairwise-secret bundle from "
                f"{actual_src} instead of {src}."
            )
        if dst != state.nid:
            raise ValueError(
                f"Node {state.nid}: received pairwise-secret bundle for "
                f"Node {dst} from Node {src}."
            )
        state.pairwise_secret_share_dict[src] = plaintext

    # Upload shares for active and dead neighbours.
    all_nids = active_nids + dead_nids
    shares: list[bytes] = []
    shares.extend(state.rd_seed_share_dict[nid] for nid in active_nids)
    shares.extend(state.pairwise_secret_share_dict[nid] for nid in dead_nids)

    log(DEBUG, "Node %d: PQ stage 3 completes. uploading shares...", state.nid)
    return {
        Key.NODE_ID_LIST: all_nids,
        Key.SHARE_LIST: shares,
        Key.ACTIVE_NODE_ID_LIST: active_nids,
        Key.DEAD_NODE_ID_LIST: dead_nids,
    }
