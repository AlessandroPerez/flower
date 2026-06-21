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
"""End-to-end tests for the post-quantum SecAgg+ protocol."""


from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.app.typing import ConfigRecordValues
from flwr.client.mod import make_ffn
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    bytes_to_ndarray,
    ndarray_to_bytes,
    parameters_to_ndarrays,
)
from flwr.common.secure_aggregation.crypto.shamir import combine_shares
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_extract,
    get_parameters_shape,
    parameters_addition,
    parameters_mod,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import dequantize
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    Key,
    Stage,
)
from flwr.common.secure_aggregation.secaggplus_utils import (
    pseudo_rand_gen,
    share_keys_plaintext_separate_ps,
)
from flwr.compat.common import recorddict_compat as compat

from .secaggplus_pq_mod import SecAggPlusPQState, secaggplus_pq_mod


def _make_ctxt(node_id: int) -> Context:
    cfg = ConfigRecord(SecAggPlusPQState().to_dict())
    return Context(
        run_id=234,
        node_id=node_id,
        node_config={},
        state=RecordDict({RECORD_KEY_STATE: cfg}),
        run_config={},
    )


def _make_handler(
    ctxt: Context,
    parameters: Parameters,
    num_examples: int,
) -> Callable[[dict[str, ConfigRecordValues]], ConfigRecord]:
    def ffn(_msg: Message, _2: Context) -> Message:
        fitres = FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters,
            num_examples=num_examples,
            metrics={},
        )
        return Message(
            compat.fitres_to_recorddict(fitres, keep_input=True), reply_to=_msg
        )

    app = make_ffn(ffn, [secaggplus_pq_mod])

    def func(configs: dict[str, ConfigRecordValues]) -> ConfigRecord:
        in_msg = Message(
            RecordDict({RECORD_KEY_CONFIGS: ConfigRecord(configs)}),
            dst_node_id=ctxt.node_id,
            message_type=MessageType.TRAIN,
        )
        out_msg = app(in_msg, ctxt)
        return out_msg.content.config_records[RECORD_KEY_CONFIGS]

    return func


# pylint: disable-next=too-many-locals,too-many-statements,too-many-branches,too-many-arguments,too-many-positional-arguments
def _run_roundtrip(
    node_ids: list[int],
    dropped_node: int | list[int] | None,
    params_per_node: dict[int, Parameters],
    clipping_range: float = 3.0,
    target_range: int = 2**10,
    single_kem: bool = False,
) -> tuple[list[NDArray[Any]], NDArray[Any]]:
    """Run the protocol and return (actual aggregate, expected aggregate)."""
    dropped_nodes_list: list[int] = []
    if isinstance(dropped_node, int):
        dropped_nodes_list = [dropped_node]
    elif dropped_node is not None:
        dropped_nodes_list = list(dropped_node)
    active_nodes = [nid for nid in node_ids if nid not in dropped_nodes_list]

    num_shares = len(node_ids)
    threshold = max(2, len(active_nodes) - 1)
    mod_range = 2**30 - 2**10
    max_weight = float(len(node_ids))

    contexts: dict[int, Context] = {}
    handlers: dict[int, Callable[..., ConfigRecord]] = {}
    public_keys: dict[int, tuple[bytes, bytes]] = {}
    original_rd_seeds: dict[int, bytes] = {}

    for nid in node_ids:
        ctxt = _make_ctxt(node_id=nid)
        handler = _make_handler(ctxt, parameters=params_per_node[nid], num_examples=1)
        setup_configs: dict[str, ConfigRecordValues] = {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: num_shares,
            Key.SHARE_NUMBER: num_shares,
            Key.THRESHOLD: threshold,
            Key.CLIPPING_RANGE: clipping_range,
            Key.TARGET_RANGE: target_range,
            Key.MOD_RANGE: mod_range,
            Key.MAX_WEIGHT: max_weight,
            Key.SINGLE_KEM: single_kem,
        }
        handler(setup_configs)
        state = SecAggPlusPQState(**ctxt.state.config_records[RECORD_KEY_STATE])
        original_rd_seeds[nid] = state.rd_seed
        public_keys[nid] = (state.pk_pairwise, state.pk_share_enc)
        contexts[nid] = ctxt
        handlers[nid] = handler

    # SHARE_KEYS
    share_keys_results: dict[int, ConfigRecord] = {}
    for nid in node_ids:
        configs: dict[str, ConfigRecordValues] = {
            str(i): list(pks) for i, pks in public_keys.items()
        }
        configs[Key.STAGE] = Stage.SHARE_KEYS
        share_keys_results[nid] = handlers[nid](configs)

    # Server forwards encapsulations and encrypted rd_seed shares.
    forward_sk: dict[int, dict[str, list[Any]]] = {
        nid: {
            Key.SOURCE_LIST: [],
            Key.CIPHERTEXT_PAIRWISE_LIST: [],
            Key.CIPHERTEXT_SHARE_ENC_LIST: [],
            Key.CIPHERTEXT_LIST: [],
        }
        for nid in node_ids
    }
    for src in node_ids:
        res = share_keys_results[src]
        for dst, ct_pw, ct_se, ct in zip(
            cast(list[int], res[Key.DESTINATION_LIST]),
            cast(list[bytes], res[Key.CIPHERTEXT_PAIRWISE_LIST]),
            cast(list[bytes], res[Key.CIPHERTEXT_SHARE_ENC_LIST]),
            cast(list[bytes], res[Key.CIPHERTEXT_LIST]),
            strict=True,
        ):
            forward_sk[dst][Key.SOURCE_LIST].append(src)
            forward_sk[dst][Key.CIPHERTEXT_PAIRWISE_LIST].append(ct_pw)
            forward_sk[dst][Key.CIPHERTEXT_SHARE_ENC_LIST].append(ct_se)
            forward_sk[dst][Key.CIPHERTEXT_LIST].append(ct)

    # COLLECT_MASKED_VECTORS (all clients send masked vectors).
    collect_results: dict[int, ConfigRecord] = {}
    for nid in node_ids:
        f = forward_sk[nid]
        configs = {
            Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
            Key.SOURCE_LIST: f[Key.SOURCE_LIST],
            Key.CIPHERTEXT_PAIRWISE_LIST: f[Key.CIPHERTEXT_PAIRWISE_LIST],
            Key.CIPHERTEXT_SHARE_ENC_LIST: f[Key.CIPHERTEXT_SHARE_ENC_LIST],
            Key.CIPHERTEXT_LIST: f[Key.CIPHERTEXT_LIST],
        }
        collect_results[nid] = handlers[nid](configs)

    # Aggregate masked vectors from active clients.
    aggregate: list[NDArray[Any]] | None = None
    for nid in active_nodes:
        bytes_list = cast(list[bytes], collect_results[nid][Key.MASKED_PARAMETERS])
        vec = [bytes_to_ndarray(b) for b in bytes_list]
        if aggregate is None:
            aggregate = vec
        else:
            aggregate = [a + b for a, b in zip(aggregate, vec, strict=True)]
    assert aggregate is not None
    aggregate = [a % mod_range for a in aggregate]

    # Server forwards pairwise-secret bundles for the unmask stage.
    forward_ps_srcs: dict[int, list[int]] = {nid: [] for nid in node_ids}
    forward_ps_ciphertexts: dict[int, list[bytes]] = {nid: [] for nid in node_ids}
    for src in node_ids:
        res = collect_results[src]
        for dst, ct in zip(
            cast(list[int], res[Key.DESTINATION_LIST]),
            cast(list[bytes], res[Key.CIPHERTEXT_LIST]),
            strict=True,
        ):
            forward_ps_srcs[dst].append(src)
            forward_ps_ciphertexts[dst].append(ct)

    # UNMASK stage (only active clients respond).
    unmask_responses: dict[int, ConfigRecord] = {}
    for nid in active_nodes:
        active_neighbours = [m for m in active_nodes if m != nid]
        unmask_configs: dict[str, ConfigRecordValues] = {
            Key.STAGE: Stage.UNMASK,
            Key.ACTIVE_NODE_ID_LIST: active_neighbours,
            Key.DEAD_NODE_ID_LIST: dropped_nodes_list,
            Key.SOURCE_LIST: forward_ps_srcs[nid],
            Key.CIPHERTEXT_LIST: forward_ps_ciphertexts[nid],
        }
        unmask_responses[nid] = handlers[nid](unmask_configs)

    # Reconstruct rd_seed for active clients and pairwise secrets for dead.
    rd_seed_shares: dict[int, list[bytes]] = {nid: [] for nid in active_nodes}
    pairwise_secret_shares: dict[tuple[int, int], list[bytes]] = {}

    for nid in active_nodes:
        res = unmask_responses[nid]
        nids = cast(list[int], res[Key.NODE_ID_LIST])
        shares = cast(list[bytes], res[Key.SHARE_LIST])
        num_active_neighbours = len(active_nodes) - 1

        for idx, (owner_nid, share) in enumerate(zip(nids, shares, strict=True)):
            if idx < num_active_neighbours:
                rd_seed_shares[owner_nid].append(share)
            else:
                plaintext = share
                actual_src, _dst, ps_shares = share_keys_plaintext_separate_ps(
                    plaintext
                )
                for m_nid, ps_share in ps_shares:
                    key = (actual_src, m_nid)
                    if key not in pairwise_secret_shares:
                        pairwise_secret_shares[key] = []
                    pairwise_secret_shares[key].append(ps_share)

    # Remove private masks for active clients.
    for nid in active_nodes:
        assert len(rd_seed_shares[nid]) >= threshold
        rd_seed = combine_shares(rd_seed_shares[nid])
        assert rd_seed == original_rd_seeds[nid]
        private_mask = pseudo_rand_gen(
            rd_seed, mod_range, get_parameters_shape(aggregate)
        )
        aggregate = parameters_subtraction(aggregate, private_mask)

    # Remove pairwise masks contributed by dropped clients.
    for dead_nid in dropped_nodes_list:
        for neighbour_nid in active_nodes:
            key = (dead_nid, neighbour_nid)
            share_list = pairwise_secret_shares.get(key, [])
            assert len(share_list) >= threshold
            pw_secret = combine_shares(share_list)
            pairwise_mask = pseudo_rand_gen(
                pw_secret, mod_range, get_parameters_shape(aggregate)
            )
            if dead_nid > neighbour_nid:
                aggregate = parameters_addition(aggregate, pairwise_mask)
            else:
                aggregate = parameters_subtraction(aggregate, pairwise_mask)

    aggregate = parameters_mod(aggregate, mod_range)
    q_total_ratio, aggregate = factor_extract(aggregate)
    inv_dq_total_ratio = target_range / q_total_ratio
    offset = -(len(active_nodes) - 1) * clipping_range
    aggregated_vector: list[NDArray[Any]] = [
        (vec + offset) * inv_dq_total_ratio
        for vec in dequantize(aggregate, clipping_range, target_range)
    ]

    # The protocol scales each client by num_examples / max_weight, so the
    # decoded result is the weighted average of active clients.
    active_params: list[NDArray[Any]] = [
        parameters_to_ndarrays(params_per_node[nid])[0] for nid in active_nodes
    ]
    expected: NDArray[Any] = np.mean(np.stack(active_params), axis=0)
    return aggregated_vector, expected


def test_no_dropout_full_roundtrip() -> None:
    """Run PQ SecAgg+ with 5 clients and no dropout; verify the aggregate."""
    node_ids = list(range(5))
    params_per_node: dict[int, Parameters] = {
        nid: Parameters(
            tensors=[ndarray_to_bytes(np.full((4,), float(nid + 1), dtype=np.float64))],
            tensor_type="numpy.ndarray",
        )
        for nid in node_ids
    }
    aggregated_vector, expected = _run_roundtrip(node_ids, None, params_per_node)
    np.testing.assert_allclose(aggregated_vector[0], expected, rtol=1e-2, atol=1e-1)


def test_no_dropout_full_roundtrip_single_kem() -> None:
    """Run single-KEM PQ SecAgg+ with 5 clients and no dropout."""
    node_ids = list(range(5))
    params_per_node: dict[int, Parameters] = {
        nid: Parameters(
            tensors=[ndarray_to_bytes(np.full((4,), float(nid + 1), dtype=np.float64))],
            tensor_type="numpy.ndarray",
        )
        for nid in node_ids
    }
    aggregated_vector, expected = _run_roundtrip(
        node_ids, None, params_per_node, single_kem=True
    )
    np.testing.assert_allclose(aggregated_vector[0], expected, rtol=1e-2, atol=1e-1)


def test_drop_one_client_full_roundtrip() -> None:
    """Run PQ SecAgg+ with 5 clients and one dropout; verify the aggregate."""
    node_ids = list(range(5))
    params_per_node: dict[int, Parameters] = {
        nid: Parameters(
            tensors=[ndarray_to_bytes(np.full((4,), float(nid + 1), dtype=np.float64))],
            tensor_type="numpy.ndarray",
        )
        for nid in node_ids
    }
    aggregated_vector, expected = _run_roundtrip(node_ids, 2, params_per_node)
    np.testing.assert_allclose(aggregated_vector[0], expected, rtol=1e-2, atol=1e-1)


def test_drop_one_client_full_roundtrip_single_kem() -> None:
    """Run single-KEM PQ SecAgg+ with 5 clients and one dropout."""
    node_ids = list(range(5))
    params_per_node: dict[int, Parameters] = {
        nid: Parameters(
            tensors=[ndarray_to_bytes(np.full((4,), float(nid + 1), dtype=np.float64))],
            tensor_type="numpy.ndarray",
        )
        for nid in node_ids
    }
    aggregated_vector, expected = _run_roundtrip(
        node_ids, 2, params_per_node, single_kem=True
    )
    np.testing.assert_allclose(aggregated_vector[0], expected, rtol=1e-2, atol=1e-1)


def test_multiple_dropouts_full_roundtrip() -> None:
    """Run PQ SecAgg+ with 20 clients and 5 dropouts; verify the aggregate."""
    node_ids = list(range(20))
    dropped_nodes = [2, 5, 8, 11, 14]
    clipping_range = 25.0
    params_per_node: dict[int, Parameters] = {
        nid: Parameters(
            tensors=[ndarray_to_bytes(np.full((4,), float(nid + 1), dtype=np.float64))],
            tensor_type="numpy.ndarray",
        )
        for nid in node_ids
    }
    aggregated_vector, expected = _run_roundtrip(
        node_ids,
        dropped_nodes,
        params_per_node,
        clipping_range=clipping_range,
        target_range=2000,
    )
    np.testing.assert_allclose(aggregated_vector[0], expected, rtol=1e-10, atol=1e-9)


def test_multiple_dropouts_full_roundtrip_single_kem() -> None:
    """Run single-KEM PQ SecAgg+ with 20 clients and 5 dropouts."""
    node_ids = list(range(20))
    dropped_nodes = [2, 5, 8, 11, 14]
    clipping_range = 25.0
    params_per_node: dict[int, Parameters] = {
        nid: Parameters(
            tensors=[ndarray_to_bytes(np.full((4,), float(nid + 1), dtype=np.float64))],
            tensor_type="numpy.ndarray",
        )
        for nid in node_ids
    }
    aggregated_vector, expected = _run_roundtrip(
        node_ids,
        dropped_nodes,
        params_per_node,
        clipping_range=clipping_range,
        target_range=2000,
        single_kem=True,
    )
    np.testing.assert_allclose(aggregated_vector[0], expected, rtol=1e-10, atol=1e-9)
