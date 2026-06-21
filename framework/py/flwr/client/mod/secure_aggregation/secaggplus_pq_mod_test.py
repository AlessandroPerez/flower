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
"""Tests for the post-quantum SecAgg+ client modifier."""


from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.app.typing import ConfigRecordValues
from flwr.client.mod import make_ffn
from flwr.common import Code, FitRes, Parameters, Status, ndarray_to_bytes
from flwr.common.secure_aggregation.crypto.pq_kem import (
    decapsulate_secret,
    derive_pairwise_secret,
    encapsulate,
    generate_key_pair,
)
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    Key,
    Stage,
)
from flwr.compat.common import recorddict_compat as compat

from .secaggplus_pq_mod import SecAggPlusPQState, check_configs, secaggplus_pq_mod


def _make_ctxt(node_id: int = 123) -> Context:
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
    parameters: Parameters | None = None,
    num_examples: int = 1,
) -> Callable[[dict[str, ConfigRecordValues]], ConfigRecord]:
    if parameters is None:
        parameters = Parameters(
            tensors=[ndarray_to_bytes(np.ones(4, dtype=np.float64))],
            tensor_type="numpy.ndarray",
        )

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


def _setup_state(
    handler: Callable[[dict[str, ConfigRecordValues]], ConfigRecord],
    params: dict[str, ConfigRecordValues] | None = None,
) -> ConfigRecord:
    default: dict[str, ConfigRecordValues] = {
        Key.STAGE: Stage.SETUP,
        Key.SAMPLE_NUMBER: 3,
        Key.SHARE_NUMBER: 3,
        Key.THRESHOLD: 2,
        Key.CLIPPING_RANGE: 3.0,
        Key.TARGET_RANGE: 2**10,
        Key.MOD_RANGE: 2**30 - 2**10,
        Key.MAX_WEIGHT: 100.0,
    }
    if params:
        default.update(params)
    return handler(default)


def test_setup_stage() -> None:
    """Setup stage returns two public keys and stores key material."""
    ctxt = _make_ctxt(node_id=7)
    handler = _make_handler(ctxt)

    out = _setup_state(handler)

    assert isinstance(out[Key.PUBLIC_KEY_PAIRWISE], bytes)
    assert isinstance(out[Key.PUBLIC_KEY_SHARE_ENC], bytes)
    state = SecAggPlusPQState(**ctxt.state.config_records[RECORD_KEY_STATE])
    assert state.nid == 7
    assert len(state.rd_seed) == 32
    assert state.pk_pairwise == out[Key.PUBLIC_KEY_PAIRWISE]
    assert state.pk_share_enc == out[Key.PUBLIC_KEY_SHARE_ENC]


def test_setup_stage_single_kem() -> None:
    """Single-KEM setup uses one key pair for both purposes."""
    ctxt = _make_ctxt(node_id=7)
    handler = _make_handler(ctxt)

    out = _setup_state(handler, {Key.SINGLE_KEM: True})

    state = SecAggPlusPQState(**ctxt.state.config_records[RECORD_KEY_STATE])
    assert state.single_kem is True
    assert state.pk_pairwise == state.pk_share_enc
    assert state.sk_pairwise == state.sk_share_enc
    assert out[Key.PUBLIC_KEY_PAIRWISE] == out[Key.PUBLIC_KEY_SHARE_ENC]


def test_check_configs_setup_missing_key() -> None:
    """Missing required setup key raises KeyError."""
    with pytest.raises(KeyError):
        check_configs(Stage.SETUP, ConfigRecord({Key.STAGE: Stage.SETUP}))


def test_check_configs_setup_bad_type() -> None:
    """Setup key with wrong type raises TypeError."""
    with pytest.raises(TypeError):
        check_configs(
            Stage.SETUP,
            ConfigRecord({Key.STAGE: Stage.SETUP, Key.SAMPLE_NUMBER: "3"}),
        )


def test_share_keys_stage() -> None:
    """Share-keys stage produces encapsulations and encrypted rd_seed shares."""
    node_ids = [0, 1, 2]
    public_keys: dict[int, tuple[bytes, bytes]] = {}
    contexts: dict[int, Context] = {}
    handlers: dict[int, Callable[..., ConfigRecord]] = {}

    for nid in node_ids:
        ctxt = _make_ctxt(node_id=nid)
        handler = _make_handler(ctxt)
        _setup_state(handler, {Key.STAGE: Stage.SETUP})
        state = SecAggPlusPQState(**ctxt.state.config_records[RECORD_KEY_STATE])
        public_keys[nid] = (state.pk_pairwise, state.pk_share_enc)
        contexts[nid] = ctxt
        handlers[nid] = handler

    # Run SHARE_KEYS for client 0.
    configs: dict[str, ConfigRecordValues] = {
        str(nid): list(pks) for nid, pks in public_keys.items()
    }
    configs[Key.STAGE] = Stage.SHARE_KEYS
    out = handlers[0](configs)

    assert Key.SOURCE_LIST in out
    assert Key.DESTINATION_LIST in out
    assert len(cast(list[int], out[Key.SOURCE_LIST])) == 2
    assert len(cast(list[bytes], out[Key.CIPHERTEXT_PAIRWISE_LIST])) == 2
    assert len(cast(list[bytes], out[Key.CIPHERTEXT_SHARE_ENC_LIST])) == 2
    assert len(cast(list[bytes], out[Key.CIPHERTEXT_LIST])) == 2


def test_share_keys_stage_single_kem() -> None:
    """Single-KEM share-keys stage produces one ciphertext per neighbour."""
    node_ids = [0, 1, 2]
    public_keys: dict[int, tuple[bytes, bytes]] = {}
    contexts: dict[int, Context] = {}
    handlers: dict[int, Callable[..., ConfigRecord]] = {}

    for nid in node_ids:
        ctxt = _make_ctxt(node_id=nid)
        handler = _make_handler(ctxt)
        _setup_state(handler, {Key.STAGE: Stage.SETUP, Key.SINGLE_KEM: True})
        state = SecAggPlusPQState(**ctxt.state.config_records[RECORD_KEY_STATE])
        public_keys[nid] = (state.pk_pairwise, state.pk_share_enc)
        contexts[nid] = ctxt
        handlers[nid] = handler

    configs: dict[str, ConfigRecordValues] = {
        str(nid): list(pks) for nid, pks in public_keys.items()
    }
    configs[Key.STAGE] = Stage.SHARE_KEYS
    out = handlers[0](configs)

    ct_pw = cast(list[bytes], out[Key.CIPHERTEXT_PAIRWISE_LIST])
    ct_se = cast(list[bytes], out[Key.CIPHERTEXT_SHARE_ENC_LIST])
    assert len(ct_pw) == 2
    assert ct_pw == ct_se


def test_pairwise_secret_symmetric() -> None:
    """Two clients derive the same pairwise secret from opposite encapsulations."""
    pk_a, sk_a = generate_key_pair()
    pk_b, sk_b = generate_key_pair()

    ss_ab, ct_ab = encapsulate(pk_b)
    ss_ba, ct_ba = encapsulate(pk_a)

    ss_ab_received = decapsulate_secret(ct_ab, sk_b)
    ss_ba_received = decapsulate_secret(ct_ba, sk_a)

    pw_a = derive_pairwise_secret(ss_ab, ss_ba_received)
    pw_b = derive_pairwise_secret(ss_ba, ss_ab_received)
    assert pw_a == pw_b


def test_collect_masked_vectors_threshold_check() -> None:
    """Collect stage requires at least threshold-1 incoming neighbours."""
    ctxt = _make_ctxt(node_id=0)
    handler = _make_handler(ctxt)
    _setup_state(
        handler,
        {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: 3,
            Key.SHARE_NUMBER: 3,
            Key.THRESHOLD: 2,
            Key.CLIPPING_RANGE: 3.0,
            Key.TARGET_RANGE: 2**10,
            Key.MOD_RANGE: 2**30 - 2**10,
            Key.MAX_WEIGHT: 3.0,
        },
    )

    configs: dict[str, ConfigRecordValues] = {
        Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
        Key.SOURCE_LIST: [],
        Key.CIPHERTEXT_PAIRWISE_LIST: [],
        Key.CIPHERTEXT_SHARE_ENC_LIST: [],
        Key.CIPHERTEXT_LIST: [],
    }
    with pytest.raises(ValueError):
        handler(configs)


def test_unmask_stage_format() -> None:  # pylint: disable=too-many-locals
    """Unmask stage returns one share per active and dead neighbour."""
    node_ids = [0, 1, 2]
    contexts: dict[int, Context] = {}
    handlers: dict[int, Callable[..., ConfigRecord]] = {}
    public_keys: dict[int, tuple[bytes, bytes]] = {}

    for nid in node_ids:
        ctxt = _make_ctxt(node_id=nid)
        handler = _make_handler(ctxt)
        _setup_state(
            handler,
            {
                Key.STAGE: Stage.SETUP,
                Key.SAMPLE_NUMBER: 3,
                Key.SHARE_NUMBER: 3,
                Key.THRESHOLD: 2,
                Key.CLIPPING_RANGE: 3.0,
                Key.TARGET_RANGE: 2**10,
                Key.MOD_RANGE: 2**30 - 2**10,
                Key.MAX_WEIGHT: 3.0,
            },
        )
        state = SecAggPlusPQState(**ctxt.state.config_records[RECORD_KEY_STATE])
        public_keys[nid] = (state.pk_pairwise, state.pk_share_enc)
        contexts[nid] = ctxt
        handlers[nid] = handler

    # SHARE_KEYS
    share_keys_results: dict[int, ConfigRecord] = {}
    for nid in node_ids:
        sk_configs: dict[str, ConfigRecordValues] = {
            str(i): list(pks) for i, pks in public_keys.items()
        }
        sk_configs[Key.STAGE] = Stage.SHARE_KEYS
        share_keys_results[nid] = handlers[nid](sk_configs)

    # Server forwards for COLLECT
    forward: dict[int, dict[str, list[int] | list[bytes]]] = {
        i: {
            Key.SOURCE_LIST: [],
            Key.CIPHERTEXT_PAIRWISE_LIST: [],
            Key.CIPHERTEXT_SHARE_ENC_LIST: [],
            Key.CIPHERTEXT_LIST: [],
        }
        for i in node_ids
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
            cast(list[int], forward[dst][Key.SOURCE_LIST]).append(src)
            cast(list[bytes], forward[dst][Key.CIPHERTEXT_PAIRWISE_LIST]).append(ct_pw)
            cast(list[bytes], forward[dst][Key.CIPHERTEXT_SHARE_ENC_LIST]).append(ct_se)
            cast(list[bytes], forward[dst][Key.CIPHERTEXT_LIST]).append(ct)

    # COLLECT_MASKED_VECTORS
    collect_results: dict[int, ConfigRecord] = {}
    for nid in node_ids:
        f = forward[nid]
        cm_configs: dict[str, ConfigRecordValues] = {
            Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
            Key.SOURCE_LIST: f[Key.SOURCE_LIST],
            Key.CIPHERTEXT_PAIRWISE_LIST: f[Key.CIPHERTEXT_PAIRWISE_LIST],
            Key.CIPHERTEXT_SHARE_ENC_LIST: f[Key.CIPHERTEXT_SHARE_ENC_LIST],
            Key.CIPHERTEXT_LIST: f[Key.CIPHERTEXT_LIST],
        }
        collect_results[nid] = handlers[nid](cm_configs)

    # Server forwards for UNMASK: pairwise-secret bundles.
    forward_ps_srcs: dict[int, list[int]] = {i: [] for i in node_ids}
    forward_ps_ciphertexts: dict[int, list[bytes]] = {i: [] for i in node_ids}
    for src in node_ids:
        res = collect_results[src]
        for dst, ct in zip(
            cast(list[int], res[Key.DESTINATION_LIST]),
            cast(list[bytes], res[Key.CIPHERTEXT_LIST]),
            strict=True,
        ):
            forward_ps_srcs[dst].append(src)
            forward_ps_ciphertexts[dst].append(ct)

    # UNMASK stage for client 0 with client 1 active and client 2 dead.
    unmask_configs: dict[str, ConfigRecordValues] = {
        Key.STAGE: Stage.UNMASK,
        Key.ACTIVE_NODE_ID_LIST: [0, 1],
        Key.DEAD_NODE_ID_LIST: [2],
        Key.SOURCE_LIST: forward_ps_srcs[0],
        Key.CIPHERTEXT_LIST: forward_ps_ciphertexts[0],
    }
    out = handlers[0](unmask_configs)

    assert len(cast(list[int], out[Key.NODE_ID_LIST])) == 3
    assert len(cast(list[bytes], out[Key.SHARE_LIST])) == 3

    # First len(active) shares are rd_seed shares; the rest are pairwise-secret
    # bundles containing Shamir shares for dead neighbours.
    active_shares = cast(list[bytes], out[Key.SHARE_LIST])[:2]
    ps_bundles = cast(list[bytes], out[Key.SHARE_LIST])[2:]
    assert all(isinstance(s, bytes) for s in active_shares)
    assert all(isinstance(b, bytes) for b in ps_bundles)


def test_state_configrecord_roundtrip_preserves_tuple_dicts() -> None:
    """State serialisation to/from ConfigRecord preserves tuple-valued dicts."""
    state = SecAggPlusPQState()
    state.nid = 7
    state.public_keys_dict = {
        1: (b"pk1a" * 32, b"pk1b" * 32),
        2: (b"pk2a" * 32, b"pk2b" * 32),
        3: (b"pk3a" * 32, b"pk3b" * 32),
    }
    state.shared_secrets_sent = {
        1: (b"ss1a" * 32, b"ss1b" * 32),
        2: (b"ss2a" * 32, b"ss2b" * 32),
        3: (b"ss3a" * 32, b"ss3b" * 32),
    }
    state.shared_secrets_received = {
        1: (b"rs1a" * 32, b"rs1b" * 32),
        2: (b"rs2a" * 32, b"rs2b" * 32),
        3: (b"rs3a" * 32, b"rs3b" * 32),
    }

    cfg = ConfigRecord(state.to_dict())
    restored = SecAggPlusPQState(**cfg)

    assert restored.public_keys_dict == state.public_keys_dict
    assert restored.shared_secrets_sent == state.shared_secrets_sent
    assert restored.shared_secrets_received == state.shared_secrets_received
