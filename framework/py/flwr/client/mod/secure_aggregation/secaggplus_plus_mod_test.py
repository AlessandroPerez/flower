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
"""Unit tests for the SecAggPlusPlus client modifier."""


from typing import cast

import numpy as np

from flwr.app import ConfigRecord
from flwr.common import bytes_to_ndarray, ndarrays_to_parameters
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    parameters_addition,
    parameters_mod,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.crypto.shamir import combine_shares
from flwr.common.secure_aggregation.secaggplus_constants import Key, Stage
from flwr.common.secure_aggregation.crypto.pq_pke import aead_decrypt, aead_encrypt
from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen_secure

from .secaggplus_plus_mod import (
    SecAggPlusPlusState,
    _collect_masked_vectors,
    _setup,
    _share_keys,
    _unmask,
)


def _server_rewrap(
    outer: bytes, src: int, dst: int, server_keys: dict[int, bytes]
) -> bytes:
    """Simulate the server-side onion re-encryption layer."""
    ad = int.to_bytes(src, 8, "little", signed=False) + int.to_bytes(
        dst, 8, "little", signed=False
    )
    inner = aead_decrypt(server_keys[src], outer, ad)
    return aead_encrypt(server_keys[dst], inner, ad)


def _make_state(nid: int) -> SecAggPlusPlusState:
    state = SecAggPlusPlusState()
    state.nid = nid
    return state


def _setup_configs() -> ConfigRecord:
    return ConfigRecord(
        {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: 2,
            Key.SHARE_NUMBER: 3,
            Key.THRESHOLD: 2,
            Key.CLIPPING_RANGE: 5.0,
            Key.TARGET_RANGE: 10,
            Key.MOD_RANGE: 1024,
            Key.MAX_WEIGHT: 10.0,
        }
    )


class TestSecAggPlusPlusMod:
    """Tests for the SecAggPlusPlus client-side modifier."""

    def test_no_dropout_end_to_end(self) -> None:
        """Two clients can aggregate with all masks cancelling."""
        # Setup both clients.
        state1 = _make_state(1)
        state2 = _make_state(2)
        setup_cfg = _setup_configs()
        _setup(state1, setup_cfg)
        _setup(state2, setup_cfg)

        # Share keys: server forwards each client's public key to the other.
        share_cfg1 = ConfigRecord(
            {
                Key.STAGE: Stage.SHARE_KEYS,
                "1": state1.pk,
                "2": state2.pk,
            }
        )
        share_cfg2 = ConfigRecord(
            {
                Key.STAGE: Stage.SHARE_KEYS,
                "1": state1.pk,
                "2": state2.pk,
            }
        )
        res1 = _share_keys(state1, share_cfg1)
        res2 = _share_keys(state2, share_cfg2)

        # Route ciphertexts: client 1 receives client 2's message and vice versa.
        # The server would have stripped the sender-side onion layer and added
        # the receiver-side onion layer; simulate that here.
        srcs1 = cast(list[int], res1[Key.SOURCE_LIST])
        dsts1 = cast(list[int], res1[Key.DESTINATION_LIST])
        cts1 = cast(list[bytes], res1[Key.CIPHERTEXT_LIST])
        srcs2 = cast(list[int], res2[Key.SOURCE_LIST])
        dsts2 = cast(list[int], res2[Key.DESTINATION_LIST])
        cts2 = cast(list[bytes], res2[Key.CIPHERTEXT_LIST])

        server_keys = {1: state1.server_key, 2: state2.server_key}
        ct_for_1 = _server_rewrap(
            cts2[dsts2.index(1)], srcs2[dsts2.index(1)], 1, server_keys
        )
        ct_for_2 = _server_rewrap(
            cts1[dsts1.index(2)], srcs1[dsts1.index(2)], 2, server_keys
        )

        collect_cfg1 = ConfigRecord(
            {
                Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                Key.SOURCE_LIST: cast(list[int], [srcs2[dsts2.index(1)]]),
                Key.CIPHERTEXT_LIST: cast(list[bytes], [ct_for_1]),
            }
        )
        collect_cfg2 = ConfigRecord(
            {
                Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                Key.SOURCE_LIST: cast(list[int], [srcs1[dsts1.index(2)]]),
                Key.CIPHERTEXT_LIST: cast(list[bytes], [ct_for_2]),
            }
        )

        params1 = ndarrays_to_parameters([np.array([1.0, 2.0])])
        params2 = ndarrays_to_parameters([np.array([3.0, -1.0])])

        c_res1 = _collect_masked_vectors(state1, collect_cfg1, 10, params1)
        c_res2 = _collect_masked_vectors(state2, collect_cfg2, 10, params2)

        # Server-side aggregate: sum(c_S) - sum(sum_S).
        c1 = [
            bytes_to_ndarray(b)
            for b in cast(list[bytes], c_res1[Key.MASKED_PARAMETERS])
        ]
        s1 = [
            bytes_to_ndarray(b) for b in cast(list[bytes], c_res1[Key.SUM_DERIVED_KEYS])
        ]
        c2 = [
            bytes_to_ndarray(b)
            for b in cast(list[bytes], c_res2[Key.MASKED_PARAMETERS])
        ]
        s2 = [
            bytes_to_ndarray(b) for b in cast(list[bytes], c_res2[Key.SUM_DERIVED_KEYS])
        ]

        aggregate = parameters_addition(c1, c2)
        sum_s = parameters_addition(s1, s2)
        aggregate = parameters_subtraction(aggregate, sum_s)
        aggregate = parameters_mod(aggregate, 1024)

        # Unmask: collect u_seed shares for all active clients and reconstruct
        # every self mask from threshold shares.
        unmask_cfg1 = ConfigRecord(
            {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: [1, 2],
                Key.DEAD_NODE_ID_LIST: [],
            }
        )
        unmask_cfg2 = ConfigRecord(
            {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: [1, 2],
                Key.DEAD_NODE_ID_LIST: [],
            }
        )
        u_res1 = _unmask(state1, unmask_cfg1)
        u_res2 = _unmask(state2, unmask_cfg2)

        u_seed_shares: dict[int, list[bytes]] = {1: [], 2: []}
        for res in (u_res1, u_res2):
            nids = cast(list[int], res[Key.NODE_ID_LIST])
            shares = cast(list[bytes], res[Key.SHARE_LIST])
            for nid, share in zip(nids, shares, strict=True):
                u_seed_shares[nid].append(share)

        dims: list[tuple[int, ...]] = [(2,)]
        params_aggregate = [aggregate[1]]
        for nid in (1, 2):
            u_seed = combine_shares(u_seed_shares[nid])
            params_aggregate = parameters_subtraction(
                params_aggregate, pseudo_rand_gen_secure(u_seed, 1024, dims)
            )
        params_aggregate = parameters_mod(params_aggregate, 1024)
        aggregate[1] = params_aggregate[0]

        # Expected quantized sum with clipping_range=5, target_range=10.
        # Weights [1,2] -> [6,7]; [3,-1] -> [8,4]. factor_combine prepends [10].
        expected = [np.array([20]), np.array([14, 11])]
        np.testing.assert_array_equal(aggregate[0], expected[0])
        np.testing.assert_array_equal(aggregate[1], expected[1])

    def test_stage3_dropout_end_to_end(self) -> None:
        """Three clients aggregate; one drops after sending its masked vector.

        The server reconstructs every self mask from threshold shares, including
        the dropped client's.
        """
        # Setup three clients in an all-to-all neighbour graph.
        states = {nid: _make_state(nid) for nid in (1, 2, 3)}
        setup_cfg = ConfigRecord(
            {
                Key.STAGE: Stage.SETUP,
                Key.SAMPLE_NUMBER: 3,
                Key.SHARE_NUMBER: 3,
                Key.THRESHOLD: 2,
                Key.CLIPPING_RANGE: 5.0,
                Key.TARGET_RANGE: 10,
                Key.MOD_RANGE: 1024,
                Key.MAX_WEIGHT: 10.0,
            }
        )
        for state in states.values():
            _setup(state, setup_cfg)

        # Share keys with all public keys visible to every client.
        pks = {str(nid): state.pk for nid, state in states.items()}
        share_cfgs = {
            nid: ConfigRecord({Key.STAGE: Stage.SHARE_KEYS, **pks}) for nid in states
        }
        share_res = {
            nid: _share_keys(state, share_cfgs[nid]) for nid, state in states.items()
        }

        # Route ciphertexts to each client, simulating the server's onion
        # re-encryption layer.
        server_keys = {nid: state.server_key for nid, state in states.items()}
        collect_cfgs: dict[int, ConfigRecord] = {}
        for nid, _state in states.items():
            srcs: list[int] = []
            cts: list[bytes] = []
            for other_nid in states:
                if other_nid == nid:
                    continue
                res = share_res[other_nid]
                dsts = cast(list[int], res[Key.DESTINATION_LIST])
                cts_for = cast(list[bytes], res[Key.CIPHERTEXT_LIST])
                idx = dsts.index(nid)
                srcs.append(other_nid)
                cts.append(_server_rewrap(cts_for[idx], other_nid, nid, server_keys))
            collect_cfgs[nid] = ConfigRecord(
                {
                    Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                    Key.SOURCE_LIST: srcs,
                    Key.CIPHERTEXT_LIST: cts,
                }
            )

        params = {
            1: ndarrays_to_parameters([np.array([1.0, 2.0])]),
            2: ndarrays_to_parameters([np.array([3.0, -1.0])]),
            3: ndarrays_to_parameters([np.array([0.0, 1.0])]),
        }
        collect_res = {
            nid: _collect_masked_vectors(state, collect_cfgs[nid], 10, params[nid])
            for nid, state in states.items()
        }

        # Server-side aggregate over all three masked vectors.
        aggregate = [
            bytes_to_ndarray(b)
            for b in cast(list[bytes], collect_res[1][Key.MASKED_PARAMETERS])
        ]
        sum_derived = [
            bytes_to_ndarray(b)
            for b in cast(list[bytes], collect_res[1][Key.SUM_DERIVED_KEYS])
        ]
        for nid in (2, 3):
            c = [
                bytes_to_ndarray(b)
                for b in cast(list[bytes], collect_res[nid][Key.MASKED_PARAMETERS])
            ]
            s = [
                bytes_to_ndarray(b)
                for b in cast(list[bytes], collect_res[nid][Key.SUM_DERIVED_KEYS])
            ]
            aggregate = parameters_addition(aggregate, c)
            sum_derived = parameters_addition(sum_derived, s)
        aggregate = parameters_subtraction(aggregate, sum_derived)
        aggregate = parameters_mod(aggregate, 1024)

        # Client 3 drops before unmask, but is still in the active list sent to
        # clients 1 and 2. They return u_seed shares for all active clients.
        u_seed_shares: dict[int, list[bytes]] = {1: [], 2: [], 3: []}
        for survivor_nid in (1, 2):
            unmask_cfg = ConfigRecord(
                {
                    Key.STAGE: Stage.UNMASK,
                    Key.ACTIVE_NODE_ID_LIST: [1, 2, 3],
                    Key.DEAD_NODE_ID_LIST: [],
                }
            )
            res = _unmask(states[survivor_nid], unmask_cfg)
            nids = cast(list[int], res[Key.NODE_ID_LIST])
            shares = cast(list[bytes], res[Key.SHARE_LIST])
            for nid, share in zip(nids, shares, strict=True):
                u_seed_shares[nid].append(share)

        # Reconstruct every self mask and remove it.
        dims: list[tuple[int, ...]] = [(2,)]
        params_aggregate = [aggregate[1]]
        for nid in (1, 2, 3):
            u_seed = combine_shares(u_seed_shares[nid])
            params_aggregate = parameters_subtraction(
                params_aggregate, pseudo_rand_gen_secure(u_seed, 1024, dims)
            )
        aggregate[1] = parameters_mod(params_aggregate, 1024)[0]
        aggregate = parameters_mod(aggregate, 1024)

        # Expected quantized sum: [1,2]->[6,7], [3,-1]->[8,4], [0,1]->[5,6].
        expected = [np.array([30]), np.array([19, 17])]
        np.testing.assert_array_equal(aggregate[0], expected[0])
        np.testing.assert_array_equal(aggregate[1], expected[1])

    def test_unmask_partitions_stage2_and_stage3_dropouts(self) -> None:
        """_unmask returns u_seed shares for active clients (including stage-3
        dropouts) and b_seed shares for stage-2 dead clients, plus the e_S
        correction for stage-2 dead only.
        """
        state = _make_state(1)
        state.mod_range = 1024
        state.dimensions_list = [(2,)]
        state.received_keys = {}
        state.received_b_shares = {}
        state.received_u_shares = {}
        state.u_seed_shares = {}
        state.derived_keys = {}

        # Client 1 is ourselves: store our own u_seed share.
        state.u_seed_shares[1] = b"u-share-self"

        # Client 2 is a stage-3 dropout: it is in the active list and we have a
        # u_seed share for it.
        state.received_keys[2] = b"key-2"
        state.received_u_shares[2] = b"u-share-2"

        # Client 3 is a stage-2 dropout: it is in the dead list and we have a
        # b_seed share for it.
        state.derived_keys[3] = b"\x01" * 32
        state.received_b_shares[3] = b"b-share-3"

        configs = ConfigRecord(
            {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: [1, 2],
                Key.DEAD_NODE_ID_LIST: [3],
            }
        )
        res = _unmask(state, configs)

        # Active clients first, then stage-2 dead clients.
        assert cast(list[int], res[Key.NODE_ID_LIST]) == [1, 2, 3]
        assert cast(list[bytes], res[Key.SHARE_LIST]) == [
            b"u-share-self",
            b"u-share-2",
            b"b-share-3",
        ]

        # e_S should contain the mask derived for the stage-2 dropout only.
        e_s = [bytes_to_ndarray(b) for b in cast(list[bytes], res[Key.E_VALUE])]
        expected_e = pseudo_rand_gen_secure(b"\x01" * 32, 1024, [(2,)])
        np.testing.assert_array_equal(e_s[0], expected_e[0])
