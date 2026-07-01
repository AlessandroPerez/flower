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
"""Workflow for the SecAggPlusPlus protocol."""


from __future__ import annotations

from dataclasses import dataclass, field
from logging import DEBUG, ERROR, INFO, WARN
from time import perf_counter
from typing import Any, cast

import flwr.compat.common.recorddict_compat as compat
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.common import FitRes, NDArrays, bytes_to_ndarray, log, ndarrays_to_parameters
from flwr.common.secure_aggregation.crypto.pq_pke import aead_decrypt, aead_encrypt
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
    Key,
    Stage,
)
from flwr.common.secure_aggregation.secaggplus_utils import (
    derive_pairwise_key,
    pseudo_rand_gen_secure,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.compat.legacy_context import LegacyContext
from flwr.serverapp.grid import Grid

from ..constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD
from ..constant import Key as WorkflowKey


def _make_ad(src_node_id: int, dst_node_id: int) -> bytes:
    """Encode (source, destination) as AEAD associated data."""
    return int.to_bytes(src_node_id, 8, "little", signed=False) + int.to_bytes(
        dst_node_id, 8, "little", signed=False
    )


@dataclass
class WorkflowState:  # pylint: disable=R0902
    """The state of the SecAggPlusPlus protocol."""

    nid_to_proxies: dict[int, ClientProxy] = field(default_factory=dict)
    nid_to_fitins: dict[int, RecordDict] = field(default_factory=dict)
    sampled_node_ids: set[int] = field(default_factory=set)
    active_node_ids: set[int] = field(default_factory=set)
    num_shares: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    quantization_range: int = 0
    mod_range: int = 0
    max_weight: float = 0.0
    nid_to_neighbours: dict[int, set[int]] = field(default_factory=dict)
    nid_to_publickeys: dict[int, bytes] = field(default_factory=dict)
    nid_to_serverkeys: dict[int, bytes] = field(default_factory=dict)
    forward_srcs: dict[int, list[int]] = field(default_factory=dict)
    forward_ciphertexts: dict[int, list[bytes]] = field(default_factory=dict)
    aggregate_ndarrays: NDArrays = field(default_factory=list)
    sum_derived: dict[int, NDArrays] = field(default_factory=dict)
    masked_params: dict[int, NDArrays] = field(default_factory=dict)
    legacy_results: list[tuple[ClientProxy, FitRes]] = field(default_factory=list)
    failures: list[Exception] = field(default_factory=list)
    timing_log: dict[str, float] = field(default_factory=dict)


class SecAggPlusPlusWorkflow:  # pylint: disable=too-many-instance-attributes
    """The SecAggPlusPlus workflow.

    This workflow implements the seed-based post-quantum SecAgg+ variant
    described in ``pqc_aggregation.md``: each client derives pairwise masks
    from a master seed shared with neighbours via a PQ PKE, and a separate
    self mask is removed after the server commits to the participant set.
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_shares: int | float,
        reconstruction_threshold: int | float,
        *,
        max_weight: float = 1000.0,
        clipping_range: float = 8.0,
        quantization_range: int = 4194304,
        modulus_range: int = 4294967296,
        timeout: float | None = None,
    ) -> None:
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.max_weight = max_weight
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range
        self.modulus_range = modulus_range
        self.timeout = timeout

        self._check_init_params()

    def __call__(self, grid: Grid, context: Context) -> None:
        """Run the SecAggPlusPlus protocol."""
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )
        state = WorkflowState()
        state.timing_log: dict[str, float] = {}

        steps = (
            ("Setup", self.setup_stage),
            ("ShareKeys", self.share_keys_stage),
            ("CollectMasked", self.collect_masked_vectors_stage),
            ("Unmask", self.unmask_stage),
        )
        round_num = cast(
            int,
            context.state.config_records[MAIN_CONFIGS_RECORD][
                WorkflowKey.CURRENT_ROUND
            ],
        )
        _call_t0 = perf_counter()
        log(INFO, "SecAggPlusPlus secure aggregation commencing (round %s).", round_num)
        for name, step in steps:
            t0 = perf_counter()
            ok = step(grid, context, state)
            state.timing_log[name] = perf_counter() - t0
            if not ok:
                log(INFO, "SecAggPlusPlus secure aggregation halted.")
                return
        _call_elapsed = perf_counter() - _call_t0
        log(
            INFO,
            "SecAggPlusPlus timing round %s: total=%.4fs stages=%s",
            round_num,
            _call_elapsed,
            state.timing_log,
        )

    def _check_init_params(self) -> None:  # pylint: disable=R0912
        # Check `num_shares`
        if not isinstance(self.num_shares, (int | float)):
            raise TypeError("`num_shares` must be of type int or float.")
        if isinstance(self.num_shares, int):
            if self.num_shares == 1:
                self.num_shares = 1.0
            elif self.num_shares <= 2:
                raise ValueError("`num_shares` as an integer must be greater than 2.")
            elif self.num_shares > self.modulus_range / self.quantization_range:
                log(
                    WARN,
                    "A `num_shares` larger than `modulus_range / quantization_range` "
                    "will potentially cause overflow when computing the aggregated "
                    "model parameters.",
                )
        elif self.num_shares <= 0:
            raise ValueError("`num_shares` as a float must be greater than 0.")

        # Check `reconstruction_threshold`
        if not isinstance(self.reconstruction_threshold, (int | float)):
            raise TypeError("`reconstruction_threshold` must be of type int or float.")
        if isinstance(self.reconstruction_threshold, int):
            if self.reconstruction_threshold == 1:
                self.reconstruction_threshold = 1.0
            elif isinstance(self.num_shares, int):
                if self.reconstruction_threshold >= self.num_shares:
                    raise ValueError(
                        "`reconstruction_threshold` must be less than `num_shares`."
                    )
        else:
            if not 0 < self.reconstruction_threshold <= 1:
                raise ValueError(
                    "If `reconstruction_threshold` is a float, "
                    "it must be greater than 0 and less than or equal to 1."
                )

        # Check `max_weight`
        if self.max_weight <= 0:
            raise ValueError("`max_weight` must be greater than 0.")

        # Check `quantization_range`
        if not isinstance(self.quantization_range, int) or self.quantization_range <= 0:
            raise ValueError(
                "`quantization_range` must be an integer and greater than 0."
            )

        # Check `modulus_range`
        if (
            not isinstance(self.modulus_range, int)
            or self.modulus_range <= self.quantization_range
        ):
            raise ValueError(
                "`modulus_range` must be an integer and "
                "greater than `quantization_range`."
            )
        if bin(self.modulus_range).count("1") != 1:
            raise ValueError("`modulus_range` must be a power of 2.")

    def _check_threshold(self, state: WorkflowState) -> bool:
        for node_id in state.sampled_node_ids:
            active_neighbors = state.nid_to_neighbours[node_id] & state.active_node_ids
            if len(active_neighbors) < state.threshold:
                log(ERROR, "Insufficient available nodes.")
                return False
        return True

    def setup_stage(  # pylint: disable=R0912, R0914, R0915
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'setup' stage."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])
        parameters = compat.arrayrecord_to_parameters(
            context.state.array_records[MAIN_PARAMS_RECORD],
            keep_input=True,
        )
        proxy_fitins_lst = context.strategy.configure_fit(
            current_round, parameters, context.client_manager
        )
        if not proxy_fitins_lst:
            log(INFO, "configure_fit: no clients selected, cancel")
            return False
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(proxy_fitins_lst),
            context.client_manager.num_available(),
        )

        state.nid_to_fitins = {
            proxy.node_id: compat.fitins_to_recorddict(fitins, True)
            for proxy, fitins in proxy_fitins_lst
        }
        state.nid_to_proxies = {proxy.node_id: proxy for proxy, _ in proxy_fitins_lst}

        sampled_node_ids = list(state.nid_to_fitins.keys())
        num_samples = len(sampled_node_ids)
        if num_samples < 2:
            log(ERROR, "The number of samples should be greater than 1.")
            return False

        if isinstance(self.num_shares, float):
            state.num_shares = round(self.num_shares * num_samples)
            if state.num_shares < num_samples and state.num_shares & 1 == 0:
                state.num_shares += 1
            if state.num_shares <= 2:
                state.num_shares = num_samples
        else:
            state.num_shares = self.num_shares

        if isinstance(self.reconstruction_threshold, float):
            state.threshold = max(
                2, round(self.reconstruction_threshold * state.num_shares)
            )
        else:
            state.threshold = self.reconstruction_threshold

        state.num_shares = min(state.num_shares, num_samples)
        state.num_shares = min(num_samples, max(state.num_shares, 3))
        if state.num_shares % 2 == 0:
            state.num_shares -= 1

        if not 1 < state.threshold < state.num_shares:
            log(ERROR, "Invalid threshold / num_shares combination.")
            return False

        state.clipping_range = self.clipping_range
        state.quantization_range = self.quantization_range
        state.mod_range = self.modulus_range
        state.max_weight = self.max_weight
        sa_params_dict = {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: num_samples,
            Key.SHARE_NUMBER: state.num_shares,
            Key.THRESHOLD: state.threshold,
            Key.CLIPPING_RANGE: self.clipping_range,
            Key.TARGET_RANGE: self.quantization_range,
            Key.MOD_RANGE: state.mod_range,
            Key.MAX_WEIGHT: self.max_weight,
        }

        # Use a deterministic ordering so the neighbour graph is stable across
        # rounds.  This lets clients reuse the ML-KEM pairwise secrets they
        # establish in round 1 instead of re-running the KEM every round.
        sampled_node_ids = sorted(sampled_node_ids)
        half_share = state.num_shares >> 1
        state.nid_to_neighbours = {
            nid: {
                sampled_node_ids[(idx + offset) % num_samples]
                for offset in range(-half_share, half_share + 1)
            }
            for idx, nid in enumerate(sampled_node_ids)
        }

        state.sampled_node_ids = set(sampled_node_ids)
        state.active_node_ids = set(sampled_node_ids)

        cfg_record = ConfigRecord(sa_params_dict)  # type: ignore
        content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})

        def make(nid: int) -> Message:
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(
            DEBUG,
            "[SecAggPlusPlus Stage 0] Sending configurations to %s clients.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [make(node_id) for node_id in state.active_node_ids], timeout=self.timeout
        )
        state.active_node_ids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[SecAggPlusPlus Stage 0] Received public keys from %s clients.",
            len(state.active_node_ids),
        )

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            key_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            node_id = msg.metadata.src_node_id
            pk = cast(bytes, key_dict[Key.PUBLIC_KEY_PAIRWISE])
            server_key = cast(bytes, key_dict[Key.SERVER_KEY])
            state.nid_to_publickeys[node_id] = pk
            state.nid_to_serverkeys[node_id] = server_key

        return self._check_threshold(state)

    def share_keys_stage(  # pylint: disable=R0914
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'share keys' stage (stage 1)."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])
        # Public keys only need to be distributed in round 1; afterwards clients
        # reuse the cached ML-KEM pairwise secrets.
        first_round = current_round == 1

        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid] & state.active_node_ids
            cfg_dict: dict[str, Any] = {Key.STAGE: Stage.SHARE_KEYS}
            if first_round:
                cfg_dict.update(
                    {str(nid): state.nid_to_publickeys[nid] for nid in neighbours}
                )
            else:
                cfg_dict[Key.NEIGHBOUR_LIST] = list(neighbours)
            cfg_record = ConfigRecord(cfg_dict)
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(current_round),
            )

        log(
            DEBUG,
            "[SecAggPlusPlus Stage 1] Forwarding public keys to %s clients.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [make(node_id) for node_id in state.active_node_ids], timeout=self.timeout
        )
        state.active_node_ids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[SecAggPlusPlus Stage 1] Received encrypted payloads from %s clients.",
            len(state.active_node_ids),
        )

        srcs: list[int] = []
        dsts: list[int] = []
        ciphertexts: list[bytes] = []

        fwd_ciphertexts: dict[int, list[bytes]] = {
            nid: [] for nid in state.active_node_ids
        }
        fwd_srcs: dict[int, list[int]] = {nid: [] for nid in state.active_node_ids}

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            dst_lst = cast(list[int], res_dict[Key.DESTINATION_LIST])
            src_lst = cast(list[int], res_dict[Key.SOURCE_LIST])
            ctxt_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_LIST])

            for src, dst, ciphertext in zip(src_lst, dst_lst, ctxt_lst, strict=True):
                srcs.append(src)
                dsts.append(dst)
                ciphertexts.append(ciphertext)

        for src, dst, ciphertext in zip(srcs, dsts, ciphertexts, strict=True):
            if dst in fwd_ciphertexts:
                ad = _make_ad(src, dst)
                inner = aead_decrypt(state.nid_to_serverkeys[src], ciphertext, ad)
                rewrapped = aead_encrypt(state.nid_to_serverkeys[dst], inner, ad)
                fwd_ciphertexts[dst].append(rewrapped)
                fwd_srcs[dst].append(src)

        state.forward_srcs = fwd_srcs
        state.forward_ciphertexts = fwd_ciphertexts

        return self._check_threshold(state)

    def collect_masked_vectors_stage(  # pylint: disable=R0914
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'collect masked vectors' stage (stage 2)."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]

        def make(nid: int) -> Message:
            cfg_dict = {
                Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                Key.SOURCE_LIST: state.forward_srcs[nid],
                Key.CIPHERTEXT_LIST: state.forward_ciphertexts[nid],
            }
            cfg_record = ConfigRecord(cfg_dict)  # type: ignore
            content = state.nid_to_fitins[nid]
            content.config_records[RECORD_KEY_CONFIGS] = cfg_record
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(
            DEBUG,
            "[SecAggPlusPlus Stage 2] Forwarding payloads to %s clients.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [make(node_id) for node_id in state.active_node_ids], timeout=self.timeout
        )
        state.active_node_ids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[SecAggPlusPlus Stage 2] Received masked vectors from %s clients.",
            len(state.active_node_ids),
        )

        del state.forward_srcs, state.forward_ciphertexts, state.nid_to_fitins

        aggregate: NDArrays | None = None
        t_deser = 0.0
        t_sum = 0.0
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            t0 = perf_counter()
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            bytes_list = cast(list[bytes], res_dict[Key.MASKED_PARAMETERS])
            c_s = [bytes_to_ndarray(b) for b in bytes_list]
            sum_s_bytes = cast(list[bytes], res_dict[Key.SUM_DERIVED_KEYS])
            sum_s = [bytes_to_ndarray(b) for b in sum_s_bytes]
            t_deser += perf_counter() - t0

            state.masked_params[msg.metadata.src_node_id] = c_s
            state.sum_derived[msg.metadata.src_node_id] = sum_s

            t0 = perf_counter()
            if aggregate is None:
                aggregate = c_s
            else:
                aggregate = parameters_addition(aggregate, c_s)
            t_sum += perf_counter() - t0

            # Backward compatibility with Strategy.
            fitres = compat.recorddict_to_fitres(msg.content, True)
            proxy = state.nid_to_proxies[msg.metadata.src_node_id]
            state.legacy_results.append((proxy, fitres))

        if aggregate is None:
            log(ERROR, "No masked vectors received.")
            return False

        t0 = perf_counter()
        # Subtract all sum_S vectors.
        for sum_s in state.sum_derived.values():
            aggregate = parameters_subtraction(aggregate, sum_s)
        t_sub = perf_counter() - t0

        aggregate = parameters_mod(aggregate, state.mod_range)
        state.timing_log["Stage2_deser"] = t_deser
        state.timing_log["Stage2_sum"] = t_sum
        state.timing_log["Stage2_sub"] = t_sub
        state.aggregate_ndarrays = aggregate

        return self._check_threshold(state)

    def unmask_stage(  # pylint: disable=R0912, R0914, R0915
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'unmask' stage (stage 3).

        ``stage2_dead`` are sampled clients that never sent a masked vector.
        Their pairwise masks with surviving clients remain in the aggregate and
        must be cancelled. Clients that sent a masked vector but dropped before
        unmask are ``stage3_dead``; their self mask must be reconstructed from
        shares. Every self mask (active or stage-3 dropped) is reconstructed
        via Shamir from u_seed shares, matching the privacy model of classical
        SecAgg+. The server cancels the pairwise masks for stage-2 dropouts and
        adds the surviving clients' ``e_S`` corrections.
        """
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])

        # Clients that sent masked vectors in stage 2.
        p2_clients = set(state.masked_params.keys())
        stage2_dead = state.sampled_node_ids - p2_clients

        def make(nid: int, active: set[int], dead: set[int]) -> Message:
            neighbours = state.nid_to_neighbours[nid]
            cfg_dict = {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: list(neighbours & active),
                Key.DEAD_NODE_ID_LIST: list(neighbours & dead),
            }
            cfg_record = ConfigRecord(cfg_dict)  # type: ignore
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(current_round),
            )

        # Ask every client that reached stage 2 for u_seed shares of all active
        # clients (so every self mask is reconstructed from threshold shares),
        # b_seed shares for stage-2 dead clients, and the e_S correction for
        # stage-2 dead neighbours.
        log(
            DEBUG,
            "[SecAggPlusPlus Stage 3] Requesting shares from %s clients.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [
                make(node_id, p2_clients, stage2_dead)
                for node_id in state.active_node_ids
            ],
            timeout=self.timeout,
        )
        responding_nids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[SecAggPlusPlus Stage 3] Received shares from %s clients.",
            len(responding_nids),
        )

        # Clients that survived through the unmask stage.
        p3_clients = responding_nids
        stage3_dead = p2_clients - p3_clients
        if stage3_dead:
            log(
                DEBUG,
                "[SecAggPlusPlus Stage 3] Stage-3 dropouts: %s",
                stage3_dead,
            )

        b_seed_shares: dict[int, list[bytes]] = {nid: [] for nid in stage2_dead}
        u_seed_shares: dict[int, list[bytes]] = {nid: [] for nid in p2_clients}

        t_sharepile = 0.0
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            nid_lst = cast(list[int], res_dict[Key.NODE_ID_LIST])
            share_lst = cast(list[bytes], res_dict[Key.SHARE_LIST])

            for idx, owner_nid in enumerate(nid_lst):
                if owner_nid in p2_clients:
                    u_seed_shares[owner_nid].append(share_lst[idx])
                elif owner_nid in stage2_dead:
                    b_seed_shares[owner_nid].append(share_lst[idx])

        masked_vector = state.aggregate_ndarrays
        del state.aggregate_ndarrays
        dimensions_list = get_parameters_shape(masked_vector)

        # Masks apply only to the parameter arrays, not to the leading
        # quantization factor.
        factor = [masked_vector[0]]
        params = masked_vector[1:]
        param_dimensions = dimensions_list[1:]

        # Cancel outgoing masks from each stage-2 dropped client to surviving
        # clients.
        t_dead_cancel = 0.0
        t_dead_combine = 0.0
        for dead_nid in stage2_dead:
            share_list = b_seed_shares[dead_nid]
            if len(share_list) < state.threshold:
                log(
                    ERROR,
                    "Not enough shares to recover b_seed for client %d",
                    dead_nid,
                )
                return False
            t0 = perf_counter()
            b_seed = combine_shares(share_list)
            mid = perf_counter()
            for survivor_nid in state.nid_to_neighbours[dead_nid] & p2_clients:
                k = derive_pairwise_key(b_seed, survivor_nid)
                mask = pseudo_rand_gen_secure(k, state.mod_range, param_dimensions)
                params = parameters_subtraction(params, mask)
            t1 = perf_counter()
            t_dead_combine += mid - t0
            t_dead_cancel += t1 - t0
        if stage2_dead:
            log(
                INFO,
                "Stage3 dead_cancel: combine=%.4fs prg+sub=%.4fs total=%.4fs n_dead=%d",
                t_dead_combine,
                t_dead_cancel - t_dead_combine,
                t_dead_cancel,
                len(stage2_dead),
            )

        # Cancel incoming masks from surviving clients to stage-2 dropped
        # clients. Each surviving client sends e_S as a vector sum.
        t_e_add = 0.0
        for msg in msgs:
            if msg.has_error():
                continue
            t0 = perf_counter()
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            e_lst = cast(list[bytes], res_dict[Key.E_VALUE])
            e_sum = [bytes_to_ndarray(b) for b in e_lst]
            params = parameters_addition(params, e_sum)
            t_e_add += perf_counter() - t0

        params = parameters_mod(params, state.mod_range)

        # Remove self masks for all clients whose c_S is in the aggregate.
        # Every u_seed is reconstructed from threshold shares, matching the
        # privacy model of classical SecAgg+.
        t_self_unmask = 0.0
        t_combine_shares = 0.0
        t_prg_and_sub = 0.0
        share_counts = set()
        for nid in p2_clients:
            share_list = u_seed_shares[nid]
            share_counts.add(len(share_list))
            if len(share_list) < state.threshold:
                log(
                    ERROR,
                    "Not enough shares to recover u_seed for client %d",
                    nid,
                )
                return False
            t0 = perf_counter()
            u_seed = combine_shares(share_list)
            mid = perf_counter()
            u_mask = pseudo_rand_gen_secure(u_seed, state.mod_range, param_dimensions)
            params = parameters_subtraction(params, u_mask)
            t1 = perf_counter()
            t_combine_shares += mid - t0
            t_prg_and_sub += t1 - mid
            t_self_unmask += t1 - t0

        log(
            INFO,
            "Stage3 self_unmask: combine=%.4fs prg+sub=%.4fs total=%.4fs n_clients=%d share_counts=%s",
            t_combine_shares,
            t_prg_and_sub,
            t_self_unmask,
            len(p2_clients),
            share_counts,
        )
        state.timing_log["Stage3_self_unmask"] = t_self_unmask

        masked_vector = [factor[0]] + parameters_mod(params, state.mod_range)

        recon_parameters = parameters_mod(masked_vector, state.mod_range)
        q_total_ratio, recon_parameters = factor_extract(recon_parameters)
        inv_dq_total_ratio = state.quantization_range / q_total_ratio
        aggregated_vector = dequantize(
            recon_parameters,
            state.clipping_range,
            state.quantization_range,
        )
        offset = -(len(state.masked_params) - 1) * state.clipping_range
        for vec in aggregated_vector:
            vec += offset
            vec *= inv_dq_total_ratio

        # Backward compatibility with Strategy.
        results = state.legacy_results
        parameters = ndarrays_to_parameters(aggregated_vector)
        for _, fitres in results:
            fitres.parameters = parameters

        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(state.failures),
        )
        aggregated_result = context.strategy.aggregate_fit(
            current_round, results, state.failures  # type: ignore
        )
        parameters_aggregated, metrics_aggregated = aggregated_result

        if parameters_aggregated:
            arr_record = compat.parameters_to_arrayrecord(parameters_aggregated, True)
            context.state.array_records[MAIN_PARAMS_RECORD] = arr_record
            context.history.add_metrics_distributed_fit(
                server_round=current_round, metrics=metrics_aggregated
            )
        return True
