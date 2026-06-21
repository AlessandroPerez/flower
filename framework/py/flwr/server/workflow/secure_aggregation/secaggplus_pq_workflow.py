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
"""Post-quantum workflow for the SecAgg+ protocol."""


import random
from dataclasses import dataclass, field
from logging import DEBUG, ERROR, INFO
from typing import cast

import flwr.compat.common.recorddict_compat as compat
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.common import FitRes, NDArrays, bytes_to_ndarray, log, ndarrays_to_parameters
from flwr.common.secure_aggregation.crypto.degree_threshold import (
    compute_degree_and_threshold,
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
    Key,
    Stage,
)
from flwr.common.secure_aggregation.secaggplus_utils import (
    pseudo_rand_gen,
    share_keys_plaintext_separate_ps,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.compat.legacy_context import LegacyContext
from flwr.serverapp.grid import Grid

from ..constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD
from ..constant import Key as WorkflowKey


@dataclass
class WorkflowState:  # pylint: disable=R0902
    """The state of the post-quantum SecAgg+ protocol."""

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
    nid_to_publickeys: dict[int, list[bytes]] = field(default_factory=dict)
    forward_rd_srcs: dict[int, list[int]] = field(default_factory=dict)
    forward_rd_ciphertexts: dict[int, list[bytes]] = field(default_factory=dict)
    forward_ct_pairwise: dict[int, list[bytes]] = field(default_factory=dict)
    forward_ct_share_enc: dict[int, list[bytes]] = field(default_factory=dict)
    forward_ps_srcs: dict[int, list[int]] = field(default_factory=dict)
    forward_ps_ciphertexts: dict[int, list[bytes]] = field(default_factory=dict)
    aggregate_ndarrays: NDArrays = field(default_factory=list)
    legacy_results: list[tuple[ClientProxy, FitRes]] = field(default_factory=list)
    failures: list[Exception] = field(default_factory=list)


class SecAggPlusPQWorkflow:  # pylint: disable=too-many-instance-attributes
    """The post-quantum workflow for the SecAgg+ protocol.

    This workflow mirrors the original SecAgg+ workflow but replaces the
    Diffie-Hellman key agreement with the post-quantum X-Wing hybrid KEM
    and uses Shamir-secret-shared pairwise secrets for dropout recovery.
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_shares: int | float | None = None,
        reconstruction_threshold: int | float | None = None,
        *,
        gamma: float = 0.2,
        delta: float = 0.2,
        sigma: int = 40,
        eta: int = 30,
        max_weight: float = 1000.0,
        clipping_range: float = 8.0,
        quantization_range: int = 4194304,
        modulus_range: int = 4294967296,
        single_kem: bool = False,
        timeout: float | None = None,
    ) -> None:
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
        self.eta = eta
        self.max_weight = max_weight
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range
        self.modulus_range = modulus_range
        self.single_kem = single_kem
        self.timeout = timeout

        self._check_init_params()

    def __call__(self, grid: Grid, context: Context) -> None:
        """Run the post-quantum SecAgg+ protocol."""
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )
        state = WorkflowState()

        steps = (
            self.setup_stage,
            self.share_keys_stage,
            self.collect_masked_vectors_stage,
            self.unmask_stage,
        )
        log(INFO, "PQ secure aggregation commencing.")
        for step in steps:
            if not step(grid, context, state):
                log(INFO, "PQ secure aggregation halted.")
                return
        log(INFO, "PQ secure aggregation completed.")

    def _check_init_params(self) -> None:  # pylint: disable=R0912
        if self.num_shares is not None and not isinstance(
            self.num_shares, (int | float)
        ):
            raise TypeError("`num_shares` must be of type int, float, or None.")
        if isinstance(self.num_shares, int) and self.num_shares <= 2:
            raise ValueError("`num_shares` as an integer must be greater than 2.")
        if isinstance(self.num_shares, float) and not 0.0 < self.num_shares <= 1.0:
            raise ValueError("If `num_shares` is a float, it must be in (0, 1].")

        if self.reconstruction_threshold is not None and not isinstance(
            self.reconstruction_threshold, (int | float)
        ):
            raise TypeError(
                "`reconstruction_threshold` must be of type int, float, or None."
            )
        if (
            isinstance(self.reconstruction_threshold, int)
            and self.reconstruction_threshold < 1
        ):
            raise ValueError(
                "`reconstruction_threshold` as an integer must be at least 1."
            )
        if (
            isinstance(self.reconstruction_threshold, float)
            and not 0.0 < self.reconstruction_threshold <= 1.0
        ):
            raise ValueError(
                "If `reconstruction_threshold` is a float, it must be in (0, 1]."
            )

        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("`gamma` must be in [0, 1).")
        if not 0.0 <= self.delta < 1.0:
            raise ValueError("`delta` must be in [0, 1).")
        if self.gamma + self.delta >= 1.0:
            raise ValueError("`gamma + delta` must be < 1.")
        if self.sigma <= 0 or self.eta <= 0:
            raise ValueError("`sigma` and `eta` must be positive.")

        if self.max_weight <= 0:
            raise ValueError("`max_weight` must be greater than 0.")
        if self.quantization_range <= 0:
            raise ValueError("`quantization_range` must be greater than 0.")
        if (
            not isinstance(self.modulus_range, int)
            or self.modulus_range <= self.quantization_range
        ):
            raise ValueError(
                "`modulus_range` must be an integer and greater than "
                "`quantization_range`."
            )
        if bin(self.modulus_range).count("1") != 1:
            raise ValueError("`modulus_range` must be a power of 2.")
        if not isinstance(self.single_kem, bool):
            raise TypeError("`single_kem` must be a boolean.")

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

        # Determine (k, t) either from explicit parameters or from the
        # paper's adaptive computation.
        if self.num_shares is None:
            k_adaptive, t_adaptive = compute_degree_and_threshold(
                num_samples, self.gamma, self.delta, self.sigma, self.eta
            )
            state.num_shares = k_adaptive
            state.threshold = t_adaptive
        else:
            if isinstance(self.num_shares, float):
                state.num_shares = round(self.num_shares * num_samples)
                if state.num_shares < num_samples and state.num_shares & 1 == 0:
                    state.num_shares += 1
                if state.num_shares <= 2:
                    state.num_shares = num_samples
            else:
                state.num_shares = self.num_shares

            if self.reconstruction_threshold is None:
                raise ValueError(
                    "`reconstruction_threshold` must be provided when "
                    "`num_shares` is explicit."
                )
            if isinstance(self.reconstruction_threshold, float):
                state.threshold = max(
                    2, round(self.reconstruction_threshold * state.num_shares)
                )
            else:
                state.threshold = self.reconstruction_threshold

        # Ensure num_shares is odd and within bounds.
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
            Key.SINGLE_KEM: self.single_kem,
        }

        random.shuffle(sampled_node_ids)
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
            "[PQ Stage 0] Sending configurations to %s clients.",
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
            "[PQ Stage 0] Received public keys from %s clients.",
            len(state.active_node_ids),
        )

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            key_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            node_id = msg.metadata.src_node_id
            pk_pw = cast(bytes, key_dict[Key.PUBLIC_KEY_PAIRWISE])
            pk_se = cast(bytes, key_dict[Key.PUBLIC_KEY_SHARE_ENC])
            state.nid_to_publickeys[node_id] = [pk_pw, pk_se]

        return self._check_threshold(state)

    def share_keys_stage(  # pylint: disable=R0914
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'share keys' stage (stage 1)."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]

        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid] & state.active_node_ids
            cfg_record = ConfigRecord(
                {str(nid): state.nid_to_publickeys[nid] for nid in neighbours}
            )
            cfg_record[Key.STAGE] = Stage.SHARE_KEYS
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(
            DEBUG,
            "[PQ Stage 1] Forwarding public keys to %s clients.",
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
            "[PQ Stage 1] Received encapsulations from %s clients.",
            len(state.active_node_ids),
        )

        # Build forward packet lists: destination -> list of packets.
        rd_srcs: list[int] = []
        rd_dsts: list[int] = []
        rd_ciphertexts: list[bytes] = []
        ct_pairwise: list[tuple[int, int, bytes]] = []
        ct_share_enc: list[tuple[int, int, bytes]] = []

        fwd_rd_ciphertexts: dict[int, list[bytes]] = {
            nid: [] for nid in state.active_node_ids
        }
        fwd_rd_srcs: dict[int, list[int]] = {nid: [] for nid in state.active_node_ids}
        fwd_ct_pairwise: dict[int, list[bytes]] = {
            nid: [] for nid in state.active_node_ids
        }
        fwd_ct_share_enc: dict[int, list[bytes]] = {
            nid: [] for nid in state.active_node_ids
        }

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            node_id = msg.metadata.src_node_id
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            dst_lst = cast(list[int], res_dict[Key.DESTINATION_LIST])
            src_lst = cast(list[int], res_dict[Key.SOURCE_LIST])
            ctxt_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_LIST])
            ct_pw_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_PAIRWISE_LIST])
            ct_se_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_SHARE_ENC_LIST])

            for src, dst, ciphertext in zip(src_lst, dst_lst, ctxt_lst, strict=True):
                rd_srcs.append(src)
                rd_dsts.append(dst)
                rd_ciphertexts.append(ciphertext)

            for dst, ct_pw in zip(dst_lst, ct_pw_lst, strict=True):
                ct_pairwise.append((node_id, dst, ct_pw))
            for dst, ct_se in zip(dst_lst, ct_se_lst, strict=True):
                ct_share_enc.append((node_id, dst, ct_se))

        for src, dst, ciphertext in zip(rd_srcs, rd_dsts, rd_ciphertexts, strict=True):
            if dst in fwd_rd_ciphertexts:
                fwd_rd_ciphertexts[dst].append(ciphertext)
                fwd_rd_srcs[dst].append(src)

        for _src, dst, ct_pw in ct_pairwise:
            if dst in fwd_ct_pairwise:
                fwd_ct_pairwise[dst].append(ct_pw)
        for _src, dst, ct_se in ct_share_enc:
            if dst in fwd_ct_share_enc:
                fwd_ct_share_enc[dst].append(ct_se)

        state.forward_rd_srcs = fwd_rd_srcs
        state.forward_rd_ciphertexts = fwd_rd_ciphertexts
        state.forward_ct_pairwise = fwd_ct_pairwise
        state.forward_ct_share_enc = fwd_ct_share_enc

        return self._check_threshold(state)

    def collect_masked_vectors_stage(  # pylint: disable=R0914
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'collect masked vectors' stage (stage 2)."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]

        def make(nid: int) -> Message:
            cfg_dict = {
                Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                Key.CIPHERTEXT_PAIRWISE_LIST: state.forward_ct_pairwise[nid],
                Key.CIPHERTEXT_SHARE_ENC_LIST: state.forward_ct_share_enc[nid],
                Key.SOURCE_LIST: state.forward_rd_srcs[nid],
                Key.CIPHERTEXT_LIST: state.forward_rd_ciphertexts[nid],
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
            "[PQ Stage 2] Forwarding encapsulations and shares to %s clients.",
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
            "[PQ Stage 2] Received masked vectors from %s clients.",
            len(state.active_node_ids),
        )

        del (
            state.forward_ct_pairwise,
            state.forward_ct_share_enc,
            state.forward_rd_srcs,
            state.forward_rd_ciphertexts,
            state.nid_to_fitins,
        )

        # Build forward list for pairwise-secret shares.
        ps_srcs: list[int] = []
        ps_dsts: list[int] = []
        ps_ciphertexts: list[bytes] = []
        fwd_ps_ciphertexts: dict[int, list[bytes]] = {
            nid: [] for nid in state.active_node_ids
        }
        fwd_ps_srcs: dict[int, list[int]] = {nid: [] for nid in state.active_node_ids}

        masked_vector = None
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            dst_lst = cast(list[int], res_dict[Key.DESTINATION_LIST])
            src_lst = cast(list[int], res_dict[Key.SOURCE_LIST])
            ctxt_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_LIST])
            bytes_list = cast(list[bytes], res_dict[Key.MASKED_PARAMETERS])
            client_masked_vec = [bytes_to_ndarray(b) for b in bytes_list]

            if masked_vector is None:
                masked_vector = client_masked_vec
            else:
                masked_vector = parameters_addition(masked_vector, client_masked_vec)

            for src, dst, ciphertext in zip(src_lst, dst_lst, ctxt_lst, strict=True):
                ps_srcs.append(src)
                ps_dsts.append(dst)
                ps_ciphertexts.append(ciphertext)

        for src, dst, ciphertext in zip(ps_srcs, ps_dsts, ps_ciphertexts, strict=True):
            if dst in fwd_ps_ciphertexts:
                fwd_ps_ciphertexts[dst].append(ciphertext)
                fwd_ps_srcs[dst].append(src)

        state.forward_ps_srcs = fwd_ps_srcs
        state.forward_ps_ciphertexts = fwd_ps_ciphertexts

        if masked_vector is not None:
            masked_vector = parameters_mod(masked_vector, state.mod_range)
            state.aggregate_ndarrays = masked_vector

        # Backward compatibility with Strategy.
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            fitres = compat.recorddict_to_fitres(msg.content, True)
            proxy = state.nid_to_proxies[msg.metadata.src_node_id]
            state.legacy_results.append((proxy, fitres))

        return self._check_threshold(state)

    def unmask_stage(  # pylint: disable=R0912, R0914, R0915
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'unmask' stage (stage 3)."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])

        active_nids = state.active_node_ids
        dead_nids = state.sampled_node_ids - active_nids

        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid]
            cfg_dict = {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: list(neighbours & active_nids),
                Key.DEAD_NODE_ID_LIST: list(neighbours & dead_nids),
                Key.SOURCE_LIST: state.forward_ps_srcs[nid],
                Key.CIPHERTEXT_LIST: state.forward_ps_ciphertexts[nid],
            }
            cfg_record = ConfigRecord(cfg_dict)  # type: ignore
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(current_round),
            )

        log(
            DEBUG,
            "[PQ Stage 3] Requesting key shares from %s clients to remove masks.",
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
            "[PQ Stage 3] Received key shares from %s clients.",
            len(state.active_node_ids),
        )

        # Build collected shares dicts.
        rd_seed_shares: dict[int, list[bytes]] = {
            nid: [] for nid in state.sampled_node_ids
        }
        pairwise_secret_shares: dict[tuple[int, int], list[bytes]] = {}

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            nids = cast(list[int], res_dict[Key.NODE_ID_LIST])
            shares = cast(list[bytes], res_dict[Key.SHARE_LIST])
            active_list = cast(list[int], res_dict[Key.ACTIVE_NODE_ID_LIST])
            num_active = len(active_list)

            for idx, (owner_nid, share) in enumerate(zip(nids, shares, strict=True)):
                if idx < num_active:
                    rd_seed_shares[owner_nid].append(share)
                else:
                    plaintext = share
                    actual_src, dst, ps_shares = share_keys_plaintext_separate_ps(
                        plaintext
                    )
                    if dst != msg.metadata.src_node_id:
                        raise ValueError(
                            "Pairwise-secret bundle destination does not match sender"
                        )
                    for m_nid, ps_share in ps_shares:
                        key = (actual_src, m_nid)
                        if key not in pairwise_secret_shares:
                            pairwise_secret_shares[key] = []
                        pairwise_secret_shares[key].append(ps_share)

        masked_vector = state.aggregate_ndarrays
        del state.aggregate_ndarrays

        # Remove private masks for active clients.
        for nid, share_list in rd_seed_shares.items():
            if nid not in active_nids:
                continue
            if len(share_list) < state.threshold:
                log(
                    ERROR,
                    "Not enough shares to recover rd_seed for client %d",
                    nid,
                )
                return False
            rd_seed = combine_shares(share_list)
            private_mask = pseudo_rand_gen(
                rd_seed, state.mod_range, get_parameters_shape(masked_vector)
            )
            masked_vector = parameters_subtraction(masked_vector, private_mask)

        # Remove pairwise masks for dead clients.
        for nid in dead_nids:
            neighbours = set(state.nid_to_neighbours[nid])
            neighbours.discard(nid)
            for neighbor_nid in neighbours:
                key = (nid, neighbor_nid)
                share_list = pairwise_secret_shares.get(key, [])
                if len(share_list) < state.threshold:
                    log(
                        ERROR,
                        "Not enough shares to recover pairwise secret (%d, %d)",
                        nid,
                        neighbor_nid,
                    )
                    return False
                pw_secret = combine_shares(share_list)
                pairwise_mask = pseudo_rand_gen(
                    pw_secret, state.mod_range, get_parameters_shape(masked_vector)
                )
                if nid > neighbor_nid:
                    masked_vector = parameters_addition(masked_vector, pairwise_mask)
                else:
                    masked_vector = parameters_subtraction(masked_vector, pairwise_mask)

        recon_parameters = parameters_mod(masked_vector, state.mod_range)
        q_total_ratio, recon_parameters = factor_extract(recon_parameters)
        inv_dq_total_ratio = state.quantization_range / q_total_ratio
        aggregated_vector = dequantize(
            recon_parameters,
            state.clipping_range,
            state.quantization_range,
        )
        offset = -(len(active_nids) - 1) * state.clipping_range
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
