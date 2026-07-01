"""End-to-end Flower SecAgg benchmark with real ClientApps.

This script runs one training round of Flower's SecAgg+ protocol in-process.
It compares:

- classical SecAgg+ (``SecAggPlusWorkflow`` + ``secaggplus_mod``)
- SecAgg++ (``SecAggPlusPlusWorkflow`` + ``secaggplus_plus_mod``)

For each (number of clients, dropout rate) configuration it records the wall
clock time of the full federated round and writes the results as JSON.
"""

from __future__ import annotations

import copy
import json
import random
import time
import uuid
from collections.abc import Callable, Iterable
from math import ceil, log2
from pathlib import Path
from typing import Any

import numpy as np
from flwr.app import Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.client.mod import make_ffn
from flwr.client.mod.secure_aggregation.secaggplus_mod import secaggplus_mod
from flwr.client.mod.secure_aggregation.secaggplus_plus_mod import secaggplus_plus_mod
from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    ndarray_to_bytes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS as SECAGG_RECORD_KEY_CONFIGS,
)
from flwr.common.secure_aggregation.secaggplus_constants import Key as SecAggKey
from flwr.common.secure_aggregation.secaggplus_constants import Stage as SecAggStage
from flwr.compat.common import recorddict_compat as compat
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow
from flwr.server.workflow.secure_aggregation.secaggplus_plus_workflow import (
    SecAggPlusPlusWorkflow,
)
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
)
from flwr.serverapp.grid import Grid
from flwr.supercore.run import Run, RunNotRunningException

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
MAX_CLIENTS: int = 100
DROPOUT_RATES: list[float] = [0.0, 0.01, 0.05, 0.1]
RUNS_PER_CONFIG: int = 1
NUM_FEATURES: int = 100_000
NUM_EXAMPLES_PER_CLIENT: int = 100
LEARNING_RATE: float = 0.01

# Output directory (relative to this script).
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "flower_secagg_benchmark.json"

SERVER_NODE_ID = 0

NDArrayFloat = np.ndarray[Any, np.dtype[np.float64]]


# ---------------------------------------------------------------------------
# In-memory Flower harness
# ---------------------------------------------------------------------------
class InMemoryGrid(Grid):
    """A minimal in-memory Grid that routes messages to local ClientApps."""

    def __init__(
        self,
        run_id: int,
        apps: dict[int, Callable[[Message, Context], Message]],
        contexts: dict[int, Context],
        dropped: set[int],
        impl: str = "classical",
    ) -> None:
        self._run = Run.create_empty(run_id)
        self._apps = apps
        self._contexts = contexts
        self._dropped = set(dropped)
        self._impl = impl
        self._closed = False
        self._replies: dict[str, Message] = {}

    def set_run(self, run: Run) -> None:
        """Set the active run for this grid."""
        self._run = run

    @property
    def run(self) -> Run:
        """Return the active run."""
        return self._run

    def create_message(
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        """Create a new message destined to ``dst_node_id``."""
        return Message(
            content,
            dst_node_id=dst_node_id,
            message_type=message_type,
            group_id=group_id,
            ttl=ttl,
        )

    def get_node_ids(self) -> Iterable[int]:
        """Return the IDs of all connected nodes."""
        if self._closed:
            raise RunNotRunningException()
        return list(self._apps.keys())

    def _copy_message(self, msg: Message) -> Message:
        """Return a message with the same metadata but a deep-copied content.

        The SecAgg client modifiers mutate the incoming config record in place
        (e.g. popping the stage key).  In a real deployment each message is
        serialized, so we copy here to avoid shared-state bugs.
        """
        return Message(
            copy.deepcopy(msg.content),
            dst_node_id=msg.metadata.dst_node_id,
            message_type=msg.metadata.message_type,
            group_id=msg.metadata.group_id,
            ttl=msg.metadata.ttl,
            dst_task_id=msg.metadata.dst_task_id,
        )

    def _is_unmask_message(self, msg: Message) -> bool:
        """Return True if the message is the final SecAgg unmask stage."""
        cfg = msg.content.config_records.get(SECAGG_RECORD_KEY_CONFIGS)
        if cfg is None:
            return False
        return cfg.get(SecAggKey.STAGE) == SecAggStage.UNMASK

    def _is_collect_masked_vectors_message(self, msg: Message) -> bool:
        """Return True if the message is the SecAgg masked-vector stage."""
        cfg = msg.content.config_records.get(SECAGG_RECORD_KEY_CONFIGS)
        if cfg is None:
            return False
        return cfg.get(SecAggKey.STAGE) == SecAggStage.COLLECT_MASKED_VECTORS

    def _should_drop(self, msg: Message) -> bool:
        """Return True if a message to a dropped client should be dropped.

        We drop the masked-vector collection message so that both
        implementations experience the same stage-2 dropout pattern: the client
        never contributes an update to the aggregate.
        """
        return self._is_collect_masked_vectors_message(msg)

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to local clients and return their message IDs."""
        ids: list[str] = []
        for msg in messages:
            mid = str(uuid.uuid4())
            ids.append(mid)
            dst = msg.metadata.dst_node_id
            if dst in self._dropped and self._should_drop(msg):
                continue
            self._replies[mid] = self._apps[dst](
                self._copy_message(msg), self._contexts[dst]
            )
        return ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Return replies for the given message IDs."""
        replies: list[Message] = []
        for mid in message_ids:
            reply = self._replies.pop(mid, None)
            if reply is not None:
                replies.append(reply)
        return replies

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: float | None = None,
    ) -> Iterable[Message]:
        """Send messages and synchronously collect all replies."""
        del timeout  # Synchronous in-memory delivery; no timeout needed.
        replies: list[Message] = []
        for msg in messages:
            dst = msg.metadata.dst_node_id
            if dst in self._dropped and self._should_drop(msg):
                continue
            replies.append(
                self._apps[dst](self._copy_message(msg), self._contexts[dst])
            )
        return replies

    def close(self) -> None:
        """Mark the grid as closed."""
        self._closed = True


class FedAvgWithInit(FedAvg):
    """FedAvg that provides deterministic initial parameters."""

    def __init__(self, initial_params: Parameters, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._initial_params = initial_params

    def initialize_parameters(
        self, client_manager: ClientManager  # noqa: ARG002
    ) -> Parameters:
        """Return the pre-defined initial parameters."""
        return self._initial_params


# ---------------------------------------------------------------------------
# Local model / data
# ---------------------------------------------------------------------------
def _local_data(node_id: int) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Return synthetic local training data for a client."""
    rng = np.random.default_rng(node_id)
    x = rng.standard_normal((NUM_EXAMPLES_PER_CLIENT, NUM_FEATURES))
    true_w = rng.standard_normal(NUM_FEATURES)
    y = x @ true_w + rng.standard_normal(NUM_EXAMPLES_PER_CLIENT)
    return x, y


def _gradient(
    params: list[NDArrayFloat], x: NDArrayFloat, y: NDArrayFloat
) -> list[NDArrayFloat]:
    """Compute the gradient for a linear model (MSE)."""
    w = params[0]
    pred = x @ w
    err = pred - y
    grad_w = (x.T @ err) / len(y)
    return [grad_w]


def _train(
    msg: Message,
    ctxt: Context,
) -> Message:
    """Local training step used by all ClientApps."""
    if msg.metadata.message_type != MessageType.TRAIN:
        # Nothing to do for non-train messages in this benchmark.
        return Message(RecordDict(), reply_to=msg)

    fitins = compat.recorddict_to_fitins(msg.content, keep_input=True)
    params = parameters_to_ndarrays(fitins.parameters)
    x, y = _local_data(ctxt.node_id)
    grads = _gradient(params, x, y)
    updated = [p - LEARNING_RATE * g for p, g in zip(params, grads, strict=True)]
    fitres = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(updated),
        num_examples=NUM_EXAMPLES_PER_CLIENT,
        metrics={},
    )
    return Message(compat.fitres_to_recorddict(fitres, keep_input=True), reply_to=msg)


def _build_client_apps(
    node_ids: list[int],
    modifier: Callable[[Message, Context, Callable[..., Message]], Message],
) -> tuple[dict[int, Callable[[Message, Context], Message]], dict[int, Context]]:
    """Build one ClientApp callable and persistent Context per client."""
    apps: dict[int, Callable[[Message, Context], Message]] = {}
    contexts: dict[int, Context] = {}
    for nid in node_ids:
        ctxt = Context(
            run_id=1,
            node_id=nid,
            node_config={},
            state=RecordDict(),
            run_config={},
        )
        contexts[nid] = ctxt
        apps[nid] = make_ffn(_train, [modifier])
    return apps, contexts


# ---------------------------------------------------------------------------
# One benchmark run
# ---------------------------------------------------------------------------
def _choose_dropped(n: int, rate: float, run: int) -> set[int]:
    """Deterministically choose which clients drop out."""
    rng = random.Random(f"{n}-{rate}-{run}")
    drop_count = int(round(n * rate))
    drop_count = max(0, min(n - 1, drop_count))
    return set(rng.sample(range(n), drop_count))


def _log_counts(max_n: int) -> list[int]:
    """Return 1-2-5 log-spaced counts from 10 up to max_n inclusive."""
    counts: set[int] = set()
    decade = 10
    while decade <= max_n:
        for factor in (1, 2, 5):
            val = decade * factor
            if val <= max_n:
                counts.add(val)
        decade *= 10
    counts.add(min(10, max_n))
    if max_n not in counts:
        counts.add(max_n)
    return sorted(counts)


def _degree_threshold_with_fallback(n: int, dropout_rate: float) -> tuple[int, int]:
    """Return a log-spaced degree and a dropout-tolerant threshold."""
    degree = min(n, 2 * ceil(log2(n)) + 1)
    active = max(2, n - int(round(n * dropout_rate)))
    threshold = min(max(2, active - 1), degree - 1)
    return degree, threshold


def _make_workflow(
    impl: str, n: int, dropout_rate: float
) -> Callable[[Grid, Context], None]:
    """Return the server-side workflow for the requested implementation."""
    k, t = _degree_threshold_with_fallback(n, dropout_rate)
    if impl == "classical":
        return SecAggPlusWorkflow(
            num_shares=k,
            reconstruction_threshold=t,
            max_weight=float(NUM_EXAMPLES_PER_CLIENT * 2),
            clipping_range=10.0,
            quantization_range=2**20,
            modulus_range=2**30,
        )
    if impl == "secaggplusplus":
        return SecAggPlusPlusWorkflow(
            num_shares=k,
            reconstruction_threshold=t,
            max_weight=float(NUM_EXAMPLES_PER_CLIENT * 2),
            clipping_range=10.0,
            quantization_range=2**20,
            modulus_range=2**30,
        )
    raise ValueError(f"Unknown implementation: {impl}")


def _make_client_modifier(
    impl: str,
) -> Callable[[Message, Context, Callable[..., Message]], Message]:
    if impl == "classical":
        return secaggplus_mod
    if impl == "secaggplusplus":
        return secaggplus_plus_mod
    raise ValueError(f"Unknown implementation: {impl}")


def _run_one(
    impl: str, n: int, target_rate: float, dropped: set[int]
) -> dict[str, Any]:
    """Run one federated round and return timing metadata."""
    node_ids = list(range(n))
    modifier = _make_client_modifier(impl)
    apps, contexts = _build_client_apps(node_ids, modifier)
    grid = InMemoryGrid(
        run_id=1, apps=apps, contexts=contexts, dropped=dropped, impl=impl
    )

    initial_params = Parameters(
        tensors=[ndarray_to_bytes(np.zeros(NUM_FEATURES, dtype=np.float64))],
        tensor_type="numpy.ndarray",
    )
    strategy = FedAvgWithInit(initial_params=initial_params)
    base_context = Context(
        run_id=1,
        node_id=SERVER_NODE_ID,
        node_config={},
        state=RecordDict(),
        run_config={},
    )
    legacy_context = LegacyContext(
        context=base_context,
        config=ServerConfig(num_rounds=1),
        strategy=strategy,
        client_manager=SimpleClientManager(),
    )

    workflow_callable = _make_workflow(impl, n, len(dropped) / max(1, n))
    default_workflow = DefaultWorkflow(
        fit_workflow=workflow_callable,
        evaluate_workflow=lambda _grid, _context: None,
    )

    start = time.perf_counter()
    try:
        default_workflow(grid, legacy_context)
        elapsed = time.perf_counter() - start
        status = "success"
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        status = f"failed: {exc}"
    finally:
        grid.close()

    return {
        "impl": impl,
        "n": n,
        "target_dropout_rate": target_rate,
        "actual_dropout_rate": len(dropped) / max(1, n),
        "dropped": len(dropped),
        "time_seconds": elapsed,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the benchmark sweep and save JSON results."""
    client_counts = _log_counts(MAX_CLIENTS)
    results: list[dict[str, Any]] = []
    impls = ["classical", "secaggplusplus"]
    total = len(client_counts) * len(DROPOUT_RATES) * RUNS_PER_CONFIG * len(impls)
    done = 0

    print(
        f"Sweep: clients={client_counts}, dropout_rates={DROPOUT_RATES}, "
        f"runs={RUNS_PER_CONFIG}, impls={impls}",
        flush=True,
    )

    for n in client_counts:
        for rate in DROPOUT_RATES:
            for run in range(RUNS_PER_CONFIG):
                dropped = _choose_dropped(n, rate, run)
                for impl in impls:
                    done += 1
                    print(
                        f"[{done}/{total}] impl={impl} n={n} dropout={rate:.0%} "
                        f"dropped={len(dropped)}",
                        flush=True,
                    )
                    res = _run_one(impl, n, rate, dropped)
                    results.append(res)
                    print(
                        f"  -> {res['status']} in {res['time_seconds']:.3f}s",
                        flush=True,
                    )

    JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {JSON_PATH.resolve()}")


if __name__ == "__main__":
    main()
