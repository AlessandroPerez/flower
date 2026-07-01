"""End-to-end Flower SecAgg+ training benchmark with real MNIST data.

This script runs federated-learning experiments on MNIST using Federated
Averaging protected by SecAgg+. It sweeps over client counts, dropout rates,
and implementations:

- classical SecAgg+ (``SecAggPlusWorkflow`` + ``secaggplus_mod``)
- SecAgg++ (``SecAggPlusPlusWorkflow`` + ``secaggplus_plus_mod``)

The local model is multinomial logistic regression (softmax) implemented in
NumPy. It is small, converges quickly, and keeps the benchmark focused on the
SecAgg+ protocol rather than deep-learning framework overhead.
"""

from __future__ import annotations

import argparse
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
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from flwr.server.workflow.secure_aggregation.secaggplus_plus_workflow import (
    SecAggPlusPlusWorkflow,
)
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import (
    SecAggPlusWorkflow,
)
from flwr.serverapp.grid import Grid
from flwr.supercore.run import Run, RunNotRunningException
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
MAX_CLIENTS: int = 100
CLIENT_COUNTS: list[int] = [10, 20, 50, 100]
DROPOUT_RATES: list[float] = [0.0, 0.01, 0.05, 0.1]
RUNS_PER_CONFIG: int = 1
NUM_ROUNDS: int = 3
LOCAL_EPOCHS: int = 3
LEARNING_RATE: float = 0.1
BATCH_SIZE: int = 64
RANDOM_SEED: int = 42

# Output directory (next to this script).
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUT_DIR / "flower_secagg_mnist_benchmark.json"

SERVER_NODE_ID = 0

NDArrayFloat = np.ndarray[Any, np.dtype[np.float64]]
NDArrayInt = np.ndarray[Any, np.dtype[np.int64]]

NUM_FEATURES = 28 * 28
NUM_CLASSES = 10
HIDDEN_DIM = 128

# Registry for the active model.  Set in main() based on CLI choice.
_init_model_fn: Callable[[int], list[NDArrayFloat]]
_train_step_fn: Callable[
    [list[NDArrayFloat], NDArrayFloat, NDArrayFloat, float, int], list[NDArrayFloat]
]
_evaluate_fn: Callable[[list[NDArrayFloat], NDArrayFloat, NDArrayInt], float]


# ---------------------------------------------------------------------------
# MNIST data loading
# ---------------------------------------------------------------------------
ClientData = tuple[NDArrayFloat, NDArrayInt]


def _images_to_numpy(images: list[Any]) -> NDArrayFloat:
    """Convert a list of grayscale PIL images to a normalized flat array."""
    arr = np.stack([np.asarray(img, dtype=np.float64) for img in images])
    return arr.reshape(-1, NUM_FEATURES) / 255.0


def _load_mnist(n: int) -> tuple[dict[int, ClientData], NDArrayFloat, NDArrayInt]:
    """Load MNIST, partition it IID across ``n`` clients, return train/test."""
    partitioner = IidPartitioner(num_partitions=n)
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
        seed=RANDOM_SEED,
    )

    client_data: dict[int, ClientData] = {}
    for node_id in range(n):
        partition = fds.load_partition(node_id)
        images = _images_to_numpy(list(partition["image"]))
        labels = np.asarray(partition["label"], dtype=np.int64)
        client_data[node_id] = (images, labels)

    test_dataset = fds.load_split("test")
    test_images = _images_to_numpy(list(test_dataset["image"]))
    test_labels = np.asarray(test_dataset["label"], dtype=np.int64)

    return client_data, test_images, test_labels


def _one_hot(labels: NDArrayInt, num_classes: int) -> NDArrayFloat:
    """Return a one-hot encoding of the integer labels."""
    encoded = np.zeros((labels.size, num_classes), dtype=np.float64)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded


# ---------------------------------------------------------------------------
# NumPy logistic regression
# ---------------------------------------------------------------------------
def _softmax(logits: NDArrayFloat) -> NDArrayFloat:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=1, keepdims=True), dtype=np.float64)


def _init_params(seed: int) -> list[NDArrayFloat]:
    """Initialize logistic-regression weights and bias."""
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((NUM_FEATURES, NUM_CLASSES)).astype(np.float64) * 0.01,
        np.zeros(NUM_CLASSES, dtype=np.float64),
    ]


def _train_step(
    params: list[NDArrayFloat],
    x: NDArrayFloat,
    y: NDArrayFloat,
    lr: float,
    batch_size: int,
) -> list[NDArrayFloat]:
    """Run one epoch of SGD and return updated parameters."""
    w, b = params
    n = x.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        xb, yb = x[batch_idx], y[batch_idx]
        probs = _softmax(xb @ w + b)
        grad_w = xb.T @ (probs - yb) / len(xb)
        grad_b = np.mean(probs - yb, axis=0)
        w -= lr * grad_w
        b -= lr * grad_b
    return [w, b]


def _evaluate(params: list[NDArrayFloat], x: NDArrayFloat, y: NDArrayInt) -> float:
    """Return classification accuracy on the given data."""
    w, b = params
    probs = _softmax(x @ w + b)
    predictions = np.argmax(probs, axis=1)
    return float(np.mean(predictions == y))


def _relu(x: NDArrayFloat) -> NDArrayFloat:
    """Element-wise ReLU activation."""
    return np.maximum(x, 0.0)


def _init_params_mlp(seed: int) -> list[NDArrayFloat]:
    """Initialize a small 2-layer MLP: 784 -> 128 -> 10."""
    rng = np.random.default_rng(seed)
    scale = 0.01
    return [
        rng.standard_normal((NUM_FEATURES, HIDDEN_DIM)).astype(np.float64) * scale,
        np.zeros(HIDDEN_DIM, dtype=np.float64),
        rng.standard_normal((HIDDEN_DIM, NUM_CLASSES)).astype(np.float64) * scale,
        np.zeros(NUM_CLASSES, dtype=np.float64),
    ]


def _mlp_forward(
    params: list[NDArrayFloat], x: NDArrayFloat
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Forward pass for the MLP; returns softmax probabilities and hidden state."""
    w1, b1, w2, b2 = params
    h = _relu(x @ w1 + b1)
    return _softmax(h @ w2 + b2), h


def _mlp_train_step(
    params: list[NDArrayFloat],
    x: NDArrayFloat,
    y: NDArrayFloat,
    lr: float,
    batch_size: int,
) -> list[NDArrayFloat]:
    """Run one epoch of SGD for the MLP and return updated parameters."""
    w1, b1, w2, b2 = params
    n = x.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        xb, yb = x[batch_idx], y[batch_idx]
        probs, h = _mlp_forward([w1, b1, w2, b2], xb)
        d2 = probs - yb
        grad_w2 = h.T @ d2 / len(xb)
        grad_b2 = np.mean(d2, axis=0)
        d1 = (d2 @ w2.T) * (h > 0).astype(np.float64)
        grad_w1 = xb.T @ d1 / len(xb)
        grad_b1 = np.mean(d1, axis=0)
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
    return [w1, b1, w2, b2]


def _evaluate_mlp(params: list[NDArrayFloat], x: NDArrayFloat, y: NDArrayInt) -> float:
    """Return classification accuracy for the MLP."""
    probs, _ = _mlp_forward(params, x)
    predictions = np.argmax(probs, axis=1)
    return float(np.mean(predictions == y))


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
    ) -> None:
        self._run = Run.create_empty(run_id)
        self._apps = apps
        self._contexts = contexts
        self._dropped = set(dropped)
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
        """Return a message with the same metadata but a deep-copied content."""
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

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to local clients and return their message IDs."""
        ids: list[str] = []
        for msg in messages:
            mid = str(uuid.uuid4())
            ids.append(mid)
            dst = msg.metadata.dst_node_id
            if dst in self._dropped and self._is_unmask_message(msg):
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
            if dst in self._dropped and self._is_unmask_message(msg):
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
# Local training app
# ---------------------------------------------------------------------------
current_client_data: dict[int, ClientData] = {}
current_test_images: NDArrayFloat = np.array([])
current_test_labels: NDArrayInt = np.array([], dtype=np.int64)


def _train(msg: Message, ctxt: Context) -> Message:
    """Local training step used by all ClientApps."""
    if msg.metadata.message_type != MessageType.TRAIN:
        return Message(RecordDict(), reply_to=msg)

    fitins = compat.recorddict_to_fitins(msg.content, keep_input=True)
    params = parameters_to_ndarrays(fitins.parameters)

    node_id = ctxt.node_id
    x, y = current_client_data[node_id]
    y_onehot = _one_hot(y, NUM_CLASSES)

    for _ in range(LOCAL_EPOCHS):
        params = _train_step_fn(params, x, y_onehot, LEARNING_RATE, BATCH_SIZE)

    fitres = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(params),
        num_examples=len(x),
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
    max_weight = float(60_000 // n * LOCAL_EPOCHS)
    if impl == "classical":
        return SecAggPlusWorkflow(
            num_shares=k,
            reconstruction_threshold=t,
            max_weight=max_weight,
            clipping_range=10.0,
            quantization_range=2**20,
            modulus_range=2**30,
        )
    if impl == "secaggplusplus":
        return SecAggPlusPlusWorkflow(
            num_shares=k,
            reconstruction_threshold=t,
            max_weight=max_weight,
            clipping_range=10.0,
            quantization_range=2**20,
            modulus_range=2**30,
        )
    raise ValueError(f"Unknown implementation: {impl}")


def _make_client_modifier(
    impl: str,
) -> Callable[[Message, Context, Callable[..., Message]], Message]:
    """Return the SecAgg client modifier for the requested implementation."""
    if impl == "classical":
        return secaggplus_mod
    if impl == "secaggplusplus":
        return secaggplus_plus_mod
    raise ValueError(f"Unknown implementation: {impl}")


def _run_one(
    impl: str, n: int, target_rate: float, dropped: set[int]
) -> dict[str, Any]:
    """Run one federated experiment and return timing and accuracy metadata."""
    global current_client_data, current_test_images, current_test_labels

    current_client_data, current_test_images, current_test_labels = _load_mnist(n)

    node_ids = list(range(n))
    modifier = _make_client_modifier(impl)
    apps, contexts = _build_client_apps(node_ids, modifier)
    grid = InMemoryGrid(run_id=1, apps=apps, contexts=contexts, dropped=dropped)

    initial_params = ndarrays_to_parameters(_init_model_fn(RANDOM_SEED))
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
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_manager=SimpleClientManager(),
    )

    workflow_callable = _make_workflow(impl, n, target_rate)
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

    # Extract final global parameters from the server context.
    arr_record = legacy_context.state.array_records[MAIN_PARAMS_RECORD]
    final_parameters = compat.arrayrecord_to_parameters(arr_record, keep_input=True)
    final_params = parameters_to_ndarrays(final_parameters)
    final_accuracy = _evaluate_fn(
        final_params, current_test_images, current_test_labels
    )

    return {
        "impl": impl,
        "n": n,
        "target_dropout_rate": target_rate,
        "actual_dropout_rate": len(dropped) / max(1, n),
        "dropped": len(dropped),
        "rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "time_seconds": elapsed,
        "final_accuracy": final_accuracy,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
_MODELS: dict[
    str,
    tuple[
        Callable[[int], list[NDArrayFloat]],
        Callable[
            [list[NDArrayFloat], NDArrayFloat, NDArrayFloat, float, int],
            list[NDArrayFloat],
        ],
        Callable[[list[NDArrayFloat], NDArrayFloat, NDArrayInt], float],
    ],
] = {
    "logistic": (_init_params, _train_step, _evaluate),
    "mlp": (_init_params_mlp, _mlp_train_step, _evaluate_mlp),
}


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(
        description="End-to-end Flower SecAgg+ training benchmark on MNIST."
    )
    parser.add_argument(
        "--model",
        choices=list(_MODELS.keys()),
        default="logistic",
        help="Local model architecture (default: logistic).",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=CLIENT_COUNTS,
        help="Client counts to benchmark (default: 10 20 50 100).",
    )
    parser.add_argument(
        "--dropout-rates",
        type=float,
        nargs="+",
        default=DROPOUT_RATES,
        help="Dropout rates to benchmark (default: 0.0 0.01 0.05 0.1).",
    )
    parser.add_argument(
        "--impls",
        nargs="+",
        default=["classical", "secaggplusplus"],
        choices=["classical", "secaggplusplus"],
        help="Implementations to benchmark (default: classical secaggplusplus).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_CONFIG,
        help="Number of repetitions per configuration (default: 1).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help="Number of federated rounds (default: 3).",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=LOCAL_EPOCHS,
        help="Local epochs per round (default: 3).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=JSON_PATH,
        help=(
            "Path to write the JSON results "
            "(default: speed_tests/flower_secagg_mnist_benchmark.json)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the training benchmark sweep and save JSON results."""
    global NUM_ROUNDS, LOCAL_EPOCHS
    args = _parse_args()

    init_fn, train_fn, eval_fn = _MODELS[args.model]
    global _init_model_fn, _train_step_fn, _evaluate_fn
    _init_model_fn = init_fn
    _train_step_fn = train_fn
    _evaluate_fn = eval_fn

    NUM_ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs

    results: list[dict[str, Any]] = []
    total = len(args.ns) * len(args.dropout_rates) * args.runs * len(args.impls)
    done = 0

    print(
        f"Sweep: model={args.model}, clients={args.ns}, "
        f"dropout_rates={args.dropout_rates}, runs={args.runs}, "
        f"impls={args.impls}",
        flush=True,
    )

    for n in args.ns:
        for rate in args.dropout_rates:
            for run in range(args.runs):
                dropped = _choose_dropped(n, rate, run)
                for impl in args.impls:
                    done += 1
                    print(
                        f"[{done}/{total}] impl={impl} n={n} "
                        f"dropout={rate:.0%} dropped={len(dropped)}",
                        flush=True,
                    )
                    res = _run_one(impl, n, rate, dropped)
                    res["model"] = args.model
                    results.append(res)
                    print(
                        f"  -> {res['status']} in {res['time_seconds']:.3f}s, "
                        f"accuracy={res['final_accuracy']:.4f}",
                        flush=True,
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
