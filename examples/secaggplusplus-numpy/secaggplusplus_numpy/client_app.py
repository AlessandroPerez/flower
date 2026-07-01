"""SecAgg++ NumPy demo client app."""

from __future__ import annotations

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_plus_mod
from flwr.common import Context

from secaggplusplus_numpy.task import evaluate, load_data, train


class FlowerClient(NumPyClient):
    """A NumPy client that trains a linear model with synthetic data."""

    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        num_features: int,
        num_examples: int,
        local_epochs: int,
        learning_rate: float,
    ) -> None:
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.num_features = num_features
        self.num_examples = num_examples
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.x, self.y = load_data(
            partition_id, num_partitions, num_features, num_examples
        )

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, float]
    ) -> tuple[list[np.ndarray], int, dict]:
        """Train the local model."""
        model = train(parameters, self.x, self.y, self.learning_rate, self.local_epochs)
        return model, len(self.y), {}

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, float]
    ) -> tuple[float, int, dict]:
        """Evaluate the local model."""
        mse, r2 = evaluate(parameters, self.x, self.y)
        return float(mse), len(self.y), {"mse": float(mse), "r2": float(r2)}


def client_fn(context: Context) -> NumPyClient:
    """Build a FlowerClient from the run context."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    num_features = int(context.run_config["num-features"])
    num_examples = int(context.run_config["num-examples-per-client"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    return FlowerClient(
        partition_id,
        num_partitions,
        num_features,
        num_examples,
        local_epochs,
        learning_rate,
    ).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_plus_mod])
