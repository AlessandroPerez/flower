"""SecAgg++ NumPy demo server app."""

from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusPlusWorkflow

from secaggplusplus_numpy.task import get_model

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run a federated learning round with SecAgg++."""
    num_rounds = int(context.run_config["num-server-rounds"])
    num_partitions = int(context.run_config["num-partitions"])
    num_features = int(context.run_config["num-features"])

    # Initial global model
    initial_arrays = get_model(num_features)
    parameters = ndarrays_to_parameters(initial_arrays)

    # FedAvg selects every client each round
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_partitions,
        min_available_clients=num_partitions,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=None,
    )

    # Wrap in a LegacyContext so the DefaultWorkflow can drive the strategy
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # SecAgg++ workflow with the same explicit parameters as classical SecAgg+
    fit_workflow = SecAggPlusPlusWorkflow(
        num_shares=int(context.run_config["num-shares"]),
        reconstruction_threshold=int(context.run_config["reconstruction-threshold"]),
        max_weight=float(context.run_config["max-weight"]),
        timeout=float(context.run_config["timeout"]),
    )

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, legacy_context)
