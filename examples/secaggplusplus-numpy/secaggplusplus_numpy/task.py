"""Synthetic data and model helpers for the SecAgg++ NumPy demo."""

from __future__ import annotations

import numpy as np


NUM_FEATURES: int = 100
NUM_EXAMPLES: int = 100


def get_model(num_features: int = NUM_FEATURES) -> list[np.ndarray]:
    """Return a zero-initialised linear model."""
    return [np.zeros(num_features, dtype=np.float64)]


def load_data(
    partition_id: int,
    num_partitions: int,  # noqa: ARG001
    num_features: int = NUM_FEATURES,
    num_examples: int = NUM_EXAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic linear regression dataset for this partition."""
    rng = np.random.default_rng(partition_id)
    x = rng.standard_normal((num_examples, num_features))
    true_w = rng.standard_normal(num_features)
    y = x @ true_w + rng.standard_normal(num_examples)
    return x, y


def train(
    model: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    local_epochs: int,
) -> list[np.ndarray]:
    """Train a linear model with gradient descent."""
    w = model[0].copy()
    for _ in range(local_epochs):
        pred = x @ w
        grad = x.T @ (pred - y) / len(y)
        w -= learning_rate * grad
    return [w]


def evaluate(
    model: list[np.ndarray], x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Return MSE and R^2 for the model on the given data."""
    w = model[0]
    pred = x @ w
    mse = float(np.mean((pred - y) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mse, r2
