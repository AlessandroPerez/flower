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
"""Tests for adaptive degree/threshold computation."""


import math

import pytest
from scipy.stats import hypergeom

from .degree_threshold import compute_degree_and_threshold


def _verify_constraints(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    n: int,
    k: int,
    t: int,
    gamma: float,
    delta: float,
    sigma: int,
    eta: int,
) -> None:
    """Check that a returned (k, t) satisfies the Bell et al. constraints."""
    assert k % 2 == 1, "k must be odd"
    assert 1 < t < k, "threshold must be strictly between 1 and k"

    n_float = float(n)
    corrupt_pop = min(n - 1, max(0, int(math.floor(gamma * n_float))))
    survive_pop = min(n - 1, max(0, int(math.floor((1.0 - delta) * n_float))))

    security_target = 2.0**-sigma / n_float
    correctness_target = 2.0**-eta / n_float
    connectivity_target = 2.0**-sigma

    e1 = 1.0 - hypergeom.cdf(t - 1, n - 1, corrupt_pop, k)
    e3 = hypergeom.cdf(t - 1, n - 1, survive_pop, k)
    half = k // 2
    e2 = n_float * (gamma + delta) ** half

    assert e1 <= security_target, f"E1 violated: {e1} > {security_target}"
    assert e3 <= correctness_target, f"E3 violated: {e3} > {correctness_target}"
    assert e2 <= connectivity_target, f"E2 violated: {e2} > {connectivity_target}"


class TestComputeDegreeAndThreshold:
    """Tests for compute_degree_and_threshold."""

    @pytest.mark.parametrize(
        "n,gamma,delta,sigma,eta",
        [
            (100, 0.1, 0.1, 30, 30),
            (1_000, 0.2, 0.2, 40, 40),
            (10_000, 0.1, 0.1, 40, 40),
            (50, 0.0, 0.1, 20, 20),
            (200, 0.05, 0.05, 30, 30),
        ],
    )
    def test_returns_valid_pair(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n: int,
        gamma: float,
        delta: float,
        sigma: int,
        eta: int,
    ) -> None:
        """Check that the function returns a feasible (k, t)."""
        k, t = compute_degree_and_threshold(n, gamma, delta, sigma, eta)
        _verify_constraints(n, k, t, gamma, delta, sigma, eta)

    def test_small_n(self) -> None:
        """A very small n may still admit parameters for relaxed targets."""
        k, t = compute_degree_and_threshold(10, 0.1, 0.1, 5, 5)
        _verify_constraints(10, k, t, 0.1, 0.1, 5, 5)

    def test_invalid_n(self) -> None:
        """N < 2 must raise ValueError."""
        with pytest.raises(ValueError):
            compute_degree_and_threshold(1, 0.1, 0.1, 10, 10)

    def test_invalid_gamma(self) -> None:
        """Gamma outside [0, 1) must raise ValueError."""
        with pytest.raises(ValueError):
            compute_degree_and_threshold(100, 1.5, 0.1, 10, 10)

    def test_invalid_delta(self) -> None:
        """Delta outside [0, 1) must raise ValueError."""
        with pytest.raises(ValueError):
            compute_degree_and_threshold(100, 0.1, -0.1, 10, 10)

    def test_gamma_plus_delta_too_large(self) -> None:
        """Gamma + delta >= 1 must raise ValueError."""
        with pytest.raises(ValueError):
            compute_degree_and_threshold(100, 0.6, 0.5, 10, 10)

    def test_impossible_parameters(self) -> None:
        """Overly tight parameters should fail gracefully."""
        with pytest.raises(ValueError):
            # Tiny n, aggressive security parameters.
            compute_degree_and_threshold(5, 0.1, 0.1, 100, 100)
