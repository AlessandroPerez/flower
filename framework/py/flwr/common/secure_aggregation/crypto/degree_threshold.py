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
"""Adaptive degree and threshold computation for SecAgg+ graphs.

This module implements the numerical parameter search described in
Bell et al., "Secure Single-Server Aggregation with (Poly)Logarithmic
Overhead" (eprint 2020/704).  Given the number of clients ``n`` and the
security/correctness parameters ``gamma``, ``delta``, ``sigma``, ``eta``,
it returns a pair ``(k, t)`` where ``k`` is the number of neighbors per
client and ``t`` is the Shamir reconstruction threshold.
"""


from __future__ import annotations

import math

from scipy.stats import hypergeom


def compute_degree_and_threshold(  # pylint: disable=too-many-locals,too-many-branches
    n: int,
    gamma: float,
    delta: float,
    sigma: int,
    eta: int,
) -> tuple[int, int]:
    """Return ``(k, t)`` satisfying the paper's graph constraints.

    The search minimises ``k`` such that the three events ``E1`` (not too
    many corrupt neighbours), ``E2`` (connectivity after dropouts), and
    ``E3`` (enough surviving neighbours) all hold with the required
    probabilities.

    Parameters
    ----------
    n : int
        Number of clients. Must be at least 2.
    gamma : float
        Maximum fraction of corrupted clients.
    delta : float
        Maximum fraction of dropout clients.
    sigma : int
        Statistical security parameter (failure probability <= 2^-sigma).
    eta : int
        Correctness parameter (failure probability <= 2^-eta).

    Returns
    -------
    tuple[int, int]
        ``(k, t)`` where ``k`` is odd and ``1 < t < k``.

    Raises
    ------
    ValueError
        If no valid ``(k, t)`` exists for the given parameters.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    if gamma + delta >= 1.0:
        raise ValueError(f"gamma + delta must be < 1, got {gamma + delta}")
    if sigma <= 0 or eta <= 0:
        raise ValueError("sigma and eta must be positive")

    n_float = float(n)
    corrupt_pop = min(n - 1, max(0, int(math.floor(gamma * n_float))))
    survive_pop = min(n - 1, max(0, int(math.floor((1.0 - delta) * n_float))))

    security_target = 2.0**-sigma / n_float
    correctness_target = 2.0**-eta / n_float
    connectivity_target = 2.0**-sigma

    # Lower bound: at least enough neighbours to admit a threshold strictly
    # between the security and correctness requirements.
    min_k = 3
    max_k = n - 1 if n % 2 == 0 else n - 2

    for k in range(min_k, max_k + 1, 2):
        # E1: not too many corrupt neighbours.
        # Find smallest t such that Pr[X >= t] <= security_target,
        # where X ~ HyperGeom(n-1, corrupt_pop, k).
        t_security = None
        for t in range(1, k + 1):
            prob = 1.0 - hypergeom.cdf(t - 1, n - 1, corrupt_pop, k)
            if prob <= security_target:
                t_security = t
                break

        if t_security is None:
            continue

        # E3: enough surviving neighbours.
        # Find largest t such that Pr[Y < t] <= correctness_target,
        # where Y ~ HyperGeom(n-1, survive_pop, k).
        t_correctness = None
        for t in range(k, 0, -1):
            prob = hypergeom.cdf(t - 1, n - 1, survive_pop, k)
            if prob <= correctness_target:
                t_correctness = t
                break

        if t_correctness is None:
            continue

        # We need a threshold that satisfies both.
        if t_security > t_correctness:
            continue

        # E2: connectivity after dropouts (Harary graph property).
        # Need n * (gamma + delta)^(floor(k/2)) <= 2^-sigma.
        half = k // 2
        if (gamma + delta) ** half > 0.0:
            connectivity_fail_prob = n_float * (gamma + delta) ** half
            if connectivity_fail_prob > connectivity_target:
                continue

        # Choose a threshold in the middle of the feasible interval.
        t = (t_security + t_correctness) // 2
        t = max(t_security, min(t_correctness, t))
        if t <= 1 or t >= k:
            continue

        return k, t

    raise ValueError(
        f"No valid (k, t) found for n={n}, gamma={gamma}, "
        f"delta={delta}, sigma={sigma}, eta={eta}"
    )
