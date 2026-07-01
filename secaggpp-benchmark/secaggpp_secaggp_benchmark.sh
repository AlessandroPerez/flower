#!/usr/bin/env bash
# Benchmark runner: classical SecAgg+ vs SecAgg++ for a variable number of clients.
# Usage: secaggpp_secaggp_benchmark.sh <N>
# Example: secaggpp_secaggp_benchmark.sh 10

set -euo pipefail

N=${1:-}
if [[ -z "$N" ]]; then
    echo "Usage: $0 <number_of_clients>" >&2
    exit 1
fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "Error: number_of_clients must be a positive integer, got '$N'" >&2
    exit 1
fi

# Resolve paths relative to this script so the benchmark works wherever the
# repo is cloned.  These defaults can be overridden with environment variables
# (e.g. for the Docker image).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSICAL_DIR="${CLASSICAL_DIR:-$SCRIPT_DIR/benchmark-apps/classical}"
PP_DIR="${PP_DIR:-$SCRIPT_DIR/benchmark-apps/pp}"

# A single shared virtualenv is enough because both apps have identical
# dependencies.  It is kept outside the project tree so `flwr run` does not
# trip over the max-directory-depth check on .venv.
CLASSICAL_ENV="${CLASSICAL_ENV:-/tmp/secagg-bench-venv}"
PP_ENV="${PP_ENV:-/tmp/secagg-bench-venv}"

PYTHON_VERSION="3.11.14"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Ensure flwr has a local-simulation connection for simulation mode.
mkdir -p ~/.flwr
if ! grep -q 'local-simulation' ~/.flwr/config.toml 2>/dev/null; then
    cat >> ~/.flwr/config.toml <<'EOF'

[superlink.local-simulation]
address = ":local:"
EOF
fi

# Use a logarithmic neighbour degree: #NG(A) ≈ ceil(log2(N)) + 1.
DEGREE=$(python3 -c "import math; print(max(3, int(math.ceil(math.log2($N))) + 1))")
THRESHOLD=$((DEGREE / 2 + 1))

# Run a single benchmark variant.
# Arguments: <display-name> <app-dir> <venv-dir>
# Progress is written to stderr; the final timing (seconds) is written to stdout.
run_benchmark() {
    local name="$1"
    local appdir="$2"
    local envdir="$3"

    echo "=== $name: $N clients ===" >&2

    # Ensure the dedicated venv exists and is in sync.
    if [[ ! -d "$envdir" ]]; then
        echo "Creating/syncing venv for $name..." >&2
        (
            cd "$appdir"
            UV_PROJECT_ENVIRONMENT="$envdir" uv sync --python="$PYTHON_VERSION" --frozen
        ) >&2
    fi

    local log
    log=$(mktemp)
    trap 'rm -f "$log"' RETURN

    # Run the Flower app in the simulation engine and stream the logs to a file.
    (
        cd "$appdir"
        UV_PROJECT_ENVIRONMENT="$envdir" uv run --no-sync --python="$PYTHON_VERSION" \
            flwr run . local-simulation --stream \
            --federation-config "num-supernodes=$N" \
            --run-config "is-demo=false" \
            --run-config "min-fit-clients=$N" \
            --run-config "min-available-clients=$N" \
            --run-config "num-shares=$DEGREE" \
            --run-config "reconstruction-threshold=$THRESHOLD" \
            --run-config "num-server-rounds=3"
    ) >"$log" 2>&1

    # Strip ANSI colour codes and extract the final timing reported by Flower.
    local timing
    timing=$(sed 's/\x1b\[[0-9;]*m//g' "$log" \
        | grep -oE 'Run finished [0-9]+ round\(s\) in [0-9.]+' \
        | grep -oE '[0-9.]+' \
        | tail -n 1)

    if [[ -z "$timing" ]]; then
        echo "$name: FAILED to extract timing" >&2
        tail -n 30 "$log" >&2
        echo "N/A"
    else
        echo "$name: ${timing}s" >&2
        echo "$timing"
    fi
}

echo "Benchmarking SecAgg+ vs SecAgg++ with $N clients, degree=$DEGREE, threshold=$THRESHOLD (3 server rounds)" >&2
echo "" >&2

# Warm-up: download and cache the CIFAR-10 dataset before the timed runs so
# the first benchmark doesn't pay a one-time download penalty.
echo "Warm-up: downloading CIFAR-10 dataset..." >&2
(
    cd "$CLASSICAL_DIR"
    UV_PROJECT_ENVIRONMENT="$CLASSICAL_ENV" uv run --no-sync --python="$PYTHON_VERSION" \
        python -c "
from flwr_datasets import FederatedDataset
fds = FederatedDataset(dataset='uoft-cs/cifar10', partitioners={'train': $N})
_ = fds.load_partition(0, 'train')
"
) >/dev/null 2>&1
echo "Warm-up complete." >&2

CLASSICAL_TIME=$(run_benchmark "Classical SecAgg+" "$CLASSICAL_DIR" "$CLASSICAL_ENV")
PP_TIME=$(run_benchmark "SecAgg++" "$PP_DIR" "$PP_ENV")

echo "" >&2
echo "===== Results ($N clients) =====" >&2
printf "Classical SecAgg+ : %ss\n" "$CLASSICAL_TIME" >&2
printf "SecAgg++          : %ss\n" "$PP_TIME" >&2

# Also emit machine-friendly results on stdout.
printf " classical=%s pp=%s\n" "$CLASSICAL_TIME" "$PP_TIME"
