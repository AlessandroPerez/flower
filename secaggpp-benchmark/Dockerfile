# syntax=docker/dockerfile:1
#
# Reproducible Docker image for the SecAgg+ vs SecAgg++ benchmark.
#
# Build (run from repo root):
#   docker build -f secaggpp-benchmark/Dockerfile -t secagg-bench:latest .
#
# Run (e.g. with 100 clients):
#   docker run --rm --shm-size=512m --network host secagg-bench:latest 100

FROM python:3.11.14-bookworm

# Install uv
RUN pip install --no-cache-dir uv==0.11.24

# Copy app sources into generic container-local paths.
COPY framework /flower/framework
COPY secaggpp-benchmark/benchmark-apps/classical /apps/classical
COPY secaggpp-benchmark/benchmark-apps/pp /apps/pp

# The pyproject.toml and uv.lock files within the apps reference the
# framework via an absolute host-specific path such as
#   /home/alessandro/Documents/flower/framework
# Instead of rewriting these files (fragile), create a symlink so that the
# original path resolves to the container-local copy.
#
# This step parses the lock file to discover whatever path the host used.
# The lock file records the framework path as `editable = "/home/.../framework"`.
# Extract it, create a symlink so the path resolves inside the container, then
# pre-build the virtualenv.  These are combined into one RUN layer so that
# later edits (e.g. to the benchmark script) don't invalidate the venv cache.
RUN LOCK_PATH=$(grep -A3 'name = "flwr"' /apps/classical/uv.lock \
    | grep -o 'editable = "[^"]*"' \
    | cut -d'"' -f2 \
    | head -1) \
    && mkdir -p "$(dirname "$LOCK_PATH")" \
    && ln -sf /flower/framework "$LOCK_PATH" \
    && UV_PROJECT_ENVIRONMENT=/venvs/bench-venv \
       uv sync --project /apps/classical --python=3.11.14 --frozen -q

# Pre-download CIFAR-10 so the first benchmark run doesn't pay download cost.
# The dataset is cached in the HF datasets cache (~160 MB) and reused across
# any number of clients (partitioners value only controls splitting).
RUN UV_PROJECT_ENVIRONMENT=/venvs/bench-venv \
    uv run --no-sync --project /apps/classical --python=3.11.14 \
        -c "exec(__import__('flwr_datasets').FederatedDataset(dataset='uoft-cs/cifar10', partitioners={'train': 100}).load_partition(0, 'train'))" \
    >/dev/null 2>&1

# Copy and fix the benchmark script last.
COPY secaggpp-benchmark/secaggpp_secaggp_benchmark.sh /benchmark.sh
RUN chmod +x /benchmark.sh

ENV CLASSICAL_DIR=/apps/classical
ENV PP_DIR=/apps/pp
ENV CLASSICAL_ENV=/venvs/bench-venv
ENV PP_ENV=/venvs/bench-venv

WORKDIR /
ENTRYPOINT ["/benchmark.sh"]
