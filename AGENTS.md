# AGENTS.md

## PR review philosophy

When reviewing or generating code, apply this checklist:

1. Necessity
   - Ask whether each added block is required for the requested behavior.
   - Flag speculative abstractions, premature generalization, and dead paths.

2. Simplicity
   - Prefer less code when readability and correctness are preserved.
   - Suggest idiomatic code for the language in that module only if it makes the result easier to understand.
   - Avoid "clever" one-liners that reduce maintainability.

3. Readability
   - Prefer explicit names, small functions, shallow nesting, and linear control flow.
   - Flag dense logic that would be easier to read if split or renamed.

4. Local consistency
   - Compare with nearby modules and existing patterns before proposing structure/style changes.
   - Follow existing naming, error-handling, typing, and test conventions.

5. PR sizing
   - Flag PRs that combine unrelated concerns.
   - Suggest a split when refactoring, behavior changes, and cleanup are mixed together.

## Review output format

When reviewing a PR, output:
- Critical issues
- Simplicity/readability suggestions
- Consistency concerns
- Whether the PR should be split
- A brief overall verdict

## Development Patterns

### Database Migrations

Python services with databases use Alembic:

```bash
cd framework
uv run --no-sync --python=3.11.14 python -m dev.generate_migration "Description"
```

For Alembic-backed services, do not write a new migration file from scratch when
the intended change is a schema diff. Help the user use
`python -m dev.generate_migration` from the `framework/` directory instead, then
review the generated revision and make only the minimal adjustments needed. This
helps avoid schema drift between SQLAlchemy models and committed migrations.

The migration generator is Alembic-branch aware. It upgrades the temporary
database to all heads, then autogenerates a revision against the selected branch
head. The default branch head is `flwr@head`; these commands are equivalent:

```bash
cd framework
uv run --no-sync --python=3.11.14 python -m dev.generate_migration "Widen integer columns"
```

```bash
cd framework
uv run --no-sync --python=3.11.14 python -m dev.generate_migration --head flwr@head "Widen integer columns"
```

After autogeneration:

- Confirm the new revision has the intended `down_revision`.
- Confirm the generated operations match the SQLAlchemy metadata change.
- Review generated `batch_alter_table` blocks for SQLite compatibility.
- Update generated schema documentation if table metadata changed.

---

## Project: SecAgg++ (Post-Quantum SecAgg+) Benchmark

### Goal
Implement a post-quantum SecAgg+ variant ("SecAgg++") using ML-KEM-768 + AES-GCM as a drop-in replacement for the classical SecAgg+ in Flower, then benchmark both at various client scales.

### Progress

#### Core Implementation (done)
- `pq_kem.py`: ML-KEM-768 wrapper (replaced X-Wing, dropped xwing-kem dep); added raw AES-key derivation.
- `pq_pke.py`: spec-compliant CCA-secure PKE using ML-KEM-768 + AES-GCM with `key = HKDF(ss || pk)` and `ml_encap` as AEAD associated data; added general `aead_encrypt`/`aead_decrypt` helpers.
- `secaggplus_plus_mod.py`: SecAgg++ client mod with key reuse, client-server onion encryption (outer AEAD layer with per-client server key), spec-compliant inner ML-KEM+AES-GCM encryption, secure per-entry mask generation via SHAKE256.
- `secaggplus_plus_workflow.py`: deterministic neighbour graph (sorted node IDs), skip public-key distribution after round 1, server-side onion re-encryption.
- `secaggplus_constants.py`: added `Key.NEIGHBOUR_LIST` and `Key.SERVER_KEY`.
- `secaggplus_utils.py`: added `pseudo_rand_gen_secure` (SHAKE256-based per-entry mask generation).
- `max_weight` type bug fixed (accepts `int` or `float`, casts to `float`).

#### Benchmark Apps (done)
- `secaggpp-benchmark/benchmark-apps/classical/` and `secaggpp-benchmark/benchmark-apps/pp/`: persistent app copies inside the benchmark suite.
- `secaggpp-benchmark/secaggpp_secaggp_benchmark.sh`: host-based runner with env-overridable paths, log-degree (`degree=ceil(log2(N))+1`), shared single venv (`bench-venv`), auto-creates `~/.flwr/config.toml` for simulation, warm-up step to pre-download CIFAR-10 before timed runs.

#### Non-Demo Mode (done)
- `is-demo=false` enforced via `--run-config` override in the script so benchmarks use real CIFAR-10 data, real training (1 epoch), and real evaluation.

#### Docker (symlink approach, works)
- `secaggpp-benchmark/Dockerfile` uses a self-discovery symlink (extracts the framework path from `uv.lock` and creates a symlink) instead of fragile `sed` rewriting — no host-specific paths baked in.
- Single shared venv for both apps.
- Build with `--network host`, run with `--shm-size=512m --network host`.

### Fixed: Even `num-shares` Bug in Classical SecAgg+

The `SecAggPlusWorkflow.setup_stage` in `secaggplus_workflow.py` built `sa_params_dict` **before** the even→odd increment, so clients received `share_num=8` while the neighbour graph had 9 nodes. This silently crashed `_share_keys` on clients (`IndexError: list index out of range`), making the aggregation fail silently — the server's `DefaultWorkflow` loop continued without actual training. The bug only manifests when `num-shares` is even (e.g. `degree=8` at N=100, but not `degree=5` at N=10).

**Fix**: moved the even→odd check (and threshold recalculation for float thresholds) before `sa_params_dict` construction in `secaggplus_workflow.py`.

The SecAgg++ workflow was not affected — it already places the even check before `sa_params_dict`.

### Key Test Results (host, N clients, is-demo=false, 3 rounds each, with fix)
| N | degree | Classical | SecAgg++ | Ratio |
|---|--------|-----------|----------|-------|
| 10 | 5 | 51.81s | 52.32s | 1.01× |
| 100 | 8 | 69.56s | 68.83s | 0.99× |

Absolute wall-clock times vary with system load, but SecAgg++ remains performance-equivalent to classical SecAgg+ despite the additional AEAD and SHAKE256 overhead. The earlier 2.12× ratio was entirely a measurement artifact caused by the silent even-`num-shares` client crash.

### Performance Analysis (N=100, with fix)

Per-round timing (5 rounds. Round 1 includes training; rounds 2-5 reuse cached keys):

**Classical SecAgg+** (rounds 2-5 ~6s each):
| Round | Setup | ShareKeys | CollectMasked | Unmask | Total |
|-------|-------|-----------|---------------|--------|-------|
| 1     | 11.2s | 0.3s      | 5.2s          | 2.5s   | 19.2s |
| 2-5   | 0.3s  | 0.35s     | 3.0s          | 2.4s   | 6.0s |

**SecAgg++** (rounds 2-5 ~5.4s each):
| Round | Setup | ShareKeys | CollectMasked | Stage3_self_unmask | Unmask | Total |
|-------|-------|-----------|---------------|--------------------|--------|-------|
| 1     | 11.4s | 0.35s     | 5.3s          | 1.2s               | 1.7s   | 18.8s |
| 2-5   | 0.3s  | 0.34s     | 3.0s          | 1.3s               | 1.8s   | 5.4s |

SecAgg++ has ~1.3s Stage3_self_unmask overhead (Shamir combine) and ~0.1s extra vector operations (SUM_DERIVED_KEYS, E_VALUE), but these are offset by key reuse that saves ML-KEM decapsulations in rounds 2+.

With the spec-compliant crypto (AES-GCM with `pk`-bound KDF, `ml_encap` as AD, client-server onion encryption, and SHAKE256 per-entry masks), the client-side mask generation rises from ~7–15ms to ~30–60ms per client and the server-side ShareKeys stage pays for the onion unwrap/rewrap.  The end-to-end wall-clock remains within ~5% of classical SecAgg+.

**Verdict**: SecAgg++ and classical SecAgg+ are **performance-equivalent** at these scales. The protocol overhead of SecAgg++'s extra vectors and stronger crypto is balanced by symmetric-only communication in rounds 2+.

### How to Run
```bash
# Host
cd secaggpp-benchmark
./secaggpp_secaggp_benchmark.sh 100    # N=100 clients

# Docker (run from repo root)
docker build -f secaggpp-benchmark/Dockerfile --network host -t secagg-bench:latest .
docker run --rm --shm-size=512m --network host secagg-bench:latest 100
```

### Key Files
- `framework/py/flwr/common/secure_aggregation/crypto/pq_kem.py`: ML-KEM-768 wrapper (replaced X-Wing); added raw AES-key derivation.
- `framework/py/flwr/common/secure_aggregation/crypto/pq_pke.py`: spec-compliant CCA-secure PKE using ML-KEM-768 + AES-GCM with `key = HKDF(ss || pk)` and `ml_encap` as AD; added general `aead_encrypt`/`aead_decrypt` helpers.
- `framework/py/flwr/common/secure_aggregation/secaggplus_utils.py`: added `pseudo_rand_gen_secure` (SHAKE256-based per-entry mask generation).
- `framework/py/flwr/client/mod/secure_aggregation/secaggplus_plus_mod.py`: SecAgg++ client mod; key reuse, client-server onion encryption, spec-compliant inner ML-KEM+AES-GCM encryption, SHAKE256 masks.
- `framework/py/flwr/server/workflow/secure_aggregation/secaggplus_plus_workflow.py`: deterministic graph, skip public-key broadcast, server-side onion re-encryption.
- `framework/py/flwr/server/workflow/secure_aggregation/secaggplus_workflow.py` (even-bug fix applied)
- `secaggpp-benchmark/benchmark-apps/classical/`, `secaggpp-benchmark/benchmark-apps/pp/`
- `secaggpp-benchmark/secaggpp_secaggp_benchmark.sh`
- `secaggpp-benchmark/Dockerfile`
- `secaggpp-benchmark/pqc_aggregation.md`


