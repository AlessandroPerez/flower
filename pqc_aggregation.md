# SecAgg++ design

This document describes the post-quantum variant of Flower's SecAgg+ protocol implemented as `SecAggPlusPlusWorkflow` / `secaggplus_plus_mod`.

## Cryptographic primitives

- **Post-quantum public-key encryption** for transporting pairwise keys and Shamir shares between neighbours is built from the **X-Wing hybrid KEM** (ML-KEM-768 + X25519), combined with **Fernet** symmetric encryption (AES-128-CTR + HMAC-SHA256).  Concretely:

  ```text
  Enc(pk, payload) = X-Wing-KEM-encaps(pk) || Fernet(KDF(shared_secret), payload)
  Dec(sk, ciphertext) = KDF(X-Wing-KEM-decaps(sk, kem_ct)) then Fernet-decrypt
  ```

- **Pairwise mask derivation** is deterministic from the sender's master seed `b_S` and the receiver's node id:

  ```text
  k_{S->R} = SHA256(b_S || encoding(n_id_R))
  ```

Where `encoding(n_id_R)` is the 32-byte little-endian representation of the receiver's node id.

- **Self mask**: each client samples `u_S` and uses it as a seed for the same pseudo-random generator that produces the pairwise masks.

- **Shamir secret sharing** is used so neighbours can reconstruct the master seeds of clients that drop before sending their masked vector.

## Protocol stages

### 0. Setup

- Each client generates an X-Wing key pair `(pub_S, prv_S)` and sends `pub_S` to the server.
- The server picks the neighbour graph `NG(S)` exactly as in classical SecAgg+ and sends the protocol parameters (sample size, share number, threshold, quantization ranges, etc.) to every client.

### 1. Mask creation / key exchange

Each client `S`:

- Samples `b_S` and `u_S` (256-bit seeds).
- For every neighbour `R != S` derives the pairwise-mask seed `k_{S->R} = SHA256(b_S || encoding(n_id_R))`. For `R = S` the client only creates Shamir shares; no pairwise mask is needed.
- Creates Shamir shares of `b_S` and `u_S` with parameters `(t, n)` where `n = |NG(S)|`.
- Receives from the server the public keys and node ids of all neighbours in `NG(S)`.
- For every neighbour `R != S`, `S` encrypts the tuple `(source_id_S, destination_id_R, k_{S->R}, share_of(b_S for R), share_of(u_S for R))` with `pub_R` using the X-Wing + Fernet PKE above and sends the ciphertext to the server.
- `S` decrypts the incoming ciphertexts from its neighbours, verifies that each plaintext is really from the claimed source and addressed to `S`, and stores the tuples `(k_{R->S}, share_of(b_R for S), share_of(u_R for S))`.

### 2. Masked-vector collection

Each surviving client `S` (one that did not drop before this stage):

- Computes its weighted, quantized model update `w_S`.
- Sends to the server:

  ```text
  c_S = w_S + sum_{R in NG(S), R sent keys} k_{R->S} + u_S
  sum_S = sum_{R in NG(S)} k_{S->R}
  ```

The sum is taken component-wise over the model parameters and reduced modulo the configured modulus.

The server computes:

```text
agg = (sum over all surviving S of c_S) - (sum over all surviving S of sum_S)
```

### 3. Pairwise-mask recovery

The implementation handles clients that **dropped before sending
`(c_S, sum_S)`** (stage-2 dropouts).

For each stage-2 dropped client `D`:

- The server collects the shares of `b_D` held by the surviving neighbours of `D` and reconstructs `b_D`.
- For every surviving neighbour `S` of `D`, the server subtracts the outgoing mask `k_{D->S} = SHA256(b_D || encoding(n_id_S))` from `agg` (this cancels the `+k_{D->S}` term that remains in `c_S`).
- Every surviving client `S` sends the `b_D` share it holds for each stage-2 dropped neighbour `D`. It also computes and sends the vector `e_S = sum_{D in NG(S), D stage-2 dropped} PRF(k_{S->D})`; the server adds `e_S` to `agg` (this cancels the `-k_{S->D}` terms that remain in `sum_S`).

Stage-2 dropouts do not contribute a self mask to `agg`, so their `u_D` shares are not needed for pairwise-mask recovery.

### 4. Self-mask removal

After pairwise-mask recovery, the server removes the self mask of every client whose masked vector is included in `agg`.  Every self mask is reconstructed from Shamir shares of `u_C` held by the surviving neighbours of `C`; surviving clients do **not** send `u_C` directly.  This matches the privacy model of classical SecAgg+, where the server needs threshold shares to learn any client's private mask seed.

Stage-3 dropouts are handled transparently: because shares of `u_C` are collected for every client `C` whose `c_C` is in the aggregate, the server can still reconstruct `u_D` for a client `D` that sent `c_D` but failed to respond to unmask, as long as enough neighbours of `D` survive.

### 5. Final aggregation

```text
final_agg = agg' - sum_{C in P} PRF(u_C)
```

Where `P` is the set of clients whose masked vector was included in `agg` and `PRF(u_C)` is the self-mask vector derived from the reconstructed `u_C`.  The result is then factor-extracted, dequantized, and averaged as in classical SecAgg+.

## Summary of client uploads

The following list matches exactly what each surviving client sends to the server in each stage of the current implementation:

- **Stage 0 (setup)**: the client's X-Wing public key `pub_S`.
- **Stage 1 (share keys)**: one X-Wing + Fernet ciphertext per neighbour `R != S`. The plaintext is the tuple

  ```text
  (source_id_S, destination_id_R, k_{S->R}, share_of(b_S for R), share_of(u_S for R))
  ```

Where `k_{S->R}` is the pairwise-mask seed and the two shares are Shamir shares intended for `R`. The source and destination identifiers let the receiver verify the ciphertext really came from `S` and was meant for `R`.

- **Stage 2 (masked vectors)**: the masked update `c_S` and the derived-key sum `sum_S`.

- **Stage 3 (unmask)**: an ordered list of shares and an `e_S` correction vector:
  - for every active client `C` (every client whose `c_C` is in the aggregate), the `u_C` share held by `S`;
  - for every stage-2 dropped neighbour `D`, the `b_D` share held by `S`;
  - `e_S = sum_{D stage-2 dropped in NG(S)} PRF(k_{S->D})`.

Surviving clients do not send their own `u_S` directly. Every self mask is reconstructed from threshold shares by the server, exactly as in classical SecAgg+.

## Benchmarks

`speed_tests/benchmark_flower_secagg.py` compares classical SecAgg+ and SecAgg++ end-to-end using an in-memory Flower run. The default workload uses a linear model with **100,000 features** so that mask generation over the model parameters dominates the runtime, as it does in realistic federated training.

Representative wall-clock speedups (`classical / secaggplusplus`) on a single machine:

| clients | 0% dropout | 10% dropout |
|--------:|-----------:|------------:|
| 10      | 1.01×      | 1.17×       |
| 20      | 1.09×      | 1.36×       |
| 50      | 1.08×      | 1.30×       |
| 100     | 1.09×      | 1.26×       |

So SecAgg++ is modestly faster in this synthetic setting: roughly **1.1× with no dropouts** and **1.2–1.4× when some clients drop before sending their masked vector**. The improvement comes from two sources:

1. **Smaller Shamir secrets.** Classical SecAgg+ shares the 306-byte EC private key `sk1` with every neighbour, whereas SecAgg++ only shares 32-byte seeds `b_S` and `u_S`. Because Flower's Shamir implementation pads secrets into 16-byte chunks and splits each chunk independently, a 306-byte secret is about 6× more expensive to create and reconstruct than a 32-byte secret.
2. **Fewer ECDH operations.** Classical SecAgg+ generates two EC key pairs per client and derives two shared secrets per neighbour (one for encrypting shares, one for pairwise masks). SecAgg++ derives pairwise masks with a single SHA256 per neighbour and uses one X-Wing KEM operation for transport.

## Security considerations

### Honest-but-curious server

Under the standard honest-but-curious server model, SecAgg++ leaks the same information as classical SecAgg+:

- For every client `C` whose masked vector `c_C` is included in the aggregate, the server reconstructs `u_C` from Shamir shares and learns `C`'s self-mask seed.
- For every client `D` that dropped before sending `c_D` (a stage-2 dropout), the server reconstructs `b_D` from Shamir shares and learns `D`'s pairwise-mask master seed.

This leakage is inherent: the server must know these secrets to cancel the masks and compute the aggregate.  In both protocols, the server does **not** learn any pairwise mask between two active clients unless it also reconstructs the corresponding master seed, which requires threshold shares.

The key difference from the previous SecAgg++ design is that surviving clients no longer send `u_S` directly.  Instead, every self mask is reconstructed from threshold shares, matching the privacy model of classical SecAgg+.

### Transport security

Classical SecAgg+ encrypts Shamir shares with keys derived from ECDH.  SecAgg++ uses X-Wing (ML-KEM-768 + X25519) + Fernet.  Against a classical adversary the two provide comparable confidentiality for in-flight shares.  Against a quantum adversary, SecAgg++ retains confidentiality because X-Wing is built on ML-KEM, whereas the ECDH-based transport of classical SecAgg+ does not.

### Malicious server

Neither protocol protects against a malicious server that lies about the active/dead client lists to make clients reveal shares they would not otherwise send.  Defending against such a server requires additional authentication and/or verification mechanisms beyond the scope of this document.
