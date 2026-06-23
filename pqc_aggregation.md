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
- The server picks the neighbour graph `NG(S)` exactly as in classical SecAgg+ and tells every client the public keys and node ids of its neighbours.

### 1. Mask creation / key exchange

Each client `S`:

- Samples `b_S` and `u_S` (256-bit seeds).
- For every neighbour `R` (including itself only for share distribution), derives `k_{S->R} = SHA256(b_S || encoding(n_id_R))`.
- Creates Shamir shares of `b_S` and `u_S` with parameters `(t, n)` where `n = |NG(S)|`.
- For every neighbour `R != S`, encrypts the tuple `(k_{S->R}, share_of(b_S for R), share_of(u_S for R))` with `pub_R` using the X-Wing + Fernet PKE above and sends the ciphertext to the server.
- Decrypts the incoming ciphertexts from its neighbours to obtain the corresponding tuples `(k_{R->S}, share_of(b_R for S), share_of(u_R for S))`.

### 2. Masked-vector collection

Each surviving client `S` (one that did not drop before this stage):

- Computes its weighted, quantized model update `w_S`.
- Sends to the server:

  ```text
  c_S = w_S + sum_{R in NG(S), R active} k_{R->S} + u_S
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

After pairwise-mask recovery, the server removes the self mask of every client whose masked vector is included in `agg`:

- For each surviving client `S`, the server uses `u_S` sent directly by `S`.
- For each client `D` that sent `(c_D, sum_D)` but dropped before unmask (stage-3 dropout), the server collects the shares of `u_D` held by the surviving neighbours of `D` and reconstructs `u_D`.

The server subtracts `PRF(u_C)` for all such clients `C` from `agg`.

### 5. Final aggregation

```text
final_agg = agg' - sum_{C in P} PRF(u_C)
```

Where `P` is the set of clients whose masked vector was included in `agg` and `PRF(u_C)` is the self-mask vector derived from `u_C`.  The result is then factor-extracted, dequantized, and averaged as in classical SecAgg+.
