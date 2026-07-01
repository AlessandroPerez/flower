# SecAgg++ design

This document describes the post-quantum variant of Flower's SecAgg+ protocol implemented as `SecAggPlusPlusWorkflow` / `secaggplus_plus_mod`.

## Cryptographic primitives

- **Post-quantum public-key encryption** for transporting pairwise keys and Shamir shares between neighbours is built from **ML-KEM-768** plus **AES-GCM** authenticated encryption. Concretely:

  ```text
  Enc_AEAD(nonce, AD, key, message)
  Dec_AEAD(nonce, AD, key, ciphertext)

  Enc(pk, message) :=
  - ml_encap = ML-KEM-encaps(pk, ss)
  - key = HKDF(ss || pk)
  - c = Enc_AEAD(nonce, ml_encap, key, m)
  Dec(sk, ciphertext) :=
  - ml_encap = ciphertext[:|ml_encap|]
  - ss = ML-KEM-decaps(sk, ml_encap)
  - key = HKDF(ss || pk)
  - m = Dec_AEAD(nonce, ml_encap, key, c)
  ```

- **Pairwise mask derivation** is deterministic from the sender's master seed `b_S` and the receiver's node id.  The pairwise PRF key is

  ```text
  k_{S->R} = H(b_S || encoding(n_id_R))
  ```

Where `H` is a 256-bit hash and `encoding(n_id_R)` is the 32-byte little-endian representation of the receiver's node id.  The full mask vector is obtained by applying a PRF to this key; each entry is produced from `ceiling(log_2(p)) + 128` bits before reduction modulo `p`:

  ```text
  mask_{S->R}[i] = PRF(k_{S->R}, i, ceiling(log_2(p)) + 128) mod p
  ```

  so that every parameter entry receives independent entropy and the modular reduction is unbiased.

- **Self mask**: each client samples `u_S` and uses it as a seed for the same PRF that produces the pairwise masks.

- **Shamir secret sharing** is used so neighbours can reconstruct the master seeds of clients that drop before sending their masked vector.

## Protocol stages

### 0. Setup

- Each client generates an ML-KEM-768 key pair `(pub_S, prv_S)` for pairwise communication and a random 256-bit symmetric key `key_{S-server}` for client-server communication. It sends both `pub_S` and `key_{S-server}` to the server.
- The server picks the neighbour graph `NG(S)` exactly as in classical SecAgg+ and sends the protocol parameters (sample size, share number, threshold, quantization ranges, p etc.) to every client.

### 1. Mask creation / key exchange

Each client `S`:

- Receives from the server the public keys and node ids of all neighbours in `NG(S)`.
- Samples `b_S` and `u_S` (in Z_p).
- For every neighbour `R` derives the pairwise-mask seed `k_{S->R} = H(b_S || encoding(n_id_R))`
- Creates Shamir shares of `b_S` and `u_S` with parameters `(t, n)` where `n = |NG(S)|`.
- For every neighbour `R != S`, `S` encrypts the tuple `m = (k_{S->R}, share_of(b_S for R), share_of(u_S for R))` first using `Enc(pk_R, m)`, then it encrypts the result using the client-server key `Enc_AEAD(nonce, (n_id_S, n_id_R), key_{S-server}, Enc(pk_R, m))` and sends it to the server.
- The server decrypts the outer client-server layer with `key_{S-server}` and re-encrypts it with the neighbour's client-server key `Enc_AEAD(nonce, (n_id_S, n_id_R), key_{R-server}, Enc(pk_R, m))` before forwarding.
- `S` decrypts the incoming ciphertexts from its neighbours, sent through the server, using `key_{S-server}` first and `prv_S` later.
- `S` stores the tuples `(k_{R->S}, share_of(b_R for S), share_of(u_R for S))`.

In subsequent rounds the pairwise ML-KEM secrets established in round 1 are reused, so the inner encryption becomes a symmetric AEAD under the cached bidirectional secret rather than a fresh ML-KEM encapsulation.

### 2. Masked-vector collection

Each surviving client `S` (one that did not drop before this stage):

- Computes its weighted, quantized model update `w_S`.
- Sends to the server:

  ```text
  c_S = w_S + sum_{R in NG(S), R sent keys} k_{R->S} + u_S mod p
  sum_S = sum_{R in NG(S)} k_{S->R} mod p
  ```

The sum is taken component-wise over the model parameters and reduced modulo p.

The server computes:

```text
agg = (sum over all surviving S of c_S) - (sum over all surviving S of sum_S) mod p
```

### 3. Pairwise-mask recovery

The implementation handles clients that **dropped before sending
`(c_S, sum_S)`** (stage-2 dropouts).

For each stage-2 dropped client `D`:

- Every surviving client `S` sends the `b_D` share it holds for each stage-2 dropped neighbour `D`. It also computes and sends the correction vector `e_S`, which is the component-wise sum of the pairwise mask vectors derived from `S`'s keys towards each dropped neighbour:

  ```text
  e_S = sum_{D in NG(S), D stage-2 dropped} PRF(k_{S->D}) mod p
  ```

  where `PRF(k)` expands the seed `k` into a full model-shaped mask vector in `[0, p)`.

- The server collects the shares of `b_D` held by the surviving neighbours of `D` and reconstructs `b_D`.
- For every surviving neighbour `S` of `D`, the server subtracts the outgoing mask vector `PRF(k_{D->S})` with `k_{D->S} = H(b_D || encoding(n_id_S))` from `agg` (this cancels the `+PRF(k_{D->S})` term that remains in `c_S`).
- The server adds each `e_S` to `agg` (this cancels the `-PRF(k_{S->D})` terms that remain in `sum_S`).

Stage-2 dropouts do not contribute a self mask to `agg`, so their `u_D` shares are not needed for pairwise-mask recovery.

### 4. Self-mask removal

After pairwise-mask recovery, the server removes the self mask of every client whose masked vector is included in `agg`.  Every self mask is reconstructed from Shamir shares of `u_C` held by the surviving neighbours of `C`; surviving clients do **not** send `u_C` directly.  This matches the privacy model of classical SecAgg+, where the server needs threshold shares to learn any client's private mask seed.

Stage-3 dropouts are handled transparently: because shares of `u_C` are collected for every client `C` whose `c_C` is in the aggregate, the server can still reconstruct `u_D` for a client `D` that sent `c_D` but failed to respond to unmask, as long as enough neighbours of `D` survive.

### 5. Final aggregation

```text
final_agg = agg' - sum_{C in P} PRF(u_C) mod p
```

Where `P` is the set of clients whose masked vector was included in `agg` and `PRF(u_C)` is the self-mask vector derived from the reconstructed `u_C`.  The result is then factor-extracted, dequantized, and averaged as in classical SecAgg+.

## Summary of client uploads

The following list matches exactly what each surviving client sends to the server in each stage of the current implementation:

- **Stage 0 (setup)**: the client's ML-KEM public key `pub_S`.
- **Stage 1 (share keys)**: one ML-KEM-768 + Fernet ciphertext per neighbour `R = S`. The plaintext is the tuple

  ```text
  (k_{S->R}, share_of(b_S for R), share_of(u_S for R))
  ```

Where `k_{S->R}` is the pairwise-mask seed and the two shares are Shamir shares intended for `R`.

- **Stage 2 (masked vectors)**: the masked update `c_S` and the derived-key sum `sum_S`.

- **Stage 3 (unmask)**: an ordered list of shares and an `e_S` correction vector:
  - for every active client `C` (every client whose `c_C` is in the aggregate), the `u_C` share held by `S`;
  - for every stage-2 dropped neighbour `D`, the `b_D` share held by `S`;
  - `e_S = sum_{D stage-2 dropped in NG(S)} PRF(k_{S->D})`, the component-wise sum of the derived pairwise mask vectors.

Surviving clients do not send their own `u_S` directly. Every self mask is reconstructed from threshold shares by the server, exactly as in classical SecAgg+. All the communications with the server are encrypted using the key established between the client and the server.

## Security considerations

### Honest-but-curious server

Under the standard honest-but-curious server model, SecAgg++ leaks the same information as classical SecAgg+:

- For every client `C` whose masked vector `c_C` is included in the aggregate, the server reconstructs `u_C` from Shamir shares and learns `C`'s self-mask seed.
- For every client `D` that dropped before sending `c_D` (a stage-2 dropout), the server reconstructs `b_D` from Shamir shares and learns `D`'s pairwise-mask master seed.

This leakage is inherent: the server must know these secrets to cancel the masks and compute the aggregate.  In both protocols, the server does **not** learn any pairwise mask between two active clients unless it also reconstructs the corresponding master seed, which requires threshold shares.

### Transport security

Classical SecAgg+ encrypts Shamir shares with keys derived from ECDH.  SecAgg++ uses ML-KEM-768 + Fernet.  Against a classical adversary the two provide comparable confidentiality for in-flight shares.  Against a quantum adversary, SecAgg++ retains confidentiality because ML-KEM is built on ML-KEM, whereas the ECDH-based transport of classical SecAgg+ does not.

### Malicious server

Neither protocol protects against a malicious server that lies about the active/dead client lists to make clients reveal shares they would not otherwise send.  Defending against such a server requires additional authentication and/or verification mechanisms beyond the scope of this document.
