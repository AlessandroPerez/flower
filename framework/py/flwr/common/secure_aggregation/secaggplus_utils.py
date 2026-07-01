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
"""Utility functions for the SecAgg/SecAgg+ protocol."""


import numpy as np

from flwr.common import NDArrayInt


def share_keys_plaintext_concat(
    src_node_id: int, dst_node_id: int, b_share: bytes, sk_share: bytes
) -> bytes:
    """Combine arguments to bytes.

    Parameters
    ----------
    src_node_id : int
        the node ID of the source.
    dst_node_id : int
        the node ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.

    Returns
    -------
    bytes
        The combined bytes of all the arguments.
    """
    return b"".join(
        [
            int.to_bytes(src_node_id, 8, "little", signed=False),
            int.to_bytes(dst_node_id, 8, "little", signed=False),
            int.to_bytes(len(b_share), 4, "little"),
            b_share,
            sk_share,
        ]
    )


def share_keys_plaintext_separate(plaintext: bytes) -> tuple[int, int, bytes, bytes]:
    """Retrieve arguments from bytes.

    Parameters
    ----------
    plaintext : bytes
        the bytes containing 4 arguments.

    Returns
    -------
    src_node_id : int
        the node ID of the source.
    dst_node_id : int
        the node ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.
    """
    src, dst, mark = (
        int.from_bytes(plaintext[:8], "little", signed=False),
        int.from_bytes(plaintext[8:16], "little", signed=False),
        int.from_bytes(plaintext[16:20], "little"),
    )
    ret = (src, dst, plaintext[20 : 20 + mark], plaintext[20 + mark :])
    return ret


def share_keys_plaintext_concat_rd(
    src_node_id: int, dst_node_id: int, rd_seed_share: bytes
) -> bytes:
    """Combine a single rd_seed share into bytes."""
    return b"".join(
        [
            int.to_bytes(src_node_id, 8, "little", signed=False),
            int.to_bytes(dst_node_id, 8, "little", signed=False),
            int.to_bytes(len(rd_seed_share), 4, "little"),
            rd_seed_share,
        ]
    )


def share_keys_plaintext_separate_rd(plaintext: bytes) -> tuple[int, int, bytes]:
    """Retrieve src, dst, and rd_seed_share from bytes."""
    src = int.from_bytes(plaintext[:8], "little", signed=False)
    dst = int.from_bytes(plaintext[8:16], "little", signed=False)
    mark = int.from_bytes(plaintext[16:20], "little")
    rd_seed_share = plaintext[20 : 20 + mark]
    return src, dst, rd_seed_share


def share_keys_plaintext_concat_ps(
    src_node_id: int, dst_node_id: int, ps_shares: list[tuple[int, bytes]]
) -> bytes:
    """Combine pairwise-secret shares into bytes.

    Each element of ``ps_shares`` is a tuple ``(owner_node_id, share_bytes)``,
    where ``owner_node_id`` is the client that created the pairwise secret.
    """
    parts: list[bytes] = [
        int.to_bytes(src_node_id, 8, "little", signed=False),
        int.to_bytes(dst_node_id, 8, "little", signed=False),
        int.to_bytes(len(ps_shares), 4, "little"),
    ]
    for owner_node_id, share in ps_shares:
        parts.append(int.to_bytes(owner_node_id, 8, "little", signed=False))
        parts.append(int.to_bytes(len(share), 4, "little"))
        parts.append(share)
    return b"".join(parts)


def share_keys_plaintext_separate_ps(
    plaintext: bytes,
) -> tuple[int, int, list[tuple[int, bytes]]]:
    """Retrieve src, dst, and pairwise-secret shares from bytes."""
    src = int.from_bytes(plaintext[:8], "little", signed=False)
    dst = int.from_bytes(plaintext[8:16], "little", signed=False)
    num_shares = int.from_bytes(plaintext[16:20], "little")
    offset = 20
    ps_shares: list[tuple[int, bytes]] = []
    for _ in range(num_shares):
        owner = int.from_bytes(plaintext[offset : offset + 8], "little", signed=False)
        offset += 8
        length = int.from_bytes(plaintext[offset : offset + 4], "little")
        offset += 4
        ps_shares.append((owner, plaintext[offset : offset + length]))
        offset += length
    return src, dst, ps_shares


def pseudo_rand_gen(
    seed: bytes, num_range: int, dimensions_list: list[tuple[int, ...]]
) -> list[NDArrayInt]:
    """Seeded pseudo-random number generator for noise generation with Numpy."""
    assert len(seed) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(seed), 4):
        seed32 ^= int.from_bytes(seed[i : i + 4], "little")
    # pylint: disable-next=no-member
    gen = np.random.RandomState(seed32)
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(gen.randint(0, num_range - 1), dtype=np.int64)
        else:
            arr = gen.randint(0, num_range - 1, dimension, dtype=np.int64)
        output.append(arr)
    return output


def derive_pairwise_key(seed: bytes, n_id: int) -> bytes:
    """Derive a 256-bit pairwise key from a master seed and a neighbour node ID.

    The derivation matches the SecAggPlusPlus design:
    ``SHA3-256(seed || encoding(n_id))`` where ``encoding(n_id)`` is the
    32-byte little-endian representation of the node ID.
    """
    from hashlib import sha3_256

    return sha3_256(seed + int.to_bytes(n_id, 32, "little")).digest()


def pseudo_rand_gen_secure(
    seed: bytes, num_range: int, dimensions_list: list[tuple[int, ...]]
) -> list[NDArrayInt]:
    """Cryptographically secure seeded pseudo-random mask generator.

    This is the SecAgg++-specific variant of :func:`pseudo_rand_gen`.  Each
    output element is produced from ``ceil(log2(num_range)) + 128`` bits of an
    AES-256-CTR keystream keyed by ``seed``, then reduced modulo ``num_range``.
    The extra 128 bits of margin keep the modular-reduction bias negligible
    even when ``num_range`` is not a power of two.

    Parameters
    ----------
    seed : bytes
        32-byte seed/key for the generator.
    num_range : int
        Upper bound (exclusive) of each generated value.
    dimensions_list : list[tuple[int, ...]]
        Shape of each output array.

    Returns
    -------
    list[NDArrayInt]
        A list of pseudo-random integer arrays matching ``dimensions_list``.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if num_range <= 0:
        raise ValueError("`num_range` must be positive.")
    if len(seed) != 32:
        raise ValueError("`seed` must be 32 bytes for AES-256.")

    total_elements = sum(int(np.prod(dim)) if dim else 1 for dim in dimensions_list)
    if total_elements == 0:
        return []

    # Each mask value must be sampled from ceil(log2(num_range)) + 128 bits
    # of randomness before reduction modulo num_range, as specified in the
    # SecAgg++ design document.
    bits_per_value = num_range.bit_length() + 128
    bytes_per_value = (bits_per_value + 7) // 8

    total_bytes = total_elements * bytes_per_value
    # Each seed is used for a single mask, so a fixed IV is safe.  The
    # deterministic output is the same as a one-time keystream expansion.
    cipher = Cipher(algorithms.AES(seed), modes.CTR(b"\x00" * 16))
    encryptor = cipher.encryptor()
    random_bytes = np.frombuffer(
        encryptor.update(b"\x00" * total_bytes) + encryptor.finalize(), dtype=np.uint8
    ).reshape(total_elements, bytes_per_value)

    # Convert big-endian byte chunks to integers and reduce mod num_range.
    powers = 256 ** np.arange(bytes_per_value - 1, -1, -1, dtype=np.int64)
    ints = random_bytes.astype(np.int64).dot(powers) % num_range

    output: list[NDArrayInt] = []
    offset = 0
    for dim in dimensions_list:
        n = int(np.prod(dim)) if dim else 1
        arr = ints[offset : offset + n]
        if dim:
            arr = arr.reshape(dim)
        output.append(arr)
        offset += n

    return output


def share_keys_plaintext_concat_plus(
    src_node_id: int,
    dst_node_id: int,
    pairwise_key: bytes,
    b_seed_share: bytes,
    self_mask_share: bytes,
) -> bytes:
    """Combine a SecAggPlusPlus payload into bytes."""
    return b"".join(
        [
            int.to_bytes(src_node_id, 8, "little", signed=False),
            int.to_bytes(dst_node_id, 8, "little", signed=False),
            int.to_bytes(len(pairwise_key), 4, "little"),
            pairwise_key,
            int.to_bytes(len(b_seed_share), 4, "little"),
            b_seed_share,
            int.to_bytes(len(self_mask_share), 4, "little"),
            self_mask_share,
        ]
    )


def share_keys_plaintext_separate_plus(
    plaintext: bytes,
) -> tuple[int, int, bytes, bytes, bytes]:
    """Retrieve src, dst, pairwise key, b-seed share, and self-mask share."""
    src = int.from_bytes(plaintext[:8], "little", signed=False)
    dst = int.from_bytes(plaintext[8:16], "little", signed=False)
    offset = 16
    length = int.from_bytes(plaintext[offset : offset + 4], "little")
    offset += 4
    pairwise_key = plaintext[offset : offset + length]
    offset += length
    length = int.from_bytes(plaintext[offset : offset + 4], "little")
    offset += 4
    b_seed_share = plaintext[offset : offset + length]
    offset += length
    length = int.from_bytes(plaintext[offset : offset + 4], "little")
    offset += 4
    self_mask_share = plaintext[offset : offset + length]
    return src, dst, pairwise_key, b_seed_share, self_mask_share
