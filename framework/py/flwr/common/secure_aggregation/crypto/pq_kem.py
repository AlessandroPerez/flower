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
"""Post-quantum KEM helpers for SecAgg+.

This module wraps ``cryptography``'s ML-KEM-768 implementation and provides the
key-derivation functions used by the PQ SecAgg+ protocol.  ML-KEM-768 is used
on its own (rather than the X-Wing hybrid) to keep public keys and ciphertexts
as small as possible while retaining NIST PQC security level 3.
"""


from __future__ import annotations

import base64
from typing import cast

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.mlkem import (
    MLKEM768PrivateKey,
    MLKEM768PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PAIRWISE_INFO = b"flwr-secaggplus-pq-pairwise"
SHARE_ENC_INFO = b"flwr-secaggplus-pq-share-enc"

# Raw byte lengths for ML-KEM-768.
PUBLIC_KEY_LEN = 1184
CIPHERTEXT_LEN = 1088
SHARED_SECRET_LEN = 32


def generate_key_pair() -> tuple[bytes, bytes]:
    """Generate an ephemeral ML-KEM-768 key pair.

    Returns
    -------
    tuple[bytes, bytes]
        ``(public_key, secret_key)``.  The secret key is the raw 64-byte seed
        that can be fed to :meth:`MLKEM768PrivateKey.from_seed_bytes`.
    """
    private_key = MLKEM768PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw()
    secret_key = private_key.private_bytes_raw()
    return public_key, secret_key


def encapsulate(public_key: bytes) -> tuple[bytes, bytes]:
    """Encapsulate to an ML-KEM-768 public key.

    Parameters
    ----------
    public_key : bytes
        The recipient's raw ML-KEM-768 public key.

    Returns
    -------
    tuple[bytes, bytes]
        ``(shared_secret, ciphertext)``.
    """
    recipient_key = MLKEM768PublicKey.from_public_bytes(public_key)
    shared_secret, ciphertext = recipient_key.encapsulate()
    return shared_secret, ciphertext


def decapsulate_secret(ciphertext: bytes, secret_key: bytes) -> bytes:
    """Decapsulate an ML-KEM-768 ciphertext using the secret key.

    Parameters
    ----------
    ciphertext : bytes
        The ML-KEM-768 ciphertext produced by :func:`encapsulate`.
    secret_key : bytes
        The recipient's raw ML-KEM-768 secret key (64-byte seed).

    Returns
    -------
    bytes
        The 32-byte shared secret.
    """
    private_key = MLKEM768PrivateKey.from_seed_bytes(secret_key)
    return private_key.decapsulate(ciphertext)


def derive_pairwise_secret(ss_forward: bytes, ss_reverse: bytes) -> bytes:
    """Derive the pairwise mask seed from two one-way ML-KEM secrets.

    The pairwise secret depends on both directions of encapsulation so that
    both clients contribute randomness.  The two secrets are sorted before
    concatenation so that both clients derive the same value regardless of
    which direction they treat as "forward".

    Parameters
    ----------
    ss_forward : bytes
        One direction of the ML-KEM shared secret.
    ss_reverse : bytes
        The opposite direction of the ML-KEM shared secret.

    Returns
    -------
    bytes
        The 32-byte pairwise mask seed.
    """
    ordered = sorted([ss_forward, ss_reverse])
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=PAIRWISE_INFO,
    ).derive(ordered[0] + ordered[1])


def derive_share_encryption_secret(shared_secret: bytes) -> bytes:
    """Derive a Fernet-compatible key from a one-way ML-KEM shared secret.

    This is kept for backwards compatibility with the older PQ variant that
    uses Fernet symmetric encryption.

    Parameters
    ----------
    shared_secret : bytes
        A one-way ML-KEM shared secret.

    Returns
    -------
    bytes
        A 32-byte raw key encoded as url-safe base64.
    """
    raw_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=SHARE_ENC_INFO,
    ).derive(shared_secret)
    return base64.urlsafe_b64encode(raw_key)


def derive_share_encryption_key(shared_secret: bytes) -> bytes:
    """Derive a raw 32-byte AES key from a one-way ML-KEM shared secret.

    Parameters
    ----------
    shared_secret : bytes
        A one-way ML-KEM shared secret.

    Returns
    -------
    bytes
        A 32-byte raw key suitable for AES-GCM.
    """
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=SHARE_ENC_INFO,
    ).derive(shared_secret)
