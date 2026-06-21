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
"""Post-quantum hybrid KEM helpers for SecAgg+.

This module wraps ``xwing-kem`` (X-Wing: ML-KEM-768 + X25519) and provides
the key-derivation functions used by the PQ SecAgg+ protocol.
"""


from __future__ import annotations

import base64
from typing import cast

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from xwing_kem import decapsulate as _xwing_decapsulate
from xwing_kem import encapsulate as _xwing_encapsulate
from xwing_kem import generate_keypair as _xwing_generate_keypair

PAIRWISE_INFO = b"flwr-secaggplus-pq-pairwise"
SHARE_ENC_INFO = b"flwr-secaggplus-pq-share-enc"


def generate_key_pair() -> tuple[bytes, bytes]:
    """Generate an ephemeral X-Wing key pair.

    Returns
    -------
    tuple[bytes, bytes]
        ``(public_key, secret_key)``.
    """
    kp = _xwing_generate_keypair()
    return kp.public_key, kp.secret_key


def encapsulate(public_key: bytes) -> tuple[bytes, bytes]:
    """Encapsulate to an X-Wing public key.

    Parameters
    ----------
    public_key : bytes
        The recipient's X-Wing public key.

    Returns
    -------
    tuple[bytes, bytes]
        ``(shared_secret, ciphertext)``.
    """
    shared_secret, ciphertext = _xwing_encapsulate(public_key)
    return shared_secret, ciphertext


def decapsulate_secret(ciphertext: bytes, secret_key: bytes) -> bytes:
    """Decapsulate an X-Wing ciphertext using the secret key.

    Parameters
    ----------
    ciphertext : bytes
        The X-Wing ciphertext produced by ``encapsulate``.
    secret_key : bytes
        The recipient's X-Wing secret key.

    Returns
    -------
    bytes
        The 32-byte shared secret.
    """
    return cast(bytes, _xwing_decapsulate(ciphertext, secret_key))


def derive_pairwise_secret(ss_forward: bytes, ss_reverse: bytes) -> bytes:
    """Derive the pairwise mask seed from two one-way X-Wing secrets.

    The pairwise secret depends on both directions of encapsulation so that
    both clients contribute randomness.  The two secrets are sorted before
    concatenation so that both clients derive the same value regardless of
    which direction they treat as "forward".

    Parameters
    ----------
    ss_forward : bytes
        One direction of the X-Wing shared secret.
    ss_reverse : bytes
        The opposite direction of the X-Wing shared secret.

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
    """Derive a Fernet-compatible key from a one-way X-Wing shared secret.

    Parameters
    ----------
    shared_secret : bytes
        A one-way X-Wing shared secret.

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
