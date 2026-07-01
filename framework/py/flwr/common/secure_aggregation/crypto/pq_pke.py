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
"""Post-quantum public-key encryption helper.

This module builds a CCA-secure PKE for arbitrary messages from ML-KEM-768
plus AES-GCM symmetric encryption, exactly as described in the SecAgg++
documentation:

  Enc(pk, m):
    ss, ml_encap = ML-KEM-encaps(pk)
    key = HKDF(ss || pk)
    c = AES-GCM(nonce, ml_encap, key, m)
    return ml_encap || nonce || c

  Dec(sk, ct):
    ml_encap = ct[:CIPHERTEXT_LEN]
    nonce = ct[CIPHERTEXT_LEN:CIPHERTEXT_LEN+NONCE_LEN]
    c = ct[CIPHERTEXT_LEN+NONCE_LEN:]
    ss = ML-KEM-decaps(sk, ml_encap)
    key = HKDF(ss || pk)
    return AES-GCM-decrypt(nonce, ml_encap, key, c)

The module also provides general AEAD helpers used for the client-server
onion-encryption layer.
"""


from __future__ import annotations

import os
from typing import cast

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from flwr.common.secure_aggregation.crypto.pq_kem import (
    CIPHERTEXT_LEN,
    PUBLIC_KEY_LEN,
    decapsulate_secret,
    encapsulate,
    generate_key_pair,
)

NONCE_LEN = 12
SHARE_ENC_INFO = b"flwr-secaggplus-pq-share-enc"


def _derive_share_key(shared_secret: bytes, public_key: bytes) -> bytes:
    """Derive the symmetric AEAD key from the ML-KEM secret and public key.

    This follows the SecAgg++ design: ``key = KDF(ss || pk)``.
    """
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=SHARE_ENC_INFO,
    ).derive(shared_secret + public_key)


def _public_key_from_secret_key(secret_key: bytes) -> bytes:
    """Re-derive the ML-KEM public key from the secret-key seed."""
    private_key = MLKEM768PrivateKey.from_seed_bytes(secret_key)
    return cast(bytes, private_key.public_key().public_bytes_raw())


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a PQ PKE key pair.

    Returns
    -------
    tuple[bytes, bytes]
        ``(public_key, secret_key)``.
    """
    return generate_key_pair()


def encrypt(public_key: bytes, plaintext: bytes) -> bytes:
    """Encrypt an arbitrary-length plaintext under a PQ public key.

    Parameters
    ----------
    public_key : bytes
        The recipient's ML-KEM-768 public key.
    plaintext : bytes
        The message to encrypt.

    Returns
    -------
    bytes
        ``ml_encap || nonce || symmetric_ciphertext`` where ``ml_encap`` is the
        ML-KEM ciphertext, ``nonce`` is the AES-GCM nonce, and
        ``symmetric_ciphertext`` is the AES-GCM encryption of ``plaintext`` with
        ``ml_encap`` as associated data.
    """
    if len(public_key) != PUBLIC_KEY_LEN:
        raise ValueError(
            f"Invalid ML-KEM public key length: expected {PUBLIC_KEY_LEN}, "
            f"got {len(public_key)}"
        )
    shared_secret, kem_ciphertext = encapsulate(public_key)
    sym_key = _derive_share_key(shared_secret, public_key)
    nonce = os.urandom(NONCE_LEN)
    aesgcm = AESGCM(sym_key)
    ct = aesgcm.encrypt(nonce, plaintext, kem_ciphertext)
    return kem_ciphertext + nonce + ct


def decrypt(
    secret_key: bytes, ciphertext: bytes, public_key: bytes | None = None
) -> bytes:
    """Decrypt a ciphertext encrypted with :func:`encrypt`.

    Parameters
    ----------
    secret_key : bytes
        The recipient's ML-KEM secret key.
    ciphertext : bytes
        The combined KEM + AEAD ciphertext produced by :func:`encrypt`.
    public_key : bytes | None
        The recipient's ML-KEM public key. If ``None``, it is re-derived from
        ``secret_key``.

    Returns
    -------
    bytes
        The original plaintext.
    """
    if len(ciphertext) < CIPHERTEXT_LEN + NONCE_LEN:
        raise ValueError("Ciphertext is too short.")
    if public_key is None:
        public_key = _public_key_from_secret_key(secret_key)
    kem_ciphertext = ciphertext[:CIPHERTEXT_LEN]
    nonce = ciphertext[CIPHERTEXT_LEN : CIPHERTEXT_LEN + NONCE_LEN]
    ct = ciphertext[CIPHERTEXT_LEN + NONCE_LEN :]
    shared_secret = decapsulate_secret(kem_ciphertext, secret_key)
    sym_key = _derive_share_key(shared_secret, public_key)
    aesgcm = AESGCM(sym_key)
    return aesgcm.decrypt(nonce, ct, kem_ciphertext)


def aead_encrypt(key: bytes, plaintext: bytes, associated_data: bytes) -> bytes:
    """AES-GCM encrypt with explicit associated data.

    Parameters
    ----------
    key : bytes
        A 32-byte AES-GCM key.
    plaintext : bytes
        Message to encrypt.
    associated_data : bytes
        Associated data that is authenticated but not encrypted.

    Returns
    -------
    bytes
        ``nonce || ciphertext``.
    """
    nonce = os.urandom(NONCE_LEN)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    return nonce + ct


def aead_decrypt(key: bytes, ciphertext: bytes, associated_data: bytes) -> bytes:
    """AES-GCM decrypt with explicit associated data.

    Parameters
    ----------
    key : bytes
        A 32-byte AES-GCM key.
    ciphertext : bytes
        ``nonce || ciphertext`` produced by :func:`aead_encrypt`.
    associated_data : bytes
        Associated data that was used during encryption.

    Returns
    -------
    bytes
        The original plaintext.
    """
    if len(ciphertext) < NONCE_LEN:
        raise ValueError("AEAD ciphertext is too short.")
    nonce = ciphertext[:NONCE_LEN]
    ct = ciphertext[NONCE_LEN:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, associated_data)
