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

This module builds a CCA-secure PKE for arbitrary messages from the X-Wing
hybrid KEM (ML-KEM-768 + X25519) plus Fernet symmetric encryption.  It is used
by SecAggPlusPlus to transport pairwise keys and Shamir shares confidentially.
"""


from __future__ import annotations

from flwr.common.secure_aggregation.crypto.pq_kem import (
    decapsulate_secret,
    encapsulate,
    generate_key_pair,
)
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    decrypt as symmetric_decrypt,
)
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    encrypt as symmetric_encrypt,
)
from flwr.common.secure_aggregation.crypto.pq_kem import (
    derive_share_encryption_secret,
)


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
        The recipient's public key.
    plaintext : bytes
        The message to encrypt.

    Returns
    -------
    bytes
        ``ciphertext || symmetric_ciphertext`` where ``ciphertext`` is the
        X-Wing KEM ciphertext and ``symmetric_ciphertext`` is the Fernet
        encryption of ``plaintext`` under the derived key.
    """
    shared_secret, kem_ciphertext = encapsulate(public_key)
    sym_key = derive_share_encryption_secret(shared_secret)
    return kem_ciphertext + symmetric_encrypt(sym_key, plaintext)


def decrypt(secret_key: bytes, ciphertext: bytes) -> bytes:
    """Decrypt a ciphertext encrypted with :func:`encrypt`.

    Parameters
    ----------
    secret_key : bytes
        The recipient's secret key.
    ciphertext : bytes
        The combined KEM + symmetric ciphertext produced by :func:`encrypt`.

    Returns
    -------
    bytes
        The original plaintext.
    """
    # X-Wing ciphertext length is fixed at 1120 bytes.
    KEM_CT_LEN = 1120
    kem_ciphertext = ciphertext[:KEM_CT_LEN]
    sym_ciphertext = ciphertext[KEM_CT_LEN:]
    shared_secret = decapsulate_secret(kem_ciphertext, secret_key)
    sym_key = derive_share_encryption_secret(shared_secret)
    return symmetric_decrypt(sym_key, sym_ciphertext)
