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
"""Tests for the post-quantum CCA-secure PKE helper."""


import pytest
from cryptography.fernet import InvalidToken

from .pq_pke import decrypt, encrypt, generate_keypair


class TestPQPKE:
    """Tests for the X-Wing + Fernet PKE wrapper."""

    def test_keypair_generation(self) -> None:
        """Generated keys are non-empty and distinct."""
        public_key, secret_key = generate_keypair()
        assert isinstance(public_key, bytes)
        assert isinstance(secret_key, bytes)
        assert len(public_key) > 0
        assert len(secret_key) > 0
        assert public_key != secret_key

    def test_encrypt_decrypt_roundtrip(self) -> None:
        """Encrypted plaintexts decrypt back to the original value."""
        public_key, secret_key = generate_keypair()
        plaintext = b"hello, secagg++"
        ciphertext = encrypt(public_key, plaintext)
        recovered = decrypt(secret_key, ciphertext)
        assert recovered == plaintext

    @pytest.mark.parametrize(
        "plaintext",
        [b"", b"x", b"\x00" * 1000, b"\xff" * 1234],
    )
    def test_encrypt_decrypt_various_lengths(self, plaintext: bytes) -> None:
        """Encryption handles arbitrary plaintext lengths."""
        public_key, secret_key = generate_keypair()
        ciphertext = encrypt(public_key, plaintext)
        assert decrypt(secret_key, ciphertext) == plaintext

    def test_ciphertext_changes_with_plaintext(self) -> None:
        """Different plaintexts produce different ciphertexts."""
        public_key, _ = generate_keypair()
        ct1 = encrypt(public_key, b"message one")
        ct2 = encrypt(public_key, b"message two")
        assert ct1 != ct2

    def test_decrypt_with_wrong_secret_key(self) -> None:
        """Decrypting with a different secret key fails."""
        public_key, _ = generate_keypair()
        _, wrong_secret_key = generate_keypair()
        ciphertext = encrypt(public_key, b"secret")
        with pytest.raises(InvalidToken):
            decrypt(wrong_secret_key, ciphertext)
