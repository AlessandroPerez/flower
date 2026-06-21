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
"""Tests for post-quantum hybrid KEM helpers."""


import pytest

from .pq_kem import (
    decapsulate_secret,
    derive_pairwise_secret,
    derive_share_encryption_secret,
    encapsulate,
    generate_key_pair,
)


class TestPQKEM:
    """Tests for the X-Wing KEM wrapper."""

    def test_keypair_generation(self) -> None:
        """Generated keys are non-empty and distinct."""
        public_key, secret_key = generate_key_pair()
        assert isinstance(public_key, bytes)
        assert isinstance(secret_key, bytes)
        assert len(public_key) > 0
        assert len(secret_key) > 0
        assert public_key != secret_key

    def test_encapsulate_decapsulate(self) -> None:
        """Encapsulated secrets can be decapsulated correctly."""
        public_key, secret_key = generate_key_pair()
        shared_secret, ciphertext = encapsulate(public_key)
        recovered = decapsulate_secret(ciphertext, secret_key)
        assert shared_secret == recovered

    def test_decapsulate_modified_ciphertext(self) -> None:
        """A modified ciphertext does not reproduce the shared secret."""
        public_key, secret_key = generate_key_pair()
        shared_secret, ciphertext = encapsulate(public_key)
        modified = bytearray(ciphertext)
        modified[0] ^= 0xFF
        recovered = decapsulate_secret(bytes(modified), secret_key)
        assert recovered != shared_secret

    def test_pairwise_secret_deterministic(self) -> None:
        """Pairwise secret derivation is deterministic given inputs."""
        ss1 = b"\x00" * 32
        ss2 = b"\x01" * 32
        pw1 = derive_pairwise_secret(ss1, ss2)
        pw2 = derive_pairwise_secret(ss1, ss2)
        assert pw1 == pw2

    def test_pairwise_secret_symmetric(self) -> None:
        """Pairwise secret derivation is symmetric in its arguments."""
        ss1 = b"\x00" * 32
        ss2 = b"\x01" * 32
        assert derive_pairwise_secret(ss1, ss2) == derive_pairwise_secret(ss2, ss1)

    def test_pairwise_secret_32_bytes(self) -> None:
        """The pairwise secret is 32 bytes long."""
        ss1 = b"\x00" * 32
        ss2 = b"\x01" * 32
        pw = derive_pairwise_secret(ss1, ss2)
        assert len(pw) == 32

    def test_share_encryption_key_fernet_format(self) -> None:
        """The share encryption key is 32 raw bytes encoded as url-safe base64."""
        ss = b"\xab" * 32
        key = derive_share_encryption_secret(ss)
        assert isinstance(key, bytes)
        assert len(key) == 44  # 32 bytes -> base64 length 44 including padding

    @pytest.mark.parametrize("seed", [b"", b"x", b"\x00" * 64])
    def test_share_encryption_key_accepts_various_secrets(self, seed: bytes) -> None:
        """Share-encryption derivation accepts arbitrary-length inputs."""
        key = derive_share_encryption_secret(seed)
        assert isinstance(key, bytes)
        assert len(key) == 44
