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
"""Tests for SecAggPlusPlus utility functions."""


import pytest

from .secaggplus_utils import (
    derive_pairwise_key,
    share_keys_plaintext_concat_plus,
    share_keys_plaintext_separate_plus,
)


class TestDerivePairwiseKey:
    """Tests for pairwise key derivation."""

    def test_deterministic(self) -> None:
        """Derivation is deterministic for the same inputs."""
        seed = b"\x00" * 32
        k1 = derive_pairwise_key(seed, 7)
        k2 = derive_pairwise_key(seed, 7)
        assert k1 == k2

    def test_depends_on_node_id(self) -> None:
        """Different neighbour IDs yield different keys."""
        seed = b"\x00" * 32
        k1 = derive_pairwise_key(seed, 1)
        k2 = derive_pairwise_key(seed, 2)
        assert k1 != k2

    def test_depends_on_seed(self) -> None:
        """Different seeds yield different keys."""
        k1 = derive_pairwise_key(b"\x00" * 32, 1)
        k2 = derive_pairwise_key(b"\x01" * 32, 1)
        assert k1 != k2

    def test_output_is_32_bytes(self) -> None:
        """The derived key is a 256-bit digest."""
        k = derive_pairwise_key(b"seed", 0)
        assert len(k) == 32


class TestShareKeysPlaintextPlus:
    """Tests for SecAggPlusPlus payload packing."""

    @pytest.mark.parametrize(
        "pairwise_key, b_seed_share, self_mask_share",
        [
            (b"k" * 32, b"b" * 16, b"u" * 16),
            (b"", b"", b""),
            (b"\x00" * 100, b"\xff" * 48, b"\xaa" * 64),
        ],
    )
    def test_roundtrip(
        self,
        pairwise_key: bytes,
        b_seed_share: bytes,
        self_mask_share: bytes,
    ) -> None:
        """Concatenation and separation are inverse operations."""
        plaintext = share_keys_plaintext_concat_plus(
            123, 456, pairwise_key, b_seed_share, self_mask_share
        )
        src, dst, pk, bs, us = share_keys_plaintext_separate_plus(plaintext)
        assert src == 123
        assert dst == 456
        assert pk == pairwise_key
        assert bs == b_seed_share
        assert us == self_mask_share
