# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch


class KVCacheOffloadMode(Enum):
    LAZY = "lazy"
    EAGER = "eager"


@dataclass
class KVLookupResult:
    # batch_size: int
    user_ids: torch.Tensor
    # total_history_lengths: torch.Tensor
    cached_start_indices: Optional[torch.Tensor] = None
    cached_lengths: Optional[torch.Tensor] = None
    gpu_cached_start_indices: Optional[torch.Tensor] = None
    gpu_cached_lengths: Optional[torch.Tensor] = None
    host_cached_start_indices: Optional[torch.Tensor] = None
    host_cached_lengths: Optional[torch.Tensor] = None

    # new_tokens_upper_bound: int
    token_ids: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def merge(cls, lookup_res1, lookup_res2):
        assert torch.equal(lookup_res1.user_ids, lookup_res2.user_ids)
        if (
            lookup_res1.gpu_cached_start_indices is not None
            and lookup_res1.gpu_cached_lengths is not None
        ):
            assert (
                lookup_res2.host_cached_start_indices is not None
                and lookup_res2.host_cached_lengths is not None
            )

        if (
            lookup_res1.host_cached_start_indices is not None
            and lookup_res1.host_cached_lengths is not None
        ):
            assert (
                lookup_res2.gpu_cached_start_indices is not None
                and lookup_res2.gpu_cached_lengths is not None
            )
            res = lookup_res1
            lookup_res1 = lookup_res2
            lookup_res2 = res

        # assume lookup_res1 is gpu lookup result, lookup_res2 is host lookup result
        batch_size = lookup_res1.user_ids.size(0)
        cached_start_indices = torch.empty_like(lookup_res1.gpu_cached_start_indices)
        cached_lengths = torch.empty_like(lookup_res1.gpu_cached_lengths)
        for i in range(batch_size):
            cached_start_ind = 0
            cached_len = 0
            if lookup_res2.host_cached_lengths[i] == 0:
                cached_start_ind = lookup_res1.gpu_cached_start_indices[i]
                cached_len = lookup_res1.gpu_cached_lengths[i]
            elif lookup_res1.gpu_cached_lengths[i] == 0:
                assert (
                    lookup_res2.host_cached_start_indices[i] == 0
                ), "Host caching from the beginning of the sequence."
                cached_start_ind = lookup_res2.host_cached_start_indices[i]
                cached_len = lookup_res2.host_cached_lengths[i]
            else:
                assert (
                    lookup_res2.host_cached_start_indices[i] == 0
                ), "Host caching from the beginning of the sequence."
                assert (
                    lookup_res1.gpu_cached_start_indices[i] >= 0
                ), "Invalid gpu cache start ind."

                assert (
                    lookup_res1.gpu_cached_start_indices[i]
                    <= lookup_res2.host_cached_lengths[i]
                ), "No gaps allowed: GPU cache start index should be smaller than or equal to host cached length."
                cached_len = max(
                    lookup_res2.host_cached_lengths[i],
                    lookup_res1.gpu_cached_start_indices[i]
                    + lookup_res1.gpu_cached_lengths[i],
                ).item()

            cached_start_indices[i] = cached_start_ind
            cached_lengths[i] = cached_len

        assert (
            getattr(lookup_res1, "extra", {}) or {}
        ) == {}, "GPU lookup results should not have extra fields."
        merged_extra = getattr(lookup_res2, "extra", {}) or {}

        merged_lookup_result = cls(
            user_ids=lookup_res1.user_ids,
            cached_start_indices=cached_start_indices,
            cached_lengths=cached_lengths,
            gpu_cached_start_indices=lookup_res1.gpu_cached_start_indices,
            gpu_cached_lengths=lookup_res1.gpu_cached_lengths,
            host_cached_start_indices=lookup_res2.host_cached_start_indices,
            host_cached_lengths=lookup_res2.host_cached_lengths,
            # token_ids=lookup_res1.token_ids if lookup_res1.token_ids is not None else lookup_res2.token_ids,
            # token_mask=lookup_res1.token_mask if lookup_res1.token_mask is not None else lookup_res2.token_mask,
            extra=merged_extra,
        )
        return merged_lookup_result


@dataclass
class KVIndexMeta:
    user_ids: torch.Tensor
    seq_lengths: torch.Tensor
