# Copyright (c) 2026 by Yuchen Wang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression coverage for sliding-window split-KV planning (issue #4972)."""

import math

import pytest
import torch

import flashinfer


@pytest.mark.parametrize("storage", ["paged", "ragged"])
@pytest.mark.parametrize("split", ["auto", "fixed", "disabled"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qo_len,kv_len,window_left,group_size",
    [(300, 4096, 333, 4), (300, 300, 33, 1), (129, 1024, 0, 4)],
)
def test_prefill_window_split(
    storage, split, causal, qo_len, kv_len, window_left, group_size
):
    torch.manual_seed(0)
    num_kv_heads, head_dim, page_size = 2, 128, 16
    num_qo_heads = num_kv_heads * group_size
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    options = dict(
        causal=causal,
        window_left=window_left,
        q_data_type=torch.float16,
        disable_split_kv=split == "disabled",
    )
    if split == "fixed":
        # The paged planner takes pages; the ragged planner takes tokens.
        options["fixed_split_size"] = 128 // page_size if storage == "paged" else 128

    if storage == "paged":
        num_pages = (kv_len + page_size - 1) // page_size
        k_cache = torch.zeros(
            num_pages * page_size,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        v_cache = torch.zeros_like(k_cache)
        k_cache[:kv_len] = k
        v_cache[:kv_len] = v
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr,
            torch.tensor([0, num_pages], dtype=torch.int32),
            torch.arange(num_pages, dtype=torch.int32),
            torch.tensor([(kv_len - 1) % page_size + 1], dtype=torch.int32),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            **options,
        )
        out, lse = wrapper.run(
            q,
            (
                k_cache.view(num_pages, page_size, num_kv_heads, head_dim),
                v_cache.view(num_pages, page_size, num_kv_heads, head_dim),
            ),
            return_lse=True,
        )
    else:
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr,
            torch.tensor([0, kv_len], dtype=torch.int32),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            **options,
        )
        out, lse = wrapper.run(q, k, v, return_lse=True)

    k_ref = k.float().repeat_interleave(group_size, dim=1)
    v_ref = v.float().repeat_interleave(group_size, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_ref) / math.sqrt(head_dim)
    q_pos = torch.arange(qo_len, device="cuda")[:, None] + kv_len - qo_len
    kv_pos = torch.arange(kv_len, device="cuda")[None, :]
    mask = kv_pos >= q_pos - window_left
    if causal:
        mask &= kv_pos <= q_pos
    scores.masked_fill_(~mask[None], float("-inf"))
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v_ref)
    expected_lse = scores.logsumexp(dim=-1).transpose(0, 1)
    torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)
