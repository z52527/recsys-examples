# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Pure-tensor output validators used by the benchmark and its tests.

Kept dependency-free (only ``torch``) so unit tests can exercise the
validation logic without importing the full benchmark runtime stack
(``commons``, ``torchrec``, Megatron, etc.).
"""
from typing import Tuple

import torch


def _check_shapes(*tensors):
    """Validate that all sids/lp tensors share a consistent shape.

    Returns an issue string on mismatch, or ``None`` if shapes line up.
    Done before any element-level comparison so callers get a clear
    error instead of a downstream indexing / broadcast failure.
    """
    # Expecting pairs of (sids, lp); sids is [B, K, H], lp is [B, K]
    if len(tensors) % 2 != 0:
        return f"_validate received {len(tensors)} tensors, expected pairs"
    sids_list = tensors[0::2]
    lp_list = tensors[1::2]
    ref_sids_shape = sids_list[0].shape
    ref_lp_shape = lp_list[0].shape
    if len(ref_sids_shape) != 3 or len(ref_lp_shape) != 2:
        return (
            f"unexpected ranks: sids should be 3-D [B, K, H], lp 2-D [B, K]; "
            f"got sids={tuple(ref_sids_shape)}, lp={tuple(ref_lp_shape)}"
        )
    if ref_sids_shape[:2] != ref_lp_shape:
        return (
            f"sids/lp batch+top-k mismatch: sids={tuple(ref_sids_shape)}, "
            f"lp={tuple(ref_lp_shape)}"
        )
    for i, s in enumerate(sids_list):
        if s.shape != ref_sids_shape:
            return f"sids shape mismatch at path {i}: {tuple(s.shape)} != {tuple(ref_sids_shape)}"
    for i, lp in enumerate(lp_list):
        if lp.shape != ref_lp_shape:
            return f"lp shape mismatch at path {i}: {tuple(lp.shape)} != {tuple(ref_lp_shape)}"
    return None


def validate_compare_outputs(
    sids_a: torch.Tensor, lp_a: torch.Tensor,
    sids_b: torch.Tensor, lp_b: torch.Tensor,
    sids_c: torch.Tensor, lp_c: torch.Tensor,
) -> Tuple[bool, str]:
    """Validate that three beam-search outputs agree within bf16 noise.

    Applies the same thresholds as the regression-guard tests, symmetrically
    across all three pairs (A vs B, B vs C, A vs C):

      - top-1 SID tuple must match exactly.
      - per-position |log_prob delta| < 0.15.
      - top-K beam SID set overlap >= 70% per sample.

    Returns ``(passed, summary)``. The summary always reports the three
    log-prob deltas and the three worst-case overlaps so a passing config
    still surfaces the actual numbers.
    """
    shape_issue = _check_shapes(sids_a, lp_a, sids_b, lp_b, sids_c, lp_c)
    if shape_issue is not None:
        return False, shape_issue

    issues = []

    top1_a = sids_a[:, 0, :]
    top1_b = sids_b[:, 0, :]
    top1_c = sids_c[:, 0, :]
    if not torch.equal(top1_a, top1_b):
        bad = (top1_a != top1_b).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
        issues.append(f"A vs B top-1 mismatch on samples {bad}")
    if not torch.equal(top1_b, top1_c):
        bad = (top1_b != top1_c).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
        issues.append(f"B vs C top-1 mismatch on samples {bad}")
    if not torch.equal(top1_a, top1_c):
        bad = (top1_a != top1_c).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
        issues.append(f"A vs C top-1 mismatch on samples {bad}")

    lp_diff_ab = (lp_a - lp_b).abs().max().item()
    lp_diff_bc = (lp_b - lp_c).abs().max().item()
    lp_diff_ac = (lp_a - lp_c).abs().max().item()
    if lp_diff_ab >= 0.15:
        issues.append(f"|lp_a - lp_b| = {lp_diff_ab:.4f} >= 0.15")
    if lp_diff_bc >= 0.15:
        issues.append(f"|lp_b - lp_c| = {lp_diff_bc:.4f} >= 0.15")
    if lp_diff_ac >= 0.15:
        issues.append(f"|lp_a - lp_c| = {lp_diff_ac:.4f} >= 0.15")

    top_k = sids_a.shape[1]
    sets_a = [
        {tuple(sids_a[b, k].tolist()) for k in range(top_k)}
        for b in range(sids_a.shape[0])
    ]
    sets_b = [
        {tuple(sids_b[b, k].tolist()) for k in range(top_k)}
        for b in range(sids_b.shape[0])
    ]
    sets_c = [
        {tuple(sids_c[b, k].tolist()) for k in range(top_k)}
        for b in range(sids_c.shape[0])
    ]

    def _min_overlap(sets_x, sets_y, label):
        worst = 1.0
        worst_sample = -1
        for b, (sx, sy) in enumerate(zip(sets_x, sets_y)):
            o = len(sx & sy) / len(sx)
            if o < worst:
                worst, worst_sample = o, b
        if worst < 0.7:
            issues.append(
                f"{label} top-{top_k} overlap {worst*100:.0f}% < 70% "
                f"on sample {worst_sample}"
            )
        return worst

    ov_ab = _min_overlap(sets_a, sets_b, "A vs B")
    ov_bc = _min_overlap(sets_b, sets_c, "B vs C")
    ov_ac = _min_overlap(sets_a, sets_c, "A vs C")

    summary = (
        f"|lp_ab|={lp_diff_ab:.3f} |lp_bc|={lp_diff_bc:.3f} |lp_ac|={lp_diff_ac:.3f} "
        f"ov_ab={ov_ab*100:.0f}% ov_bc={ov_bc*100:.0f}% ov_ac={ov_ac*100:.0f}%"
    )
    if issues:
        return False, "; ".join(issues) + " | " + summary
    return True, summary


def validate_pair_outputs(
    sids_a: torch.Tensor, lp_a: torch.Tensor,
    sids_b: torch.Tensor, lp_b: torch.Tensor,
) -> Tuple[bool, str]:
    """Two-path version of ``validate_compare_outputs`` for
    ``run_sweep`` (A vs B only). Same thresholds, single pair.
    """
    shape_issue = _check_shapes(sids_a, lp_a, sids_b, lp_b)
    if shape_issue is not None:
        return False, shape_issue

    issues = []

    top1_a = sids_a[:, 0, :]
    top1_b = sids_b[:, 0, :]
    if not torch.equal(top1_a, top1_b):
        bad = (top1_a != top1_b).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
        issues.append(f"A vs B top-1 mismatch on samples {bad}")

    lp_diff = (lp_a - lp_b).abs().max().item()
    if lp_diff >= 0.15:
        issues.append(f"|lp_a - lp_b| = {lp_diff:.4f} >= 0.15")

    top_k = sids_a.shape[1]
    worst = 1.0
    worst_sample = -1
    for b in range(sids_a.shape[0]):
        sa = {tuple(sids_a[b, k].tolist()) for k in range(top_k)}
        sb = {tuple(sids_b[b, k].tolist()) for k in range(top_k)}
        o = len(sa & sb) / len(sa)
        if o < worst:
            worst, worst_sample = o, b
    if worst < 0.7:
        issues.append(
            f"A vs B top-{top_k} overlap {worst*100:.0f}% < 70% "
            f"on sample {worst_sample}"
        )

    summary = f"|lp_ab|={lp_diff:.3f} ov_ab={worst*100:.0f}%"
    if issues:
        return False, "; ".join(issues) + " | " + summary
    return True, summary
