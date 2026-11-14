---
id: STATE-0001
title: ALAC-V Active State
status: active
date: 2026-11-14
owner: Owner
---

# ALAC-V Active State

## Repository baseline

- Repository: https://github.com/zcx369658780/ALAC-V
- Visibility: Public
- Default branch: `main`
- Baseline HEAD at governance bootstrap: `10cb05d379c39043a17dc840643535fd150bce20`
- Baseline tree: `308e3403dbc52784a751a47766508e26a0690f81`
- Governance setup commit before the active-state snapshot: `43ce7155e682fb7f5783503f4dae7f8885b8652e`
- The current `main` HEAD is authoritative and must be read from GitHub; this file intentionally does not self-reference its own commit SHA.
- Initial delivery tag: `phase0-1` (reported by DSH; points to the initial delivery commit)
- Local DSH workspace reported: `D:\ALAC_V`

## Current phase

**Phase 0: implementation reported complete. Phase 1: implementation reported complete; independent acceptance is CONDITIONAL / REMEDIATION REQUIRED. Phase 2: BLOCKED.**

The current engineering baseline is committed, but the first independent review identified evidence and metric-validity issues. Do not begin video, temporal learning, or Phase 2 work until the Owner accepts the remediation and issues a separate task order.

## Current open task

- [ALAC-0001 — Phase 0/1 independent acceptance and evidence remediation](https://github.com/zcx369658780/ALAC-V/issues/1)
- State: Open; **Owner authorized implementation on 2026-11-14**. DSH remediation submitted on branch
  `task/1-phase0-1-remediation` (PR pending review). Acceptance remains reserved to the Owner; the Issue must
  not be closed by the implementing agent.

## Verified project facts from repository

- Hardware reported by DSH: Intel Core Ultra 7 265K, 32 GB DDR5, RTX 5070 12 GB.
- Phase 1 backbone: frozen DINOv2 ViT-S/14, HuggingFace revision recorded in model manifest, Apache-2.0.
- Pipeline: content-addressed raw archive, multi-scale pyramid, foveated ROI, CLS/Patch FP16 features, SQLite + SafeTensors, cosine retrieval and ROI backtracking.
- DSH reports 83 tests passing (final-state run logged under `artifacts/evidence/test-runs/`). This has not yet been independently rerun by ChatGPT.
- DS-0001 is a programmatically generated synthetic benchmark with 24 gallery scenes, 72 query variants, and 2 local wallpaper samples.
- A/B/C results do not support a general efficiency-improvement claim; the report explicitly says so.

## Key review findings (first review, 2026-11-14 — historical)

> These are the **first-round** findings that triggered TASK-0002. They are kept for history and are
> **not** a description of the current tree. Current remediation status is in
> [Remediation status](#remediation-status-task-0002-2026-11-14) below; each item is closed or
> superseded as recorded there.

1. Public repository excludes `artifacts/runs/*`; only aggregate summaries are public. This does not meet the repository's own run-level evidence rule for formal independent audit.
2. A-variant backtrack reconstruction in `scripts/run_experiments.py` appears not to apply the center-crop offset used by `fit_to_input()`; A backtrack PSNR/SSIM require correction and re-evaluation.
3. ADR-0007 calls the ROI gradient method Sobel, while the implementation uses central finite differences. Code and ADR must be reconciled.
4. RGB conversion records ICC metadata but does not currently apply an ICC profile transform to sRGB; documentation must state this accurately or implementation must be extended.
5. TASK-0001 refers to `tests/test_pyramid.py`, while the current repository test is `tests/test_imaging.py::test_pyramid_sizes`.
6. CPU peak percentage and OS product/build identification need source-level validation before reuse as formal claims.

See `PHASE0_1_ACCEPTANCE.md` and Issue #1 for the complete original list; see the remediation table below for current status.

## Remediation status (TASK-0002, 2026-11-14)

**Round 3 (final polish).** Second-round review accepted R9 GT bbox, R10 per-axis scale,
R11 manifest hash and R12 provenance, and asked for small documentation/robustness fixes:
- C ROI mechanism clarified: ROI_00 hit 58.3%, any stored crop 66.7%, union coverage 47.7%
  (GT area covered by full-resolution crops); remaining ~52% relies on the 1/2 mosaic.
- Backtracking is now **fail-closed**: missing L0 (B), missing foveated asset (C) or unknown
  variant raises instead of silently returning the original image.
- Contradictory statements in AMD §3, remediation report §11, PHASE1 §6.2, README and this file
  are corrected; final tests are re-anchored to the final code commit.
- PR #2 remains open; acceptance still reserved to the Owner; Phase 2 BLOCKED.

| Review finding | Remediation | Status |
|---|---|---|
| R1 run-level evidence | `artifacts/evidence/EXP-0001/` (9 sanitized run bundles) | submitted |
| R2 A-variant center crop | `src/alacv/backtrack.py`, AMD-0001, recomputed metrics | submitted; **H3 rejected** (GT bbox: C −3.14 dB vs A) |
| R3 Sobel naming | ADR-0008 (central difference), ADR-0007 supersede pointer | submitted |
| R4 ICC claim | docs corrected to "mode conversion + ICC metadata only" | submitted |
| R5 test pointer | `TASK-0001-erratum-0001.md` | submitted |
| R6 CPU peak metric | `peak_cpu_pct` withdrawn; `peak_cpu_cores` introduced | submitted |
| R7 OS identity | `docs/02_audit/os-evidence.json` (direct registry read) | submitted |
| R8 manifest paths | generic `os-wallpaper-sample-NN` labels; old/new SHA recorded | submitted |

Phase 2 remains **BLOCKED** until Owner acceptance of this remediation.

## Owner decisions / workflow

- GitHub Issues are the authoritative online task and acceptance records.
- `docs/01_tasks/` stores versioned task snapshots and history.
- DSH is the primary implementation agent.
- ClawsGO is an independent reviewer and defaults to read-only.
- No direct pushes to `main`; use task branches and PRs.
- No Phase 2 without a separate Owner authorization.
