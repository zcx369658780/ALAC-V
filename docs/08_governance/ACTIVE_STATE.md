---
id: STATE-0001
title: ALAC-V Active State
status: active
date: 2026-11-15
owner: Owner
---

# ALAC-V Active State

## Repository baseline

- Repository: https://github.com/zcx369658780/ALAC-V
- Visibility: Public
- Default branch: `main`
- Phase 0/1 accepted merge commit: `367035318172ba1bbc0103b6231847750143d1d7`
- Phase 2A design accepted merge commit: `ef39be01cf7c7bec2650f26af9e2bcefc667c526`
- The current `main` HEAD is authoritative and must be read from GitHub; this file intentionally does not self-reference its own commit SHA.
- Initial delivery tag: `phase0-1`
- Local DSH workspace reported: `D:\ALAC_V`

## Current phase and authorization

**Phase 0: complete. Phase 1: complete and independently accepted (PASS). Phase 2A design: independently accepted and merged. Phase 2A.1: Owner-authorized and active under ALAC-0003 / Issue #6. DS-0002 generation and frozen teacher-cache creation are authorized now, using only the already-pinned local Phase 1 DINOv2 weights. Formal model training is NOT yet authorized until Gate 2 (independent DS-0002 acceptance) and Gate 3 (implementation review) both pass. Phase 2B and Phase 2C are BLOCKED. Phase 3 (video/real-time streams) is BLOCKED.**

Owner decisions on 2026-11-15:
- D-2A-1: **ACCEPTED** — enter Phase 2A.1 implementation and conduct EXP-0002 subject to mandatory stage gates.
- D-2A-2: **ACCEPTED** — generate DS-0002 and frozen DINOv2 teacher feature cache using the existing pinned local weights/revision only.
- D-2A-3: **ACCEPTED** — Phase 2A conclusions are limited to the synthetic domain.
- D-2A-4: **ACCEPTED** — A2 is the sole primary arm; B2/C2 are secondary comparisons.

## Phase 2A.1 Gate 1 status (2026-11-15)

- Task: ALAC-0003 / TASK-0004 (Issue #6), branch `task/6-phase2a1-ds0002-generation`.
- DS-0002 **v3** generated: 3,200 base scenes / 22,000 views (train 2,000/16,000; val 300/1,500; test 900/4,500).
- Split isolation: zero overlap of base seeds, view IDs and image SHA-256 across splits.
- Deterministic regeneration spot-check: 9/9 byte-identical.
- Evaluation labels isolated in `data/ds0002/eval/`; DataLoader allowlist = `{view_id, image, augment}`.
- Labels reflect real drawing parameters (texture, palette, shape); occlusion is measured on the true target mask with opaque occluders (later targets and distractors); three target shapes.
- Training loader reads only the projected index (`view_id/path/augment/pair_key`), holds no label manifest, and views use opaque IDs/paths with no scene digits; the eval-side salt and mapping live only under eval/ (never read by training) and are not claimed to be cryptographic anonymity.
- All 22,000 image SHA-256 recomputed from disk: 0 mismatches, 0 byte mismatches, 0 duplicates; filesystem/manifest/index counts all 22,000.
- Frozen DINOv2 teacher cache: 23 shards, 23.15 GB, FP16; full view-set equality, all shard tensors and counts verified,
  33/33 sampled views bit-identical to recomputation; sanitized cache index published.
- Evidence: `artifacts/evidence/DS-0002/`; report: `docs/05_reports/TASK-0004-GATE1-REPORT.md`.
- Gate 2 independent acceptance **PENDING**; no model implementation or optimizer step has been run.

## Phase 2A.1 Gate 3 status (2026-11-16)

- Branch `task/6-phase2a1-gate3-model-implementation`, stacked PR base `task/6-phase2a1-ds0002-generation`.
- Six arms implemented: F0 / A1 / A2 / B2 / C2 / R0; 384-d CLS readout; A2/C2 7.10M params, B2 5.03M.
- Mask: 821 patches, 4 non-overlapping quadrant blocks (A2) or random (C2), deterministic per seed.
- A1/F0 invariant verified on real cache: bit-identical descriptors, Recall@1 and Kendall tau equal.
- Dry-run VRAM (batch 8): A2/C2 ~1.43 GB; no optimizer step; EXP-0002 NOT RUN.
- Gate 2 follow-ups (cache dataset id, pair_key wording, isolation test, independent occlusion test) closed.
- Gate 4 remains **NOT AUTHORIZED** pending Gate 3 acceptance.

## Current open task

- [ALAC-0003 — Phase 2A.1 DS-0002 generation, model implementation and EXP-0002](https://github.com/zcx369658780/ALAC-V/issues/6)
- Task snapshot: `docs/01_tasks/TASK-0004-phase2a1-implementation-and-ds0002.md`
- State: Owner-authorized. DSH may execute Gate 0 and Gate 1 only now.
- Current gate: **Gate 1 — DS-0002 generation + frozen teacher cache**.
- Next stop: submit complete data/cache evidence and wait for ChatGPT independent review and Owner Gate 2 acceptance.
- No formal training before Gate 2 and Gate 3 PASS.

## Accepted Phase 2A design baseline

- Design PR #5 merged: `ef39be01cf7c7bec2650f26af9e2bcefc667c526`; design Issue #3 closed as completed.
- Literature review: `docs/research/PHASE2A_AUTONOMOUS_LEARNING_LITERATURE.md`
- Architecture: `docs/03_design/PHASE2A_AUTONOMOUS_LEARNING_DESIGN.md`
- EXP-0002 pre-registration: `docs/04_experiments/EXP-0002-autonomous-visual-learning-preregistration-draft.md`
- Model license matrix: `docs/02_audit/PHASE2A_MODEL_LICENSE_MATRIX.md`
- Owner decision summary: `docs/05_reports/PHASE2A_OWNER_DECISION_SUMMARY.md`

Frozen design constraints:
- Teacher: DINOv2 ViT-S/14, pinned Phase 1 revision, Apache-2.0, frozen.
- Input: 518×518, 37×37 = 1,369 patches; all arms retrieve using 384-d L2-normalized CLS.
- A2/C2 mask exactly 821 patches (59.97%); A2 uses four strictly non-overlapping contiguous blocks; C2 random 821 patches; mask pixels before student encoder forward.
- A2 is sole primary; B2/C2 secondary; A1 predictor-only invariant control; F0 frozen baseline; R0 random adapter control.
- Trainable parameters ≤10M; A2/C2 7.0M, B2 4.7M; no claim of equal parameter budget.
- Seeds: 2026, 2027, 2028.
- DS-0002 scenes: train/validation/test = 2,000/300/900, grouped by base-scene seed; train 8 views/scene, validation/test 5 views/scene.
- Train-time data loader may access pixels, augmentation parameters and mask seeds only; scene IDs, bboxes, factor labels and evaluation labels are evaluation-only.
- H1-D, H1-P, H1-S and all statistical limits are frozen in EXP-0002. Three seeds support feasibility reporting only, not robust cross-initialization claims.
- Training budget ≤21 runs; estimated GPU time 21–35 h. Stop if a run exceeds 2 h or peak GPU memory exceeds 9 GB.

## Mandatory Phase 2A.1 stage gates

1. **Gate 0 — sync and freeze:** verify latest main, clean workspace, pinned EXP-0002 hash and local hardware/disk.
2. **Gate 1 — data generation (authorized now):** generate 3,200 DS-0002 base scenes, expected views, manifests, evaluation sidecar, and teacher feature cache using existing local pinned DINOv2 only. No downloads.
3. **Gate 2 — independent data acceptance:** verify split isolation, deterministic regeneration, file integrity, label separation, DataLoader allowlist, teacher-cache alignment/fingerprint and resource evidence. Stop for ChatGPT/Owner acceptance.
4. **Gate 3 — model implementation (after Gate 2 PASS):** implement F0/A1/A2/B2/C2/R0, tests and configs. Stop for code/implementation review before training.
5. **Gate 4 — training (after Gate 3 PASS):** run EXP-0002 within the pre-registered 21-run limit; no unregistered changes.
6. **Gate 5 — final independent acceptance:** report per-run evidence, per-seed outcomes, H1 decisions, secondary comparisons, collapse and resource diagnostics.

## Completed Phase 0/1 milestone

- PR #2 merged: https://github.com/zcx369658780/ALAC-V/pull/2
- Merge commit: `367035318172ba1bbc0103b6231847750143d1d7`
- Issue #1: closed as completed.
- TASK-0002 independent acceptance: PASS.
- DSH-reported 83 tests passed in 8.04s, anchored to clean code commit `07a723c`; subsequent commits changed only documentation/evidence pointers. ChatGPT verified the code-tree comparison, not a local execution on the Owner's workstation.
- Corrected GT bbox result: A/B/C PSNR 31.66 / 31.05 / 28.53 dB; H3 rejected (C −3.14 dB vs A).
- C ROI mechanism: roi_00 hit 58.3%, any stored crop hit 66.7%, union coverage of GT area 47.7%.
- Nine sanitized historical run bundles published. Eight runs are anchored to code commit; C_r2's original code commit remains unanchored and explicitly marked inferred.

## Scientific limitations to preserve

1. DS-0001 is a small, programmatically generated synthetic benchmark; its results do not establish natural-image generalization.
2. Phase 2A.1 conclusions are limited to DS-0002 synthetic domain; do not generalize to natural images.
3. C_r2's original code commit cannot be anchored; retain this provenance limitation.
4. No complete EXP-0001 rerun, ICC profile-to-sRGB transform, Sobel comparison, natural-image benchmark or independent workstation reproduction has been performed.
5. Phase 1 does not demonstrate a general efficiency improvement; do not claim one.
6. The Phase 2A power simulation does not include empirical seed-to-seed variance; three seeds are feasibility scale only.

## Owner-approved research order

1. **Phase 2A — Autonomous visual learning:** design accepted; Phase 2A.1 implementation/EXP-0002 now active with gates.
2. **Phase 2B — Long-term, multi-level visual memory:** blocked until Phase 2A learning mechanism is validated.
3. **Phase 2C — Dynamic compute allocation:** blocked until learned representations and memory interfaces are stable.
4. **Phase 3 — Offline video and real-time visual streams:** blocked until Phase 2 gates are accepted.

## Owner decisions / workflow

- GitHub Issues are the authoritative online task and acceptance records.
- `docs/01_tasks/` stores versioned task snapshots and history.
- DSH is the primary implementation/research agent.
- ClawsGO is an independent reviewer and defaults to read-only.
- Use task branches and PRs; do not push directly to `main`.
- Current authorization is limited to ALAC-0003 / Issue #6 and its explicit stage gates.
- **Authorized now:** Gate 0 and Gate 1 DS-0002 + teacher cache with existing local pinned DINOv2 weights.
- **Not yet authorized:** formal training before independent Gate 2 and Gate 3 PASS; new weight downloads; natural-image data; Phase 2B/2C; Phase 3.
