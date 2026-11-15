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
- The current `main` HEAD is authoritative and must be read from GitHub; this file intentionally does not self-reference its own commit SHA.
- Initial delivery tag: `phase0-1`
- Local DSH workspace reported: `D:\ALAC_V`

## Current phase and authorization

**Phase 0: complete. Phase 1: complete and independently accepted (PASS). Phase 2A: research design and pre-registration submitted for review (not yet accepted). Model training is NOT authorized. Phase 2B and Phase 2C are BLOCKED. Phase 3 (video/real-time streams) is BLOCKED.**

The Owner approved PR #2 merge and Issue #1 closure on 2026-11-15. Phase 2 is now entered only through the separately authorized Phase 2A design task. No model training, backbone fine-tuning, natural-image dataset download, long-term memory implementation, dynamic compute controller, video, Event Camera or SNN work is authorized by the current task.

## Current open task

- [ALAC-0002 — Phase 2A autonomous visual learning research design](https://github.com/zcx369658780/ALAC-V/issues/3)
- Task snapshot: `docs/01_tasks/TASK-0003-phase2a-autonomous-visual-learning-design.md`
- State: deliverables submitted on branch `task/3-phase2a-autonomous-learning-design` (PR pending review); acceptance reserved to the Owner.
- Stop condition: do not train models, download new weights/datasets, generate DS-0002, or enter Phase 2B/2C.

## Phase 2A design artifacts (submitted 2026-11-15)

- Literature review: `docs/research/PHASE2A_AUTONOMOUS_LEARNING_LITERATURE.md`
- Architecture design: `docs/03_design/PHASE2A_AUTONOMOUS_LEARNING_DESIGN.md`
- EXP-0002 pre-registration draft: `docs/04_experiments/EXP-0002-autonomous-visual-learning-preregistration-draft.md`
- License matrix: `docs/02_audit/PHASE2A_MODEL_LICENSE_MATRIX.md`
- Owner decision summary: `docs/05_reports/PHASE2A_OWNER_DECISION_SUMMARY.md`

Revised 2026-11-15 after the first independent review of PR #5 (P1-1..P1-5, plus the cached-feature
leakage finding): unified 384-d CLS readout, mask-before-encode student path, 768-d B2 target,
A2 as sole primary with Holm-controlled secondary comparisons, validation-only arm selection for the
data-efficiency study, and corrected V-JEPA/V-JEPA 2 and BYOL license entries.

Revised again 2026-11-15 after the third review: mask count fixed at 821 patches (59.97%) with a
reproducible block-sampling rule; primary checkpoint fixed to best validation Recall@1 with a
deterministic tie-break; data-efficiency protocol completed (validation-only selection, frozen seed
set, single test pass, descriptive-only status); cluster-level power analysis added, which showed that
300 test scenes give only ~50% power at 3 pp, so DS-0002 test is raised to 900 scenes (~0.90 power).
Re-review pending.

Key design constraints recorded: frozen DINOv2 teacher (Apache-2.0), ≤10M trainable parameters,
synthetic-only data (DS-0002 design, not generated), group split by base scene, training never uses
labels, Phase 2A.1 requires a separate task order.

## Completed Phase 0/1 milestone

- PR #2 merged: https://github.com/zcx369658780/ALAC-V/pull/2
- Merge commit: `367035318172ba1bbc0103b6231847750143d1d7`
- Issue #1: closed as completed: https://github.com/zcx369658780/ALAC-V/issues/1
- TASK-0002 independent acceptance: PASS.
- DSH-reported 83 tests passed in 8.04s, anchored to clean code commit `07a723c`; subsequent commits changed only documentation/evidence pointers. ChatGPT verified the code-tree comparison, not a local execution on the Owner's workstation.
- Corrected GT bbox result: A/B/C PSNR 31.66 / 31.05 / 28.53 dB; H3 rejected (C −3.14 dB vs A).
- C ROI mechanism: roi_00 hit 58.3%, any stored crop hit 66.7%, union coverage of GT area 47.7%.
- Nine sanitized historical run bundles published. Eight runs are anchored to code commit; C_r2's original code commit remains unanchored and explicitly marked inferred.

## Accepted Phase 1 technical baseline

- Frozen DINOv2 ViT-S/14, pinned revision and Apache-2.0 model license.
- Immutable SHA-256 raw image archive.
- Multi-scale image pyramid and Foveated ROI.
- CLS/Patch FP16 features.
- SQLite metadata + SafeTensors feature storage.
- Cosine retrieval, ROI backtracking and fail-closed validation for required assets.
- EXP-0001 A/B/C evidence and corrected GT bbox analysis.

## Scientific limitations to preserve

1. DS-0001 is a small, programmatically generated synthetic benchmark; its results do not establish natural-image generalization.
2. C_r2's original code commit cannot be anchored; retain this provenance limitation.
3. No complete EXP-0001 rerun, EXP-0002 training experiment, ICC profile-to-sRGB transform, Sobel comparison, natural-image benchmark or independent workstation reproduction has been performed.
4. Phase 1 does not demonstrate a general efficiency improvement; do not claim one.

## Owner-approved research order

1. **Phase 2A — Autonomous visual learning** (current task: research design and pre-registration only).
2. **Phase 2B — Long-term, multi-level visual memory** (blocked until Phase 2A learning mechanism is validated).
3. **Phase 2C — Dynamic compute allocation** (blocked until learned representations and memory interfaces are stable).
4. **Phase 3 — Offline video and real-time visual streams** (blocked until Phase 2 gates are accepted).

## Owner decisions / workflow

- GitHub Issues are the authoritative online task and acceptance records.
- `docs/01_tasks/` stores versioned task snapshots and history.
- DSH is the primary implementation/research agent.
- ClawsGO is an independent reviewer and defaults to read-only.
- Use task branches and PRs; do not push directly to `main`.
- Only the explicitly scoped work in Issue #3 is currently authorized.
- Model training, weight downloads, public natural-image datasets and later phases require separate Owner-approved task orders.
