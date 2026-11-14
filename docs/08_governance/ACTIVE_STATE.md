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

**Phase 0: complete. Phase 1: complete and independently accepted (PASS). Phase 2A: research design and experiment pre-registration authorized. Model training is NOT yet authorized. Phase 2B and Phase 2C are BLOCKED. Phase 3 (video/real-time streams) is BLOCKED.**

The Owner approved PR #2 merge and Issue #1 closure on 2026-11-15. Phase 2 is now entered only through the separately authorized Phase 2A design task. No model training, backbone fine-tuning, natural-image dataset download, long-term memory implementation, dynamic compute controller, video, Event Camera or SNN work is authorized by the current task.

## Current open task

- [ALAC-0002 — Phase 2A autonomous visual learning research design](https://github.com/zcx369658780/ALAC-V/issues/3)
- Task snapshot: `docs/01_tasks/TASK-0003-phase2a-autonomous-visual-learning-design.md`
- State: Owner-authorized for literature review, architecture comparison, unlabeled-data design, experiment pre-registration and resource estimation only.
- Stop condition: submit a PR and wait for independent review. Do not train models or download new weights/datasets.

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
