---
id: INDEX-0001
title: ALAC-V Project Source Index
status: active
date: 2026-11-14
owner: Owner
---

# ALAC-V Project Source Index

> This file is the navigation entry for the ChatGPT Project Source pack and for agents joining the repository. The GitHub repository is canonical; any uploaded ZIP is a dated snapshot, not a replacement for live GitHub state.

## 1. Canonical project source

- Repository: https://github.com/zcx369658780/ALAC-V
- Default branch: `main`
- Current accepted Phase 0/1 merge commit: `367035318172ba1bbc0103b6231847750143d1d7`
- Baseline at initial governance bootstrap: `10cb05d379c39043a17dc840643535fd150bce20`
- Local implementation workspace reported by DSH: `D:\ALAC_V`

## 2. Recommended reading order for a new agent

1. [Root AGENTS.md](../../AGENTS.md) — operating rules, authority order, safety, and phase gates.
2. [Active State](ACTIVE_STATE.md) — current phase, blockers, and task status.
3. [Project Charter](../00_charter/ALAC-V-charter.md) — research goals, non-goals, and stage gates.
4. [Documentation Index](../README.md) — document taxonomy and evidence rules.
5. [Phase 0/1 Task Snapshot](../01_tasks/TASK-0001-phase0-phase1.md) — original authorized scope.
6. [Phase 1 Architecture](../03_design/architecture.md) and [Data Model](../03_design/data-model.md).
7. [ADR directory](../03_design/decisions/) — accepted design decisions.
8. [EXP-0001](../04_experiments/EXP-0001-abc-comparison.md) — preregistration and reported results.
9. [Phase 1 Report](../05_reports/PHASE1-REPORT.md) and [Submission Record](../submissions/PHASE_0_1_SUBMISSION.md).
10. [Final Phase 0/1 Acceptance](PHASE0_1_ACCEPTANCE.md) and [Closed remediation Issue](https://github.com/zcx369658780/ALAC-V/issues/1).
11. [TASK-0003 — Phase 2A autonomous visual learning design](../01_tasks/TASK-0003-phase2a-autonomous-visual-learning-design.md) and [Open Issue #3](https://github.com/zcx369658780/ALAC-V/issues/3).

## 3. Repository document map

| Area | Canonical path | Purpose |
|---|---|---|
| Project entry | `README.md` | Overview, status, install and quick start |
| Agent rules | `AGENTS.md` | Rules every coding/review agent must follow |
| Change history | `CHANGELOG.md` | Versioned project changes |
| Charter | `docs/00_charter/ALAC-V-charter.md` | Long-term goals, principles, phase boundaries |
| Tasks | `docs/01_tasks/` | Immutable/versioned task snapshots |
| Environment | `docs/02_audit/` | Hardware, runtime, dependency and model license audits |
| Design | `docs/03_design/` | Architecture, data model and ADRs |
| Experiments | `docs/04_experiments/` | Preregistration, metrics and interpretation |
| Research | `docs/research/` | Literature review and external-evidence notes |
| Reports | `docs/05_reports/` | Phase reports |
| Operations | `docs/06_ops/` | Setup, runbook and troubleshooting |
| Risks | `docs/07_risks/` | Risk register |
| Governance | `docs/08_governance/` | Active state, acceptance protocol, Issue workflow and source index |
| Submission history | `docs/submissions/` | Immutable delivery records |
| Source code | `src/alacv/` | Image archive, pyramid, foveation, features, store, retrieval, metrics and pipeline |
| Scripts | `scripts/` | Audit, model fetch, dataset generation, experiments and integrity checks |
| Tests | `tests/` | Unit and integration tests |
| Configurations | `configs/` | Defaults, experiment and dataset parameters |
| Data manifests | `data/manifests/` | Dataset and model fingerprints; raw datasets are not committed |
| Experiment summaries | `artifacts/experiments/` | Aggregated JSON/Markdown results and confidence intervals |
| Figures | `artifacts/figures/` | Report figures |
| Evidence bundles | `artifacts/evidence/EXP-0001/` | Sanitized per-run evidence (9 runs) for independent review |

## 4. Important implementation files

| File | Responsibility |
|---|---|
| `src/alacv/ingest.py` | Decode validation, SHA-256 content-addressed raw archive, metadata |
| `src/alacv/imaging.py` | RGB conversion, image decode, multi-scale pyramid and input fitting |
| `src/alacv/foveation.py` | ROI scoring and foveated mosaic / ROI crop generation |
| `src/alacv/features.py` | Frozen DINOv2 encoder and CLS/Patch feature extraction |
| `src/alacv/store.py` | SQLite metadata and SafeTensors feature storage |
| `src/alacv/retrieval.py` | Cosine retrieval, metrics and ROI backtracking |
| `src/alacv/pipeline.py` | A/B/C image pipeline orchestration |
| `scripts/run_experiments.py` | EXP-0001 experiment execution and run artifact generation |
| `scripts/bootstrap_ci.py` | Local CI/bootstrap helper |
| `tests/` | Regression and unit tests |

## 4b. Phase 2A design documents (submitted for review)

- `docs/research/PHASE2A_AUTONOMOUS_LEARNING_LITERATURE.md`
- `docs/03_design/PHASE2A_AUTONOMOUS_LEARNING_DESIGN.md`
- `docs/04_experiments/EXP-0002-autonomous-visual-learning-preregistration-draft.md`
- `docs/02_audit/PHASE2A_MODEL_LICENSE_MATRIX.md`
- `docs/05_reports/PHASE2A_OWNER_DECISION_SUMMARY.md`

## 5. Key accepted design records

- ADR-0001: HuggingFace DINOv2 ViT-S/14, frozen.
- ADR-0002: SHA-256 content-addressed immutable raw archive.
- ADR-0003: SQLite metadata + SafeTensors features; no vector database in Phase 1.
- ADR-0004: independent 518×518 forward passes for pyramid levels.
- ADR-0005: dual foveation artifacts (mosaic + full-resolution ROI crop).
- ADR-0006: FP16 feature storage, FP32 retrieval accumulation.
- ADR-0007: gradient-energy ROI importance (original wording retained; superseded on terminology by ADR-0008).
- ADR-0008: ROI importance uses central-difference gradient energy; behavior frozen, no Sobel substitution.

## 6. Data and evidence boundaries

The public repository intentionally excludes raw private images/videos, local dataset files, model weights, full local run stores, caches, and credentials. Dataset manifests and aggregate experiment summaries are versioned. For formal acceptance, sanitized run-level evidence belongs in `artifacts/evidence/`.

## 7. Current project state

- Phase 0/1: independently accepted; PR #2 merged; Issue #1 closed.
- Phase 2A: research design and experiment pre-registration authorized under Issue #3; no model training or new dataset/weight download authorized.
- Phase 2B/2C and Phase 3: blocked pending separate gates.

## 8. Uploaded Project Source pack

The companion ZIP should contain this index plus the project charter, current state, acceptance protocol, architecture, data model, task snapshot, ADRs, experiment record, audit, report, runbook, risk register, and submission record. Its timestamp/commit must be recorded when uploaded; always check GitHub for newer changes.
