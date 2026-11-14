---
id: ACCEPT-0001
title: Preliminary Independent Review of Phase 0 and Phase 1
status: remediation-required
date: 2026-11-14
owner: ChatGPT reviewer; final decision reserved to Owner
task: TASK-0001
baseline: 10cb05d379c39043a17dc840643535fd150bce20
issue: https://github.com/zcx369658780/ALAC-V/issues/1
---

# Preliminary Independent Review — Phase 0/1

## 1. Review scope and method

Reviewed the live public repository at baseline `10cb05d379c39043a17dc840643535fd150bce20`, including:
- README and documentation index;
- project charter and TASK-0001;
- environment and model-license audit;
- architecture, data model, and ADR-0001 through ADR-0007;
- EXP-0001 preregistration and result sections;
- Phase 1 report and submission record;
- selected source files: imaging, ingest, foveation, features, pipeline, retrieval, metrics, and experiment runner;
- aggregate experiment JSON and dataset manifest.

This was a repository/document/source inspection. The reviewer did **not** execute code on the Owner's workstation, access the local D:\ALAC_V working tree, or independently reproduce the 56 tests and GPU runs. DSH-reported local execution is therefore distinguished from independent runtime verification.

## 2. Preliminary verdict

**Engineering baseline: PASS WITH LIMITATIONS (repository structure and implementation are present).**

**Formal Phase 1 scientific acceptance: BLOCKED pending remediation and evidence completion.**

This is not a rejection of the delivered work. The repository has a coherent modular structure, a useful task/ADR/experiment document system, a frozen DINOv2 image pipeline, and transparent reporting that does not claim an overall efficiency gain.

However, several source-level and evidence-chain issues affect the validity or independent auditability of reported metrics. Phase 2 remains BLOCKED.

## 3. Positive findings

1. Public repository exists, is public, and uses `main`; the current baseline and initial commit ancestry were read from GitHub.
2. The repository is already organized into source, scripts, tests, configs, docs, manifests, experiment summaries, and figures. No broad directory reshuffle is needed.
3. The charter clearly states raw-data immutability, local/open-source preference, LLM non-dependence for core visual reasoning, evidence-backed efficiency claims, and phase gates.
4. The architecture and data model are sufficiently explicit for the Phase 1 static-image pipeline.
5. DINOv2 ViT-S/14 is pinned to a HuggingFace revision and the license is recorded as Apache-2.0.
6. The report honestly states that A/B/C do not establish a general efficiency improvement and identifies ROI selection as weak.
7. The reported DS-0001 benchmark is explicitly synthetic and the report warns against generalizing it to natural-image performance.
8. The repository excludes raw images, model weights, local caches, and full local run stores from public Git history.

## 4. Findings requiring remediation

### R1 — Run-level evidence is not in the public repository (blocking)

The repository's own `docs/README.md` §7 requires every formal metric to link to run-level `metrics.json`, config snapshot/hash, dataset manifest, and code commit/dirty state.

However, `.gitignore` excludes `artifacts/runs/*`. The public repository contains aggregate files under `artifacts/experiments/`, but not the nine run-level evidence directories. The submission record also confirms that raw run directories were not committed.

**Required:** publish a sanitized evidence subset under `artifacts/evidence/EXP-0001/<run-id>/`, including at minimum metrics, effective config and hash, environment, model fingerprint, per-item results, dataset manifest hash, and code commit/dirty state. Do not publish raw images, full local stores, secrets, caches, or model weights. If an item cannot be recovered, mark it missing and downgrade the affected claim.

### R2 — A-variant backtrack reconstruction does not match the encoder's center crop (blocking for A backtrack metrics)

The encoder uses `fit_to_input()`: resize the short side to 518 and then center-crop a 518×518 square.

The A-variant reconstruction in `scripts/run_experiments.py::recover()` resizes the image but crops using scaled original coordinates without subtracting the horizontal/vertical center-crop offset. For non-square inputs such as the 1024×768 synthetic scenes, this does not reproduce the actual encoder-visible crop.

**Required:** implement the exact inverse/visibility mapping of the preprocessing transform; define behavior for ROIs partly or fully outside the center crop; add regression tests; recompute A backtrack PSNR/SSIM and update the report. Do not continue treating the old A value as a valid apples-to-apples reconstruction metric.

### R3 — ROI gradient method name differs from implementation (design consistency)

ADR-0007 and the function docstring describe Sobel gradient magnitude. The current `gradient_energy_map()` implementation uses two-pixel central differences (`arr[:, 2:] - arr[:, :-2]` and vertical equivalent), not the standard 3×3 Sobel operator.

**Required:** either implement and test Sobel, or preserve the current method and add a new ADR accurately naming it as central-difference gradient energy. Do not silently rewrite an accepted ADR.

### R4 — ICC profile is recorded but not color-managed

The ingest path identifies and records ICC profile descriptions and hashes. The RGB conversion path uses PIL mode conversion/compositing, but does not apply an ICC profile transform (for example, profile-to-sRGB) before feature extraction.

**Required:** either implement a tested color-managed conversion to a declared target space, or correct documentation to say “mode conversion + ICC metadata recording; no ICC color transform.” This is a scope/claim correction, not necessarily a Phase 1 blocker if the target is explicitly limited to ordinary RGB inputs.

### R5 — Task evidence pointer mismatch

TASK-0001 refers to `tests/test_pyramid.py`, while the repository contains pyramid size tests in `tests/test_imaging.py::test_pyramid_sizes`.

**Required:** correct the navigation/evidence index or add the intended test file. Preserve TASK-0001 as the historical authorization; record the correction in this review or a follow-up document.

### R6 — CPU peak percentage needs validation

The aggregate experiment JSON reports `peak_cpu_pct` values above 4000 while the audited machine has 20 logical processors. The average CPU-core calculation is separately reported, but the peak percentage needs validation against psutil's Windows semantics and the sampling code.

**Required:** explain the measurement semantics or correct/rename the metric. Do not use the current peak percentage as a formal resource claim until validated.

### R7 — OS product name/build needs a direct system-source check

The environment audit records “Windows 10 Pro 26H2, build 26300.9711” and also notes that the build is atypical for common Windows 10 releases. This review did not independently verify the OS identity.

**Required:** preserve raw `winver` or registry/ProductName/DisplayVersion/CurrentBuild evidence and state the exact product/version without inference.

### R8 — Public sample manifest source path

The public DS-0001 manifest includes local Windows wallpaper source paths. They do not appear to include a user profile, but are machine-specific and unnecessary for reproducing the synthetic dataset.

**Required:** replace with a generic source label in the public manifest/generator, while preserving the local-only source provenance in private local records if needed.

## 5. DSH-reported results (not independently reproduced)

| Variant | End-to-end median | Total derived storage / image | Recall@1 | mAP@10 |
|---|---:|---:|---:|---:|
| A | 34.4 ms | 1.00 MB | 0.833 | 0.883 |
| B | 216.2 ms | 4.10 MB | 0.875 | 0.918 |
| C | 361.3 ms | 6.14 MB | 0.889 | 0.922 |

Reported ablations show C2 (foveated + ROI) at 2.01 MB and Recall@1 0.903, and C3 at 4.02 MB and Recall@1 0.931. These are DSH-reported aggregate results until run-level evidence is available.

The report's conclusion that no overall efficiency improvement is established is appropriately conservative and should be retained.

## 6. Required next actions

Tracked in [ALAC-0001](https://github.com/zcx369658780/ALAC-V/issues/1). The Issue is an audit/remediation backlog. DSH must first synchronize governance files and wait for explicit Owner authorization before changing code or rerunning experiments.

## 7. Phase gate

**Phase 2: BLOCKED.** No video model training, temporal model implementation, Event Camera work, SNN work, or autonomous learning expansion is authorized by this review.
