---
id: TASK-0004
title: Phase 2A.1 DS-0002 数据生成、模型实现与 EXP-0002 实验
status: owner-authorized; gate-1-authorized
date: 2026-11-15
owner: Owner
source: ALAC-0003
issue: https://github.com/zcx369658780/ALAC-V/issues/6
---

# ALAC-0003 — Phase 2A.1：DS-0002 数据生成、模型实现与 EXP-0002 实验

- **Task snapshot:** `docs/01_tasks/TASK-0004-phase2a1-implementation-and-ds0002.md`
- **Phase:** Phase 2A.1（自主视觉学习，合成静态图像）
- **执行 Agent：** DSH
- **独立审查：** ChatGPT；ClawsGO 可按 Owner 安排只读审查
- **授权基线：** Phase 2A design merge `ef39be01cf7c7bec2650f26af9e2bcefc667c526`
- **Owner authorization:** **ACCEPTED on 2026-11-15**
- **Design baseline:** `docs/03_design/PHASE2A_AUTONOMOUS_LEARNING_DESIGN.md`
- **Pre-registration:** `docs/04_experiments/EXP-0002-autonomous-visual-learning-preregistration-draft.md`

## 1. Owner decisions — explicit authorization

Owner explicitly approved both:
- **D-2A-1: ACCEPTED** — enter Phase 2A.1 implementation and conduct the pre-registered feasibility experiment, subject to the stage gates below.
- **D-2A-2: ACCEPTED** — generate DS-0002, including the frozen DINOv2 teacher-feature cache using only the already-pinned Phase 1 local weights/revision.

Previously accepted:
- **D-2A-3: ACCEPTED** — Phase 2A conclusions are limited to the synthetic domain.
- **D-2A-4: ACCEPTED** — A2 is the sole primary arm; B2/C2 are secondary comparisons.

This Issue is the authoritative task order. The work MUST follow the ordered gates. **No formal model training is permitted before the DS-0002 independent acceptance gate is passed and recorded in this Issue.**

## 2. Research objective

Implement and evaluate the pre-registered self-supervised visual representation-learning candidates under the frozen DINOv2 ViT-S/14 teacher, on a procedurally generated synthetic static-image dataset.

Primary question: does A2 (JEPA-style contiguous-block masked latent prediction + adapter) improve held-out synthetic-scene retrieval Recall@1 over F0 (frozen DINOv2), with the pre-registered H1-D/H1-P/H1-S interpretation?

B2 (cross-view self-distillation) and C2 (random-mask latent prediction) are secondary. A1 is the predictor-only invariant control; R0 is the random-initialized adapter control; F0 is the frozen teacher baseline.

## 3. Stage gates — strict execution order

### Gate 0 — synchronize and freeze the accepted design

1. Fetch latest `main`; verify merge commit `ef39be01cf7c7bec2650f26af9e2bcefc667c526`.
2. Read `AGENTS.md`, `ACTIVE_STATE.md`, the accepted Phase 2A architecture, EXP-0002, license matrix, and this Issue.
3. Verify local repository clean state and available disk/RAM/GPU; preserve any uncommitted work.
4. Freeze the exact EXP-0002 file SHA and configuration defaults in the task branch. Any change to arm, threshold, seed, data split, mask, checkpoint, optimizer, or primary endpoint requires a pre-run amendment and Owner/independent review.

### Gate 1 — generate DS-0002 and teacher cache (authorized now)

Dataset specification:
- **Base scenes:** train / validation / test = **2,000 / 300 / 900**.
- Split by base-scene seed/group; no scene or derivative view may cross splits.
- Train: 8 views per scene (16,000 views).
- Validation/test: 5 views per scene (1 gallery + 4 queries).
- Resolution: 1024×768 source images; DINOv2 input preprocessing follows accepted Phase 1 / EXP-0002 specification (518×518, 37×37 patches).
- Synthetic scene composition follows the accepted design: 1–3 textured targets and 2–4 distractor blocks; controlled variation in shape, position, scale, occlusion, texture, palette, lighting/contrast and view augmentation.
- Training-visible fields are pixels, training-side augmentation parameters, and deterministic mask seeds only.
- Scene IDs, target boxes, factor labels and evaluation metadata must be isolated from the training DataLoader behind an explicit field allowlist.

Generation requirements:
1. Use a deterministic, versioned procedural generator; record generator code commit, config, PRNG algorithm/version, seed derivation and per-file SHA-256.
2. Produce train/val/test manifests and a separate evaluation-label sidecar. Record split group IDs in audit metadata; training loader must not expose them.
3. Ensure exactly 3,200 base scenes and expected view counts (16,000 train; 1,500 validation; 4,500 test).
4. Generate the frozen DINOv2 teacher feature cache using **only the already-existing pinned Phase 1 local weights/revision**. No network fetch. If the local pinned weight is unavailable or its fingerprint differs, stop and request Owner direction; do not download.
5. Cache CLS and 1,369×384 patch features in the accepted FP16/SafeTensors or memory-mapped format, with explicit view-to-feature index mapping, checksums and model fingerprint.
6. Keep image files, caches and large artifacts local/ignored by Git. Commit only generator code/config, sanitized manifests, hashes, audit summaries and compact evidence. Do not commit raw images, feature caches or weights unless separately authorized.
7. Estimate disk use before generation. Expected images ~2.8 GB and teacher cache up to ~22 GB; require at least 30 GB free on the target volume before starting. Record actual disk/RAM/GPU use.

Stop at the end of Gate 1 and report; do not proceed to model implementation until Gate 2 is independently accepted.

### Gate 2 — independent DS-0002 acceptance (hard gate)

Submit:
- scene/view counts and per-split manifest;
- proof of zero base-scene/seed overlap across splits;
- deterministic regeneration spot-check (same seed/config yields identical bytes/hashes);
- image dimensions, decode checks, corruption/duplicate checks;
- evaluation-label sidecar completeness and separation;
- training DataLoader field-allowlist audit/test proving no scene ID, bbox, factor or test labels are exposed;
- teacher cache count, shape, dtype, index alignment, missing/duplicate detection, model revision and weight fingerprint;
- SHA-256 inventory and storage/RAM/GPU usage;
- privacy/license scan and NOT RUN statement.

**Stop and wait for ChatGPT independent review and Owner gate acceptance.** A DSH self-reported PASS is not sufficient. No model implementation requiring DS-0002 and no optimizer step before this gate.

### Gate 3 — model implementation (authorized only after Gate 2 PASS)

Implement according to the accepted architecture and EXP-0002:
- F0 frozen DINOv2 baseline;
- A1 predictor-only invariant control;
- A2 sole primary (continuous-block masked latent prediction);
- B2 secondary (cross-view self-distillation; conditional VICReg trigger exactly as pre-registered; any B2+VR result separately labelled);
- C2 secondary (random 821-patch masked latent prediction);
- R0 random-initialized adapter control.

Required modules/scripts/configs:
- `src/alacv/ssl/`: adapter, predictor, deterministic mask samplers, objectives, collapse diagnostics;
- teacher-cache read/validation utilities (reuse Gate 1 cache format);
- `scripts/train_ssl.py`, `scripts/eval_ssl.py`;
- `configs/exp0002.yaml`;
- unit/integration tests for tensor shapes, loss targets, stop-gradient, mask counts/geometry/determinism, A1/F0 exact descriptor invariance, no-label leakage, checkpoint selection, early stopping and fail-closed cache validation.

Implementation constraints:
- Freeze the exact teacher revision and weight SHA from Phase 1; no backbone fine-tuning.
- Trainable parameters <=10M per arm; report actual parameter counts.
- All arms use 384-d L2-normalized CLS retrieval readout.
- A2/C2 mask exactly 821 of 1,369 patches; A2 four strictly non-overlapping contiguous blocks, C2 random 821; mask pixels before student encoder forward. Full-image cached teacher features are targets only.
- A2/C2 share adapter/predictor architecture; B2 target/predictor are 768-d as pre-registered.
- A1 retrieval descriptors must be bit-identical to F0 (descriptor SHA-256 equal, Kendall tau=1, Recall delta=0).
- H1 checkpoint selected by validation Recall@1; ties: lower validation prediction loss, then earlier epoch. Early stopping independently uses validation prediction loss with patience 3. Test never selects checkpoints or hyperparameters.
- Preserve train/validation/test group isolation and strict label allowlist.

After implementation, run all tests and submit a PR. **Do not start training until ChatGPT/Owner has reviewed implementation, tests, and the exact frozen EXP-0002 configuration.**

### Gate 4 — pre-registered training and experiment (authorized only after Gate 3 acceptance)

After explicit Gate 3 PASS:
- Run 4 trainable arms (A1/A2/B2/C2) × 3 seeds {2026, 2027, 2028} = 12 main runs.
- F0 and R0 are controls and do not consume optimizer training runs.
- Data-efficiency study: selected arm determined only from validation; 25%/50%/100% data × the same 3 seeds = 9 secondary descriptive runs. Run the full 9 before the single batched test evaluation for this secondary study.
- Maximum **21 training runs** total. Any B2+VR trigger follows the exact preregistered rule and is separately labelled, not silently pooled with plain B2.
- Use the exact accepted EXP-0002 H1-D/H1-P/H1-S definitions, Holm correction, seed-wise reporting and scene-level hierarchical bootstrap (n=900 within each seed).
- Do not change thresholds, seeds, data splits, mask budget, checkpoint rules or endpoints after observing test results.
- Runtime stop: if any run exceeds 2 h or peak GPU memory exceeds 9 GB, stop and report; do not silently reduce resolution or alter the method.
- Record per-run ID, config SHA, data manifest SHA, teacher/model fingerprint, code commit/dirty, command, environment, metrics, checkpoint SHA and resource telemetry.
- Store weights/checkpoints locally; publish sanitized evidence only, not weights or raw images.

### Gate 5 — final independent acceptance

Submit a report with:
- all run IDs and complete evidence;
- per-seed and aggregated results;
- H1-D/H1-P/H1-S decisions and uncertainty;
- secondary B2/C2 results with Holm correction;
- A1/F0 invariant proof;
- collapse diagnostics and B2+VR events;
- compute/memory/storage/latency costs;
- synthetic-domain limitations and negative results;
- exact code/data/model fingerprints and reproduction commands.

Stop for ChatGPT independent review and Owner acceptance. No automatic Phase 2B/2C transition.

## 4. Explicit non-goals and prohibitions

- No new model-weight downloads; only the already pinned Phase 1 DINOv2 local weights may be used.
- No public natural-image dataset downloads, no user private images, no cloud inference or external LLM APIs.
- No backbone fine-tuning.
- No video, optical flow, real-time stream, Event Camera or SNN.
- No Phase 2B long-term memory implementation.
- No Phase 2C dynamic compute controller.
- No changes to accepted Phase 1 results/ADRs.
- No direct push to main, no force push, no history rewrite.
- No closing this Issue before Owner final acceptance.

## 5. Deliverables

1. DS-0002 generator, configs, sanitized manifests, checksums and data audit report.
2. Frozen teacher feature cache and alignment/provenance evidence (local only; publish fingerprints and summary).
3. Independent Gate 2 data acceptance report.
4. A1/A2/B2/C2/F0/R0 implementation, tests, configs and code PR.
5. EXP-0002 run/evidence bundles and final scientific report.
6. Updated `ACTIVE_STATE.md`, task snapshot, project index and changelog through PRs.

## 6. Acceptance criteria

- [ ] All 3,200 base scenes and expected views generated with deterministic hashes.
- [ ] Split isolation proven at base-scene/seed level.
- [ ] Training DataLoader cannot access evaluation labels; enforced by tests.
- [ ] Teacher cache complete, aligned, checksummed and uses exact pinned Phase 1 model/weights.
- [ ] Gate 2 independently accepted before model implementation/training advances.
- [ ] Model code conforms to accepted architecture and has complete tests.
- [ ] Gate 3 independently accepted before any optimizer step.
- [ ] EXP-0002 executed without unregistered deviations; max 21 runs.
- [ ] All scientific claims supported by per-run evidence and uncertainty.
- [ ] No unapproved downloads, private data uploads or later-phase work.
- [ ] PRs submitted; DSH stops at each gate and waits for independent/Owner acceptance.

## 7. Current authorization state

**Owner-authorized now:** Gate 0, Gate 1 DS-0002 generation and frozen teacher-cache creation with existing local pinned weights.

**Authorized after Gate 2 PASS:** Gate 3 model implementation.

**Authorized after Gate 3 PASS:** Gate 4 training/EXP-0002 (up to 21 runs).

**Not authorized:** natural-image data/weights, backbone fine-tuning, Phase 2B/2C, Phase 3.

The sequence is mandatory. Do not treat overall Phase 2A.1 approval as permission to skip a gate.
