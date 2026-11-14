---
id: GOV-0001
title: ALAC-V Acceptance Protocol
status: active
date: 2026-11-14
owner: Owner
---

# ALAC-V Acceptance Protocol

## 1. Acceptance statuses

| Status | Meaning |
|---|---|
| PASS | All required criteria are met and evidence is independently verifiable. |
| PASS WITH LIMITATIONS | Core deliverable works, but explicit non-blocking limitations remain; Owner accepts those limitations. |
| FAIL | A required criterion is not met or a critical defect is confirmed. |
| BLOCKED | Required evidence, access, dependency, or Owner decision is missing. |
| NOT RUN | A test or experiment was not executed. Never treat as PASS. |
| AGENT-REPORTED | The executing agent reports success, but independent evidence/reproduction is not yet available. |

## 2. Four review layers

1. **L1 Engineering**: source structure, dependency lock, tests, code paths, data integrity, security.
2. **L2 Model**: model identity, revision, architecture, preprocessing, license, weight hash, input/output contract.
3. **L3 Experiment**: pre-registration, dataset manifest, run-level metrics, config snapshots, resource measurements, statistics, repeatability.
4. **L4 Scientific interpretation**: whether evidence supports the hypothesis, whether limitations and alternative explanations are stated, and whether claims exceed the data.

## 3. Mandatory evidence chain

Every formal metric must be linked to:
- unique run ID;
- `metrics.json` and, where relevant, per-item `items.jsonl`;
- effective `config.snapshot.yaml` and `config.sha256`;
- environment and model snapshots;
- dataset manifest and its SHA-256;
- code commit and dirty/clean state;
- exact execution command.

Local full runs may remain outside Git if they contain large stores or private data. A sanitized evidence subset must be published in `artifacts/evidence/<experiment-id>/<run-id>/`. Do not publish raw private images, full local stores, secrets, or model weights.

If a required item is absent, the affected number is AGENT-REPORTED or NOT independently verified, not a formal reproducible result.

## 4. Review discipline

- Read the task Issue and accepted ADRs before reviewing implementation.
- Check source code against design claims; do not rely only on README or agent summaries.
- Distinguish static inspection from execution. ChatGPT must not claim to have run code on the Owner's workstation.
- Recompute arithmetic and inspect raw run evidence where available.
- Check the actual preprocessing path when evaluating image/video quality metrics.
- Check all claims of compression or efficiency against both resource cost and task performance.
- Preserve pre-registered hypotheses and thresholds. Changes require a new experiment ID or an explicitly documented protocol amendment.

## 5. PR and closure rules

- Every implementation change must be in a task branch and submitted as a PR.
- PRs reference Issues; they do not automatically close them.
- ClawsGO review is independent evidence, not Owner acceptance.
- Only the Owner may record final acceptance and close a task Issue.
- Update `docs/08_governance/ACTIVE_STATE.md` after each accepted milestone.

## 6. Phase gates

Phase 0/1 acceptance is required before Phase 2. Phase 2 acceptance is required before Phase 3. A completed task does not authorize the next phase automatically.
