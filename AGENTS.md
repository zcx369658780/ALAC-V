# ALAC-V Agent Operating Rules

> This file is the repository-level instruction entry for DSH, ClawsGO, and any future agent.

## 1. Project identity

ALAC-V means **Visual-first Autonomous Learning and Adaptive Cognition**. The long-term research goal is a visual-first intelligent system whose core perception, memory, prediction, and action do not depend on an LLM.

The accepted project charter is `docs/00_charter/ALAC-V-charter.md`. The current state is `docs/08_governance/ACTIVE_STATE.md`.

## 2. Authority order

When instructions conflict, follow this order:

1. Explicit Owner decisions in the current conversation or recorded as accepted decisions.
2. An Owner-authorized, open GitHub Issue task order in this repository.
3. Accepted project charter and accepted ADRs.
4. Current implementation and tests (evidence of what exists, not authorization to expand scope).
5. Reports, proposals, and agent suggestions.

A report marked PASS by an executing agent is not Owner acceptance.

## 3. Task authorization and phase gates

- GitHub Issues are the authoritative online task orders and acceptance threads.
- Files in `docs/01_tasks/` are versioned snapshots or historical records; they do not independently authorize work.
- Do not start implementation merely because an Issue exists. The Issue must explicitly state that Owner authorization has been granted, or the Owner must authorize it in the current conversation.
- Do not enter Phase 2 or later without a separate Owner-approved task order.
- Current gate: Phase 0/1 and Phase 2A design are accepted and merged. Issue #6 / ALAC-0003 authorizes Phase 2A.1 under strict gates: Gate 1 DS-0002 generation and teacher-cache creation with existing local pinned DINOv2 weights is authorized now; Gate 2 independent data acceptance must pass before model implementation advances; Gate 3 implementation review must pass before any formal training. No new model weights or natural-image datasets may be downloaded. Phase 2B/2C and Phase 3 remain BLOCKED.

## 4. Agent roles

- **DSH**: primary implementation agent. Owns local code changes, tests, experiments, and PR preparation.
- **ClawsGO**: independent reviewer. Default to read-only review; do not edit the same working tree or push competing changes unless Owner explicitly assigns implementation.
- **ChatGPT / Owner**: research direction, task definition, independent review, acceptance decisions, and phase authorization.

Do not have DSH and ClawsGO concurrently modify the same branch or files.

## 5. Git workflow

- Repository: https://github.com/zcx369658780/ALAC-V
- Default branch: `main`.
- For each authorized task, create a branch named `task/<issue-number>-<short-slug>`.
- Commit in small, reviewable units. Link the PR to the Issue using `Refs #N`; do not use an auto-close keyword before Owner acceptance.
- Do not push directly to `main`, force-push, rewrite published history, or close an Issue without Owner approval.
- Before work: fetch the configured remote that points to this repository (do not assume its name is `origin`), inspect `git status`, preserve uncommitted changes, and confirm the base SHA.
- After work: run tests, record the exact commands and outputs, push the branch, open a PR, and report commit SHA(s).

## 6. Evidence and scientific integrity

Every reported number must be traceable to:
- run ID and per-run metrics;
- effective configuration snapshot and SHA-256;
- dataset manifest and SHA-256;
- model ID, revision, license, and weight fingerprint;
- code commit and dirty/clean state;
- exact command and environment.

Large raw runs may remain local, but a sanitized evidence bundle must be published under `artifacts/evidence/<experiment-id>/` when the result is used for formal project conclusions. Never invent missing evidence. Mark unavailable items NOT RUN or NOT IN REPOSITORY.

Separate:
- code inspection;
- agent-reported local execution;
- independently reproduced results;
- Owner acceptance.

Do not claim efficiency improvement from storage or latency alone. Report task performance and resource costs together.

## 7. Data, privacy, and licensing

- Keep original user images/videos immutable and local unless Owner explicitly authorizes publication.
- Never commit API keys, tokens, SSH private keys, `.env`, private images/videos, local caches, or model weights unless a task explicitly authorizes a permitted artifact.
- Do not upload user data to cloud services or external LLM APIs.
- Record model and dataset licenses and exact upstream revisions.
- Use relative paths in public documentation; redact usernames, machine-specific paths, and secrets.

## 8. Engineering constraints

- Target machine: Intel Core Ultra 7 265K, 32 GB RAM, RTX 5070 12 GB, Windows; verify actual resources before each major training task.
- Prefer open-source models and local inference.
- Keep the pipeline modular and reproducible.
- Do not introduce heavyweight services or dependencies without an ADR and Owner approval.
- Preserve old accepted ADRs. Supersede them with a new ADR; do not silently rewrite history.

## 9. Completion report

Every PR must state:
1. Issue/task ID and scope.
2. Changed files and design decisions.
3. Tests and exact commands.
4. Experiment IDs, run IDs, configuration/data/model fingerprints.
5. Results and limitations.
6. Security/license review.
7. Remaining work and explicit NOT RUN items.

The Owner decides PASS / PASS WITH LIMITATIONS / FAIL / BLOCKED.
