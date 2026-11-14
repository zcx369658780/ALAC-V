---
id: GOV-0002
title: GitHub Issue and PR Workflow
status: active
date: 2026-11-14
owner: Owner
---

# GitHub Issue and PR Workflow

## 1. Source of truth

The public repository is the shared source of truth for project code, accepted decisions, tasks, experiments, and acceptance records.

- GitHub Issue = authoritative online task order and discussion thread.
- PR = proposed code/documentation change linked to an Issue.
- `docs/01_tasks/` = durable Markdown snapshot of task scope and status.
- `docs/08_governance/ACTIVE_STATE.md` = current phase, baseline, blockers, and Owner decisions.
- Accepted ADRs = current design decisions.
- Reports = evidence-backed results, not task authorization.

## 2. Issue lifecycle

1. ChatGPT drafts a task with scope, non-goals, acceptance criteria, and evidence requirements.
2. Owner explicitly authorizes implementation.
3. DSH creates a task branch and executes only the authorized scope.
4. DSH opens a PR referencing the Issue.
5. ClawsGO may review independently in read-only mode.
6. ChatGPT reviews code, tests, metrics, evidence, and scientific claims.
7. Owner records final acceptance.
8. Only after acceptance may the Issue be closed and ACTIVE_STATE advanced.

An Issue existing or being assigned does not, by itself, grant implementation permission.

## 3. IDs and naming

- Online task key: `ALAC-NNNN` in the Issue title/body.
- Versioned task snapshot: `TASK-NNNN-<slug>.md`.
- ADR: `ADR-NNNN-<slug>.md`.
- Experiment: `EXP-NNNN-<slug>.md`.
- Run: `artifacts/runs/<UTC>_<EXP>_<variant>/`.
- Evidence: `artifacts/evidence/<EXP>/<run-id>/`.

Numbers are monotonic and never reused. Historical task files are retained; do not rewrite an old authorization to match later findings.

## 4. Branch and PR rules

- Branch: `task/<issue-number>-<short-slug>`.
- Base: `main`.
- Use normal fast-forward PR merge; no force push.
- Use `Refs #N` rather than auto-closing keywords before Owner acceptance.
- A PR should be narrow enough to review and independently test.
- Do not make concurrent conflicting edits from DSH and ClawsGO.

## 5. Required Issue content

Every task must define:
- background and research question;
- objective and scope;
- non-goals and phase boundaries;
- relevant charter/ADR;
- implementation work packages;
- acceptance criteria;
- exact test and evidence requirements;
- privacy/license requirements;
- explicit Owner authorization state.

## 6. Required completion report

The executing agent reports exact commands, test output, run IDs, config/data/model fingerprints, commit SHA, changed files, limitations, and NOT RUN items. Agent self-report is not final acceptance.

## 7. Current transition

- Phase 0/1 initial delivery: TASK-0001.
- Phase 0/1 remediation: TASK-0002; [ALAC-0001 / Issue #1](https://github.com/zcx369658780/ALAC-V/issues/1) is closed as completed after PR #2 merged at `367035318172ba1bbc0103b6231847750143d1d7`.
- Current task: [ALAC-0002 / Issue #3 — Phase 2A autonomous visual learning design](https://github.com/zcx369658780/ALAC-V/issues/3), snapshot `TASK-0003`.
- Current authorization is limited to literature review, design comparison, data plan, resource estimation and experiment pre-registration. Model training, weight/data downloads and Phase 2B/2C/3 work are not authorized.
