---
id: TASK-0002
title: Phase 0/1 independent acceptance remediation
status: owner-authorized; implementation submitted for review
date: 2026-11-14
owner: Owner
source: ALAC-0001
issue: https://github.com/zcx369658780/ALAC-V/issues/1
---

# TASK-0002 — Phase 0/1 independent acceptance remediation

## Status

**Owner-authorized / implementation in progress (2026-11-14).** Issue #1 records the authorization and
is the authoritative online discussion and acceptance record. This snapshot tracks the eight remediation
work packages; acceptance remains reserved to the Owner.

## Objective

Resolve source-level inconsistencies and publish a sanitized run-level evidence bundle so Phase 0/1 can be independently audited. Phase 2 remains blocked.

## Scope

1. Publish sanitized per-run evidence for EXP-0001.
2. Correct A-variant backtrack reconstruction to match the exact resize + center-crop transform.
3. Reconcile ADR-0007 with the actual ROI gradient implementation.
4. Correct ICC/color-management documentation or implement a tested profile transform.
5. Fix the TASK-0001 test evidence pointer mismatch without rewriting the historical authorization.
6. Validate CPU peak percentage semantics.
7. Verify OS product/version/build from direct system evidence.
8. Sanitize machine-specific sample source labels in public manifests.

## Non-goals

- No Phase 2 implementation.
- No video processing, temporal models, Event Camera, SNN, or real-time stream work.
- No DINOv2 backbone training or large dataset download.
- No redesign of the project charter or wholesale replacement of the Phase 1 architecture.

## Acceptance criteria

- Run-level evidence can be traced to config, dataset, model, and code fingerprints.
- A backtrack metric is recomputed using the exact encoder preprocessing transform.
- ROI algorithm code, tests, and ADR terminology agree.
- Public documentation and manifests contain no unnecessary machine-specific paths.
- Full test suite passes and exact output is recorded.
- Phase 1 report clearly separates original results, corrected results, and limitations.
- Owner independently reviews the PR and records final acceptance.

## Authorization

No implementation is authorized by this snapshot alone. Wait for explicit Owner approval in the current conversation or Issue.
