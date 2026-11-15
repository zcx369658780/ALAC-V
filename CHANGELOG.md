# CHANGELOG

## 2026-11-15 — Phase 2A design accepted; Phase 2A.1 authorized

- Owner approved D-2A-1 (Phase 2A.1 implementation) and D-2A-2 (DS-0002 generation).
- Merged PR #5; accepted design merge commit: `ef39be01cf7c7bec2650f26af9e2bcefc667c526`.
- Closed ALAC-0002 / Issue #3 as completed.
- Opened ALAC-0003 / Issue #6 for DS-0002 generation, teacher feature cache, model implementation and EXP-0002.
- Gate 1 DS-0002 generation and frozen teacher-cache creation are authorized using existing local pinned Phase 1 DINOv2 weights only.
- Formal training remains gated on independent DS-0002 acceptance (Gate 2) and implementation review (Gate 3).
- No new model-weight or natural-image dataset downloads. Phase 2B/2C and Phase 3 remain blocked.

## 2026-11-15 — Phase 0/1 accepted; Phase 2A design authorized

- Owner approved and merged PR #2; merge commit: `367035318172ba1bbc0103b6231847750143d1d7`.
- Closed ALAC-0001 / Issue #1 as completed after final independent acceptance PASS.
- Recorded corrected GT bbox result: H3 rejected; C is 3.14 dB below A on the preregistered target-region fidelity metric.
- Opened ALAC-0002 / Issue #3 for Phase 2A autonomous visual learning literature review, architecture comparison, data design and experiment pre-registration.
- Owner-approved sequence: Phase 2A autonomous visual learning → Phase 2B long-term visual memory → Phase 2C dynamic compute allocation → Phase 3 video/real-time streams.

## 2026-11-14 — Governance bootstrap

- Added root-level `AGENTS.md` and GitHub Issue/PR templates.
- Added `docs/08_governance/` for active state, acceptance protocol, Issue workflow, project source index, and Phase 0/1 independent review.
- Clarified that GitHub Issues are authoritative online task orders; Markdown task files are versioned snapshots.
- Recorded Phase 0/1 as engineering-complete per DSH report but not yet independently accepted; Phase 2 remains BLOCKED.
- Opened ALAC-0001 to track evidence-chain and metric-validity remediation.

## Initial Phase 0/1 delivery (2026-11-14)

## [Unreleased] — Phase 1

### Added
- Phase 0 环境审计脚本与报告（硬件、驱动、CUDA、Python、依赖、磁盘、GPU FP16 自检）。
- 原始图像归档：内容寻址、SHA-256、只读、EXIF 摆正、JPEG/PNG/WebP/AVIF 支持。
- 多尺度金字塔：可配置 1/1、1/2、1/4、1/8，WebP 落盘。
- Foveated ROI：梯度能量重要性、分级马赛克、全分辨率 ROI 裁片、ROI 坐标入库。
- DINOv2 ViT-S/14 编码器：CLS + Patch、FP16、预处理参数与模型指纹记录。
- 视觉知识库：SQLite（7 表）+ SafeTensors 特征 + 完整性校验。
- 检索与回溯：余弦 Top-K、Recall@K/mAP、ROI 高清回溯与 PSNR/SSIM 评估。
- EXP-0001：A/B/C 三组对照实验（预注册 + 自动汇总）。
- 文档系统：宪章、任务单、审计、架构、数据模型、7 篇 ADR、实验、报告、运维、风险。
- 单元测试：配置、归档、金字塔、foveation、存储、检索、指标、管线、特征。

### Notes
- 已知限制与风险见 `docs/07_risks/risk-register.md`。
- Phase 1 完成后停止，不自动进入 Phase 2。
