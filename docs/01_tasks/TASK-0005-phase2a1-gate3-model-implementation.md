---
id: TASK-0005
title: Phase 2A.1 Gate 3 自主视觉学习模型实现
status: gate-3-submitted; awaiting independent review
date: 2026-11-16
owner: Owner
source: ALAC-0003
issue: https://github.com/zcx369658780/ALAC-V/issues/6
base: 6f8a5ec1d03c90871168dbf97e863a51e38305ce
branch: task/6-phase2a1-gate3-model-implementation
---

# TASK-0005 — Gate 3 模型实现

- **Issue**：#6（ALAC-0003）
- **授权**：Owner 2026-11-16 Gate 2 PASS / Gate 3 implementation only
- **基线**：`6f8a5ec1d03c90871168dbf97e863a51e38305ce`（PR #8 HEAD，DS-0002 v3）
- **分支**：`task/6-phase2a1-gate3-model-implementation`
- **PR**：stacked，base = `task/6-phase2a1-ds0002-generation`，`Refs #6`

## 范围

1. WP0：关闭 Gate 2 四项遗留（cache dataset_id、pair_key 措辞、标签隔离测试、独立遮挡参考测试）。
2. WP1：实现 F0 / A1 / A2 / B2 / C2 / R0 六臂、mask 采样、目标函数、坍塌诊断、checkpoint/早停/VICReg 规则。
3. WP2：单元/集成测试、forward/backward 冒烟、显存报告、A1/F0 不变量证据、证据归档。

## 严格禁止（本次未执行）

- 任何 `optimizer.step()`（`optimizer_steps = 0`）
- 正式训练 epoch / EXP-0002 / 数据效率（`training_runs = 0`，`EXP-0002 = NOT RUN`）
- 新权重或自然图像下载
- Phase 2B/2C、视频、Event Camera、SNN、Phase 3

## 交付

- `src/alacv/ssl/`：`masking.py`、`modules.py`、`arms.py`、`checkpoint.py`、`data.py`
- `scripts/train_ssl.py`（默认 dry-run，拒绝训练模式）、`scripts/eval_ssl.py`
- `configs/exp0002.yaml`（冻结超参）
- `tests/test_ssl.py`、`tests/test_ssl_invariants.py`、`tests/test_occlusion_reference.py`
- 证据：`artifacts/evidence/GATE3/`、`artifacts/evidence/DS-0002/test-runs/gate3-pytest.txt`
- 报告：`docs/05_reports/TASK-0005-GATE3-REPORT.md`
