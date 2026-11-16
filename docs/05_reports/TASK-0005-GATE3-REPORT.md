---
id: REPORT-0005
title: TASK-0005 Gate 3 模型实现报告
status: 待独立复核
date: 2026-11-16
issue: https://github.com/zcx369658780/ALAC-V/issues/6
---

# TASK-0005 Gate 3 报告

> 仅实现与验证；**optimizer_steps = 0，training_runs = 0，EXP-0002 = NOT RUN**。
> 等待 ChatGPT 独立 Gate 3 复核与 Owner 验收。

## 1. WP0 — 四项遗留

| 项 | 处理 | 证据 |
|---|---|---|
| cache summary dataset_id | `build_teacher_cache.py` 从 `manifests/ds0002.json` 读取并写入；本地 `cache.json` 已更新为 `ds-0002-synthetic-v3`，未重算 23 GB 张量 | `cache.json: dataset_id`；audit `teacher_cache.dataset_id_matches` |
| pair_key 措辞 | 全仓库改为 “sampler-only opaque grouping token”，删除“不可逆” | `src/alacv/ds0002.py`、`scripts/generate_ds0002.py` |
| 标签隔离测试 | 在 Dataset 构造前 patch `Path.open` / `io.open` / `builtins.open`，守卫持续到 init、取样、sampler | `tests/test_ds0002.py::test_loader_never_opens_labels_manifests_or_identity` |
| 独立遮挡参考测试 | 从 sidecar 几何自行 rasterize 三种形状，重算 union/阈值；未调用生产 mask 实现 | `tests/test_occlusion_reference.py`（3 项） |

独立遮挡复算未发现 v3 标签与实际几何不一致，因此**未修改已验收数据**。

## 2. 六臂实现

| 模块 | 内容 |
|---|---|
| `ssl/masking.py` | A2 四象限块（821）、C2 随机 821、seed 规则、像素展开 |
| `ssl/modules.py` | 2-block Adapter/Predictor（384, 6 heads, MLP 4）、B2 MLP、固定 2D 正弦位置 |
| `ssl/arms.py` | F0/A1/A2/B2/C2/R0 与 latent L2 / cosine 目标 |
| `ssl/checkpoint.py` | Recall/mAP/tau、checkpoint 选择、patience=3 早停、VICReg 状态机 |
| `ssl/data.py` | 像素层 mask → 冻结学生前向 → 可见 patch；教师缓存常量 |
| `scripts/train_ssl.py` | 默认 dry-run；`--allow-training` 也会被拒绝（Gate 4 未授权） |
| `scripts/eval_ssl.py` | 384-d CLS 检索评估（validation 默认） |

### 参数量

| Arm | Trainable | Total |
|---|---:|---:|
| F0 | 0 | 0 |
| A1 | 3,549,696 | 3,549,696 |
| A2 | 7,100,160 | 7,100,160 |
| B2 | 5,027,328 | 5,027,328 |
| C2 | 7,100,160 | 7,100,160 |
| R0 | 0 | 3,550,464（冻结） |

A2/C2 ≤ 10M 上限。

### ⚠ 设计歧义（需要 Owner 裁定，未静默修改方法）

设计文档写 “B2 predictor MLP 768→768→768 ≈1.18M，总 4.7M”，但 B2 的 q 输入是 **384 维 adapter CLS**，
必须有一个 384→768 投影。实现为 `384→768→768→768`，实际 **1.475M**，B2 总计 **5.03M**（仍 <10M）。
若要严格 1.18M，需要把 B2 的 adapter readout 改成 768 维（例如 CLS⊕patch-mean），这会改变已接受架构。
**本实现保留 384-d CLS 语义，并在报告中标记该偏差，等待 Owner 决定。**

## 3. Mask 确定性

| seed | block count | block sha256 | random count | random sha256 |
|---|---:|---|---:|---|
| 2026 | 821 | `9e146788…` | 821 | `402fb5fa…` |
| 2027 | 821 | `b16f6dde…` | 821 | `9a7f6af4…` |
| 2028 | 821 | `fcb4b590…` | 821 | `1b3509fb…` |

四块位于互斥象限，连通分量恰为 4，每块 ≥196 patch；抽样测试覆盖数量、确定性、不重叠。

## 4. A1 / F0 不变量（真实缓存，8 views）

- `torch.equal(A1_CLS, F0_CLS) = true`
- descriptor SHA-256 相同：`0ee21aed…`
- Recall@1：A1 = F0 = 1.0；Kendall τ = 1.0

## 5. 资源（batch 8，RTX 5070 12 GB，dry-run）

| Arm | Peak VRAM | Forward | Backward | Loss |
|---|---:|---:|---:|---:|
| F0 | 16 MB | — | — | — |
| A1 | 847 MB | 0.67 s | 0.09 s | 2.93 |
| A2 | 1,432 MB | 0.30 s | 0.09 s | 6.66 |
| B2 | 785 MB | 15.2 s* | 0.23 s | 1.02 |
| C2 | 1,432 MB | 0.43 s | 0.08 s | 6.75 |
| R0 | 288 MB | 14.8 s* | — | — |

\* 首次调用的 CUDA kernel 编译开销；后续调用亚秒级。峰值显存远低于 9 GB 停止线。

## 6. 测试

- 新增 `tests/test_ssl.py`（12）、`tests/test_ssl_invariants.py`（1）、`tests/test_occlusion_reference.py`（3）。
- 全量测试 113 passed（原始日志 `artifacts/evidence/DS-0002/test-runs/gate3-pytest.txt`，锚定干净提交）。

## 7. NOT RUN

```
optimizer_steps = 0
training_runs = 0
EXP-0002 = NOT RUN
```

未下载新权重/自然图像；未进入 Phase 2B/2C/3。

## 8. 复现

```powershell
python scripts/train_ssl.py --arm A2 --batch 8     # dry-run only
python scripts/eval_ssl.py --arm F0 --split validation --limit-scenes 5
python -m pytest -q
```
