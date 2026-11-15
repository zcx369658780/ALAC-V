---
id: DESIGN-0002
title: Phase 2A 自主视觉学习架构设计
status: 草案（待审）
date: 2026-11-15
owner: lead
task: TASK-0003
issue: https://github.com/zcx369658780/ALAC-V/issues/3
---

# Phase 2A 自主视觉学习架构设计

> 标注约定同文献调研：**[PAPER] / [REPO] / [INFER] / [HYP]**。
> **本文件是设计，不是训练授权。** 任何模型训练需 Phase 2A.1 独立任务单。

## 1. 目标与非目标

### 目标

1. 在**冻结 DINOv2 ViT-S/14** 之上，定义三种可公平比较的无标签学习机制：
   A. JEPA 式潜空间预测（context→target）；
   B. 跨视图自蒸馏；
   C. 掩码潜表征预测（masked latent prediction）。
2. 明确区分 **predictor learning** 与 **representation learning**，并把"只训 predictor"作为对照臂。
3. 给出可在 **RTX 5070 12GB / 32GB RAM / 单卡** 上执行的数据、训练、评估与资源方案。
4. 产出一份可证伪的预注册草案（EXP-0002），供 Phase 2A.1 执行。

### 非目标（本阶段明确不做）

- 不训练任何模型、不微调 DINOv2 主干；
- 不下载新权重或自然图像数据集；
- 不实现 Phase 2B 长期记忆 / Phase 2C 动态计算；
- 不处理视频、光流、Event Camera、SNN；
- 不追求 SOTA，不宣称"自主视觉学习成功"，除非预注册判据全部满足。

## 2. 已确认的基线（来自 Phase 1，实测）

| 项 | 值 | 来源 |
|---|---|---|
| 主干 | DINOv2 ViT-S/14，22.06M 参数，384 维 | `data/manifests/model-dinov2_vits14.json` |
| 输入 | 518×518，patch 14 → 37×37 = 1369 token | 同上 |
| 前向 | 5.23 ms/张（FP16，batch=1） | `artifacts/runs/*_AUDIT/gpu_selfcheck.json` |
| 权重 | 42.1 MB（FP16） | 同上 |
| 单图激活 | 61.1 MB | 同上 |
| Patch 特征 | 1369×384×2 B = **1.05 MB/视图** | 本设计计算 |
| 现有数据 | DS-0001：24 base scenes + 72 queries + 2 真实图 | `data/manifests/ds-0001-*.json` |
| 检索基线 | A/B/C Recall@1 0.833 / 0.875 / 0.889 | EXP-0001 |

**[INFER]** 关键含义：**前向不是瓶颈，数据规模和评估设计才是**。16,000 个视图的教师特征
只需约 83 秒编码（5.2 ms × 16k），但缓存占 16.8 GB 磁盘。

## 3. 设计原则

1. **教师冻结**：唯一权重 `facebook/dinov2-small`（Apache-2.0）。所有目标由教师特征产生，
   学生只学"如何组织这些特征"，避免 12GB 上的联合训练与坍塌风险。
2. **参数预算封顶**：可训练参数 ≤ 10M（≈ 主干的 45%）。三臂实际为 7.0M / 4.7M / 7.0M，**不声称同预算**；方法比较必须同时报告参数量与单位参数收益。
3. **公平臂**：同一数据、同一训练步数、同一评估集；**A2/C2 的遮挡数量统一为 821 个 patch（37×37×60% = 821.4 → 取 821，占 59.97%）**（结构不同：连续块 vs 随机散点），B2 不使用掩码但单独记录其跨视图计算预算。
4. **无标签训练**：训练期只读图像与增强参数；合成器的 scene id / bbox / 因子标签**仅在评估**使用。
5. **fail-closed**：缺教师特征、缺 mask 元数据、缺 checkpoint 时直接报错，不静默降级。
6. **可证伪**：每条假设都有阈值与失败判据（EXP-0002 §6）。

## 4. 三种候选机制

记教师为 \(f_\theta\)（冻结，DINOv2 ViT-S/14），adapter 为 \(g_\phi\)，predictor 为 \(h_\psi\)。
对图像 \(x\)，教师输出 CLS \(c\in\mathbb{R}^{384}\) 与 patch 特征 \(Z\in\mathbb{R}^{1369\times384}\)。

### 4.0 全臂统一的检索 readout（P1-1）

**所有实验臂（F0/A1/A2/B2/C2/R0）的检索特征一律为 384 维 CLS**，L2 归一化后取余弦：

\[
R(x)=\frac{\mathrm{CLS}(x)}{\lVert \mathrm{CLS}(x)\rVert}\in\mathbb{R}^{384}
\]

- F0 / A1：\(\mathrm{CLS}(x)=c\)（教师 CLS，二者必须逐位相同）；
- A2 / B2 / C2：\(\mathrm{CLS}(x)=g_\phi(\cdot)\) 输出的 CLS 位置；
- R0：随机初始化 adapter 的 CLS 位置。
- 实现约定：adapter 的输入序列为 `[教师 CLS token; patch tokens]`，输出同序，因此第 0 个 token 即检索用 CLS。
- Patch 特征只用于**训练目标与诊断**，不参与检索，避免不同臂使用不同 readout 造成不可比。

### 4.1 方案 A — JEPA 式块预测（同一图像、同一坐标系）

**主实验采用同一张图像、同一坐标系**：先做图像级增强得到 \(x\)，在 \(x\) 的 37×37 网格上
采样 4 个连续目标块 \(\{B_k\}\)，**四块合计恰好 821 个 patch**（59.97%），其余 548 个为 context。
**可复现的块采样算法**（所有机器一致）：

1. 固定随机种子 `mask_seed = base_seed * 1000 + step`（base_seed ∈ {2026,2027,2028}），
   同一次训练步在 CPU/GPU、不同机器上产生相同掩码；
2. 块尺寸：`w_k ≈ h_k ≈ round(sqrt(821/4)) = 14`，按 `w×h` 生成 4 个 14×14 块（共 784），
   剩余 37 个 patch 从块边缘逐块扩展（固定顺序：块 0→3，先右后下），直到恰好 821；
3. 块左上角在 `[0, 37-w_k] × [0, 37-h_k]` 内均匀采样，块间 IoU > 0.1 时重采样该块（最多 50 次）；
4. 掩码集合 `M` 写出到该 step 的日志（`mask_sha256`），用于复现与审计。
**不跨视图**，因此不存在"不同 crop 后 patch 索引不对应"的问题（P1-2）。

**关键实现约束（防止目标泄漏）**：掩码必须施加在**像素/输入层**，先于 DINOv2 的上下文编码。
教师目标来自**未掩码**图像的缓存特征；学生输入是**掩码后的图像**，重新做一次冻结前向。

\[
\mathcal{L}_A=\frac{1}{|M|}\sum_{i\in M}\left\|h_\psi\big(g_\phi(f_\theta(\tilde{x}))\big)_i-\mathrm{sg}\big(Z_i\big)\right\|_2^2
\]

- \(\tilde{x}\)：目标块像素被替换为数据集均值（归一化后为 0）的图像；
- \(M\)：被遮挡的 patch 索引集合；\(Z_i\)：**完整图像**教师特征中对应位置（stop-gradient）；
- 位置编码：patch 的 2D 位置嵌入在 adapter 与 predictor 中均保留，确保学生知道"预测哪里"。
- **adapter 输入**：`[教师 CLS token（来自掩码图像）; 可见位置 patch 特征 ; 每个被遮挡位置一个共享 mask token]`，
  三者都加 2D 位置编码；adapter 输出同序列，第 0 个 token 即检索用 CLS（P1-2）。
- **predictor 输入/输出**：直接对 adapter 输出序列做 2 层 self-attention，**取 mask token 位置的输出**
  作为该 patch 的 384 维预测；不使用"块查询"，因此不存在"一个块查询如何展开整块"的歧义。
  损失只在 \(M\) 上计算。A2 与 C2 共享同一 adapter/predictor 结构，**仅掩码结构不同**（连续块 vs 随机散点）。
- **训练/评估的掩码差异**：训练时 adapter 看到掩码图像；**评估时不做任何掩码**（与 MAE 的
  fine-tune 协议一致 [PAPER]），检索特征由完整图像计算。该差异在报告中明示。

**两种设定（必须分开报告）**：

| 设定 | 冻结 | 可训练 | 检索表征 |
|---|---|---|---|
| **A1 predictor-only** | \(f_\theta\) 且 adapter=\(\mathrm{Id}\) | 仅 \(h_\psi\) | \(c\)（与 F0 **逐位相同**） |
| **A2 adapter+predictor** | \(f_\theta\) | \(g_\phi\)（2 block，3.5M）+ \(h_\psi\)（2 block，3.5M） | \(g_\phi\) 的 CLS |

**A1 的意义**：预测器学习 ≠ 表征学习。A1 的检索特征按定义与 F0 相同，因此 Recall/排序必须
完全一致；这是实现不变量（见 §5），不是"预期结果"。

### 4.2 方案 B — 跨视图自蒸馏（仅全局目标，不涉及 patch 对齐）

**目标**：同一场景两个增强视图 \(x_1,x_2\)；学生从 \(x_1\) 的 CLS 预测教师对 \(x_2\) 的全局表征。

\[
\mathcal{L}_B=-\frac{\big\langle q,\ \mathrm{sg}(t)\big\rangle}{\lVert q\rVert\,\lVert t\rVert},
\qquad
q=h_\psi\big(\mathrm{CLS}[g_\phi(Z_1)]\big)\in\mathbb{R}^{768},
\qquad
t=\big[\,c_2\ ;\ \bar{Z}_2\,\big]\in\mathbb{R}^{768}
\]

- \(\mathrm{CLS}[g_\phi(Z_1)]\)：adapter 在视图 1 全部 patch token 上输出后取 CLS 位置（384 维）；
- \(c_2\)：教师对 \(x_2\) 的 CLS（384 维）；\(\bar{Z}_2\)：教师 patch 特征的均值（384 维）；
- **target 768 维，predictor 输出必须为 768 维**（P1-1）：MLP `768→768→768`，约 1.18M 参数；
- 无负样本、无 EMA（教师冻结）；增强：随机裁剪/缩放、旋转 ±10°、亮度/对比度/饱和、JPEG；
- 检索仍用 adapter 的 **CLS 384 维**（§4.0），与 target 的 768 维解耦。

**坍塌控制（预注册的条件规则）**：若某次评估的有效秩低于 F0 的 60% 并连续 2 个 epoch 成立，
则在下一 epoch 启用 VICReg 方差项 \(\lambda=25\)（VICReg 原文默认量级 [PAPER]），
记录启用 epoch，产物单独标记为 **B2+VR**；不得把加了正则的 run 与纯 B2 合并报告。

### 4.3 方案 C — 随机掩码潜表征预测（同一图像）

**目标**：随机遮挡 **821 个 patch（59.97%）**，用可见部分预测被遮挡位置的教师表征（data2vec 式 [PAPER]）。

\[
\mathcal{L}_C=\frac{1}{|M|}\sum_{i\in M}\left\|h_\psi\big(g_\phi(f_\theta(\tilde{x}))\big)_i-\mathrm{sg}\big(Z_i\big)\right\|_2^2
\]

**与 MAE 的区别**：预测潜表征，不需要 decoder [PAPER]。**与 A2 的区别**：掩码是随机散点（821 个，与 A2 数量相同），不是连续块；其余管线相同。

**实现细节（P1-2 / Codex P1）**：

| 组件 | 定义 |
|---|---|
| 掩码位置 | 像素层：被选中 patch 的像素替换为数据集均值；**先掩码、后编码**，杜绝上下文泄漏 |
| 学生输入 | \(f_\theta(\tilde{x})\) 中**可见位置**的 patch 特征（不含掩码位置特征）；与共享 mask token 拼接后送入 adapter |
| mask token | 每个被遮挡位置一个**共享的可学习向量** \(m\in\mathbb{R}^{384}\) |
| 位置编码 | 可见 token 与 mask token 都加 2D 位置嵌入，学生知道绝对位置 |
| predictor | 对 adapter 输出 `[可见 token; mask token]` 做 2 层 self-attention，取 mask token 位置输出（384 维） |
| 教师 | **完整图像** \(x\) 的缓存 patch 特征；只取 \(M\) 位置做目标 |
| 损失位置 | 仅 \(M\) 上计算，\(|M|=0.6\times1369\approx821\) |

### 4.4 三条臂的对齐

| 项 | A2 | B2 | C2 |
|---|---|---|---|
| 可训练参数 | adapter 3.5M + predictor 3.5M = **7.0M** | adapter 3.5M + predictor 1.18M = **4.7M** | adapter 3.5M + predictor 3.5M = **7.0M** |
| 检索 readout | CLS 384 | CLS 384 | CLS 384 |
| 掩码 | 4 个连续块，合计 821 个（59.97%） | 无 | 随机 821 个（59.97%） |
| 学生前向 | 掩码图像（在线） | 完整视图 1（在线） | 掩码图像（在线） |
| 教师目标 | 完整图块的缓存 patch | 完整视图 2 的缓存 CLS⊕mean | 完整图的缓存 patch（仅 M） |
| 增强依赖 | 弱（图像级增强后同图） | 强（跨视图） | 弱 |

**[INFER]** 三臂参数并不相等（7.0M vs 4.7M），只满足"≤10M 上限"。H3 的方法排序必须同时报告
参数量与**单位参数收益**（ΔRecall@1 per 1M trainable params），不得把差异全部归因于目标函数。

## 5. Predictor-only 与 Adapter+Predictor 的形式化对比

设下游检索使用表征 \(R(x)\)。三种情形：

| 情形 | 训练后 \(R(x)\) | 预期 |
|---|---|---|
| F0 冻结 | \(f_\theta(x)\) | 基线 |
| A1 只训 predictor | \(f_\theta(x)\) **不变** | 检索指标与 F0 完全一致（工程上可验证） |
| A2 adapter | \(g_\phi(f_\theta(x))\) | 可能提升或退化 |
| Backbone 微调（不授权） | \(f_{\theta'}(x)\) | 可能最强，但 12GB 不可行且破坏 Phase 1 可复现性 |

**[INFER]** A1 与 F0 的检索指标应在数值上**完全相同**（同一特征）。若出现差异，说明评估管线
有随机性/泄漏，必须先排查，而不是当作"predictor 有效"。

## 6. 数据方案

### 6.1 DS-0001 是否足够？

**结论：不够支持学习结论，只够做管线验证。**

- 24 个 base scene，按 base-scene 分组切分（如 16/4/4）后，测试集只有 4 个场景，
  Recall@1 的 95% CI 宽度约 ±20 pp，无法检出 3–5 pp 的效应；
- 24 个场景的因子组合有限，任何提升都可能是对生成器的过拟合。

因此：**DS-0001 仅用于 EXP-0002 的冒烟测试（pipeline smoke），不进入正式结论**。

### 6.2 DS-0002（设计稿，尚未生成）

程序化、无标签、可分组切分的合成数据：

| 参数 | 设计值 | 理由 |
|---|---|---|
| base scenes（train/val/test） | 2,000 / 300 / **900** | 900 个测试场景：3 pp 效应在 80% 功效下可检出（见 EXP-0002 §8.1 功效分析） |
| 每场景视图数 | train 8；val/test 5（1 gallery + 4 queries） | 跨视图目标需要 ≥2；评测需要多查询 |
| 分辨率 | 1024×768 | 与 Phase 1 一致 |
| 目标/干扰 | 1–3 个纹理目标 + 2–4 个干扰块 | 与 DS-0001 同分布族 |
| 增强 | 裁剪/缩放/旋转/亮度/对比度/饱和/JPEG | 训练只用图像与增强参数 |
| 隐藏标签 | scene id、bbox、因子（颜色/纹理/位置） | **仅评估**使用 |
| 切分 | 按 base scene seed 分组 | 杜绝同场景不同视图跨 split |

**泄漏规则**（硬约束）：

1. 训练脚本不得读取 manifest 中的 `scene_id`、`gt_bbox`、`factor` 字段；
2. 增强参数只来自训练侧随机数，不来自场景标签；
3. 评估脚本只能在全流程冻结后运行；
4. 数据 manifest 记录每个文件的 SHA-256 与 split，可审计。

**数据量**：2,000 × 8 = 16,000 训练视图 + 300×5 val + 900×5 test；图像约 **2.8 GB**；
教师特征缓存 16.8 GB（train）+ 4.7 GB（test，可在线计算，不必常驻）。

### 6.3 自然图像

不在本任务授权内。若 Owner 批准 D-1（小规模公开自然图像），需另立数据许可决策与 manifest；
在此之前，所有结论限定"合成域"。

## 7. 评估协议

评估集：DS-0002 test 的 **900** 个 base scene，每场景 4 个查询视图 + 1 个 gallery 视图（共 3,600 查询）。

| 指标 | 定义 | 用途 |
|---|---|---|
| Recall@1 / @5、mAP@10 | 检索同场景 | 主任务表现 |
| 同源/异源 AUC | 同场景 vs 异场景余弦分布 | 判别性 |
| 跨视图一致性 | 同场景不同增强的余弦均值 | 不变性 |
| 线性探针 | 冻结特征 + 逻辑回归（标签仅评估用） | 语义可分性 |
| 有效秩 / 参与比 | 特征矩阵奇异值 | 坍塌诊断 |
| 可训练参数 / 峰值显存 / 训练时长 / 推理延迟 | 运行时测量 | 资源成本 |

**禁止**：训练期使用任何上述标签；"训练 loss 下降"不作为成功证据。

## 8. 资源预算（静态估算）

以 518×518、1369 token、FP16 教师、缓存特征训练为基准：

| 项 | 估算 | 依据 |
|---|---|---|
| 教师前向（缓存，只算一次） | 5.2 ms/图；16k 图 ≈ 83 s | Phase 1 实测 [REAL] |
| 学生前向（每步在线，掩码图像） | 32 图 × 5.2 ms ≈ 0.17 s/step；10,000 步 ≈ 28 min/run | 实测 + 估算 |
| 教师特征缓存 | 1.05 MB/视图 → 16.8 GB | 计算 |
| Adapter+predictor 参数 | 3.5M + 3.5M ≈ 7M | 12d² × blocks |
| 训练显存（batch 32，缓存特征） | 约 2–4 GB | 激活 32×1369×384×4B ≈ 67 MB/张量 × 层数 |
| 单步时间（估） | 30–60 ms | 7M 参数 × 1369 token × batch 32 |
| 每 epoch（16k 视图 / batch 32 = 500 步） | 约 3–5 min（含学生在线前向） | 估算 |
| 每 run（20 epoch × 3–5 min/epoch） | **1.0–1.7 h**（含验证） | 估算 |
| 主实验 12 runs（4 臂 × 3 seed） | **12–20 h** | 1.0–1.7 h × 12 |
| 数据效率 9 runs（3 档 × 3 seed） | **9–15 h** | 同上 |
| **总计（不含数据生成/评测）** | **21–35 h** | 算术一致；这是粗略估计 |
| 数据生成 | 16k 图，约 10–20 min | 估算 |
| 推理延迟 | 5.2 ms + adapter（< 1 ms） | 实测 + 估算 |

**[INFER]** 12GB 显存的主要约束不是训练本身，而是**同时驻留的教师特征与数据加载**；
设计上采用 memmap + 小 batch，显存余量充足（桌面占用 2.6–2.8 GB，可用约 9 GB）。

## 9. 失败模式与诊断

| 失败模式 | 症状 | 诊断 | 处置 |
|---|---|---|---|
| 坍塌 | 有效秩骤降、样本对余弦→1 | 每 epoch 记录有效秩 | 启用方差正则；仍失败则判该臂失败 |
| Adapter 无学习 | 训练 loss 平、评估≈F0 | 对比 A1/F0 与 A2 | 报告"无表征收益" |
| 过拟合合成器 | val 升 test 降 | 按未见 seed 评估 | 结论限定合成域 |
| 评估泄漏 | 指标异常高 | 审计训练数据读取路径 | 作废该 run |
| 教师特征缓存错位 | 目标与位置不匹配 | 抽样可视化/相关性检查 | fail-closed 重建缓存 |
| 显存溢出 | OOM | 降 batch / 梯度累积 | 记录并重跑 |

## 10. Phase 2A.1 实施计划（供后续任务单）

1. `src/alacv/ssl/`：adapter、predictor、mask 采样、三种目标、坍塌监控；
2. `scripts/encode_teacher_cache.py`：教师特征缓存（SafeTensors + manifest）；
3. `scripts/train_ssl.py`：配置驱动、AMP、梯度累积、checkpoint、run evidence；
4. `scripts/eval_ssl.py`：检索/不变性/探针/资源指标；
5. `configs/exp0002.yaml`；`tests/` 覆盖 mask、目标、坍塌指标、泄漏检查；
5b. **主结果 checkpoint 规则（预注册）**：H1 使用**验证集 Recall@1 最优**的 checkpoint；
   并列时依次比较 (a) validation prediction loss 更低、(b) epoch 更早；
   最终 epoch 作为**敏感性分析**单独报告，不用于 H1 判定；不得依据 test 结果选 checkpoint；
6. 证据：`artifacts/evidence/EXP-0002/<run-id>/`（config、metrics、checkpoint 指纹，不含大数据）。

## 11. 风险与未决问题

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 合成域结论不可外推 | 高 | 预注册声明边界；自然图像另立任务 |
| R2 | 冻结教师上界 | 中 | 明确"adapter 只能重排/筛选教师信息" |
| R3 | 评估标签泄漏 | 高 | 硬编码读取白名单 + 测试 |
| R4 | I-JEPA/MAE 代码为 NC 许可 | 中 | 独立实现，不复制代码；只记录方法 |
| R5 | 12GB 显存与桌面占用 | 中 | 缓存特征、batch 8–32、梯度累积 |
| R6 | 3 seed 统计功效不足 | 中 | 300 测试场景 + 配对 bootstrap |

## 12. Owner 决策点（见决策摘要）

- D-2A-1：是否批准本设计进入 Phase 2A.1 实施；
- D-2A-2：是否授权生成 DS-0002（纯程序化，约 2.4 GB）；
- D-2A-3：是否保持"只用合成域"作为 Phase 2A 的结论边界。
