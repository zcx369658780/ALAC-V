---
id: EXP-0002
title: EXP-0002 自主视觉学习预注册（草案）
status: 草案（未授权运行）
date: 2026-11-15
owner: lead
task: TASK-0003
issue: https://github.com/zcx369658780/ALAC-V/issues/3
preregistered: false
---

# EXP-0002 自主视觉学习预注册（草案）

> **状态：草案，未授权运行。** 本文件只定义"如果 Owner 批准 Phase 2A.1，将如何做实验"。
> 在 Owner 批准与独立复核前：不训练、不生成 DS-0002、不下载权重。
>
> 写法遵守 EXP-0001 的教训：假设、阈值、判定规则在任何运行之前写定；结果章节运行后只追加。

## 1. 研究问题

RQ：在**无人工标签**、**冻结 DINOv2 ViT-S/14**、**RTX 5070 12GB** 的约束下，三种无标签机制
（JEPA 式潜空间预测 / 跨视图自蒸馏 / 掩码潜表征预测）中，是否存在一种能在**合成域**上
稳定提升检索与不变性、且不牺牲判别性的表征学习方案？

RQ2：只训练 predictor（不改变表征）与训练 adapter（改变表征）在指标上是否可区分？

## 2. 实验臂

**统一检索 readout（P1-1）**：所有臂一律使用 **384 维 CLS**，L2 归一化后余弦检索；
patch 特征只用于训练目标与诊断，不参与检索。

| 臂 | 可训练部分 | 训练目标 | 检索表征 | 目的 |
|---|---|---|---|---|
| **F0** | 无 | — | 教师 CLS 384 | 冻结基线 |
| **A1** | predictor（3.5M） | A 的块预测 | 教师 CLS 384（与 F0 逐位相同） | predictor-only 不变量 |
| **A2** | adapter（3.5M）+ predictor（3.5M） | A 的块预测 | adapter CLS 384 | JEPA + 表征 |
| **B2** | adapter（3.5M）+ predictor（1.18M，768→768→768） | 跨视图 768 维余弦 | adapter CLS 384 | 跨视图蒸馏 |
| **C2** | adapter（3.5M）+ predictor（3.5M） | 随机掩码潜表征 | adapter CLS 384 | 掩码潜表征 |
| **R0** | 同 A2 结构、随机初始化、不训练 | — | 随机 adapter CLS 384 | 容量/随机投影对照 |

**遮挡预算**：A2 = 4 个连续块、合计 **821** 个 patch（59.97%）；C2 = 随机 **821** 个；B2 无掩码（单独记录跨视图预算）。
**约束**：可训练参数上限 10M；实际 A2/C2 = 7.0M、B2 = 4.7M，**不声称同预算**；
H3 的方法比较必须同时报告参数量与单位参数收益（ΔRecall@1 per 1M params）。
训练步数、batch、数据、增强、评估协议对四条训练臂完全相同。

## 3. 数据

| 项 | 规定 |
|---|---|
| 主数据 | DS-0002（**尚未生成**；2,000 / 300 / **900** base scenes 按 seed 分组切分） |
| 冒烟 | DS-0001 的 24 场景仅用于管线冒烟，不进入结论 |
| 训练可见信息 | 图像像素 + 增强参数 + mask 位置（由随机种子决定） |
| 训练不可见 | scene id、bbox、因子标签、split 之外的信息 |
| 评估标签 | 仅在第 7 节评估阶段使用 |
| 泄漏检查 | 训练 DataLoader 的字段白名单 + 单元测试；manifest 按 base scene 分组 |

## 4. 训练协议（预注册）

| 项 | 值 |
|---|---|
| 输入 | 518×518，37×37 patch |
| 教师 | 冻结 DINOv2 ViT-S/14，FP16；**完整图像**的 patch 特征预计算并缓存（仅作目标） |
| Adapter | 2×Transformer block（384 维，6 头，MLP 4×） |
| Predictor | A2/C2：2×block（384 维）；B2：MLP 768→768→768 |
| 优化器 | AdamW，lr 1e-3（adapter）/ 1e-3（predictor），wd 0.05 |
| batch | 32（显存不足时 16 + 梯度累积 2） |
| 精度 | AMP（bf16），教师前向 FP16 |
| 掩码实现 | **先掩码像素、后编码**：student 对掩码图像（掩码块像素=数据集均值）做一次冻结前向；教师目标来自完整图像缓存，杜绝上下文泄漏 |
| 掩码数量 | **821**（37×37×0.6=821.4 → 取整 821，59.97%）；A2 四块合计 821，C2 随机 821；`mask_seed = base_seed*1000 + step` |
| 主结果 checkpoint | **validation Recall@1 最优**；并列先比 validation prediction loss（低者优），再比 epoch（早者优）；最终 epoch 作为敏感性分析另行报告；不得按 test 选择 |
| VICReg 条件规则 | 有效秩 < F0 的 60% 连续 2 epoch → 下一 epoch 起加方差项 λ=25；记录启用 epoch，产物标记为 `B2+VR` 并单独报告 |
| 步数 | 20 epoch × 500 step（16,000 视图 / batch 32）= 10,000 step |
| 随机种子 | **3 个**：2026, 2027, 2028 |
| checkpoint | 每 2 epoch 保存；另存最终 epoch；**H1 只用 validation Recall@1 最优者**（破平：val prediction loss → 更早 epoch）。early stopping 独立按 validation prediction loss（patience 3）判定 |
| 早停 | val loss 连续 3 次不降则停 |

## 5. 评价指标

1. **检索**：Recall@1、Recall@5、mAP@10（test **900** 场景 × 4 查询 = 3,600 查询）；
2. **判别性**：同源/异源余弦 AUC；
3. **不变性**：同场景跨视图余弦均值；
4. **线性探针**：在**冻结的检索表征**（384 维 CLS）上训练逻辑回归，预测生成因子标签
   （纹理类型 / 调色板族 / 目标位置八分区），报告 top-1。标签使用边界：
   在 **train split** 上拟合 probe 与调参（仅评估阶段执行），**validation** 选正则强度，
   **test** 只做一次最终评估；test 标签不得用于拟合或调参；
5. **坍塌诊断**：有效秩、参与比、最小维方差；
6. **资源**：可训练参数、峰值显存、训练时长、推理延迟、特征体积；
7. **预测质量**（A1/A2/C2）：held-out 场景/增强上的 latent prediction error（掩码位置 L2）。

## 6. 假设与判定规则（先定后测）

| 编号 | 假设 | 判定 |
|---|---|---|
| **H1-D**（primary，有功效） | A2 方向为正 | 95% CI 下界 > 0；900 场景、真实 3 pp 时功效 ≈ **0.92** |
| **H1-P**（practical，筛选） | A2 观察效应 ≥ 3 pp | 点估计 Δ ≥ 3 pp；真实效应恰为 3 pp 时 P ≈ **0.50**，且**不随场景数增加**（见 §8.1） |
| **H1-S**（strong，探索） | 证明真实效应 ≥ 3 pp | 95% CI 下界 > 3 pp；需真实效应约 **6 pp**（900 场景时功效 ≈ 0.92） |
| **H2（不变量）** | A1 与 F0 的检索结果**完全相同** | 描述子 SHA-256 相同、Kendall τ=1、ΔRecall=0；任一不满足即判评估管线缺陷 |
| **H3（次要）** | B2、C2 各自相对 F0 的效应 | 作为 secondary family，用 **Holm 校正**（m=2）控制 FWER；未校正显著的只写"探索性观察" |
| **H3b（探索）** | A2 vs B2、A2 vs C2 的差异 | 预注册两个对比（不选"最优 vs 次优"），Holm 校正；CI 不含 0 才报告差异 |
| **H4** | adapter 臂提升跨视图不变性且不牺牲判别性 | 跨视图余弦 +≥0.02 且 AUC 下降 ≤2 pp → 支持 |
| **H5** | 收益可泛化到未见 seed | test 与 val 的 Recall@1 差距 ≤2 pp 且 test 增益方向一致 → 支持 |

**多重比较控制（P1-3）**：A2 是唯一 primary；B2/C2 与 H3b 的两个对比构成 secondary family，
统一用 Holm–Bonferroni 校正（m=2 与 m=2 分别校正），报告校正前后 CI；
**不做事后选择"最优臂"再检验**。

## 7. 失败判据（必须预先接受）

满足任一条件即判定**本阶段未成功**，必须在报告首段写明：

1. **A2 未同时满足 H1-D 与 H1-P**（方向不显著，或观察效应 <3 pp）；B2/C2 的次要结果不改变主判定；
   但必须同时声明：未达到 3 pp **不等于**证明 A2 无效应（见 §8.1 的功效边界）；
2. 任一报告臂的 val→test 落差 >5 pp（合成器过拟合）；
3. 任一臂有效秩 < F0 的 50%（坍塌）；
4. 评估集/标签泄漏被证实；
5. 训练无法在 12GB 内完成且降 batch 后仍 OOM。

**不允许**的替代叙事：
- 用训练 loss 下降代替 H1；
- 用"特征更紧凑"代替任务指标；
- 在 H1 失败后改阈值或换测试集。

## 8. 统计方案

### 8.1 配对场景级 Power / MDE 敏感性分析（不训练，仅模拟）

**方法**：以 test scene 为 cluster，每场景 4 个查询；场景难度 \(d\sim N(0,\sigma^2)\)，
查询命中概率 \(p=\mathrm{sigmoid}(\mathrm{logit}(0.80)+d+\delta)\)。\(\delta\) 用 400,000 次
场景抽样**校准**，使混合后的总体平均提升恰好等于目标 Δpp（不是只在 \(d=0\) 处校准）。
配对差按场景求均值，用 cluster-robust SE（\(sd/\sqrt{n}\)）构造 95% CI；每格 4,000 次模拟；
\sigma ∈ {0.5, 1.0, 1.5}。

**三个事件必须分开报告**（P1）：

- \(E_{dir}\)：95% CI 下界 > 0（方向性）
- \(E_{thr}\)：点估计 ≥ 3 pp（实践阈值）
- \(E_{joint}=E_{dir}\land E_{thr}\)：H1 实际采用的联合判定

| 场景数 | 真实效应 | P(E_dir) | P(E_thr) | **P(E_joint)** |
|---|---|---|---|---|
| 300 | 3 pp | 0.47–0.50 | 0.51 | **0.47–0.49** |
| 300 | 4 pp | 0.73–0.75 | 0.75–0.76 | 0.72–0.74 |
| 900 | 3 pp | 0.91–0.93 | 0.49–0.50 | **0.49–0.50** |
| 900 | 3.5 pp | 0.97–0.98 | 0.71–0.72 | 0.71–0.72 |
| 900 | 4 pp | 0.99–1.00 | 0.87–0.88 | 0.87–0.88 |
| 900 | 5 pp | 1.00 | 0.99 | 0.99 |

**关键结论**：

1. 当真实效应**恰好等于阈值**（3 pp）时，\(P(E_{thr})\approx 0.5\) **与样本量无关**——
   因为"点估计 ≥ 阈值"在真值等于阈值时本质是抛硬币。因此 900 场景**不能**把 H1 联合判定
   的功效提高到 90%；它提高的是**方向性检验**的功效（3 pp 时约 0.92）。
2. 900 场景下，联合判定达到 80% 需要真实效应约 **3.8–4.0 pp**；达到 95% 约需 4.5 pp。
3. 若要求强结论 \(CI_{lower}>3\,\mathrm{pp}\)，需要真实效应约 **6 pp**（900 场景下约 0.92）。
4. 因此预注册把 H1 拆为三个端点（见 §6），**Phase 2A 定性为可行性研究**：
   联合判定通过只说明"观察到 ≥3 pp 且方向为正"，不构成"已证明真实效应 ≥3 pp"。

### 8.2 统计方案

- 3 seeds：**逐 seed 报告**全部主指标，并给 mean ± std；3 seeds 是小规模可行性实验，**不声称跨随机初始化的稳健性**；
- **层级 bootstrap**：先对 3 个 seed 做配对（同一 seed 内 A2 与 F0 共享场景），
  再在 seed 内对 test scene 重采样（n=900），两层各 10,000 次，取 95% 百分位 CI；
  不把 3×300 当作 900 个独立样本；
- 次要比较用 Holm 校正后的 CI；
- 效应量：ΔRecall@1（pp）、ΔAUC、Δ余弦；
- 不做事后分层挑选；预注册的子组只有"未见 seed"。

## 9. 数据效率（次要）

**选臂与超参数只允许使用 validation**，流程固定为：

1. 在主实验完成后，按 **validation Recall@1** 选出候选臂与超参数；并列时依次比较
   (a) validation mAP@10、(b) validation prediction loss、(c) 训练 epoch 更少；
2. 冻结该臂、超参数与**三个种子的集合 {2026, 2027, 2028}**（不是单个 seed）；
3. 在 25% / 50% / 100% 训练场景上各训练 3 seeds（每档 seed 与主实验相同）；
4. **等全部 9 个数据效率 run 完成后**，再统一进行一次 test 评估；
5. 数据效率曲线属于**次要描述性分析**，不产生新的确认性结论，也不改变 H1 判定。

禁止：用 test 指标选择"最佳臂"、用 test 结果调整训练轮数、在报告中对不同臂使用不同 test 子集。
若 validation 与 test 结论不一致，如实报告两者，不重新选臂。
F0 作为不随数据变化的水平线。

## 10. 资源与停止条件

- 主实验：4 个训练臂（A1/A2/B2/C2）× 3 seeds = 12 runs；F0 与 R0 不需要训练；
  数据效率 +9 runs；合计 ≤ 21 个训练 run；
- 预估 GPU 时间：**1.0–1.7 h/run**（20 epoch × 3–5 min/epoch，含验证），主实验 12 runs =
  **12–20 h**；数据效率 9 runs = **9–15 h**；合计 **21–35 h**（粗略估计，实测后修正）；
- 若实际单 run 超过 2 h 或显存峰值 > 9 GB，暂停并重新设计，不得硬扛。

## 11. 产物与证据

每个 run：`artifacts/runs/<UTC>_EXP-0002_<arm>_s<seed>/`
- `config.snapshot.yaml` + `config.sha256`；
- `metrics.json`（训练损失、val 指标、坍塌指标、资源）；
- `items.jsonl`（逐场景检索结果）；
- `checkpoint.sha256`（权重文件不入库，只存指纹）；
- `env.json`、`model.json`（教师指纹）。
证据包：`artifacts/evidence/EXP-0002/<run-id>/`（脱敏，不含图像/权重）。

## 12. 修订政策

任何假设/阈值/臂定义的变更必须：
1. 在运行前写入本文件的新版本（`status: amended`）；
2. 说明变更理由与影响；
3. 若已运行，另立 EXP-0003，不得覆盖 EXP-0002 的预注册条款。

## 13. 与 Phase 1 的关系

- 复用 Phase 1 的归档、特征存储、检索与评估实现；
- 不修改 Phase 1 代码与 A/B/C 结论；
- 教师权重与 revision 沿用 Phase 1 记录（Apache-2.0）。

## 14. NOT RUN（当前）

- DS-0002 生成：未执行；
- 任何训练：未执行；
- 任何新权重下载：未执行；
- 自然图像评估：未执行（需 D-1 授权）。

## 附录 A：功效模拟代码（无训练）

```python
import numpy as np

def logit(p): return np.log(p / (1 - p))

def calibrate_delta(target_pp, sigma, base=0.80, n=400_000, seed=7):
    """让混合后的平均提升恰好等于 target_pp（不是只在 d=0 校准）。"""
    rng = np.random.default_rng(seed); d = rng.normal(0, sigma, n)
    p0 = 1 / (1 + np.exp(-(logit(base) + d))); lo, hi = 0.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p1 = 1 / (1 + np.exp(-(logit(base) + d + mid)))
        if (p1 - p0).mean() < target_pp: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def joint_power(n, true_pp, sigma, thr_pp=0.03, reps=4000, base=0.80, q=4, seed=11):
    """返回 P(CI 下界>0), P(点估计>=thr), P(两者同时)。"""
    delta = calibrate_delta(true_pp, sigma); rng = np.random.default_rng(seed)
    pos = thr = both = 0
    for _ in range(reps):
        d = rng.normal(0, sigma, n)
        p0 = 1 / (1 + np.exp(-(logit(base) + d)))
        p1 = 1 / (1 + np.exp(-(logit(base) + d + delta)))
        h0 = (rng.random((n, q)) < p0[:, None]).mean(1)
        h1 = (rng.random((n, q)) < p1[:, None]).mean(1)
        diff = h1 - h0; m = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n)
        a = (m - 1.96 * se) > 0; b = m >= thr_pp
        pos += a; thr += b; both += (a and b)
    return pos / reps, thr / reps, both / reps
```

复现示例：`joint_power(900, 0.03, 1.0)` 应给出约 `(0.92, 0.50, 0.50)`。
