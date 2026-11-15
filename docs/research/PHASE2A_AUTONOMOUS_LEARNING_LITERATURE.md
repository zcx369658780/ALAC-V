---
id: RES-0001
title: Phase 2A 自主视觉学习文献调研
status: 草案（待审）
date: 2026-11-15
owner: lead
task: TASK-0003
issue: https://github.com/zcx369658780/ALAC-V/issues/3
---

# Phase 2A 自主视觉学习文献调研

> **证据标注约定**（全文强制）：
> - **[PAPER]** 论文原文可查事实（含 arXiv ID；ID 已用 arXiv API 核验，见 §9）。
> - **[REPO]** 官方仓库/权重页可查事实（2026-11-15 通过 GitHub / HuggingFace API 核验，见许可矩阵）。
> - **[INFER]** 本文档作者基于事实的推断，不是论文结论。
> - **[HYP]** 尚待实验验证的假设，写入 EXP-0002 预注册草案。
>
> 本调研只覆盖静态图像与"冻结主干 + 轻量可训练模块"的可行情景；视频、Event Camera、SNN、
> 世界模型规划不在本任务范围。

## 1. 研究问题与边界

Phase 1 已交付**冻结的 DINOv2 ViT-S/14** 表征、多尺度 / Foveated ROI 管线与检索回溯基线。
Phase 2A 要回答的核心问题是：

> 在**无人工标签、无外部 LLM、单卡 RTX 5070 12GB** 的约束下，能否通过"观察图像内部/跨视图的
> 预测关系"学到**比冻结 DINOv2 更有用或更稳健**的视觉表征？

必须先把三个容易混淆的概念分开（Issue #3 §2）：

| 概念 | 训练对象 | 表征是否改变 | 典型价值 |
|---|---|---|---|
| **Predictor learning** | 只有 predictor | 否 | 学"世界如何变化"的预测器，可服务 Phase 2C 的规划/主动感知 |
| **Representation learning (adapter/student)** | 轻量 adapter 或 student | 是（在冻结特征之上的重参数化） | 可能提升下游检索/不变性 |
| **Backbone fine-tuning** | 整个 ViT | 是 | 最强但也最贵，本阶段**不授权** |

**[INFER]** 许多"JEPA 有效"的结论来自**联合训练 encoder + predictor**（I-JEPA / V-JEPA），
其收益不能直接归因于 predictor 本身。在 12GB 单卡、不允许微调主干的约束下，我们必须把
"只训 predictor"作为**对照臂**，而不是把它当作方法本身。

## 2. 联合嵌入预测架构（JEPA 家族）

### 2.1 概念起点

LeCun 的位置论文 *A Path Towards Autonomous Machine Intelligence*（OpenReview, 2022）提出
JEPA 作为世界模型的核心：在**表征空间**预测缺失/未来的表征，而不是重建像素。[PAPER]

**[INFER]** 该文是研究纲领而非可复现实验；它给出的关键可检验命题是"潜空间预测优于像素重建"，
但这一命题在具体数据/预算下需要独立验证，不能当作既定事实。

### 2.2 I-JEPA（图像）

- 论文：Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive
  Architecture*, arXiv:2301.08243（2023-01），CVPR 2023。[PAPER]
- 结构：ViT context encoder + 轻量 predictor；目标块的表征由**目标编码器**产生，该编码器是
  context encoder 的 EMA（指数滑动平均）。[PAPER]
- 目标函数：在潜空间对多个目标块做 L2 回归；**无像素 decoder、无负样本、无人工标签**。[PAPER]
- 掩码：采样若干目标块（block-wise），context 由其余 patch 组成。[PAPER]
- 资源：论文主实验为 ViT-H/14 级模型 + ImageNet-1k 级数据，远超本机预算。[PAPER][INFER]

### 2.3 V-JEPA / V-JEPA 2

- V-JEPA：Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from
  Video*, arXiv:2404.08471（2024-02）。把 I-JEPA 扩展到时空管（tubelet）掩码预测。[PAPER]
- V-JEPA 2：arXiv:2506.09985（2025-06），多阶段视频预训练 + 机器人规划，模型规模达十亿级、
  数据达百万小时级。[PAPER]
- **[INFER]** 与本项目的关系：V-JEPA 的方法论（掩码潜空间预测 + 冻结评估）可借鉴；其规模不可复制，
  且视频属 Phase 3，**本阶段不实施**。

### 2.4 世界模型视角

- Garrido et al., *Learning and Leveraging World Models in Visual Representation Learning*,
  arXiv:2403.00504（2024-03）：系统比较"预测式世界模型目标"与"增强不变性目标"，指出两者
  捕获的信息不同，组合可能更好。[PAPER]
- **[INFER]** 这为我们的候选设计提供了直接依据：**预测目标（context→target）与不变性目标
  （view→view）不应被当成同一件事**，应作为不同候选臂比较。

## 3. 自蒸馏与跨视图学习

### 3.1 DINO

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, arXiv:2104.14294
  （2021-04），ICCV 2021。[PAPER]
- 结构：student + EMA teacher；多裁剪（multi-crop）跨视图；teacher 输出做 centering + sharpening；
  **无负样本、无标签**。[PAPER]
- **[REPO]** 官方仓库 `facebookresearch/dino` 为 Apache-2.0。

### 3.2 iBOT

- Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer*, arXiv:2111.07832（2021-11），
  ICLR 2022。[PAPER]
- 结构：DINO 式自蒸馏 + 掩码 patch 的 token 级预测（在线 tokenizer 提供目标）。[PAPER]
- **[REPO]** `bytedance/ibot` 为 Apache-2.0。

### 3.3 BYOL

- Grill et al., *Bootstrap your own latent*, arXiv:2006.07733（2020-06），NeurIPS 2020。[PAPER]
- 结构：online 网络 + 目标网络（EMA）+ predictor；**stop-gradient**、无负样本。[PAPER]
- **[INFER]** 无负样本方法避免坍塌依赖"不对称性"（predictor、stop-grad、EMA、BN 等），
  这些机制在"冻结 teacher"设定下需要重新审视：冻结 teacher 本身就是最强的反坍塌约束。

### 3.4 SimSiam

- Chen & He, *Exploring Simple Siamese Representation Learning*, arXiv:2011.10566（2020-11），
  CVPR 2021。[PAPER]
- 结构：共享 encoder + predictor + stop-gradient；**无 EMA、无负样本**。[PAPER]
- **[REPO]** 官方仓库为 **CC-BY-NC-4.0**（见许可矩阵），商用受限。

### 3.5 DINOv2（本项目基线）

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193
  （2023-04）。[PAPER]
- **[REPO]** 官方仓库与 `facebook/dinov2-small` 权重均为 **Apache-2.0**；本项目 Phase 1 已固定
  revision 与权重的 SHA-256。

## 4. 掩码建模与掩码潜空间预测

### 4.1 MAE

- He et al., *Masked Autoencoders Are Scalable Vision Learners*, arXiv:2111.06377（2021-11），
  CVPR 2022。[PAPER]
- 结构：随机遮挡高比例 patch（75%），非对称 encoder-decoder，**重建像素**。[PAPER]
- **[INFER]** 像素重建目标迫使模型编码低频/纹理细节，与"检索/语义"目标不完全一致；
  在合成图像上尤其容易"重建得很好但表征不好"。

### 4.2 data2vec

- Baevski et al., *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and
  Language*, arXiv:2202.03555（2022-02）。[PAPER]
- 结构：对掩码位置预测**教师网络的潜表征**（教师为学生的 EMA），无 decoder。[PAPER]
- **[INFER]** 这正是"masked latent prediction"最干净的参考实现思想，也最适合我们
  "冻结 DINOv2 当教师"的设定。

### 4.3 像素重建 vs 潜空间预测

| 维度 | 像素重建（MAE） | 潜空间预测（I-JEPA/data2vec） |
|---|---|---|
| 目标 | 原始像素 | 教师表征 |
| 是否需 decoder | 是 | 否（或极轻） |
| 对纹理/细节 | 强 | 取决于教师表征 |
| 坍塌风险 | 低（像素目标不退化） | 中（需 stop-grad/EMA/正则） |
| 与本项目兼容性 | 需改输入管线 | 直接复用 DINOv2 patch 特征 |

## 5. 表征坍塌：机制与诊断

无负样本/自蒸馏方法的核心风险是**表征坍塌**（所有输入映射到同一常数向量）。文献中的主要对策：

| 机制 | 出处 | 作用 |
|---|---|---|
| stop-gradient | BYOL / SimSiam [PAPER] | 阻断"捷径解" |
| EMA teacher | BYOL / DINO / I-JEPA [PAPER] | 提供稳定目标 |
| centering + sharpening | DINO [PAPER] | 防止 teacher 输出退化 |
| 方差/协方差正则 | VICReg（arXiv:2105.04906）、Barlow Twins（arXiv:2103.03230）[PAPER] | 显式维持各维方差与去相关 |
| 负样本 | SimCLR/MoCo 系 [PAPER] | 直接推开样本 |

**[INFER]** 本项目最有利的一点是：**教师是冻结的、预训练好的 DINOv2**，其目标表征本身
不会坍塌；学生只要不退化即可。这使"冻结教师 + 轻量学生"比"从零联合训练"风险低得多，
但也意味着**学生能学到的东西受教师表征上界约束**。

诊断指标（写入预注册）：
1. 有效秩（effective rank）与参与比（participation ratio）；
2. 各维方差的最小值/中位数；
3. 样本对平均余弦相似度（坍塌时趋近 1）；
4. 下游探针准确率（标签仅用于评估）。

## 6. 预测编码与在线自适应

- 经典 predictive coding 以"预测误差最小化"解释皮层表征；在深度学习中与自监督预测目标同源。[PAPER/INFER 边界模糊]
- 在线/持续自适应（test-time training、在线自监督）在分布漂移下有价值，但会引入"评估不可复现"
  与"标签泄漏"风险。[INFER]
- **本阶段决定**：只在设计中保留"在线适配"作为 Phase 2A.1 之后的候选，不进入本轮预注册主臂；
  理由：单机、无标签、需先建立可靠的离线评估闭环。

## 7. 方法对照表（WP1 要求字段）

> “训练资源”为论文量级描述；标 † 者为未逐字核对原文、按公开资料估计的 **[INFER]**，仅用于判断
> 与本机（单卡 12GB）的差距，不作为可复现承诺。

| 方法 | 年份/arXiv | 核心目标 | Teacher | Decoder | 负样本 | 典型规模 | 代表训练资源† | 官方代码许可 | 权重许可 |
|---|---|---|---|---|---|---|---|---|---|
| DINO | 2021 / 2104.14294 | 跨视图自蒸馏（centering+sharpening） | EMA | 否 | 否 | ViT-S/B | 8×V100 级，ImageNet 数天 | Apache-2.0 | 未逐 checkpoint 核验 |
| DINOv2 | 2023 / 2304.07193 | 自蒸馏 + 掩码 token + 大规模数据管线 | EMA | 否 | 否 | ViT-S/B/L/g | 多卡 A100，数十万步 | Apache-2.0 | Apache-2.0 |
| iBOT | 2021 / 2111.07832 | DINO + 掩码 token 预测（在线 tokenizer） | EMA | 否 | 否 | ViT-S/B | ImageNet 多卡 | Apache-2.0 | 未逐 checkpoint 核验 |
| BYOL | 2020 / 2006.07733 | 在线网络预测 EMA 目标 | EMA | 否 | 否（BN 不对称） | ResNet-50 | 8–64 卡级 | Apache-2.0（`deepmind-research/byol/`） | 未核验 |
| SimSiam | 2020 / 2011.10566 | 孪生 + stop-grad + predictor | 无（共享权重） | 否 | 否 | ResNet-50 | 8 卡级 | CC-BY-NC-4.0 | 未提供 |
| I-JEPA | 2023 / 2301.08243 | 潜空间块预测 | EMA | 否 | 否 | ViT-B/H | 16–32×A100 级 | CC-BY-NC-4.0 | CC-BY-NC-4.0 |
| V-JEPA（原始） | 2024 / 2404.08471 | 时空管潜空间预测 | EMA | 否 | 否 | ViT-L/H | 多卡 A100 级 | CC-BY-NC-4.0（`facebookresearch/jepa`） | 未核验 |
| V-JEPA 2 | 2025 / 2506.09985 | 视频预训练 + 规划 | EMA | 否 | 否 | ViT-L/g | 十亿级参数、百万小时级 | **MIT + 少数文件 Apache-2.0**（`facebookresearch/vjepa2`） | MIT（`facebook/vjepa2-vitl-fpc64-256`） |
| MAE | 2021 / 2111.06377 | 像素重建（75% 掩码） | 无 | 是 | 否 | ViT-B/L/H | 32×A100 级 | CC-BY-NC-4.0 | Apache-2.0（vit-mae-base） |
| data2vec | 2022 / 2202.03555 | 掩码位置预测教师潜表征 | EMA | 否 | 否 | ViT-B/L | 多卡级 | 未核验（fairseq） | 未核验 |
| VICReg | 2021 / 2105.04906 | 不变性 + 方差/协方差正则 | 无 | 否 | 否 | ResNet-50 | 8 卡级 | MIT | — |
| Barlow Twins | 2021 / 2103.03230 | 互相关矩阵去冗余 | 无 | 否 | 否 | ResNet-50 | 8 卡级 | MIT | — |

**与本项目（冻结 DINOv2、单卡 12GB）的兼容性判断 [INFER]**：

| 方法 | 兼容性 | 原因 |
|---|---|---|
| DINO / iBOT | 中 | 需 EMA 教师与联合训练；12GB 下只能做极小规模 |
| DINOv2 | **高（作为冻结教师）** | 项目已用；权重 Apache-2.0 |
| BYOL / SimSiam | 中 | 可在冻结教师上做学生端；无负样本，需监控坍塌 |
| I-JEPA | **高（方法）/ 低（代码）** | 目标可直接用教师 patch 特征；代码 NC 需独立实现 |
| MAE | 中 | 像素重建需 decoder 与原始图像；与“潜空间”方向不同 |
| data2vec / masked latent | **高** | 与冻结教师特征天然兼容 |
| VICReg / Barlow | 高（作为正则项） | 实现简单，MIT |

## 8. 与本项目约束的匹配分析

| 约束 | 含义 |
|---|---|
| RTX 5070 12GB，单卡 | 任何 ViT-L/H 联合训练不可行；只能"冻结主干 + ≤10M 可训练参数" |
| 32GB RAM | 不能把全量 patch 特征常驻内存；需 memmap / 按需读取 |
| 无标签 | 训练目标只能用图像自身；合成因子与 bbox 仅在评估时使用 |
| 无自然图像授权 | 现阶段只能用程序化合成数据，结论限定在合成域 |
| 不训练主干 | 表征变化只能来自 adapter/student 或 predictor 后的读出层 |

**[INFER]** 由此得到三条设计推论：

1. **必须把"表征是否改变"作为实验因素**，而不是论文里的隐含假设；
2. **冻结教师让"坍塌"从致命问题降级为可监控风险**；
3. **在合成域上，最可能被检验的是"不变性/预测性"而非"语义泛化"**，预注册必须承认这一点。

## 9. 文献不足以回答的问题

1. 冻结 DINOv2 之上加 1–10M 参数的 adapter，能否在**小数据合成域**上稳定提升检索？
2. 潜空间预测目标相对纯增强不变性目标，是否在**同一参数预算**下更优？文献多为大规模联合训练，
   缺少"冻结教师 + 轻量学生"的受控比较。
3. 预测器单独训练（predictor-only）能否作为 Phase 2C 的可用世界模型组件？文献几乎没有把
   "predictor 质量"与"表征质量"分开报告。

这三条正是 EXP-0002 预注册的出发点。

## 10. 参考文献（ID 已用 arXiv API 核验）

| # | 文献 | arXiv | 说明 |
|---|---|---|---|
| 1 | LeCun, A Path Towards Autonomous Machine Intelligence | OpenReview 2022（无 arXiv） | JEPA 纲领 |
| 2 | Assran et al., I-JEPA | 2301.08243 | 潜空间块预测 |
| 3 | Bardes et al., V-JEPA | 2404.08471 | 视频扩展 |
| 4 | V-JEPA 2 | 2506.09985 | 十亿级视频模型 |
| 5 | Caron et al., DINO | 2104.14294 | 自蒸馏 |
| 6 | Zhou et al., iBOT | 2111.07832 | 掩码 token 蒸馏 |
| 7 | Grill et al., BYOL | 2006.07733 | 无负样本 |
| 8 | Chen & He, SimSiam | 2011.10566 | 简单孪生 |
| 9 | He et al., MAE | 2111.06377 | 像素重建 |
| 10 | Baevski et al., data2vec | 2202.03555 | 掩码潜表征 |
| 11 | Bardes et al., VICReg | 2105.04906 | 方差正则 |
| 12 | Zbontar et al., Barlow Twins | 2103.03230 | 去相关 |
| 13 | Oquab et al., DINOv2 | 2304.07193 | 项目基线 |
| 14 | Garrido et al., World Models in Visual Representation Learning | 2403.00504 | 预测 vs 不变性 |

核验命令：`https://export.arxiv.org/api/query?id_list=<id>`（2026-11-15 执行，标题与日期已比对）。
许可核验见 [PHASE2A_MODEL_LICENSE_MATRIX.md](../02_audit/PHASE2A_MODEL_LICENSE_MATRIX.md)。
