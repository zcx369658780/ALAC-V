---
id: TASK-0003
title: Phase 2A 自主视觉学习研究设计与静态图像可行性方案
status: owner-authorized; design-only
date: 2026-11-15
owner: Owner
source: ALAC-0002
issue: https://github.com/zcx369658780/ALAC-V/issues/3
---

# ALAC-0002 — Phase 2A：自主视觉学习研究设计与静态图像可行性方案

- **Task snapshot:** `docs/01_tasks/TASK-0003-phase2a-autonomous-visual-learning-design.md`
- **Phase:** Phase 2A（自主视觉学习，静态图像）
- **执行 Agent：** DSH
- **独立审查：** ChatGPT；ClawsGO 可按 Owner 后续安排进行只读审查
- **授权基线：** `367035318172ba1bbc0103b6231847750143d1d7`
- **授权日期：** 2026-11-15

## 1. Owner 决策与授权

Owner 已明确批准合并 PR #2 并关闭 Issue #1，并确认后续研究顺序：
1. **Phase 2A：自主视觉学习**
2. **Phase 2B：长期、多层次视觉记忆**
3. **Phase 2C：动态计算资源分配**
4. Phase 3：离线视频与实时视觉流。

据此，Owner **正式授权本 Issue 的研究设计、文献调研、架构比较、数据方案与实验预注册工作**。

**本授权仅覆盖 Phase 2A 的设计与预注册，不授权模型训练、主干微调、正式实验运行或 Phase 2B/2C。** 设计方案通过 ChatGPT/Owner 审核后，再建立独立实施任务单。

## 2. 项目基线与研究问题

Phase 0/1 已完成独立验收并合并至 `main`。合并提交：
`367035318172ba1bbc0103b6231847750143d1d7`。

现有静态图像基线：
- 冻结的 DINOv2 ViT-S/14；
- 原始图像内容寻址归档；
- 多尺度金字塔与 Foveated ROI；
- CLS/Patch 特征、SQLite + SafeTensors；
- 图像检索、ROI 回溯；
- EXP-0001 的 A/B/C 结果与九次脱敏 run evidence。

硬件边界：Intel Core Ultra 7 265K、32GB RAM、RTX 5070 12GB，单机单卡。

核心研究问题：

**在不使用人工标签、不依赖外部 LLM API 的前提下，能否让视觉模型通过观察图像、不同视角/增强之间的关系，以及上下文—目标区域之间的预测任务，自主学习到比固定 DINOv2 表征更有用、更稳健的视觉表示？**

必须区分：
- 只训练 predictor：学习预测视觉状态，但不一定改变 encoder representation；
- 训练 adapter/student：可能改变可用于下游任务的视觉表示；
- 微调完整 backbone：计算与过拟合风险较高，本任务不授权。

## 3. 本任务范围：只做研究设计，不做训练

### WP0 — 基线与代码审计

1. 从最新 `main` 同步，核验 Phase 0/1 merge commit。
2. 阅读项目宪章、Phase 1 architecture/data model、ADR-0001 至 ADR-0008、EXP-0001、PHASE1-REPORT、TASK-0002 remediation report。
3. 确认 DINOv2 预处理、CLS/Patch 输出、特征维度、现有数据集限制及本机硬件实测基线。
4. 不修改现有 Phase 1 代码和实验结果。

### WP1 — 相关研究与开源方案调研

至少比较以下方向，并允许补充更合适的路线：

- JEPA / I-JEPA / V-JEPA：潜空间上下文—目标预测；
- DINO / iBOT / BYOL / SimSiam：自蒸馏、跨视图一致性与无负样本学习；
- MAE / masked latent modeling：掩码区域的像素或特征预测；
- predictive coding / online self-supervised adaptation：持续观察与增量学习。

每项需记录：
- 论文原始链接、年份、核心训练目标；
- 网络结构与训练信号；
- 是否需要 teacher、decoder、负样本或标签；
- 参数规模与训练资源；
- 官方代码、预训练权重、许可证；
- 与现有 DINOv2 特征管线的兼容性；
- 在 RTX 5070 12GB 上的可行性和预期限制。

不能把模型开源代码、开放权重和完整训练数据视为同一概念；逐项核实许可。

### WP2 — 候选机制比较

提出最多 3 个候选方案，至少覆盖：

A. **JEPA-style latent prediction**：上下文 Patch 预测目标 Patch 的 teacher latent；比较 predictor-only 与 adapter+predictor 两种设定。

B. **Self-distillation / cross-view learning**：冻结 teacher，以不同 crop/augmentation 产生目标，训练轻量 student adapter/projector。

C. **Masked latent prediction**：遮蔽部分 Patch，以未遮蔽上下文预测冻结 teacher 的目标特征。

每种方案都要明确：
- 哪些参数冻结、哪些可训练；
- 训练目标的数学表达；
- 数据增强与 mask 策略；
- 防止 representation collapse 的机制；
- 训练显存、参数量和吞吐估计；
- 失败模式及诊断方法。

不得预先假定 JEPA 一定优于 DINO/BYOL；必须提出可证伪比较。

### WP3 — 无标签数据方案

当前 Owner 尚未批准下载公开自然图像集，因此：
- 只允许设计基于 DS-0001 与程序生成图像的初期方案；
- 不得下载自然图像数据集或上传用户图片；
- 训练阶段不得使用合成器内部类别、对象 ID、场景因子或 bbox 标签；
- 这些隐藏元数据只能在评估阶段作为 ground truth 使用；
- 必须按基础场景/生成 seed 分组切分 train/validation/test，不能把同一 base scene 的不同 augmentation 分散到训练集和测试集造成泄漏。

请评估 DS-0001 的 24 个 base scenes 是否足以支持任何学习结论；若不足，提出可控的程序化无标签数据生成方案，并清楚声明结论仅适用于合成域。

自然图像外部验证需要另行向 Owner 提交 D-1 数据许可决策，不在本 Issue 授权范围内。

### WP4 — 实验预注册草案

设计一份独立实验预注册（建议 EXP-0002 draft），至少包括：

**基线**
- Frozen DINOv2 原始特征；
- 仅训练 predictor 的方案；
- 训练轻量 adapter/student 的方案；
- 随机初始化或简单投影对照（若有必要）。

**评估指标**
- 无标签训练损失与 latent prediction error；
- 同一场景不同增强下的表示一致性；
- Recall@1、Recall@5、mAP@10、同源/异源 AUC；
- 对未见生成 seed / 未见组合的泛化；
- 数据效率（不同训练样本量下的性能曲线）；
- 可训练参数、GPU 峰值显存、训练时间、推理延迟、特征存储量。

训练过程不得使用评价标签；评价标签只允许用于训练后测量。

至少 3 个随机种子，预先定义统计方法、置信区间和失败判据。不能只报告训练 loss 下降就声称自主视觉学习成功。

### WP5 — 资源与实施方案

以 RTX 5070 12GB 为硬约束，提出：
- 推荐的输入分辨率和 Patch 数；
- batch size、梯度累积、混合精度；
- 冻结主干与可训练模块的显存预算；
- 预计训练时间与数据规模；
- checkpoint、配置快照、数据 manifest 和 run evidence 方案。

本任务可做静态资源估算与代码路径审查；**不运行正式训练，不下载权重，不安装新训练框架**。

## 4. 交付文件

在任务分支中提交：

1. `docs/research/PHASE2A_AUTONOMOUS_LEARNING_LITERATURE.md`
2. `docs/03_design/PHASE2A_AUTONOMOUS_LEARNING_DESIGN.md`
3. `docs/04_experiments/EXP-0002-autonomous-visual-learning-preregistration-draft.md`
4. `docs/02_audit/PHASE2A_MODEL_LICENSE_MATRIX.md`
5. 更新 `docs/08_governance/ACTIVE_STATE.md` 与项目索引；
6. 一份简明 Owner 决策摘要：推荐方案、替代方案、风险、待决策事项。

文档中应区分“论文事实”“官方仓库事实”“Agent 推断”和“尚待实验验证的假设”。

## 5. 明确不在本任务授权范围内

- 不训练任何模型，不微调 DINOv2 backbone；
- 不下载任何新模型权重；
- 不下载公开自然图像数据集；
- 不使用用户私人图像；
- 不实现长期视觉记忆系统（Phase 2B）；
- 不实现动态计算控制器（Phase 2C）；
- 不处理视频、光流、实时流、Event Camera 或 SNN（Phase 3+）；
- 不改变 Phase 1 A/B/C 结果或已接受 ADR；
- 不引入向量数据库、分布式训练或云端服务。

## 6. 验收标准

- [ ] 文献与官方代码/权重/许可信息可追溯；
- [ ] 至少 3 种自监督/预测学习机制有公平比较；
- [ ] 清楚区分 predictor learning 与 representation learning；
- [ ] 数据划分无 base-scene / augmentation leakage；
- [ ] 训练不使用人工标签，评价标签仅用于评估；
- [ ] 有可证伪假设、基线、指标、统计方案和失败判据；
- [ ] 资源预算符合 12GB VRAM 单卡约束；
- [ ] 无未经授权的数据/权重下载或模型训练；
- [ ] 文档与 Phase 1 已接受架构和治理规则一致；
- [ ] PR 提交后停止，等待独立审查和 Owner 决策。

## 7. 停止条件与下一阶段门

本 Issue 完成后，DSH 只提交研究设计与预注册草案，不得自动开始训练。

ChatGPT 与 Owner 审核设计后，若认可，将另行创建 **Phase 2A.1 模型训练与静态图像可行性实验**任务单。

Phase 2B（长期视觉记忆）必须等待 Phase 2A 的学习机制验证；Phase 2C（动态计算分配）必须等待前两项有稳定的视觉表征和记忆接口。视频/实时流属于 Phase 3，不得提前启动。
