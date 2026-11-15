# ALAC-V 文档系统

> 本文件是 ALAC-V 项目文档体系的入口。任何人（包括后续接手的 Agent）应先读本文件，再按索引进入对应文档。项目源导航见 `docs/08_governance/PROJECT_SOURCE_INDEX.md`。

## 1. 为什么需要文档系统

ALAC-V 是一个长周期、多阶段、由 GitHub Issue 任务单驱动的研究工程项目。文档系统承担三件事：

1. **可追溯**：每个结论都能回溯到任务、运行、配置、数据版本和代码版本。
2. **可复现**：他人仅凭文档和证据即可重建环境、重跑实验、检查结论。
3. **可交接**：Agent 会话可能中断，GitHub 与版本化文档是跨会话记忆。

## 2. 目录职责

| 目录 | 职责 | 谁维护 |
|---|---|---|
| `docs/00_charter/` | 项目宪章：目标、原则、边界、成功标准。变更需 Owner 批准 | 主工程 Agent 起草，Owner 批准 |
| `docs/01_tasks/` | TASK-XXXX 任务单快照与历史状态；正式在线授权以 GitHub Issue 为准 | 顾问/Owner 下达，Agent 归档 |
| `docs/02_audit/` | 环境审计、依赖清单、硬件基线、模型许可 | 主工程 Agent |
| `docs/03_design/` | 架构、数据模型、接口契约、ADR 决策记录 | 主工程 Agent |
| `docs/04_experiments/` | 实验预注册（假设、变量、指标、判定规则）与结果 | 主工程 Agent |
| `docs/05_reports/` | 面向 Owner 的阶段交付报告 | 主工程 Agent |
| `docs/06_ops/` | 运行手册：安装、下载、跑管线、排障 | 主工程 Agent |
| `docs/07_risks/` | 风险登记册与已知问题 | 主工程 Agent |
| `docs/08_governance/` | 当前状态、验收协议、Issue/PR 工作流、项目源索引、独立验收记录 | ChatGPT/Owner 主导，Agent 同步 |
| `docs/research/` | Phase 2A 自主视觉学习文献调研与研究综述 | DSH 起草，ChatGPT/Owner 审核 |
| `docs/submissions/` | 不可变的阶段交付提交记录 | 主工程 Agent |

代码与数据的对应关系见 `docs/03_design/data-model.md`。

## 3. 权威来源与任务流

- GitHub Issue 是正式任务单、Owner 授权、问题讨论和验收状态的权威在线记录。
- `docs/01_tasks/TASK-XXXX.md` 是 Issue 的版本化快照或历史授权记录，不替代 Issue。
- PR 是候选变更，不等于验收；必须关联 Issue。
- 只有 Owner 可以记录最终 PASS、关闭 Issue 或授权进入下一阶段。
- `docs/08_governance/ACTIVE_STATE.md` 是当前项目状态的单一导航入口。

## 4. 编号与命名规范

| 前缀 | 含义 | 格式 | 示例 |
|---|---|---|---|
| `TASK-` | 任务快照 | `TASK-XXXX-<slug>.md` | `TASK-0001-phase0-phase1.md` |
| `ALAC-` | GitHub Issue 任务键 | `ALAC-NNNN` | `ALAC-0001` |
| `ADR-` | 架构决策记录 | `ADR-XXXX-<slug>.md` | `ADR-0003-store-sqlite-safetensors.md` |
| `EXP-` | 实验（预注册 + 结果） | `EXP-XXXX-<slug>.md` | `EXP-0001-abc-comparison.md` |
| `RUN-` | 一次实际运行 | `artifacts/runs/<UTC>_<exp>_<variant>/` | `20261114T103000Z_EXP0001_C/` |
| `DS-` | 数据集 | `data/manifests/*.json` | `ds-0001-alacv-synthetic-v1.json` |

规则：
- 编号只增不减，不复用；废弃文档保留并标记状态。
- 文件名使用小写英文 + 连字符；正文中文，术语首次出现给出英文原文。
- 时间统一用 UTC（ISO 8601）；本地时区仅用于展示。

## 5. 文档头部元信息（强制）

每份长期有效文档开头必须包含 YAML front-matter：

```yaml
---
id: ADR-0003
title: 元信息与特征用 SQLite + SafeTensors 存储
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0001
supersedes: null
---
```

## 6. 状态标签

- `草案`：正在写，结论不可引用。
- `提议`：可评审，等待 Owner 确认。
- `已接受`：生效，代码必须与之保持一致。
- `已废弃`：不再有效，保留历史。
- `已取代`：被新编号文档替代。
- 治理状态另可使用 `active`、`remediation-required`、`blocked`。

## 7. 证据链规则

任何正式报告数字必须同时具备：

1. **来源运行**：`artifacts/runs/<run_id>/metrics.json` / `run.log`；
2. **配置快照**：`config.snapshot.yaml` + `config.sha256`；
3. **数据版本**：`data/manifests/*.json` 与 SHA-256；
4. **模型版本**：模型 ID、revision、license、weight SHA-256；
5. **代码版本**：`git rev-parse HEAD` 与 dirty 标记。

完整 run 可以本地保存；用于正式验收的脱敏证据子集必须放在 `artifacts/evidence/<experiment-id>/<run-id>/`。缺失任一项的数字只能标记为 AGENT-REPORTED 或未独立复核，不得作为已复现结论。

## 8. 更新纪律

- 任务开始：更新对应 Issue 和 TASK 快照状态。
- 关键决策：新增 ADR，不静默改写旧 ADR。
- 实验前：预注册假设、变量、指标和判定规则。
- 实验后：追加结果、run 证据和限制；不得事后修改预注册标准。
- 阶段结束：写阶段报告、更新 CHANGELOG 和 ACTIVE_STATE。
- PR 合并不自动代表 Owner 验收；只有 Owner 确认后才能关闭 Issue 或进入下一阶段。

## 9. 当前文档索引

| 文档 | 说明 | 当前状态 |
|---|---|---|
| `00_charter/ALAC-V-charter.md` | 项目宪章 | 已接受 |
| `01_tasks/TASK-0001-phase0-phase1.md` | Phase 0/1 原始任务单快照 | 已完成并验收 |
| `01_tasks/TASK-0002-phase0-1-remediation.md` | Phase 0/1 独立验收修复任务快照 | 已完成；PASS；PR #2 已合并 |
| `02_audit/environment-audit.md` | 环境审计报告 | DSH 审计完成；OS 产品/版本另有注册表直接证据；未声称独立重跑审计 |
| `02_audit/model-licenses.md` | 模型许可证登记 | 已记录 |
| `03_design/architecture.md` | 系统架构 | 已接受（Phase 1） |
| `03_design/data-model.md` | 数据模型与目录布局 | 已接受（Phase 1） |
| `03_design/decisions/` | ADR 决策记录 | ADR-0001 至 ADR-0008；Phase 1 设计已接受 |
| `04_experiments/EXP-0001-abc-comparison.md` | A/B/C 预注册与结果 | 证据链已补齐；GT bbox 修订结果已验收；H3 拒绝 |
| `05_reports/PHASE1-REPORT.md` | Phase 1 交付报告 | 已验收；H3 按 GT bbox 口径拒绝 |
| `06_ops/runbook.md` | 运行手册 | 已接受（初版） |
| `07_risks/risk-register.md` | 风险登记册 | 已接受（初版） |
| `08_governance/ACTIVE_STATE.md` | 当前项目状态 | 当前入口 |
| `08_governance/ACCEPTANCE_PROTOCOL.md` | 验收标准 | 生效 |
| `08_governance/ISSUE_WORKFLOW.md` | GitHub 任务工作流 | 生效 |
| `08_governance/PROJECT_SOURCE_INDEX.md` | 项目源索引 | 当前入口 |
| `08_governance/PHASE0_1_ACCEPTANCE.md` | Phase 0/1 最终独立验收 | PASS；PR #2 merged；Issue #1 closed |
| `01_tasks/TASK-0003-phase2a-autonomous-visual-learning-design.md` | Phase 2A 自主视觉学习研究设计任务 | Owner 已授权；仅设计/预注册 |
| `submissions/PHASE_0_1_SUBMISSION.md` | 首次公开仓库交付记录 | 历史记录 |
| `research/PHASE2A_AUTONOMOUS_LEARNING_LITERATURE.md` | Phase 2A 文献调研（arXiv/仓库/许可核验） | 草案；待 ChatGPT/Owner 审阅 |
| `03_design/PHASE2A_AUTONOMOUS_LEARNING_DESIGN.md` | Phase 2A 架构设计与资源方案 | 草案；待审 |
| `04_experiments/EXP-0002-autonomous-visual-learning-preregistration-draft.md` | EXP-0002 预注册草案 | 草案；未授权运行 |
| `02_audit/PHASE2A_MODEL_LICENSE_MATRIX.md` | Phase 2A 代码/权重许可矩阵 | 草案；已核验来源 |
| `05_reports/PHASE2A_OWNER_DECISION_SUMMARY.md` | Phase 2A Owner 决策摘要 | 草案；待审 |


## 10. 当前阶段与开放任务

- Phase 0/1：已通过独立验收；PR #2 已合并，Issue #1 已关闭。
- Phase 2A：设计、文献、许可矩阵与 EXP-0002 预注册草案已提交（PR 待审）；训练仍未授权。
- 模型训练、权重下载、公开自然图像数据下载均未授权。
- 当前开放任务：[ALAC-0002 / Issue #3](https://github.com/zcx369658780/ALAC-V/issues/3)。
- Phase 2B 长期视觉记忆、Phase 2C 动态计算与 Phase 3 视频/实时流均需后续独立授权。
