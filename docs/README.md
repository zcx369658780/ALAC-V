# ALAC-V 文档系统

> 本文件是 ALAC-V 项目文档体系的入口。任何人（包括后续接手的 Agent）应先读本文件，再按索引进入对应文档。

## 1. 为什么需要文档系统

ALAC-V 是一个**长周期、多阶段、由顾问下达任务单驱动**的研究工程项目。文档系统承担三件事：

1. **可追溯**：每个结论都能回溯到"哪张任务单、哪次运行、哪份配置、哪个数据版本"。
2. **可复现**：他人仅凭文档即可重建环境、重跑实验、得到同等结论。
3. **可交接**：Agent 会话可能中断，文档是唯一可靠的跨会话记忆。

## 2. 目录职责

| 目录 | 职责 | 谁维护 |
|---|---|---|
| `docs/00_charter/` | 项目宪章：目标、原则、边界、成功标准。变更需 Owner 批准 | 主工程 Agent 起草，Owner 批准 |
| `docs/01_tasks/` | 任务单（TASK-XXXX）与执行状态日志。任务单是唯一的工作授权来源 | 顾问下达，Agent 记录执行 |
| `docs/02_audit/` | 环境审计、依赖清单、硬件基线 | 主工程 Agent |
| `docs/03_design/` | 架构、数据模型、接口契约、ADR 决策记录 | 主工程 Agent |
| `docs/04_experiments/` | 实验预注册（假设、变量、指标、判定规则）与实验结果 | 主工程 Agent |
| `docs/05_reports/` | 面向 Owner 的阶段交付报告 | 主工程 Agent |
| `docs/06_ops/` | 运行手册：安装、下载、跑管线、排障 | 主工程 Agent |
| `docs/07_risks/` | 风险登记册与已知问题 | 主工程 Agent |

代码与数据的对应关系见 `docs/03_design/data-model.md`。

## 3. 编号与命名规范

| 前缀 | 含义 | 格式 | 示例 |
|---|---|---|---|
| `TASK-` | 任务单 | `TASK-0001-<slug>.md` | `TASK-0001-phase0-phase1.md` |
| `ADR-` | 架构决策记录 | `ADR-0001-<slug>.md` | `ADR-0003-store-sqlite-safetensors.md` |
| `EXP-` | 实验（预注册 + 结果） | `EXP-0001-<slug>.md` | `EXP-0001-abc-comparison.md` |
| `RUN-` | 一次实际运行 | 目录 `artifacts/runs/<UTC时间戳>_<exp>_<variant>/` | `20261114T103000Z_EXP0001_C/` |
| `DS-` | 数据集 | `data/manifests/ds-0001-*.json` | `ds-0001-alacv-synthetic-v1.json` |

规则：
- 编号只增不减，不复用；废弃文档保留并标记 `状态: 废弃`。
- 文件名一律小写英文 + 连字符；正文用中文，术语首次出现给出英文原文。
- 时间统一用 UTC（ISO 8601，`2026-11-14T10:30:00Z`），本地时区仅用于展示。

## 4. 文档头部元信息（强制）

每份文档开头必须包含 YAML front-matter：

```yaml
---
id: ADR-0003
title: 元信息与特征用 SQLite + SafeTensors 存储
status: 已接受        # 草案 | 提议 | 已接受 | 已废弃 | 已取代
date: 2026-11-14
owner: lead
task: TASK-0001
supersedes: null
---
```

## 5. 状态标签

- `草案`：正在写，结论不可引用。
- `提议`：可评审，等待 Owner/顾问确认。
- `已接受`：生效，代码必须与之保持一致。
- `已废弃`：不再有效，保留历史。
- `已取代`：被新编号文档替代，头部 `supersedes` 字段互相指向。

## 6. 证据链规则

任何写入报告的数字必须同时具备：

1. **来源运行**：`artifacts/runs/<run_id>/` 下的 `metrics.json` / `run.log`；
2. **配置快照**：该目录下的 `config.snapshot.yaml` + `config.sha256`；
3. **数据版本**：`data/manifests/*.json` 中的数据集清单与 SHA-256；
4. **代码版本**：`git rev-parse HEAD` 与该次运行的 `dirty` 标记。

缺任一项的数字只能标记为"非正式观察"，不得进入结论。

## 7. 更新纪律

- 任务开始：更新 `docs/01_tasks/TASK-XXXX.md` 的状态日志（开始时间、计划）。
- 关键决策：新增 ADR，不修改旧 ADR 结论。
- 实验前后：先写预注册（计划），跑完追加结果；**不得事后修改假设与判定规则**。
- 阶段结束：写 `docs/05_reports/` 报告 + 更新 `CHANGELOG.md`。

## 8. 当前索引

| 文档 | 说明 | 状态 |
|---|---|---|
| `00_charter/ALAC-V-charter.md` | 项目宪章 | 已接受 |
| `01_tasks/TASK-0001-phase0-phase1.md` | Phase 0 + Phase 1 任务单 | 已完成（待审） |
| `02_audit/environment-audit.md` | 环境审计报告 | 已接受 |
| `02_audit/model-licenses.md` | 模型许可证登记 | 已接受 |
| `03_design/architecture.md` | 系统架构 | 已接受 |
| `03_design/data-model.md` | 数据模型与目录布局 | 已接受 |
| `03_design/decisions/` | ADR 决策记录 | 见目录 |
| `04_experiments/EXP-0001-abc-comparison.md` | 三组对照实验 | 已完成 |
| `05_reports/PHASE1-REPORT.md` | Phase 1 交付报告 | 待 Owner 审核 |
| `06_ops/runbook.md` | 运行手册 | 已接受 |
| `07_risks/risk-register.md` | 风险登记册 | 已接受 |
