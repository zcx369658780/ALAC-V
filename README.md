# ALAC-V

**Visual-first Autonomous Learning and Adaptive Cognition** —— 一种不依赖 LLM 作为核心推理引擎的视觉智能系统的工程基础。

> **当前状态（2026-11-14）：** Phase 0/1 工程实现与整改已提交为 PR #2（`task/1-phase0-1-remediation`），
> 第二轮独立复核已通过核心修复（GT bbox、逐轴缩放、manifest 哈希、provenance），第三轮为收尾修订；
> 等待最终复核与 Owner 验收。**Phase 2 保持 BLOCKED。**

## 当前实验基线（DSH 报告值，尚未独立复现）

| 变体 | 单图时间 (ms) | 派生存储/图 | Recall@1 | 回溯 PSNR（GT bbox） |
|---|---:|---:|---:|---:|
| A 原始分辨率 | 34.4 | 1.00 MB | 0.833 | 31.66 dB* |
| B 多尺度 | 216.2 | 4.10 MB | 0.875 | 31.05 dB* |
| C 多尺度 + Foveated ROI | 361.3 | 6.14 MB | 0.889 | 28.53 dB* |

（DS-0001：24 张合成基准 + 72 条查询；3 次重复中位数。详见 `docs/04_experiments/EXP-0001-abc-comparison.md`。）

\* 回溯列为 TASK-0002 修正后、按预注册 **ground-truth bbox** 区域重算的 PSNR；原始值（26.76 / 29.19 / 29.10 dB）
与修订说明见 `docs/04_experiments/EXP-0001-AMD-0001-backtrack-coordinates.md`。

**重要说明：** 九次 run 的脱敏证据、GT bbox 主指标、逐轴缩放修正、LF manifest 哈希规范均已随 PR 提交
（`artifacts/evidence/EXP-0001/`）。上述数字仍为 **DSH 报告值**，尚未在 Owner 工作站独立复现；
当前报告没有宣称整体效率提升。

## 它做什么

给定任意图像目录，一条命令完成：

```
原始归档（SHA-256 内容寻址、只读）
   → 四档多尺度金字塔（1/1, 1/2, 1/4, 1/8）
   → Foveated ROI（高分中央凹 + 分级周边 + 全分辨率 ROI 裁片）
   → DINOv2 ViT-S/14 特征（CLS 全局 + Patch 局部，FP16）
   → SQLite + SafeTensors 视觉知识库
   → 余弦检索 + ROI 坐标 + 原始高清回溯
```

## 硬件基线（DSH 本机审计）

| 项 | 报告值 |
|---|---|
| CPU | Intel Core Ultra 7 265K（20 线程） |
| RAM | 31.48 GB 识别总量；审计时可用 10.9 GB |
| GPU | NVIDIA GeForce RTX 5070，12227 MiB，sm_120 |
| 驱动 / CUDA | 616.92 / UMD 13.4 / toolkit 12.8 / PyTorch cu130 |

单卡 12 GB 是本项目的硬约束。详见 `docs/02_audit/environment-audit.md`；OS 产品名/版本取自注册表直接证据（`docs/02_audit/os-evidence.json`）。

## 快速开始

```powershell
cd D:\ALAC_V

# 0) 网络（本机特有）
$env:HTTPS_PROXY='http://127.0.0.1:7897'; $env:HTTP_PROXY=$env:HTTPS_PROXY

# 1) 隔离环境
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe -e . --no-deps

# 2) 环境审计、模型、数据
.\.venv\Scripts\python.exe scripts\audit_environment.py
.\.venv\Scripts\python.exe scripts\fetch_model.py
.\.venv\Scripts\python.exe scripts\make_dataset.py

# 3) 单图入库 / 检索 / 回溯
.\.venv\Scripts\python.exe -m alacv.cli ingest --input data\datasets\ds-0001\base --variant C
.\.venv\Scripts\python.exe -m alacv.cli query --image data\datasets\ds-0001\query\scene_000_q0.jpg --variant C
.\.venv\Scripts\python.exe -m alacv.cli verify

# 4) 测试
.\.venv\Scripts\python.exe -m pytest -q
```

## 目录结构

```
AGENTS.md         Agent 工作规则与权限边界
.github/          GitHub Issue / PR 模板
src/alacv/        图像归档、金字塔、foveation、特征、存储、检索与管线
scripts/          审计、模型下载、数据集、实验、完整性校验
tests/            单元与集成测试
configs/          可复现配置
data/             本地原始数据、派生数据、特征和索引（默认不入库）
artifacts/
  experiments/    实验汇总
  figures/        报告插图
  evidence/       脱敏 run 级证据包（正式验收需要）
  runs/            完整本地 run（默认不入库）
docs/
  00_charter/      项目宪章
  01_tasks/        任务单快照
  02_audit/        环境和模型许可审计
  03_design/       架构、数据模型和 ADR
  04_experiments/  预注册与实验结果
  05_reports/      阶段报告
  06_ops/          运行手册
  07_risks/        风险登记
  08_governance/   当前状态、验收协议、工作流和项目源索引
  submissions/     历史交付记录
```

## 文档与治理入口

- [Agent 工作规则](AGENTS.md)
- [项目源索引](docs/08_governance/PROJECT_SOURCE_INDEX.md)
- [当前状态](docs/08_governance/ACTIVE_STATE.md)
- [验收协议](docs/08_governance/ACCEPTANCE_PROTOCOL.md)
- [GitHub Issue / PR 工作流](docs/08_governance/ISSUE_WORKFLOW.md)
- [Phase 0/1 独立审查](docs/08_governance/PHASE0_1_ACCEPTANCE.md)
- [全部文档索引](docs/README.md)
- [当前开放任务 ALAC-0001](https://github.com/zcx369658780/ALAC-V/issues/1)

## 设计原则

1. 原始数据不可变：内容寻址 + 只读 + 校验。
2. 视觉为主：LLM 不参与核心视觉推理。
3. 本地优先、开源优先：模型许可、revision 和权重指纹可查。
4. 效率声明必须附带能力证据。
5. 每次实验配置化、版本化、日志化、可追溯。
6. 不擅自扩张：每个阶段必须有 Owner 授权。
7. 用户图像不上传云端；公开仓库不保存私人数据和模型权重。

## 许可证

- 项目代码：**许可证尚未确定/尚未发布 LICENSE 文件**；在 Owner 明确选择前，不应假定为 Apache-2.0
- Phase 1 模型：DINOv2（Meta AI），Apache-2.0
- DS-0001 合成数据集：CC0-1.0
