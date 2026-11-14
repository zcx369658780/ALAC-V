# ALAC-V

**Visual-first Autonomous Learning and Adaptive Cognition** —— 一种不依赖 LLM 作为核心推理引擎的视觉智能系统的工程基础。

本仓库当前处于 **Phase 1：静态图像视觉编码 — 已完成，等待 Owner 审核**。

| 变体 | 单图时间 (ms) | 派生存储/图 | Recall@1 | 回溯 PSNR |
|---|---|---|---|---|
| A 原始分辨率 | 34.4 | 1.00 MB | 0.833 | 26.76 dB |
| B 多尺度 | 216.2 | 4.10 MB | 0.875 | 29.19 dB |
| C 多尺度 + Foveated ROI | 361.3 | 6.14 MB | 0.889 | 29.10 dB |

（DS-0001：24 张合成基准 + 72 条查询；3 次重复中位数。详见 `docs/05_reports/PHASE1-REPORT.md`。）

**关键结论**：三组准确率差异有限而代价差异显著，本阶段**不宣称效率提升**；消融显示 foveated+ROI 描述子的信息密度最高（同为 2 MB 时 +8.4 pp）。**未进入 Phase 2。**

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

## 硬件基线

| 项 | 实测 |
|---|---|
| CPU | Intel Core Ultra 7 265K（20 线程） |
| RAM | 31.48 GB 可用 |
| GPU | NVIDIA GeForce RTX 5070，12227 MiB，sm_120 |
| 驱动 / CUDA | 616.92 / UDA 13.4，toolkit 12.8 |

单卡 12 GB 是本项目的硬约束：所有设计必须在"单机 + 单卡 + 32 GB 内存"下可运行。

## 快速开始

```powershell
cd D:\ALAC_V

# 0) 网络（本机特有：schannel 不可用，须走代理 + Python/uv）
$env:HTTPS_PROXY='http://127.0.0.1:7897'; $env:HTTP_PROXY=$env:HTTPS_PROXY

# 1) 隔离环境
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe -e . --no-deps

# 2) 环境审计（写 docs/02_audit/environment-audit.json）
.\.venv\Scripts\python.exe scripts\audit_environment.py

# 3) 模型（DINOv2 ViT-S/14，Apache-2.0，记录 SHA-256 与许可证）
.\.venv\Scripts\python.exe scripts\fetch_model.py

# 4) 数据（纯程序化生成，无外部下载）
.\.venv\Scripts\python.exe scripts\make_dataset.py

# 5) 单图入库 + 检索 + 回溯
.\.venv\Scripts\python.exe -m alacv.cli ingest --input data\datasets\ds-0001\base --variant C
.\.venv\Scripts\python.exe -m alacv.cli query --image data\datasets\ds-0001\query\scene_000_q0.jpg --variant C
.\.venv\Scripts\python.exe -m alacv.cli backtrack --image-id <id> --roi 0 --out out.png

# 6) 三组对照实验
.\.venv\Scripts\python.exe scripts\run_experiments.py

# 7) 测试
.\.venv\Scripts\python.exe -m pytest -q
```

## 目录结构

```
docs/            文档系统（宪章、任务单、审计、设计、实验、报告、运维、风险）
configs/         可复现配置（默认 / 实验 / 数据集）
src/alacv/       管线代码
scripts/         审计、下载、数据集、实验、校验
tests/           单元测试
data/            raw（只读原始）/ processed（派生）/ features / index / manifests / datasets
artifacts/       runs（每次运行的全部证据）/ models / experiments / figures
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档系统与编号规范、证据链规则 |
| [docs/00_charter/ALAC-V-charter.md](docs/00_charter/ALAC-V-charter.md) | 项目宪章：目标、原则、阶段边界 |
| [docs/01_tasks/TASK-0001-phase0-phase1.md](docs/01_tasks/TASK-0001-phase0-phase1.md) | 本次任务单与验收清单 |
| [docs/02_audit/environment-audit.md](docs/02_audit/environment-audit.md) | 硬件/驱动/CUDA/Python/磁盘审计 |
| [docs/03_design/architecture.md](docs/03_design/architecture.md) | 架构与数据流 |
| [docs/03_design/data-model.md](docs/03_design/data-model.md) | 数据模型、SQLite schema、SafeTensors 格式 |
| [docs/03_design/decisions/](docs/03_design/decisions/) | ADR 决策记录 |
| [docs/04_experiments/EXP-0001-abc-comparison.md](docs/04_experiments/EXP-0001-abc-comparison.md) | A/B/C 对照实验（预注册 + 结果） |
| [docs/05_reports/PHASE1-REPORT.md](docs/05_reports/PHASE1-REPORT.md) | Phase 1 交付报告 |
| [docs/06_ops/runbook.md](docs/06_ops/runbook.md) | 运行手册与排障 |
| [docs/07_risks/risk-register.md](docs/07_risks/risk-register.md) | 风险与已知问题 |

## 设计原则（摘要）

1. 原始数据不可变：内容寻址 + 只读 + 校验。
2. 视觉为主：不调用外部 LLM API 参与视觉推理。
3. 本地优先、开源优先：模型可离线运行、许可证可查。
4. 效率声明必须附带能力证据：压缩/降采样/注意力之外的每一次"省"，都要用检索与回溯指标交代代价。
5. 不擅自扩张：不下载大数据集、不训练主干、不引入重型基础设施。

## 许可证

- 本项目代码：Apache-2.0
- 模型：DINOv2（Meta AI），Apache-2.0，见 `artifacts/models/dinov2_vits14/LICENSE`
- 合成数据集：CC0-1.0（程序化生成）
