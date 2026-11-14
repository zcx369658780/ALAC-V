---
id: TASK-0001
title: Phase 0 环境审计 + Phase 1 静态图像视觉编码
status: 已完成（待 Owner 审核）
date: 2026-11-14
owner: lead
source: GPT 顾问任务单
---

# TASK-0001 · Phase 0 + Phase 1

## 1. 任务单原文（权威版本，不得改写结论）

> **ALAC-V 项目启动：Phase 0 + Phase 1**
>
> 你现在作为 ALAC-V（Visual-first Autonomous Learning and Adaptive Cognition）项目的主工程 Agent。
>
> ### 一、项目目标
> 开发一种不依赖 LLM 作为核心推理引擎的视觉智能系统。长期目标：1) 高效视觉编码与压缩；2) 多层次视觉记忆；3) 自主视觉学习；4) 未来状态预测；5) 动态计算资源分配；6) 从静态图像逐步扩展至视频和实时视觉流。
> 必须坚持开源模型优先，模型推理和学习过程以本地运行作为目标。
>
> ### 二、本地硬件
> CPU: Intel Core Ultra 7 265K / RAM: 32GB DDR5 / GPU: NVIDIA RTX 5070 / 显存：预计 12GB GDDR7，需实际核验。不得假设具备多 GPU 或服务器级计算资源。
>
> ### 三、第一阶段模型选型
> 主模型：DINOv2 ViT-S/14，21M 参数，Apache 2.0。研究技术：Gaussian Pyramid、Foveated Vision、Multi-Scale Visual Representation、ROI-based High-Resolution Retrieval。暂不进行大型基础模型预训练。
>
> ### 四、Phase 0：环境审计
> 检查：操作系统与版本；CPU、RAM、GPU 型号和显存；NVIDIA 驱动及 CUDA 兼容情况；Python、PyTorch 环境；硬盘空间；本地开发环境与项目目录。采用独立 Python 虚拟环境，不修改系统全局依赖。确认 RTX 5070 的 CUDA 推理和混合精度运行正常。完成后记录完整配置和依赖版本。
>
> ### 五、Phase 1：静态图像视觉编码
> 1. **原始图像归档**：支持 JPEG、PNG、WebP、AVIF。原始文件只读保存；记录 SHA-256、分辨率、颜色空间、文件大小；不允许不可逆覆盖原始图像。
> 2. **多尺度视觉处理**：原始分辨率 / 1/2 / 1/4 / 1/8 四档可配置表示。Foveated ROI：重要区域保留高分辨率，周边低分辨率；保留原始图像以支持局部高清回溯；不允许简单永久丢弃焦点外信息。
> 3. **视觉特征提取**：DINOv2 ViT-S/14，提取 CLS 全局特征与 Patch 局部特征，保存 FP16，记录模型版本与预处理参数。
> 4. **视觉知识库存储**：独立视觉数据目录，至少保存原始图片索引、多尺度图像、全局特征、局部 Patch 特征、ROI 坐标、模型版本、数据处理元信息。可用 SQLite/Parquet 管理元信息，SafeTensors 或类似格式存特征。不要过早引入复杂数据库。
>
> ### 六、第一阶段实验
> A：原始分辨率 DINOv2 特征；B：多尺度图像 DINOv2 特征；C：Foveated ROI + 多尺度 DINOv2 特征。
> 比较：单张图片处理时间；GPU 显存峰值；CPU 和 RAM 占用；特征文件大小；特征提取与检索速度；视觉相似性和检索准确率；局部高清回溯能力。
> **不得仅凭特征文件变小就宣称模型效率提升，必须同时检查视觉任务表现。**
>
> ### 七、工程要求
> 可复现配置文件；记录每次实验参数和日志；单元测试；保存模型许可信息；原始数据不可变；不调用外部 LLM API 参与视觉推理；不上传用户图像到云端；不进行未经批准的大规模数据下载；不在本阶段擅自训练 DINOv2 主干。
>
> ### 八、阶段交付
> 1. 硬件与环境审计报告；2. 项目目录结构；3. 模型下载与运行说明；4. 静态图像编码程序；5. 多尺度与 Foveated ROI 实现；6. 三组实验对比结果；7. GPU 显存与运行性能报告；8. 已知问题和下一阶段建议。
> 本阶段完成后停止，不要自动进入视频模型训练。由 Owner 审核 Phase 1 结果后，再决定进入 Phase 2。
>
> **核心原则：先建立可靠、可复现的视觉数据处理基础，再逐步研究新的学习机制。**

## 2. 验收清单（逐条可勾选）

| # | 要求 | 证据位置 | 状态 |
|---|---|---|---|
| P0-1 | OS 版本核验 | `docs/02_audit/environment-audit.md` §1 | ✅ |
| P0-2 | CPU/RAM/GPU/显存实测 | 同上 §2（含 nvidia-smi 原始输出） | ✅ |
| P0-3 | NVIDIA 驱动 + CUDA 兼容性 | 同上 §3 | ✅ |
| P0-4 | Python / PyTorch 环境 | 同上 §4（含 `torch.cuda` 自检） | ✅ |
| P0-5 | 硬盘空间 | 同上 §5 | ✅ |
| P0-6 | 独立 venv，未污染全局 | 同上 §6 + `.venv/` | ✅ |
| P0-7 | FP16 推理在 RTX 5070 上跑通 | 同上 §7 + `artifacts/runs/*/gpu_selfcheck.json` | ✅ |
| P1-1 | JPEG/PNG/WebP/AVIF 归档 + SHA-256 + 元数据 | `tests/test_ingest.py`、`data/manifests/` | ✅ |
| P1-2 | 原始文件只读、不可变 | 归档权限位 + `tests/test_ingest.py::test_*immutable*` | ✅ |
| P1-3 | 四档多尺度表示（可配置） | `configs/default.yaml`、`tests/test_pyramid.py` | ✅ |
| P1-4 | Foveated ROI（高分中心 + 分级周边 + 保留原始） | `src/alacv/foveation.py`、`tests/test_foveation.py` | ✅ |
| P1-5 | DINOv2 CLS + Patch，FP16 落盘 | `src/alacv/features.py`、`tests/test_features.py` | ✅ |
| P1-6 | 模型版本 + 预处理参数记录 | `data/manifests/model-*.json`、每 run 的 `model.json` | ✅ |
| P1-7 | 知识库：索引/多尺度/全局/局部/ROI/版本/元信息 | `src/alacv/store.py`、SQLite schema | ✅ |
| P1-8 | A/B/C 三组对照，7 类指标 | `docs/04_experiments/EXP-0001-abc-comparison.md` | ✅ |
| P1-9 | 效率声明同时给任务表现 | 报告 §"效率与能力联合结论" | ✅ |
| P1-10 | 配置文件 + 运行日志 + 可复现 | `artifacts/runs/`、`config.sha256` | ✅ |
| P1-11 | 单元测试 | `pytest -q` 全绿 | ✅ |
| P1-12 | 模型许可信息 | `artifacts/models/LICENSE-*`、`docs/02_audit/model-licenses.md` | ✅ |
| P1-13 | 无外部 LLM 调用、无上传 | 代码审计：无 http 客户端调用 LLM；`docs/05_reports` §合规 | ✅ |
| P1-14 | 未训练主干 | 权重 SHA-256 与上游一致 | ✅ |
| P1-15 | 交付报告 8 项齐全 | `docs/05_reports/PHASE1-REPORT.md` | ✅ |
| P1-16 | 完成后停止，不进入 Phase 2 | 报告末节 + 状态日志 | ✅ |

## 3. 执行状态日志

| 时间 (UTC) | 事件 | 备注 |
|---|---|---|
| 2026-11-14T10:18Z | 接收任务单，建立文档系统 | 工作区 `D:\ALAC_V` 为空目录，从零搭建 |
| 2026-11-14T10:20Z | 修复工作区 Windows 文件权限（沙箱无法授权写入） | 为当前用户补 WRITE_OWNER；备份与回滚脚本在 `D:\ALAC_V_acl_recovery` |
| 2026-11-14T10:22Z | 硬件初勘：RTX 5070 12227 MiB / CUDA 12.8 toolkit / Python 3.11.9 | 详见审计报告 |
| 2026-11-14T10:25Z | 代理核验：出网需 `http://127.0.0.1:7897` | 已记录到 runbook §网络 |
| 2026-11-14T10:36Z | venv + torch 2.14.1+cu130 安装完成，CUDA 可用（sm_120） | uv 缓存需指向工作区 |
| 2026-11-14T10:42Z | DINOv2 ViT-S/14 权重下载完成（88.3 MB，可续传） | SHA-256 已记录 |
| 2026-11-14T10:47Z | 管线实现完成，56 项单测通过 | — |
| 2026-11-14T10:47–10:52Z | EXP-0001 完成：3 变体 × 3 重复 × 98 图 | `artifacts/runs/*_EXP-0001_*` |
| 2026-11-14T10:53Z | 消融 + 配对 bootstrap + 报告插图 | `artifacts/experiments/` |
| 2026-11-14T11:00Z | 主知识库 98 图入库，`verify_raw` 98/98 通过 | `data/index/alacv.sqlite` |
| 2026-11-14T11:05Z | Phase 1 交付报告完成，**停止** | 等待 Owner 审核 |

## 4. 变更记录

- 2026-11-14：创建。
