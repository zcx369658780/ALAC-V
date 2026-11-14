---
id: AUDIT-0001
title: Phase 0 硬件与环境审计报告
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0001
evidence: docs/02_audit/environment-audit.json, artifacts/runs/20261114T105248Z_AUDIT/
---

# Phase 0：硬件与环境审计报告

> 所有数字来自 `scripts/audit_environment.py` 的实际执行，原始 JSON 见
> `docs/02_audit/environment-audit.json` 与 `artifacts/runs/20261114T105248Z_AUDIT/`
> （`environment.json`、`gpu_selfcheck.json`）。审计时间：2026-11-14T10:52Z。

## 0. 结论摘要

| 检查项 | 结果 |
|---|---|
| OS | Windows 10 Pro 26H2（build 26300.9711），x86-64 ✅ |
| CPU | Intel Core Ultra 7 265K，20 核 / 20 线程 ✅ |
| RAM | 31.48 GB 可用（标称 32 GB）✅ |
| GPU | NVIDIA GeForce RTX 5070，**12227 MiB ≈ 11.94 GiB** ✅（与"预计 12 GB"一致） |
| 驱动 / CUDA | 驱动 616.92，CUDA UMD 13.4，toolkit 12.8，PyTorch cu130 运行时 13.0 ✅ |
| 算力架构 | **sm_120（Blackwell）**，PyTorch `arch_list` 含 `sm_120` ✅ |
| Python / PyTorch | 3.11.9 venv + torch 2.14.1+cu130，`cuda.is_available()=True` ✅ |
| 混合精度 | FP16 matmul 无 NaN/Inf，相对误差 2.1e-3；DINOv2 FP16 前向 5.23 ms ✅ |
| 磁盘 | C: 247 GB 可用；D: 987 GB 可用 ✅ |
| 隔离性 | 独立 `.venv`，系统 Python 与全局包未改动 ✅ |
| **总判定** | **通过（GPU 可用、FP16 可用、可进入 Phase 1）** |

## 1. 操作系统

| 项 | 值 |
|---|---|
| 产品名 | Windows 10 Pro |
| 版本 | 26H2 |
| 构建 | 26300.9711 |
| 架构 | AMD64 |
| `platform.platform()` | `Windows-10-10.0.26300-SP0` |

说明：该构建号高于同期公开的 Windows 10 版本号（常见为 19045），本机为 26H2 / 26300 分支。审计只记录事实，不对系统来源做推断。**Windows 的进程调度与 Defender 实时扫描会引入计时抖动**，已在指标中用中位数与 P95 处理（见风险 R-09）。

## 2. CPU 与内存

| 项 | 值 | 说明 |
|---|---|---|
| 型号 | Intel(R) Core(TM) Ultra 7 265K | Arrow Lake，**无超线程** |
| 物理核 / 逻辑核 | 20 / 20 | 8 P-core + 12 E-core，32 线程的说法不成立 |
| 最大频率 | 3900 MHz | 任务单标称 265K |
| 内存总量 | 31.48 GB | 标称 32 GB，扣硬件保留 |
| 审计时可用 | 10.9 GB | 桌面环境已占用约 20 GB（浏览器、Docker、IDE 等） |

对设计的含义：32 GB RAM 中约 1/3 被桌面占用，**图像批处理必须流式、不可把整个数据集读进内存**；本实现逐图处理，RSS 峰值约 1.3–1.5 GB（见 Phase 1 报告）。

## 3. GPU 与显存核验（任务单要求"需实际核验"）

```
NVIDIA-SMI 616.92      KMD Version: 616.92      CUDA UMD Version: 13.4
GPU 0: NVIDIA GeForce RTX 5070   WDDM
       12227 MiB / 12227 MiB 总显存      PCIe 4.0 x16      250 W
       Compute Capability 12.0 (sm_120)
```

| 项 | nvidia-smi | PyTorch |
|---|---|---|
| 名称 | NVIDIA GeForce RTX 5070 | 同 |
| 显存总量 | 12227 MiB | 12199.4 MB（`total_memory`） |
| 计算能力 | 12.0 | `(12, 0)` |
| 显存占用（审计时） | 2783 MiB | — |
| SM 数量 | — | 48 |
| 架构支持 | — | `sm_75, 80, 86, 90, 100, 120` |

**核验结论**：任务单"预计 12 GB GDDR7"成立，实际可用约 **11.9 GiB**。但**桌面合成与其他应用常驻占用 2.6–2.8 GB**，实际可供模型使用的约 **9.2–9.4 GB**，这是 12 GB 卡做视觉实验必须计入的现实约束。

## 4. 驱动与 CUDA 兼容性

| 层 | 版本 | 兼容性判断 |
|---|---|---|
| NVIDIA 驱动 | 616.92 | 支持 CUDA 13.x（UMD 13.4） |
| CUDA UMD（驱动内） | 13.4 | 向后兼容 12.x / 13.0 运行时 |
| CUDA Toolkit（`nvcc`） | 12.8.93 | 仅用于编译；本项目不写自定义 CUDA kernel |
| PyTorch 运行时 | cu130（CUDA 13.0） | 与驱动匹配，`torch.cuda.is_available()=True` |
| cuDNN | 92400 | 已随 torch 轮子安装 |
| GPU 架构 | sm_120 | 轮子 `arch_list` 明确包含，**无 "no kernel image" 风险** |

**关键决策**：RTX 5070 为 Blackwell（sm_120），只有 CUDA 12.8+ 的轮子包含该架构。因此选用 `torch==2.14.1+cu130`（SJTU 镜像，见 runbook §1），而不是 PyPI 默认轮子（Windows 上为 CPU-only）。

## 5. Python 与依赖

| 项 | 值 |
|---|---|
| 解释器 | `D:\ALAC_V\.venv\Scripts\python.exe` |
| 版本 | CPython 3.11.9（MSC v.1938 64-bit） |
| 虚拟环境 | ✅ 独立 venv，`sys.prefix != base_prefix` |
| 全局 Python | 未安装 torch，未被本项目修改 |

核心依赖（完整锁文件见 `requirements.lock.txt`）：

| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.14.1+cu130 | 张量 / GPU / FP16 |
| torchvision | 0.29.1+cu130 | 图像变换算子 |
| transformers | 4.57.1 | `Dinov2Model` 加载与推理 |
| safetensors | 0.8.0 | 特征存储 |
| numpy | 2.4.6 | 数值与检索 |
| pillow | 13.0.0 | 解码 JPEG/PNG/WebP/AVIF |
| opencv-python-headless | 5.0.0.93 | 保留（本阶段核心路径未依赖） |
| pydantic | 2.14.3 | 配置校验辅助 |
| pytest | 9.2.0 | 单元测试 |
| psutil | 7.2.2 | CPU/RAM 采样 |
| matplotlib | 3.11.2 | 报告图表 |
| pyarrow | 26.0.1 | 可选 Parquet 导出 |

## 6. 磁盘

| 盘 | 总量 | 可用 | 用途 |
|---|---|---|---|
| C: | 930.34 GB | 247.03 GB | 系统、Python |
| D: | 1907.73 GB | 986.64 GB | **项目工作区 `D:\ALAC_V`** |

单次 Phase 1 实验的磁盘占用：每个 run 的 store 约 60–80 MB（96 张图 + 派生 + 特征），9 次运行合计约 0.6 GB。空间充裕。

## 7. GPU 功能自检（RTX 5070 + FP16）

### 7.1 FP16 数值正确性（1024×1024 矩阵乘，对照 FP32）

| 指标 | 值 |
|---|---|
| 最大绝对误差 | 0.0701 |
| 平均绝对误差 | 0.00905 |
| 相对误差（逐元素 / |ref|） | 2.14e-3 |
| NaN / Inf | 无 |

参考值量级为 O(32)（K=1024 的随机矩阵乘），0.07 的最大误差与 FP16 的 10 位尾数一致，**属于正常舍入，不是缺陷**。

### 7.2 吞吐（2048×2048 矩阵乘，50 次）

| 精度 | 单次耗时 | 等效 TFLOPS |
|---|---|---|
| FP16 | 0.316 ms | 54.33 |
| FP32 | 0.900 ms | 19.09 |

FP16 ≈ 2.8× FP32 吞吐 —— 混合精度在 RTX 5070 上**确实生效**（不是被降级到 FP32 模拟）。

### 7.3 DINOv2 ViT-S/14 实际前向（FP16，518×518，batch=1）

| 指标 | 值 |
|---|---|
| 单次前向 | **5.23 ms**（20 次均值，GPU 已预热） |
| 参数量 | 22.06 M |
| 权重体积 | 42.1 MB（FP16） |
| 激活峰值 | 61.1 MB |
| 输出形状 | `[1, 1370, 384]` = CLS + 1369 patch |
| NaN/Inf | 无 |

结论：**RTX 5070 的 CUDA 推理与混合精度运行正常**，单图单尺度前向 5.2 ms，12 GB 显存对 ViT-S/14 @ 518 完全无压力。

## 8. 兼容性结论

1. **可以**在本地 GPU 上以 FP16 运行 DINOv2 ViT-S/14，吞吐与显存均满足 Phase 1 需求。
2. **必须**把"桌面常驻占用 2.6–2.8 GB"计入显存预算；本阶段 batch=1 时激活仅 ~61 MB，余量充足。
3. **不得**假设多卡/大显存：后续阶段若要更大模型或更大 batch，需显式评估 12 GB 上限。
4. Windows 计时抖动明显（同一图 p95 约为中位数 1.15 倍），所有性能结论均以中位数 + P95 呈现。

## 9. 复现命令

```powershell
cd D:\ALAC_V
$env:HTTPS_PROXY='http://127.0.0.1:7897'; $env:HTTP_PROXY=$env:HTTPS_PROXY
$env:UV_CACHE_DIR='D:\ALAC_V\.uv-cache'
.\.venv\Scripts\python.exe scripts\audit_environment.py
```

输出：
- `artifacts/runs/<UTC>_AUDIT/environment.json`
- `artifacts/runs/<UTC>_AUDIT/gpu_selfcheck.json`
- `docs/02_audit/environment-audit.json`

## 10. 与任务单的对照

| 任务单要求 | 完成情况 |
|---|---|
| 操作系统与版本 | §1 |
| CPU、RAM、GPU 型号和显存 | §2、§3（显存实测 12227 MiB） |
| NVIDIA 驱动及 CUDA 兼容情况 | §4 |
| Python、PyTorch 环境 | §5 |
| 硬盘空间 | §6 |
| 本地开发环境与项目目录 | §5、§6、README |
| 独立虚拟环境、不修改全局 | §5（venv 验证 `venv_active=True`） |
| 确认 CUDA 推理与混合精度正常 | §7（FP16 前向 5.23 ms，无 NaN） |
| 记录完整配置与依赖版本 | 本文 + `requirements.lock.txt` + JSON 证据 |
