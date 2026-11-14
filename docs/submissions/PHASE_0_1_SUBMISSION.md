---
id: SUBMIT-0001
title: Phase 0 + Phase 1 交付提交说明
status: 待验收
date: 2026-11-14
owner: lead
task: TASK-0001
repo: https://github.com/zcx369658780/ALAC-V
---

# Phase 0 + Phase 1 交付提交说明

本文件是 ALAC-V 首次公开仓库归档的提交说明，如实记录已完成内容、运行方式、测试与实验数据。
**不包含任何编造或推断的测试结果**；所有数字均可在本地 `artifacts/` 中重放生成。

---

## 1. 实际完成内容

### Phase 0 — 环境审计（已完成）

- 采集并核验：OS、CPU、RAM、GPU 型号与显存、NVIDIA 驱动、CUDA 兼容性、Python/PyTorch、磁盘、项目目录。
- 建立独立 venv（不修改系统全局依赖）。
- 实测并确认 RTX 5070（sm_120）上的 CUDA 推理与 FP16 混合精度可用：
  - FP16 2048×2048 矩阵乘 **54.33 TFLOPS**（FP32 为 19.09 TFLOPS）；
  - 1024×1024 FP16 矩阵乘最大绝对误差 0.070、相对误差 2.14e-3、无 NaN/Inf；
  - DINOv2 ViT-S/14 FP16 单次前向 **5.23 ms**，激活峰值 61.1 MB。
- 产出：审计报告 + 原始 JSON 证据 + 模型许可证登记。

### Phase 1 — 静态图像视觉编码（已完成）

| 能力 | 实现 | 状态 |
|---|---|---|
| 原始图像归档 | JPEG/PNG/WebP/AVIF；SHA-256 内容寻址；只读；元数据入库 | ✅ |
| 多尺度表示 | 1/1、1/2、1/4、1/8，WebP q95，可配置 | ✅ |
| Foveated ROI | 梯度能量选 ROI；1/2 画布分级马赛克；全分辨率 ROI 裁片；ROI 坐标入库 | ✅ |
| 视觉特征 | DINOv2 ViT-S/14，CLS(384) + Patch(37×37×384)，FP16，记录预处理与模型指纹 | ✅ |
| 视觉知识库 | SQLite 7 表 + SafeTensors；原始索引/多尺度/全局/局部/ROI/版本/元信息 | ✅ |
| 检索与回溯 | 余弦 Top-K、Recall@K/mAP、按 image_id+ROI 从原图取回高清局部 | ✅ |
| 三组对照实验 | A/B/C × 3 重复 × 98 图，7 类指标 + 消融 + 配对 bootstrap CI | ✅ |
| 单元测试 | 56 项，全部通过 | ✅ |

**明确未做**（遵守任务单边界）：没有训练/微调 DINOv2 主干；没有调用外部 LLM API；没有上传图像；没有下载大规模数据集；没有进入视频/Phase 2。

---

## 2. 代码文件位置

```
src/alacv/
  config.py        配置加载/校验/哈希/快照
  logging_utils.py JSONL 结构化日志
  ingest.py        只读内容寻址归档
  imaging.py       解码、色彩归一、金字塔
  foveation.py     ROI 选择与分级合成
  features.py      DINOv2 封装（CLS+Patch, FP16）
  store.py         SQLite + SafeTensors
  retrieval.py     余弦检索、回溯、指标
  metrics.py       计时/显存/CPU/RAM/PSNR/SSIM
  pipeline.py      变体 A/B/C 编排
  cli.py           命令行入口

scripts/
  audit_environment.py     Phase 0 审计
  fetch_model.py           权重下载（可续传）+ 许可证 + SHA-256
  make_dataset.py          DS-0001 程序化生成（无外部下载）
  run_experiments.py       A/B/C 对照实验
  eval_ablations.py        描述子消融（复用已存特征）
  bootstrap_ci.py          配对 bootstrap 置信区间
  make_figures.py          报告插图
  verify_raw_integrity.py  归档不可变性校验

tests/                56 项单元测试（9 个文件）
configs/              default.yaml / experiment_abc.yaml / dataset_ds0001.yaml
docs/                 文档系统（宪章、任务单、审计、ADR、实验、报告、运维、风险）
artifacts/experiments/ 实验汇总（summary / ablations / ci）
artifacts/figures/     报告插图
```

实验原始 run 目录（每 run 含 config 快照、日志、逐图指标与完整 store）体积较大，**未入库**，可由命令完整重放。

---

## 3. 环境配置

| 项 | 值 |
|---|---|
| OS | Windows 10 Pro 26H2，build 26300.9711，x64 |
| CPU | Intel Core Ultra 7 265K，20 核 / 20 线程 |
| RAM | 31.48 GB |
| GPU | NVIDIA GeForce RTX 5070，12227 MiB，CC 12.0 (sm_120) |
| 驱动 / CUDA | 驱动 616.92（UMD 13.4）；toolkit 12.8.93；PyTorch cu130（运行时 13.0）；cuDNN 92400 |
| Python | CPython 3.11.9（独立 `.venv`） |
| PyTorch | 2.14.1+cu130，`cuda.is_available()=True` |
| 关键依赖 | transformers 4.57.1、safetensors 0.8.0、numpy 2.4.6、pillow 13.0.0、psutil 7.2.2、pytest 9.2.0 |
| 完整依赖锁 | `requirements.lock.txt`（逐包精确版本） |

安装（Windows，需 CUDA 13.0 轮子源）：

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 torchvision==0.29.1 `
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe -e . --no-deps
```

---

## 4. 使用的模型：版本、来源与许可

| 字段 | 值 |
|---|---|
| 模型 | DINOv2 ViT-S/14（`facebook/dinov2-small`） |
| 上游 | Meta AI Research |
| 分发源 | HuggingFace `facebook/dinov2-small` |
| Revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| 参数量 | 22.06 M |
| 输入 / patch | 518×518 / 14（37×37 patch） |
| 特征维度 | 384（CLS 与 patch 共享） |
| 权重文件 | `model.safetensors`，88,249,960 字节 |
| 权重 SHA-256 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| 许可证 | **Apache-2.0**（全文见 `artifacts/models/dinov2_vits14/LICENSE`；仓库未收录权重） |
| 是否微调 | **否**，全程 `eval()` + `inference_mode()`，权重 SHA-256 与上游一致 |

**权重不入库**：通过 `scripts/fetch_model.py` 从官方渠道下载，并记录 revision 与 SHA-256。脚本对每个文件实现 Range 续传与重试（本机代理长连接不稳定）。

```powershell
python scripts/fetch_model.py
# 产物: artifacts/models/dinov2_vits14/{model.safetensors, config.json, LICENSE, SHA256SUMS}
#       data/manifests/model-dinov2_vits14.json
```

---

## 5. 实际运行的命令

以下命令均为本机真实执行过的命令（按执行顺序）：

```powershell
# 环境与审计
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe -e . --no-deps
python scripts/audit_environment.py

# 模型与数据
python scripts/fetch_model.py
python scripts/make_dataset.py            # 98 张（24 基准 + 72 查询 + 2 张本机真实图）

# 测试
python -m pytest -q                        # 56 项全部通过

# 建库（主知识库，98 图，变体 C）
python -m alacv.cli ingest --input data\datasets\ds-0001 --variant C --dataset-id ds-0001
python -m alacv.cli verify                 # 98/98 通过
python -m alacv.cli query --image data\datasets\ds-0001\query\scene_003_q1.jpg --variant C --topk 3
python -m alacv.cli backtrack --image-id fce4f5ac90eb9d2a --roi 0 --out artifacts\figures\roi_backtrack_demo.png

# 三组对照实验（9 runs：3 变体 × 3 重复）
python scripts/run_experiments.py

# 二次分析
python scripts/eval_ablations.py           # 描述子消融
python scripts/bootstrap_ci.py             # 配对 bootstrap 95% CI
python scripts/make_figures.py             # 报告插图
```

---

## 6. 测试结果

```
> .\.venv\Scripts\python.exe -m pytest -q
........................................................                 [100%]
```

- **56 项测试，全部通过**，0 failure、0 error、0 skip。
- 覆盖：配置校验与哈希、归档（格式/去重/只读/损坏拒绝/篡改检测/AVIF）、金字塔尺寸、Foveated ROI（能量峰值、ROI 命中、纯色兜底、裁片分辨率）、存储（schema、特征往返、去重、描述子归一、资产唯一性）、检索（最近邻、Recall/AP、相似度分离）、指标（PSNR/SSIM）、管线（A/B/C 全流程、幂等、拒绝）、DINOv2（形状、确定性、FP16/FP32 余弦一致性）。

> 说明：测试输出为 56 个通过点，未附带 `-v` 汇总行；该结果来自本机实际执行，未做任何修改。

---

## 7. 实验数据

**数据**：DS-0001，24 张 1024×768 程序化合成基准 + 72 条扰动查询（亮度/对比度/旋转/缩放/JPEG 重压缩）+ 2 张本机真实图；无外部下载。
**协议**：3 变体 × 3 重复，每 run 独立 store，处理顺序随机化，取中位数。预注册见 `docs/04_experiments/EXP-0001-abc-comparison.md`。

### 7.1 主结果

| 变体 | 维度 | 总时间 中位/P95 (ms) | 编码 (ms) | 显存峰值 (MB) | 特征/图 | 合计/图 | Recall@1 | Recall@5 | mAP@10 | 回溯 PSNR | 回溯 SSIM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A 原始分辨率 | 384 | 34.4 / 40.3 | 19.6 | 91.1 | 1.00 MB | 1.00 MB | 0.833 | 0.944 | 0.883 | 26.76 dB | 0.869 |
| B 多尺度 | 1536 | 216.2 / 258.9 | 63.8 | 91.1 | 4.02 MB | 4.10 MB | 0.875 | 0.986 | 0.918 | 29.19 dB | 0.852 |
| C 多尺度+ROI | 2304 | 361.3 / 412.0 | 97.4 | 91.1 | 6.02 MB | 6.14 MB | 0.889 | 0.972 | 0.922 | 29.10 dB | 0.851 |

### 7.2 消融（复用已存特征，无额外前向）

| 描述子 | 维度 | 特征/图 | Recall@1 | mAP@10 |
|---|---|---|---|---|
| A0 = L0 | 384 | 1.00 MB | 0.833 | 0.883 |
| B2 = L0+L2 | 768 | 2.01 MB | 0.819 | 0.886 |
| **C2 = foveated+ROI** | 768 | **2.01 MB** | **0.903** | 0.927 |
| B = L0–L3 | 1536 | 4.02 MB | 0.875 | 0.918 |
| **C3 = L2+L3+foveated+ROI** | 1536 | 4.02 MB | **0.931** | 0.953 |
| C = 6 源 | 2304 | 6.02 MB | 0.889 | 0.922 |

### 7.3 统计判定

| 假设 | 观测 | 判定 |
|---|---|---|
| B 优于 A | +4.2 pp，95% CI [0.0, +9.7] | 未达判定标准 |
| C 不劣于 B | +1.4 pp，CI [−5.6, +8.3] | 支持 |
| C 回溯优于 A | +2.34 dB | 支持 |
| C 比 B 省存储 | 1.50×（更大） | 拒绝 |

**结论**：三组准确率差异有限而代价差异显著，**不宣称效率提升**；同字节预算下 foveated+ROI 描述子精度更高（+5.6～8.4 pp）。结论受合成数据与 72 条查询限制，不外推自然图像。

---

## 8. 未完成内容

1. **自然图像基准未做**：无公开自然图像集的检索真值，准确率仅来自合成数据。
2. **ROI 语义化未做**：重要性仅用无监督梯度能量，未使用学习式显著性。
3. **视频 / 时序 / 光流**：属于 Phase 2+，未开始。
4. **ANN 索引**：当前为暴力余弦检索（Phase 1 规模足够），未引入向量库。
5. **多 ROI 编码**：实现了至多 2 个 ROI 的检测与裁片，但描述子只用 `roi_00`。
6. **Parquet 导出**：`pyarrow` 已安装，导出脚本未实现（非必需）。

---

## 9. 已知问题

| # | 问题 | 影响 |
|---|---|---|
| 1 | ROI 语义命中率 58.3%（梯度能量法） | C 的焦点可能落在非目标高纹理区 |
| 2 | 准确率仅在合成数据上测（72 查询） | CI 宽 ±5–8 pp，不能外推 |
| 3 | C 的 6 源定义不最优（6.14 MB 且低于 C3） | 下一阶段建议默认改 C3 |
| 4 | 金字塔 WebP 编码约占 100 ms/图 | B/C 端到端时间被非模型部分主导 |
| 5 | 回溯指标口径为"派生表示保真度" | 原始层对三组均永久保留，勿误读 |
| 6 | FP16 存储的余弦量化误差 ~1e-3 | 检索已用 FP32 累加规避 |
| 7 | Windows 计时抖动（P95 ≈ 1.15×中位） | 已用多次重复 + 中位数 |

完整清单见 `docs/07_risks/risk-register.md` 与 `docs/05_reports/PHASE1-REPORT.md` §8.1。

---

## 10. 本次提交 SHA

| 项 | 值 |
|---|---|
| 成果基线提交（本文件之前） | `178a7f3032303d7b8d6f8890e86baabf6d7b9e0e` |
| 提交说明 | `chore: redact local user paths before public release` |
| 更早历史 | `291e974`（Phase 1 全部成果）、`0f83806`（骨架与文档系统） |
| 本文件所在提交 | 紧随基线提交之后；文档无法记录自身 SHA（自引用），最终 HEAD 以仓库 `main` 的 `git ls-remote` 回读为准 |
| 推送目标 | `git@github.com:zcx369658780/ALAC-V.git` → `main` |

推送采用普通快进推送（`git push -u alacv HEAD:main`），**未使用 force push**，保留全部本地提交历史。

---

## 附：仓库安全自查

| 检查项 | 结果 |
|---|---|
| SSH 私钥 / `.pem` / `id_ed25519` | 未入库（`.gitignore` 覆盖；`git ls-files` 无匹配） |
| API Key / Token / `.env` | 未发现（正则扫描 `PRIVATE KEY`、`api_key`、`sk-`、`token=` 均无命中） |
| 用户私人图像 / 视频 | 未入库；`data/` 全部被忽略，仅保留清单与 `.gitkeep` |
| 未获许可的第三方数据集 | 仅程序化生成的 DS-0001（CC0），无第三方素材 |
| 大型模型权重 | 未入库（88 MB 权重由下载脚本获取） |
| 本地缓存 / 无关日志 | `.venv/`、`.uv-cache/`、`.hf-cache/`、`artifacts/runs/` 均忽略 |
| 用户绝对路径 / 账户信息 | 已脱敏：`C:\Users\<user>` → `%USERPROFILE%`，全仓库无残留 |
| 仓库体积 | 约 1.4 MB，最大单文件 936 KB（报告插图） |
