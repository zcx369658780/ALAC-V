---
id: ARCH-0001
title: ALAC-V Phase 1 系统架构
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0001
---

# Phase 1 系统架构

## 1. 设计约束

- 单机、单 GPU（RTX 5070 12 GB）、32 GB RAM、Windows。
- 输入是**静态图像文件**，不是视频帧、不是流。
- 主模型冻结：DINOv2 ViT-S/14（21M，384 维，patch 14）。
- 不引入向量库/服务/消息队列；元信息用一个 SQLite 文件，特征用 SafeTensors 文件。
- 一切派生数据可由"原始文件 + 配置"重放生成。

## 2. 分层视图

```
┌──────────────────────────────────────────────────────────────────────┐
│ L0  CLI / 编排层        alacv.cli  ·  scripts/run_pipeline.py        │
├──────────────────────────────────────────────────────────────────────┤
│ L1  管线层              ingest → imaging(pyramid) → foveation        │
│                         → features(DINOv2) → store → retrieval       │
├──────────────────────────────────────────────────────────────────────┤
│ L2  能力层              归档  金字塔  Foveated  编码  索引  检索     │
├──────────────────────────────────────────────────────────────────────┤
│ L3  模型层              Dinov2Model(frozen) + 预处理 + FP16           │
├──────────────────────────────────────────────────────────────────────┤
│ L4  存储层              data/raw(只读) · processed · features · index │
└──────────────────────────────────────────────────────────────────────┘
```

## 3. 数据流（单张图像）

```
原始文件 (JPEG/PNG/WebP/AVIF)
   │  ① ingest：解码校验 → SHA-256 → 元数据 → 只读归档（内容寻址）
   ▼
data/raw/<sha256 前2位>/<sha256>.<ext>        ← 只读，永不改写
   │  ② imaging：RGB 模式归一（不做 ICC 变换）→ 四档金字塔 (1/1, 1/2, 1/4, 1/8)
   ▼
data/processed/<image_id>/pyramid/L0..L3.webp
   │  ③ foveation：ROI 选择 → 分级合成 → ROI 坐标 + 高清 ROI 裁片
   ▼
data/processed/<image_id>/foveated/foveated.webp + roi/roi_00.webp + roi.json
   │  ④ features：每个表示 → CLS(384) + Patch(N×384)，FP16
   ▼
data/features/<image_id>/<variant>.safetensors
   │  ⑤ store：SQLite 元信息 + 特征文件路径 + 校验和
   ▼
data/index/alacv.sqlite
   │  ⑥ retrieval：查询特征 → 归一化余弦 → Top-K → 回溯原始高清
   ▼
结果 (image_id, score, roi bbox, 原始图路径)
```

## 4. 三种实验变体（A/B/C）

| 变体 | 输入给 DINOv2 的表示 | 存储 | 检索描述子 |
|---|---|---|---|
| **A** 原始分辨率 | 原图 resize 到 518×518（短边→518，中心裁） | 仅特征 | CLS(384)，L2 归一 |
| **B** 多尺度 | 4 个尺度各自 resize 到 518，独立前向 | 4 份特征 | 4×CLS 拼接(1536) 归一，或逐级融合 |
| **C** Foveated + 多尺度 | 多尺度 4 级 + foveated 合成图 + ROI 高清裁片 | 特征 + foveated + ROI | 多尺度 CLS + foveated CLS + ROI CLS 加权拼接 |

> 关键公平性：三组使用**同一模型、同一精度、同一 batch=1、同一 518 输入**；差异只在"输入表示"与"派生存储"。
> 因此 A 的耗时是 B/C 的下界参照，B/C 的额外收益必须用检索/回溯指标证明。

## 5. 模块职责

| 模块 | 职责 | 不负责 |
|---|---|---|
| `config.py` | 读取 YAML、校验、补默认值、算配置 SHA-256、快照 | 业务逻辑 |
| `ingest.py` | 格式识别、解码验证、SHA-256、EXIF/ICC **元数据记录**、只读归档、清单 | 缩放、特征、ICC 色彩变换 |
| `imaging.py` | RGB 模式归一（PIL convert / alpha 合成，**不做 ICC profile 变换**）、金字塔生成、可配置插值、尺寸对齐 | ROI 选择、ICC 色彩管理 |
| `foveation.py` | 重要性图 → ROI 选择 → 分级 paste 合成 → ROI 裁片与坐标 | 特征 |
| `features.py` | 模型加载、预处理、前向、CLS/Patch 提取、FP16 转换 | 存储路径 |
| `store.py` | SQLite schema、事务、SafeTensors 落盘/读取、完整性校验 | 检索算法 |
| `retrieval.py` | 描述子归一、余弦检索、ROI 回溯、指标计算 | 建库 |
| `metrics.py` | 时间/显存/CPU/RAM 采样、PSNR/SSIM | 业务判断 |
| `pipeline.py` | 按变体编排、幂等、断点续跑 | CLI 解析 |

## 6. 幂等与重放

- `image_id = sha256(原始字节)[:16]`；同一张图重复入库得到同一 ID，不重复计算。
- 派生文件写入前先写临时文件再原子重命名；SQLite 用 WAL + 显式事务。
- 配置变化（如插值算法）会使派生文件"过期"：`store` 记录 `config_sha256`，`--force` 才重算。

## 7. 失败模式

| 失败 | 行为 |
|---|---|
| 图像损坏/截断 | 记录 `status=rejected` + 原因，不归档，不中断批处理 |
| 不支持格式 | 同上，明确报出格式名 |
| CUDA OOM | 降 batch 到 1；仍失败则记录并跳过，不静默 |
| 磁盘不足 | 写入前检查阈值，低于阈值拒绝新任务 |
| 特征文件损坏 | 入库时写 SHA-256；读取时校验，失配则报错并标记需重建 |

## 8. 与 Phase 2 的接口（预留，不在本阶段实现）

- `VisualMemory` 抽象：`write(observation) -> id` / `recall(query, k)` / `replay(id)`。
- Phase 1 的 `store` 是其最底层实现（像素层 + 特征层）；Phase 2 增加"记忆层"而不改像素层。
