---
id: REPORT-0004
title: TASK-0004 Gate 0/1 交付报告 — DS-0002 生成与冻结教师缓存
status: 待审（Gate 2）
date: 2026-11-15
owner: lead
task: TASK-0004
issue: https://github.com/zcx369658780/ALAC-V/issues/6
---

# TASK-0004 Gate 0/1 交付报告

> **范围**：Gate 0（同步与冻结）+ Gate 1（DS-0002 生成 + 冻结 DINOv2 教师特征缓存）。
> **未执行**：模型实现、任何 optimizer step、A1/A2/B2/C2 训练、新权重/自然图像下载。
> 本报告是 DSH 自报，**不构成 Gate 2 验收**；等待 ChatGPT 独立复核与 Owner 批准。

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| Gate 0 基线 | `aaa5a75e3ffb44096c22fcf81b21413d2828ef17`（与 GitHub `main` 一致） |
| 场景 / 视图 | **3,200 scenes / 22,000 views**（train 2,000/16,000；val 300/1,500；test 900/4,500） |
| 图像 | 970,362,145 B（**0.97 GB**），全部 1024×768 RGB，解码失败 0，重复 0 |
| 分组隔离 | train/val/test 的 base seed、view_id、图像 SHA-256 交集均为 **0** |
| 确定性 | 9/9 抽样重生成字节一致 |
| 评估标签 | 3 个 split 的 sidecar 完整；bbox 合法；gallery/query 计数正确 |
| DataLoader 白名单 | 实际返回键 = `{view_id, image, augment}`，禁用字段 0 |
| 教师缓存 | 23 shards、22,000 views、**23.15 GB**、FP16、`[1369,384]`；60 个抽样 view 的 `max_abs_err = 0.0` |
| 模型身份 | revision `ed25f3a3…`、权重 SHA-256 `ae1e99fc…`（与 Phase 1 一致，未下载） |
| 审计总判定 | **PASS（DSH 自检）** |

## 1. Gate 0：基线与冻结

- 同步 `alacv/main` → `aaa5a75e3ffb44096c22fcf81b21413d2828ef17`；工作区干净，无未提交修改。
- 分支：`task/6-phase2a1-ds0002-generation`（基于上述 main）。
- 冻结文件：`docs/01_tasks/TASK-0004-gate0-freeze.json`
  - 设计 SHA-256 `e88abadab34670209747832904cca8ae57ff2e626ac401c4f6dcdde12c2980de`
  - 预注册 SHA-256 `76b20389f25c768c4d22931004f10e695d08946baa397aa26ff73dddacfa8cd2`
  - seeds `{2026,2027,2028}`、mask 821、参数上限 10M、最大 21 runs 等均固化。
- 环境：磁盘可用 981.4 GB（开始前）、RAM 31.48 GB、RTX 5070 12,227 MiB。
- 模型：本地权重 SHA-256 与 `data/manifests/model-dinov2_vits14.json` 完全一致；未联网。

## 2. DS-0002 规格与生成

### 2.1 计数（实测）

| Split | Scenes | Views | gallery | query | Seed 范围 | 生成耗时 |
|---|---:|---:|---:|---:|---|---:|
| train | 2,000 | 16,000 | — | — | 0–1,999 | 975 s |
| validation | 300 | 1,500 | 300 | 1,200 | 100,000–100,299 | 115 s |
| test | 900 | 4,500 | 900 | 3,600 | 200,000–200,899 | 325 s |
| **合计** | **3,200** | **22,000** | 1,200 | 4,800 | 互斥 | 1,415 s |

- 分辨率 1024×768；每场景 1–3 个纹理目标 + 2–4 个干扰块，含位置/尺度/遮挡/纹理/调色板/光照变化。
- 增强：裁剪缩放、旋转 ±10°、亮度/对比度/饱和度、JPEG 重压缩（训练视图 q60–85；gallery q92）。
- 图像总量 0.97 GB（低于预估 2.8 GB：强增强视图的重压缩显著减小文件）。

### 2.2 生成器与指纹

| 项 | 值 |
|---|---|
| 生成器 | `scripts/generate_ds0002.py`，`ds0002-v1` |
| 场景代码 | `src/alacv/synth/scene.py` SHA-256 `22739391…c36a7` |
| 生成脚本 | SHA-256 `7f08d937…38a1` |
| 配置 | `configs/ds0002.yaml`，规范化 SHA-256 `86fe54f9…122c` |
| PRNG | `numpy.random.Generator(PCG64)` |
| 种子派生 | `base_seed = split_base + index`；`view_seed = base_seed*1000 + view` |
| 确定性 | 9/9 抽样（每 split 3 个场景）重渲染 + 重增强后字节完全一致 |

### 2.3 Manifest 与评估标签

- `data/ds0002/manifests/{train,validation,test}.json`：逐 view 的 `view_id / scene_id / base_seed /
  role / path / sha256 / bytes / width / height / augment`。
- `data/ds0002/manifests/ds0002.json`：全局计数、seed 范围、生成器/配置指纹。
- `data/ds0002/eval/labels_{split}.json`：scene 级 targets、factors、gallery/query view 列表；
  文件头明确标注 **evaluation-only**。
- Manifest SHA-256：train `394747cc…b1fd7`；validation `0407fc4a…b0a`；test `5dcb2826…6573`。

### 2.4 隔离证明

审计直接计算集合交集：

| 交集 | 数量 |
|---|---:|
| train ∩ validation base seeds | 0 |
| train ∩ test base seeds | 0 |
| validation ∩ test base seeds | 0 |
| 跨 split view_id | 0 |
| 跨 split 图像 SHA-256 | 0 |

seed 分段由构造保证（0–1999 / 100000–100299 / 200000–200899），同一 base scene 的所有视图只属于
一个 split。

### 2.5 完整性与重复

- 22,000 张全部通过 `Image.open().verify()` + 二次解码；尺寸集合 `{(1024,768)}`；模式全 RGB。
- 解码失败 0；尺寸不符 0；重复 SHA-256 **0**。

## 3. 教师特征缓存

| 项 | 值 |
|---|---|
| 模型 | DINOv2 ViT-S/14，revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| 权重 | `model.safetensors`，SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| 许可 | Apache-2.0；未下载、未微调 |
| 格式 | SafeTensors，`cls float16[384]` + `patch float16[1369,384]`，每 shard 1,000 views |
| 分片 | train 16、validation 2、test 5，共 **23 shards / 23.15 GB** |
| 对齐 | 60 个抽样 view 重新编码，`max_abs_err = 0.0`（FP16 逐位一致） |
| 索引 | `index_{split}.json` 记录 view_ids 顺序、shard SHA-256、模型与 manifest 指纹 |
| 完整性 | 23/23 shard SHA-256 校验通过；缺失/重复 view 0 |
| 构建耗时 | 622 s（含 22,000 次冻结前向与 23 GB 落盘） |

缓存读取实现 `src/alacv/teacher_cache.py` 为 **fail-closed**：缺 index、缺 shard、SHA 不符、
形状/dtype 不符、view 不存在均抛错。

## 4. DataLoader 字段白名单审计

`src/alacv/ds0002.py::Ds0002Dataset` 的 `__getitem__` 返回键被限制为：

```
{"view_id", "image", "augment"}
```

审计结果：实际键 = 允许键，禁用字段（`scene_id / base_seed / targets / factors / role`）出现 0 次。
同场景配对（B2 需要）由 `PairBatchSampler` 在**采样层**完成，不把 scene_id 传入模型输入。
评估标签只由 `load_eval_labels()` 读取，训练代码禁止调用（测试覆盖）。

## 5. 资源使用

| 项 | 实测 |
|---|---|
| 磁盘（审计时可用） | 958.81 GB（数据集 + 缓存共约 24.1 GB） |
| RAM | 总量 31.48 GB；缓存构建峰值进程 RSS 约 1.5 GB |
| GPU | RTX 5070，12,199.4 MiB；缓存构建时占用约 2.0 GB、利用率约 34% |
| 生成总耗时 | 1,415 s（3 个 split 串行） |
| 缓存构建 | 622 s |

## 6. 隐私与许可证

- 数据为程序化生成，**无第三方素材**；数据集以 CC0 处理。
- 未下载任何新权重/自然图像；未使用用户私人图像；未调用外部服务。
- 提交内容不含图像、特征缓存、模型权重、密钥或用户绝对路径；`data/ds0002/*` 已在 `.gitignore`。
- 仓库扫描：用户名/`C:\Users` 0 命中（见审计与 PR 说明）。

## 7. 全量 SHA-256 inventory

`artifacts/evidence/DS-0002/sha256-inventory.txt`：22,000 行 `sha256  path`，与三个 manifest 一一对应。

## 8. 已知限制

1. 合成域结论不可外推自然图像（D-2A-3 已接受该边界）。
2. 强增强视图的 JPEG 质量随视图变化（q60–85），这是预注册的增强设计，不是压缩伪影缺陷；
   gallery 使用 q92。
3. 训练/验证/测试的 `role` 在 manifest 中可见（审计需要），但 DataLoader 不返回该字段。
4. 生成是单进程完成的（沙箱限制 multiprocessing 管道），总耗时约 24 分钟；不影响确定性。

## 9. NOT RUN

- 模型实现（`src/alacv/ssl/`）、A1/A2/B2/C2/F0/R0 代码：**未开始**；
- 任何 optimizer step / 训练 / 微调：**未执行**；
- 新模型权重或自然图像下载：**未执行**；
- EXP-0002 训练运行：**未执行**（Gate 4 未授权）；
- Phase 2B / 2C / 3：**BLOCKED**。

## 10. 复现命令

```powershell
python scripts/generate_ds0002.py --config configs/ds0002.yaml --workers 1
python scripts/build_teacher_cache.py --shard-size 1000 --batch 8
python scripts/audit_gate1.py --cache-sample 20
```

## 11. 停止声明

Gate 1 已完成并停止。未进入模型实现、未执行任何 optimizer step、未开始 A1/A2/B2/C2 训练。
等待 ChatGPT 独立复核与 Owner 对 Gate 2 的批准。
