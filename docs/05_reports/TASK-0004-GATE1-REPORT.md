---
id: REPORT-0004
title: TASK-0004 Gate 0/1 交付报告（DS-0002 v2 整改）
status: 待审（Gate 2 第二轮）
date: 2026-11-15
owner: lead
task: TASK-0004
issue: https://github.com/zcx369658780/ALAC-V/issues/6
---

# TASK-0004 Gate 0/1 交付报告（v2 整改）

> 本版对应 PR #8 第二轮整改：P1-1 标签真实性、P1-2 训练/标签隔离、P1-3 全量 SHA 重算、
> P1-4 缓存完整性，以及 P2-1/2/3。**数据已全部重新生成（v2），v1 数据不再使用。**
> 未执行模型实现或任何 optimizer step；等待 Gate 2 独立验收。

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| 数据集 | `ds-0002-synthetic-v2`（生成器 `ds0002-v2`） |
| 场景 / 视图 | 3,200 / 22,000（train 2,000/16,000；val 300/1,500；test 900/4,500） |
| 图像 | 955,340,256 B（0.96 GB），1024×768 RGB；**全量重算 SHA：0 失配、0 字节失配、0 重复** |
| 分组隔离 | base seed / view_id 跨 split 交集均为 0 |
| 确定性 | 12/12 抽样一致（含 **6 个 strong augmentation** 视图） |
| 标签真实性 | 纹理/调色板/形状/遮挡均来自**实际绘图参数与几何** |
| 训练隔离 | Dataset 只读投影索引；无 `manifest` 属性；索引禁用字段 0 |
| 教师缓存 | 23 shards / 23.15 GB；**view 集合完全一致**；tensor keys/shape/dtype/计数全部校验；33 抽样 `max_abs_err=0.0` |
| 审计 | **PASS**（`gate1-audit.json`，失败即 exit 1） |
| 测试 | 97 项通过（完整日志见 `artifacts/evidence/DS-0002/test-runs/gate1-pytest.txt`） |

## 1. P1-1：标签与图像内容一致性

v1 的缺陷：纹理/调色板标签与实际绘制参数是两次独立抽样；`occlusion` 是随机布尔值且干扰块先于目标绘制。
v2 修复（`src/alacv/synth/scene.py`）：

| 标签 | v2 生成方式 |
|---|---|
| `texture_kind` | 先抽样，传入 `_texture(kind, ...)` 绘制，再原值写入标签 |
| `palette_family` | 先抽样，`_palette_colors(family)` 生成实际颜色，再原值写入 |
| `shape` | `rect / ellipse / triangle` 三选一，用真实 mask 合成，标签即该形状 |
| `occlusion` | 先画目标、后画干扰块；`occlusion_fraction = max(干扰块与主目标重叠面积 / 目标面积)`；`occlusion = fraction ≥ 0.05` |
| 宽高比 | `aspect ∈ [0.65, 1.55]`，记录真实 `w,h` |

数据集版本升级为 **v2**，全部 22,000 张图像、manifest、index、labels、教师缓存均已重新生成；
v1 文件已从本地删除，不进入 EXP-0002。

## 2. P1-2：训练与评估标签真正隔离

- 新增投影训练索引 `data/ds0002/index/{split}.json`，每 view 仅含
  `view_id / path / augment / pair_key`；`pair_key = "g"+sha256("ds0002-pair:"+scene_id)[:12]`，
  不可逆且不暴露 scene_id。
- `Ds0002Dataset` 只加载该索引；**不存在 `manifest` 属性**，也不持有 targets/factors/scene_id。
- B2 同场景配对由 `PairBatchSampler` 在采样层使用 `pair_key` 完成，模型输入仍只有图像。
- 负向测试：
  - `test_loader_rejects_index_with_label_fields`：索引含 `scene_id` 直接报错；
  - `test_loader_holds_no_label_manifest`：断言无 manifest 属性；
  - `test_loader_never_opens_eval_labels`：monkeypatch `open`，任何 `labels_*` 读取即失败。

审计结果：`index_violations = 0`，`item_keys = {augment, image, view_id}`，`has_manifest_attr = False`。

## 3. P1-3：22,000 张图像全量重算 SHA-256

审计不再信任 manifest：

- 逐文件重新计算 SHA-256，与 manifest 比对：**0 失配**；
- 逐文件 `stat().st_size` 与 manifest `bytes` 比对：**0 失配**；
- 用**实际 SHA** 做重复检测：**0 重复**；
- `sha256-inventory.txt` 由实际文件哈希重新生成（22,000 行）；
- 全量 `Image.verify()` + 二次解码：0 失败；尺寸集合 `{(1024,768)}`；模式全 RGB。

## 4. P1-4：教师缓存完整性与索引公开

- **完整 view 集合比对**（非仅数量）：三个 split 的 cache view 列表与 manifest 逐一相等（`view_set_equal=True`）。
- **每个 shard**：SafeTensors keys 必须为 `{cls, patch}`；dtype 必须 `float16`；
  shape 必须 `(n,384)` 与 `(n,1369,384)`；`n` 与索引一致；SHA-256 逐个校验。
- 重复 view_id → Reader 初始化即抛错；缺失 view → `KeyError`；SHA 失配 → `RuntimeError`；
  Reader **默认 `verify_sha=True`**，并采用最多 2 个 shard 的 LRU 缓存（P2 内存建议）。
- 公开脱敏索引元数据：`artifacts/evidence/DS-0002/cache-index/index_{split}.json`
  （含每 shard 的 SHA-256、view 映射、tensor shape/dtype、数量、模型 revision 与权重指纹）。
- 缺缓存时审计 `ok=False`（fail-closed），不会出现 "NOT RUN 但 PASS"。

## 5. P2 修正

| 编号 | 处理 |
|---|---|
| P2-1 | 确定性抽样覆盖 **strong augmentation**：每 split 2 场景 × (view 0, view 1)，共 12 个样本，其中 6 个 strong |
| P2-2 | 配置与实现对齐：`width/height/seed_base` 与生成器常量校验，不一致直接退出；`jpeg_quality` 保留在 `augment` 中并用于实际保存 |
| P2-3 | 完整 pytest 日志（HEAD / branch / dirty / 环境）提交到 `artifacts/evidence/DS-0002/test-runs/gate1-pytest.txt`（97 passed） |
| 内存建议 | Reader 已改为有界 LRU（默认 2 shards），避免训练时驻留全部 16 个 train shards |

## 6. 计数与指纹

| 项 | 值 |
|---|---|
| dataset_id | `ds-0002-synthetic-v2` |
| 配置 SHA-256 | `1bd8ca549c84b28f20eed2da1640ed58270c9f27a0664489900f2dec763049f1` |
| 生成脚本 SHA-256 | `ab0db849a793f55184426c3d8b87bf6706e5535a1db0b99d507abc23e8fafa3f` |
| 场景代码 SHA-256 | `124b53982ba81c7b5f619d91187a3396e932aff7705e594326622cf34c74157f` |
| train manifest | `781d20bfd0d425aecc10f8b442f6cda3fa96e46be1a435ca287a67b47b661fb5` |
| validation manifest | `91e7ec4f0cbc932619dcc20dc3a996d14da0f2a3e459f8cfa377c16030d64d1d` |
| test manifest | `52517963cdc874deea8cb0e228a2a2e7cd92ad869f0e597812b29ec190c5ff86` |
| 模型 | revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`；权重 `ae1e99fc…14be1`，Apache-2.0 |
| 缓存 | 23 shards，23,147,523,680 B，22,000 views，FP16 |

## 7. 资源

| 项 | 实测 |
|---|---|
| 生成 | train 1,103 s；validation 111 s；test 320 s（单进程） |
| 缓存构建 | 610 s |
| 审计 | 325 s（含 22,000 次 SHA 重算与全量解码） |
| 磁盘可用 | 958.75 GB（数据 + 缓存约 24.1 GB） |
| RAM / GPU | 31.48 GB / RTX 5070 12,199.4 MiB（构建时约 2 GB、34%） |

## 8. 可视抽样

`artifacts/evidence/DS-0002/sample-grid.png`：4×4 合成图像拼图，覆盖三种形状、不同纹理、
遮挡与场景组合，供独立复核直观检查。

## 9. 已知限制

1. 合成域结论不可外推自然图像（D-2A-3 已接受）。
2. 位置/尺度等因子标签用于评估探针，不进入训练；`pair_key` 不可逆但训练采样器仍可利用同场景关系（B2 方法所需）。
3. `occlusion_fraction` 为矩形重叠面积近似（目标形状非矩形时略高估），标签定义已如实写入 sidecar。
4. 生成单进程约 25 分钟（沙箱禁止 multiprocessing 管道），不影响确定性。

## 10. NOT RUN

模型实现、A1/A2/B2/C2/F0/R0、任何 optimizer step、EXP-0002 训练、新权重/自然图像下载、
Phase 2B/2C/3 —— 均未执行。

## 11. 复现

```powershell
python scripts/generate_ds0002.py --config configs/ds0002.yaml
python scripts/build_teacher_cache.py --shard-size 1000 --batch 8
python scripts/audit_gate1.py --cache-sample 30
python -m pytest -q
```

## 12. 停止声明

Gate 1 v2 已完成并停止。等待 ChatGPT 独立复核与 Owner Gate 2 批准；在此之前不进入模型实现与训练。
