---
id: REPORT-0004
title: TASK-0004 Gate 0/1 交付报告（DS-0002 v3）
status: 待审（Gate 2 第三轮）
date: 2026-11-15
owner: lead
task: TASK-0004
issue: https://github.com/zcx369658780/ALAC-V/issues/6
---

# TASK-0004 Gate 0/1 交付报告（v3）

> 本版对应 PR #8 第三轮整改：shape-aware 遮挡、训练侧隐去 scene identity、JPEG 配置生效、
> 文件系统/缓存 fail-closed 审计与测试日志。数据全部重新生成（v3），v2 废弃。
> 未执行模型实现或任何 optimizer step；等待 Gate 2 验收。

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| 数据集 | `ds-0002-synthetic-v3`（生成器 `ds0002-v3`） |
| 场景 / 视图 | 3,200 / 22,000（train 2,000/16,000；val 300/1,500；test 900/4,500） |
| 图像 | 938,689,519 B；**全量重算 SHA：0 失配、0 字节失配、0 重复**；文件系统/清单/索引计数均 22,000 |
| 遮挡标签 | **按目标真实 mask 逐像素计算**，干扰块不透明；抽样 9 个场景复算误差 ≤5e-4 |
| 分组隔离 | base seed 跨 split 交集 0 |
| 训练身份隔离 | view_id/path 不含场景编号；salt 仅存在于 eval 侧；索引不含 salt |
| 确定性 | 12/12 抽样一致（6 个 strong augmentation） |
| 教师缓存 | 23 shards / 23.15 GB；完整 view 集合、tensor、计数、SHA 校验；33 抽样 `max_abs_err=0.0` |
| JPEG 配置 | strong 视图 q∈[60,84]、gallery q=92，全部符合配置 |
| 测试 | 97 passed（完整日志已锚定干净提交） |
| 审计 | **PASS**（任一检查失败即 exit 1） |

## 1. P1-1：shape-aware 遮挡（本轮核心修正）

v2 的缺陷：`occlusion_fraction` 用矩形 bbox 重叠近似；目标形状为 triangle/ellipse 时，
分母是 bbox 面积而非真实形状面积；干扰块半透明，且"遮挡"并非物理遮挡。

v3 定义（`src/alacv/synth/scene.py`）：

1. 每个目标按真实形状生成 mask（rect / ellipse / triangle），逐像素写入画布；
2. 目标按顺序绘制，**后画的目标会遮住先画的目标**；
3. 干扰块在目标之后绘制，且 **alpha = 1.0（不透明）**，是物理遮挡；
4. 遮挡分数：

\[
O = 1 - \frac{\text{可见目标 mask 像素数}}{\text{目标原始 mask 像素数}}
\]

5. `occlusion = (O ≥ 0.05)`，`occlusion_definition` 字段写明公式与"不透明遮挡物"前提。

`Scene.masks` 保留每个目标的完整 mask，审计可独立复算；抽样 9 个场景（train/val/test × 3）
复算结果与标签最大误差 ≤5e-4，审计 `occlusion.bad_count = 0`。

> 说明：若将干扰块改为半透明，应把指标改名为"半透明覆盖"而非物理遮挡；本版选择不透明以保持定义清晰。

## 2. P1-2：训练侧不再暴露 scene identity

- **opaque view_id**：`v<sha256(salt|split|index|view)[:16]>`，不含任何场景编号。
- **opaque path**：`images/<view_id[:2]>/<view_id>.jpg`，路径同样不含编号。
- **salt**：随机 16 hex，保存在 `data/ds0002/eval/identity_salt.json`（eval-only 目录），
  **不写入训练索引**；审计验证 `salt_in_index = False`。
- **pair_key**：`g<sha256(salt|pair|scene_id)[:16]>`，仅供采样器分组；训练索引与模型输入均不含 scene_id。
- 诚实说明：对**只读取训练索引**的代码，view 身份是 opaque；但 eval 侧的 salt 与 labels 一起公开在仓库中，
  任何读者都能据此重放映射。因此这不是密码学匿名化，也不被这样描述。
- 审计逐条检查：索引中 view_id/path 不匹配 `(train|validation|test)_\d{5}`，`identity.issue_count = 0`。

## 3. P1-3：22,000 张图像全量重算（并补齐文件系统闭环）

- 逐文件重算 SHA-256 与 `stat().st_size`：**0 失配、0 字节失配**；
- 用实际 SHA 检测重复：**0 重复**；
- `Image.verify()` + `Image.load()` + 二次解码：0 失败；尺寸全 1024×768、模式 RGB；
- 计数闭环：文件系统 22,000 = manifest 22,000 = index 22,000；
- `sha256-inventory.txt` 由实际哈希重建（22,000 行）。

## 4. P1-4：教师缓存 fail-closed 审计

- 每个 split 的 cache view 列表与 manifest **逐项相等**（`view_set_equal=True`）。
- 每个 shard：keys `{cls, patch}`、dtype `float16`、shape `(n,384)` / `(n,1369,384)`；
  `n` 与索引一致；`n_shards` 与 `rows` 计数一致；SHA-256 全部通过。
- Reader：默认 `verify_sha=True`；重复 view_id 初始化即抛错；缺失 view 抛 `KeyError`；
  有界 LRU（2 shards）避免训练时占满内存。
- 缓存缺失时审计 `ok=False`（fail-closed）。
- 公开脱敏索引：`artifacts/evidence/DS-0002/cache-index/index_{split}.json`。

## 5. P2 修正

| 编号 | 处理 |
|---|---|
| P2-1 | JPEG 配置真正生效：`augment()` 接收 `q_gallery` 与 `q_strong_range`；审计验证 strong ∈[60,84]、gallery=92（bad=0） |
| P2-2 | 审计新增 `im.load()`、文件系统计数、manifest/index 数组长度一致性 |
| P2-3 | 完整 `pytest -v` 日志（HEAD/branch/dirty/环境）提交到 `evidence/DS-0002/test-runs/gate1-pytest.txt`；缓存检查补充 shard 数与 row 数一致 |

## 6. 指纹

| 项 | 值 |
|---|---|
| dataset_id | `ds-0002-synthetic-v3` |
| 配置 SHA-256 | `4945c95c866230b022d74c87e5642db4b7a5464e13ac6c86013570505ea80cb2` |
| 生成脚本 SHA-256 | `4dbaaee27f00e1777ada5fc1fab10a5a76da47f6f322a4c3bcf43f87f34bb876` |
| 场景代码 SHA-256 | `6bf4fbdee66a72c3e1ec848419c25ea6c8e2e85f3928fed812a10820f117fa9f` |
| train manifest | `6083576f23c439b85ad2bc8a519df06951bb648da104bef709f51033c5528df2` |
| validation manifest | `9a266fc8387a7670a403b701766c02968123fb15ed7694a9e4ecd4055d98d44a` |
| test manifest | `5f640ce70c24e4e3b0960bcf78eb573936e8f165d5b4cba0fc40debe4d9b3d2e` |
| 模型 | revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`；权重 `ae1e99fc…14be1`；Apache-2.0 |
| 缓存 | 23 shards，23,147,523,680 B，22,000 views，FP16 |

## 7. 资源

| 项 | 实测 |
|---|---|
| 生成 | train 1,079 s；validation 100 s；test 309 s |
| 缓存构建 | 626 s |
| 审计 | 375 s（22,000 次 SHA 重算 + 全量解码 + 33 抽样重编码） |
| 磁盘可用 | 958.53 GB（数据 + 缓存约 24.1 GB） |
| RAM / GPU | 31.48 GB / RTX 5070 12,199.4 MiB |

## 8. 可视抽样

`artifacts/evidence/DS-0002/sample-grid.png`：4×4 拼图，覆盖三种形状、不同纹理、真实遮挡。

## 9. 已知限制

1. 合成域结论不可外推自然图像（D-2A-3 已接受）。
2. opaque ID 由 eval 侧 salt 派生：训练索引本身不含 scene 编号，但 eval 侧 salt/labels 公开，
   读者可据此重放映射；这是"训练侧隔离"而非密码学匿名。
3. 遮挡只建模"不透明遮挡物"；半透明覆盖不作为物理遮挡，若未来需要应另立标签。
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

Gate 1 v3 已完成并停止。等待 ChatGPT 独立复核与 Owner Gate 2 批准；在此之前不进入模型实现与训练。
