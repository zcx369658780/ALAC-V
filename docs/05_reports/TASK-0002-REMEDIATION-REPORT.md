---
id: REPORT-0002
title: TASK-0002 / ALAC-0001 整改交付报告
status: 待审
date: 2026-11-14
owner: lead
task: TASK-0002
issue: https://github.com/zcx369658780/ALAC-V/issues/1
baseline: 2bc71dfba3d6ccebcda44124a324f697ba2f0b1d
branch: task/1-phase0-1-remediation
pr: https://github.com/zcx369658780/ALAC-V/pull/2
---

# TASK-0002 整改交付报告

**任务**：ALAC-0001 / Phase 0/1 独立验收整改
**基线**：`2bc71dfba3d6ccebcda44124a324f697ba2f0b1d`
**分支**：`task/1-phase0-1-remediation`
**PR**：[#2](https://github.com/zcx369658780/ALAC-V/pull/2)
**状态**：已提交 PR，等待 ChatGPT 独立复核与 Owner 最终验收；**Phase 2 仍 BLOCKED**。

---

## 1. 九次历史 run 证据盘点（WP0）

完整清单：`artifacts/evidence/EXP-0001/inventory.json` / `inventory.md`。

| 项目 | 结果 |
|---|---|
| 发现 run 数 | **9**（3 变体 × 3 重复） |
| 必需文件完整性 | 9/9 完整（`metrics.json`、`config.snapshot.yaml`、`config.sha256`、`env.json`、`model.json`、`items.jsonl`） |
| 配置哈希可复算 | 9/9：由 `config.snapshot.yaml` 重新规范化计算得到与 `config.sha256` 完全一致的 SHA-256 |
| 代码 commit 锚定 | **8/9** 记录 `0f83806303e5aeb18f8e4ac8798ddc5931f9817c`，`git_dirty=false` |
| 代码身份无法锚定 | **1/9**：`20261114T104749Z_EXP-0001_C_r2` 启动于首次提交之前（`env.json` 的 `git_commit` 为空），只能推断代码与 `0f83806` 相同，**未伪造 commit** |
| 数据清单 | 公开 LF 规范化 SHA-256 `e604b3ffe6ab39e3fbcf71703aa87fa95a057255011db16493bb395055ada6e7`（GitHub 可复算）；运行期原始字节（CRLF）`596fcb3c…` 仅存档，不混用 |
| 模型指纹 | `facebook/dinov2-small`，revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`，权重 SHA-256 `ae1e99fce…14be1` |
| 原始 run 目录 | **未修改**（只读盘点；公开副本另存） |

各 run 的配置哈希互不相同，原因是 `paths.*` 为每个 run 独立的绝对路径；公开快照已脱敏为
`<RUN_STORE>` 占位符，并同时保留原始哈希与公开快照哈希，二者用途不同。

## 2. 脱敏证据包（WP1）

发布位置：`artifacts/evidence/EXP-0001/<run-id>/`（9 套，共 74 个文件，约 0.61 MB）

每套包含：

| 文件 | 说明 |
|---|---|
| `metrics.json` | 原样复制，数值未改动 |
| `config.sha256` | 原始配置哈希（未改动） |
| `config.snapshot.public.yaml` | 路径脱敏后的配置快照 |
| `config.snapshot.public.sha256` | 公开快照自身哈希 |
| `env.public.json` | 环境（脱敏 `D:\ALAC_V` → `<REPO>`） |
| `model.public.json` | 模型指纹（revision / 权重 SHA / 许可证） |
| `items.public.jsonl` | 逐图明细；2 张本机真实图为稳定匿名 ID |
| `evidence-manifest.json` | 源文件清单与 SHA-256、脱敏映射、数据/模型指纹、代码 commit 与 dirty 状态、原始命令 |

另附 `inventory.json`、`inventory.md`、`README.md`（数据集与复现说明）。

**未发布**：原始图像、完整 store、SQLite 数据库、模型权重、缓存、密钥、用户名。
对证据目录的隐私正则扫描（用户名、`C:\Users`、`D:\ALAC_V`、密钥前缀）结果为 **0 命中**。

## 3. A 变体中心裁剪修复（WP2）

缺陷：`recover()` 的 A 分支未应用 `fit_to_input()` 的 `(nw-518)//2, (nh-518)//2` 中心裁剪偏移，
缩放尺寸用截断而非四舍五入，也未排除编码器不可见区域。

修复：

- 新增 `src/alacv/backtrack.py`：`fit_geometry` / `visible_region` / `reconstruct_from_visible` /
  `recover_variant`，与 `imaging.fit_to_input()` 共用同一几何函数（单一事实来源）。
- `scripts/run_experiments.py::recover()` 改为委托 `recover_variant`；
  `retrieval.recovery_from_variant()` 同步修正。
- 新指标在 **共同可见交集** 上比较 A/B/C，记录 `coverage`、`excluded`。

### 修复前后对照（24 张基准图 × 3 次重复，中位数）

**主指标（预注册口径：GT bbox 区域）**

| 变体 | 原始运行 PSNR | 第一轮（ROI 区域，已作废） | **第二轮（GT bbox，现行）** | 相对 A |
|---|---|---|---|---|
| A | 26.76 | 29.13 | **31.66** | — |
| B | 29.19 | 29.43 | **31.05** | −0.62 dB |
| C | 29.10 | 29.35 | **28.53** | **−3.14 dB** |

- 有效样本 24/24；**排除 0**；GT 区域平均可见覆盖率 **0.928**。
- **H3 被拒绝**：C 在真实目标区域反而比 A 低 3.14 dB。原「C 优于 A」的结论来自评价区域错误
  （使用了模型自己选中的 ROI）。机制：ROI_00 命中率 58.3%、任一已存裁片命中 66.7%、
  两个裁片并集仅覆盖 GT 面积 47.7%，C 仍有约一半目标区域依赖 1/2 画布马赛克。
- 副指标（attended ROI，探索性，不替代主指标）：A 29.44 / B 29.43 / C 29.35 dB。

### 坐标与评价口径修正（第二轮复核 R9/R10）

1. **评价区域**：`scripts/recompute_backtrack.py` 改为读取 manifest 的 `gt_bbox`；
   `recover_gt_region()` 与 `recover_variant()`（attended ROI）严格分离命名。
2. **C 的 GT 回溯规则**：ROI 裁片覆盖处用全分辨率像素，其余用 foveated 马赛克上采样，
   逐像素合成（`_c_full_canvas`），不因 ROI 未命中而只评价 ROI 内部。
3. **逐轴缩放**：`FitGeometry` 新增 `x_scale = resized_w/W`、`y_scale = resized_h/H`；
   可见区域逆映射与 A 重建均使用逐轴比例，消除 1 px 偏差；新增回归测试
   `test_per_axis_scales_differ_and_cause_1px_shift` 与 `test_reconstruction_matches_per_axis_mapping`。
4. **复算 provenance**：`EXP-0001-backtrack-corrected.json` 记录 9 个 run_id、脚本 SHA-256、
   git commit/dirty、执行命令、环境、九次配置哈希与参数摘要（已验证九次参数一致）、
   数据清单 LF 哈希、模型指纹；配置直接取自各 run 的 `config.snapshot.yaml`，不依赖当前配置文件。

## 4. 新增测试与完整输出（WP2/WP3）

新增 `tests/test_backtrack.py`（19 项）与 `tests/test_foveation.py` 中 2 项算子测试：

| 测试 | 覆盖 |
|---|---|
| `test_fit_to_input_matches_geometry[5 种尺寸]` | 几何函数与 `fit_to_input` 逐像素一致 |
| `test_geometry_values_landscape/portrait` | 缩放、取整、偏移的精确值 |
| `test_visible_region_center_is_full` | ROI 全可见 |
| `test_visible_region_partial_at_crop_edge` | 部分可见（覆盖率 0.375） |
| `test_visible_region_fully_outside` / `portrait_outside` | 完全不可见 |
| `test_reconstruct_raises_on_invisible` | 不可见不得伪造像素 |
| `test_constant_image_reconstruction_is_lossless` | 重建数值正确性 |
| `test_full_resolution_reconstruction_is_identity` | B/C 资产不做无谓重采样 |
| `test_reconstruction_uses_518_scale_not_original` | A 必须来自 518 输入 |
| `test_gradient_operator_matches_central_difference` | 逐像素核对中心差分，并证明 ≠ Sobel |
| `test_gradient_energy_zero_on_flat` | 平坦图边界 |
| `test_per_axis_scales_differ_and_cause_1px_shift` | 逐轴比例导致 1 px 偏移的回归 |
| `test_reconstruction_matches_per_axis_mapping` | 重建与逐轴映射逐像素一致 |
| `test_c_gt_region_composes_roi_crop_over_mosaic` | C 的 GT 区域合成规则 |
| `tests/test_manifest_hash.py`（3 项） | 公开 manifest 的 LF 哈希规范、无 CRLF、git blob 一致 |

完整测试输出（锚定到干净提交 `04d66d1`，`git_dirty=false`）：`artifacts/evidence/test-runs/2026-11-14-task-0002-pytest.txt`

```
============================= 83 passed in 9.43s ==============================
```

（原有 56 项 + 新增 27 项；0 failure / 0 error / 0 skip。）

## 5. ADR-0008（WP3）

[docs/03_design/decisions/ADR-0008-roi-central-difference.md](../03_design/decisions/ADR-0008-roi-central-difference.md)

- 保留中心差分实现，**未替换为 Sobel**，实验输入不变。
- 精确定义：`gx=I[:,2:]-I[:,:-2]`、`gy=I[2:,:]-I[:-2,:]`，`M=√(gx²+gy²)`，归一化。
- ADR-0007 原文保留，仅增加 `superseded_by` 与更正说明，不重写决策正文。
- 代码 docstring 与报告用语同步更正。

## 6. ICC 与色彩处理声明（WP4）

未新增任何色彩变换。文档已统一为：

> 管线只做 **PIL 模式转换**（统一 RGB，alpha 合成白底），并把 ICC 描述/SHA-256 作为**元数据记录**；
> **不执行 ICC profile → sRGB 色彩管理变换**，特征反映的是未做色域映射的 RGB 数值。

修改位置：`README`/`docs/03_design/architecture.md`、`docs/05_reports/PHASE1-REPORT.md`、
`docs/submissions/PHASE_0_1_SUBMISSION.md`、`src/alacv/imaging.py`、`src/alacv/__init__.py`。

## 7. 其他审计修正（WP5）

| 项 | 处理 |
|---|---|
| TASK-0001 测试指针 | 新增 [TASK-0001-erratum-0001.md](../01_tasks/TASK-0001-erratum-0001.md)；原任务单不改写；导航索引已更新 |
| `peak_cpu_pct` | 判定无效（可超过 `100×20`）；`ResourceSampler` 改为 `peak_cpu_cores = max(ΔCPU时间/Δwall)`；历史值保留但不再引用 |
| OS 版本 | 直接读注册表并保存：`docs/02_audit/os-evidence.json`（ProductName `Windows 10 Pro`、DisplayVersion `26H2`、CurrentBuild `26300`、UBR `9711`、EditionID `Professional`），不据 build 推断产品版本 |
| 公开 manifest | 2 条本机壁纸 origin 改为 `os-wallpaper-sample-NN`；旧/新 SHA 记录于 `data/manifests/ds-0001-manifest-sanitization.json`；图像文件与 SHA 未变 |

## 8. 变更文件清单

| 类型 | 文件 |
|---|---|
| 新增代码 | `src/alacv/backtrack.py` |
| 修改代码 | `src/alacv/imaging.py`、`src/alacv/retrieval.py`、`src/alacv/metrics.py`、`src/alacv/foveation.py`、`src/alacv/__init__.py`、`scripts/run_experiments.py`、`scripts/make_dataset.py` |
| 新增脚本 | `scripts/publish_evidence.py`、`scripts/recompute_backtrack.py` |
| 新增测试 | `tests/test_backtrack.py`；`tests/test_foveation.py` 新增 2 项 |
| 新增证据 | `artifacts/evidence/EXP-0001/**`、`artifacts/evidence/test-runs/**`、`artifacts/experiments/EXP-0001-backtrack-corrected.json` |
| 新增文档 | `ADR-0008`、`EXP-0001-AMD-0001`、`TASK-0001-erratum-0001`、本报告 |
| 更新文档 | `PHASE1-REPORT`、`EXP-0001`、`README`、`ACTIVE_STATE`、`risk-register`、`docs/README`、`PROJECT_SOURCE_INDEX`、`TASK-0002`、`data/manifests/*` |

## 8.5 第三轮复核修复（本轮）

| 复核项 | 处理 |
|---|---|
| 文档自相矛盾 | AMD §3 改为 GT bbox 主指标；本报告 §11 删除"三组持平"；PHASE1 §6.2 标注"原始未修正历史判定"；README 移除已修复的待办；ACTIVE_STATE 更新到第二轮结果 |
| C 的 ROI 机制 | 区分 ROI_00 命中率（58.3%）、任一已存裁片命中率（66.7%）、并集覆盖 GT 面积（47.7%）；逐图 `roi_boxes` / `roi_union_coverage_of_gt` 写入 corrected JSON |
| 缺失资产 fail-open | B 缺 L0、C 缺 foveated、未知 variant 均改为抛错（fail-closed），新增 3 项测试 |
| 测试锚点 | 在最终代码提交上重跑 83 项并更新日志（HEAD / dirty=false / `<REPO>`） |

## 9. NOT RUN / 未完成事项

| 项 | 原因 |
|---|---|
| 完整重跑 EXP-0001 | 未获 Owner 再次批准；本任务只做重算与证据补齐 |
| EXP-0002（Sobel 对照等） | 明确禁止 |
| ICC profile → sRGB 变换实现 | WP4 明确不新增 |
| ClawsGO 只读审查 | 由 Owner 安排 |
| 独立复现（ChatGPT 运行） | 需在 Owner 工作站进行，非本 Agent 职责 |
| `ruff` / 静态类型检查 | 环境未安装；仅执行 `python -m compileall`（exit 0） |
| 自然图像基准 | 不在本任务范围 |
| Phase 2 / 视频 / 时序 | 明确禁止，仍 BLOCKED |

## 10. 交付提交

| 项 | 值 |
|---|---|
| 基线 | `2bc71dfba3d6ccebcda44124a324f697ba2f0b1d` |
| 分支 | `task/1-phase0-1-remediation`（HEAD 见 PR #2） |
| 第一轮提交 | `d6db736` → `af9bfe6` → `05feedf` → `d01c7d4` |
| 第二轮提交 | `9a8e1d1`（GT/逐轴/哈希/provenance）→ `a257b09`（GT 重算）→ `d8604aa`（锚定测试日志）→ `9a693ce`（校验工具 ref） |
| PR | https://github.com/zcx369658780/ALAC-V/pull/2 （`Refs #1`，不自动关闭） |
| 合并 / 关闭 Issue | **未执行**，等待 Owner |

## 11. 结论与限制

- 证据链、坐标口径、术语一致性、测量口径、隐私与路径问题均已按 Issue #1 处理。
- **科学结论发生实质改变**：A 变体的回溯劣势是测量缺陷；按预注册 GT bbox 口径，**H3 被拒绝**——
  C 的真实目标区域保真度比 A 低 3.14 dB（31.66 vs 28.53），并非三组持平。
- 残余限制：1 次 run 的代码身份只能推断；样本仅合成数据；`peak_cpu_pct` 历史值不可用。
- Phase 2 保持 **BLOCKED**，等待独立复核与 Owner 验收。
