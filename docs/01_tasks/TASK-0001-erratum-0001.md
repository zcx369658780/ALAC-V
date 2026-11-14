---
id: ERRATUM-0001
title: TASK-0001 证据指针勘误（tests/test_pyramid.py）
status: 提议（待 Owner 验收）
date: 2026-11-14
owner: lead
task: TASK-0002
references: TASK-0001, Issue #1 R5
---

# TASK-0001 证据指针勘误

## 1. 勘误事项

`docs/01_tasks/TASK-0001-phase0-phase1.md` 的验收清单 P1-3 行引用
`tests/test_pyramid.py`，但仓库实际不存在该文件；金字塔尺寸测试位于：

```
tests/test_imaging.py::test_pyramid_sizes
```

## 2. 处理原则

- TASK-0001 是**历史授权记录**，不重写正文、不修改原始验收条目。
- 本勘误单独记录正确指针，并由 `docs/README.md` 与 `docs/08_governance/PROJECT_SOURCE_INDEX.md` 建立索引。
- 原任务单中的引用保持原样，阅读时应以本勘误为准。

## 3. 正确的证据映射

| 原引用 | 实际位置 | 覆盖内容 |
|---|---|---|
| `tests/test_pyramid.py` | `tests/test_imaging.py::test_pyramid_sizes` | 1/1、1/2、1/4、1/8 尺寸断言 |
| — | `tests/test_imaging.py::test_fit_to_input_square_landscape` | 518 短边缩放 + 中心裁剪 |
| — | `tests/test_imaging.py::test_fit_to_input_portrait` | 竖图同一变换 |
| — | `tests/test_backtrack.py`（TASK-0002 新增） | 可见交集与重建坐标 |

## 4. 影响

仅为导航/审计指针错误，不影响任何实验数值或代码行为。
