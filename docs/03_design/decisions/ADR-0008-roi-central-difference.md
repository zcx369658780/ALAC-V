---
id: ADR-0008
title: ROI 重要性使用中心差分梯度能量（术语修订，行为冻结）
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0002
supersedes: ADR-0007（仅"算子名称/描述"部分；决策本身继续有效）
---

# ADR-0008：ROI 重要性使用中心差分梯度能量

## 背景

独立审查（`docs/08_governance/PHASE0_1_ACCEPTANCE.md` R3，Issue #1 §C）指出：ADR-0007 与
`src/alacv/foveation.py::gradient_energy_map()` 的 docstring 都把 ROI 梯度称为 "Sobel"，
但实现使用的是**两像素中心差分**（`arr[:, 2:] - arr[:, :-2]` 与纵向对应），
并非标准 3×3 Sobel 算子。

## 决策

1. **保留现有中心差分实现，不替换为 Sobel。** 原因：EXP-0001 已完成，ROI 选择条件必须冻结；
   本任务是证据与文档修正，不是算法变更。替换算子会静默改变实验输入，使已发布结果不可比。
2. 新增本 ADR，准确命名当前算子，并明确 supersede ADR-0007 中关于 "Sobel" 的错误描述。
3. ADR-0007 的历史正文**不删除、不改写**，仅增加取代指针与更正说明。
4. 只修正代码 docstring 与文档用语，不改变 `gradient_energy_map()` 的任何数值行为。

## 当前算子的精确定义

对灰度图 `I`（先做 σ=1.0 高斯模糊），取

```
gx[y, x] = I[y, x+1] - I[y, x-1]      (边界一列/一行置 0)
gy[y, x] = I[y+1, x] - I[y-1, x]
M[y, x]  = sqrt(gx² + gy²)
E        = M / max(M)                  (全零时返回全零图)
```

即**中心差分梯度幅值**（central finite difference gradient magnitude），
等价于 3×3 卷积核 `[-1, 0, 1]` 的可分离形式，**不是** Sobel 的
`[[-1,0,1],[-2,0,2],[-1,0,1]]`（Sobel 在垂直方向额外加权 2）。

## 后果

- **正面**：代码、文档、ADR 三者一致；实验输入未发生任何变化，历史结果仍可比。
- **负面**：中心差分对噪声比 Sobel 更敏感（少了纵向平滑），但这正是 EXP-0001 实际使用的算子，
  改变它需要新的实验编号与重新预注册。
- 若未来要比较 Sobel，必须另立实验（EXP-0002）与新任务单，不得在证据修正任务中顺手替换。

## 验证

`tests/test_foveation.py::test_gradient_operator_matches_central_difference` 用已知图案
逐像素核对算子输出与上述公式一致，并证明它与 Sobel 的差异（同一输入下两者不相等）。
