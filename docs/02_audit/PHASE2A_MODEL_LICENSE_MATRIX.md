---
id: AUDIT-0003
title: Phase 2A 模型与代码许可证矩阵
status: 草案（待审）
date: 2026-11-15
owner: lead
task: TASK-0003
issue: https://github.com/zcx369658780/ALAC-V/issues/3
---

# Phase 2A 模型与代码许可证矩阵

> 核验时间：2026-11-15。方法：
> - 代码仓库：GitHub REST `GET /repos/{owner}/{repo}/license`；返回 `NOASSERTION` 时读取 `LICENSE` 原文前几行人工判定。
> - 模型权重：HuggingFace API `GET /api/models/{model_id}` 的 `cardData.license` 字段。
>
> **重要区分**：仓库代码许可证 ≠ 权重许可证 ≠ 训练数据许可证。下表逐列记录，不合并三者。

## 1. 官方代码仓库

| 项目 | 仓库 | 许可证 | 商用 | 说明 |
|---|---|---|---|---|
| DINOv2（项目基线） | `facebookresearch/dinov2` | **Apache-2.0** | ✅ | Phase 1 已使用；代码与权重许可一致 |
| DINO | `facebookresearch/dino` | **Apache-2.0** | ✅ | 自蒸馏参考实现 |
| iBOT | `bytedance/ibot` | **Apache-2.0** | ✅ | 掩码 token 蒸馏 |
| VICReg | `facebookresearch/vicreg` | **MIT** | ✅ | 方差/协方差正则 |
| Barlow Twins | `facebookresearch/barlowtwins` | **MIT** | ✅ | 去相关正则 |
| I-JEPA | `facebookresearch/ijepa` | **CC-BY-NC-4.0** | ❌ 非商用 | GitHub 返回 `NOASSERTION`，LICENSE 原文为 CC-BY-NC |
| V-JEPA | `facebookresearch/jepa` | **CC-BY-NC-4.0** | ❌ 非商用 | 同上 |
| MAE | `facebookresearch/mae` | **CC-BY-NC-4.0** | ❌ 非商用 | 同上 |
| SimSiam | `facebookresearch/simsiam` | **CC-BY-NC-4.0** | ❌ 非商用 | 同上 |
| MSN | `facebookresearch/msn` | **CC-BY-NC-4.0** | ❌ 非商用 | 同上 |
| BYOL | `deepmind/byol` | 未核验 | ？ | `gh api` 返回 404（该路径下无独立仓库）；若使用需单独确认 |
| timm | `huggingface/pytorch-image-models` | Apache-2.0 | ✅ | 第三方实现来源（仅代码） |
| PyTorch | `pytorch/pytorch` | BSD-3-Clause | ✅ | 运行时 |

## 2. 公开权重

| 权重 | 来源 | 许可证 | 商用 | 备注 |
|---|---|---|---|---|
| `facebook/dinov2-small` | HF | **Apache-2.0** | ✅ | **本项目唯一在用权重**；Phase 1 已记录 revision 与 SHA-256 |
| `facebook/dinov2-base` | HF | Apache-2.0 | ✅ | 可选升级，需重新评估显存 |
| `facebook/ijepa_vith14_1k` | HF | **CC-BY-NC-4.0** | ❌ | 与代码仓库许可一致（NC） |
| `facebook/vjepa2-vitl-fpc64-256` | HF | **MIT** | ✅ | 视频模型；本阶段不下载、不使用 |
| `facebook/vit-mae-base` | HF | Apache-2.0 | ✅ | 像素重建基线；本阶段不下载 |

## 3. 结论与项目约束

1. **唯一允许在本阶段继续使用的权重是 `facebook/dinov2-small`（Apache-2.0）**；本任务不下载
   任何新权重。
2. **I-JEPA / V-JEPA / MAE / SimSiam / MSN 的官方代码为 CC-BY-NC-4.0**：
   - 不得把其代码复制进本仓库（即使是"参考实现"也不应逐行搬运）；
   - 若未来需要其**权重**（如 I-JEPA ViT-H），只能用于非商业研究，且需单独审批与记录；
   - **方法与思想不受版权保护**，可以独立实现目标函数（本设计的 A/C 臂即为独立实现）。
3. **V-JEPA 2 权重是 MIT，但其代码仓库是 CC-BY-NC**：二者不一致，未来若使用必须分别核对，
   不能以权重许可推断代码许可。
4. **合成数据无第三方许可问题**：DS-0001 为程序化生成（CC0），DS-0002 设计同样为程序化生成。
5. 若未来引入 timm 实现，只能使用 **代码**（Apache-2.0），其加载的权重仍需单独核验。

## 4. 本任务的合规声明

- 未下载任何新模型权重；
- 未复制任何 CC-BY-NC 仓库代码；
- 未下载任何自然图像数据集；
- 未使用用户私人图像；
- 设计中的 JEPA/MAE 目标均为**独立数学描述**，不依赖受限制代码。
