---
id: ADR-0001
title: 主模型用 HuggingFace 分发版 DINOv2 ViT-S/14（冻结）
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0001
---

# ADR-0001：主模型用 HuggingFace 分发版 DINOv2 ViT-S/14（冻结）

## 背景

任务单指定 DINOv2 ViT-S/14（21M，Apache-2.0）。可选获取途径：

1. `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` — 需 GitHub clone + `dl.fbaipublicfiles.com` 下载 `.pth`；
2. HuggingFace `facebook/dinov2-small` + `transformers.Dinov2Model`；
3. timm `vit_small_patch14_dinov2.lvd142m`。

本机网络需经本地代理（`127.0.0.1:7897`），Windows 的 schannel 栈在该环境下握手失败（`SEC_E_NO_CREDENTIALS`），而 Python 的 OpenSSL 栈可正常工作。GitHub/`dl.fbaipublicfiles.com` 的可用性未验证；HuggingFace 已验证 200。

## 决策

使用 **HuggingFace `facebook/dinov2-small` + `transformers.Dinov2Model`**，并：

- 固定 `revision`（commit hash），不用 `main`；
- 下载后计算权重文件 SHA-256 写入 `data/manifests/model-*.json` 与 `artifacts/models/*/SHA256SUMS`；
- 保存上游 LICENSE 全文到 `artifacts/models/dinov2_vits14/LICENSE`；
- `model.eval()` + `requires_grad_(False)`，全程 `torch.inference_mode()`。

## 理由

- 权重与许可证随模型卡一起可得，来源可审计；
- 免去 git clone 与随仓库代码变化的风险；
- `transformers` 的 `Dinov2Model` 输出 `last_hidden_state`（含 CLS + register tokens）与 `pooler_output`，满足 CLS/Patch 双提取需求；
- 单文件约 88 MB，本地加载 <2 s。

## 后果

- **正面**：安装简单、版本可锁、许可可查；与 `safetensors` 原生兼容（HF 权重即 safetensors）。
- **负面**：`transformers` 引入额外依赖（约 10 个传递依赖），需在锁文件中固定；模型结构以 HF 实现为准，极端情况下与官方仓库数值有 ~1e-6 级差异。
- **缓解**：在 `tests/test_features.py` 中做**确定性测试**（同图两次前向 bitwise 相同）与**语义合理性测试**（同图不同变换的 CLS 余弦 > 不同图），不依赖官方仓库做逐位比对。

## 验证

Phase 0 自检中打印 `torch.__version__`、`transformers.__version__`、模型参数量（应约 21–22M）、权重 SHA-256。
