---
id: AUDIT-0002
title: 模型许可证与来源登记
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0001
---

# 模型许可证与来源登记

| 字段 | 值 |
|---|---|
| 模型 | DINOv2 ViT-S/14（`dinov2-small`） |
| 上游 | Meta AI Research（facebookresearch/dinov2） |
| 分发源 | HuggingFace `facebook/dinov2-small` |
| Revision（commit） | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| 许可证 | **Apache License 2.0** |
| 许可证全文 | `artifacts/models/dinov2_vits14/LICENSE`（11,358 字节） |
| 权重文件 | `model.safetensors`，88,249,960 字节 |
| 权重 SHA-256 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| config.json SHA-256 | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| 参数量 | 22.06 M |
| 输入分辨率 | 518×518（37×37 patch，patch=14） |
| 特征维度 | 384（CLS 与 patch 共享） |
| 是否微调 | **否**。权重与上游 SHA-256 一致，全程 `eval()` + `inference_mode()` |
| 下载时间 | 2026-11-14T10:42:58Z |

## 合规要点

1. **许可证允许**商业与非商业使用、修改与再分发，需保留版权与许可声明 → 已随权重保存 `LICENSE` 全文。
2. **未做训练/微调**：任务单禁止在本阶段训练主干。代码路径中不存在 `optimizer`、`backward()`、`state_dict` 写回。
3. **不调用外部 LLM API**：推理全部本地完成；`transformers` 仅用于加载本地权重（`local_files_only=True`）。
4. **不上传用户图像**：管线无任何 HTTP 客户端上传路径；网络仅用于下载权重与依赖。
5. 权重来源可复核：`data/manifests/model-dinov2_vits14.json` 记录 revision 与 SHA-256，`artifacts/models/dinov2_vits14/SHA256SUMS` 可离线校验。

## 依赖许可证（关键项）

| 依赖 | 许可证 |
|---|---|
| PyTorch | BSD-3-Clause |
| torchvision | BSD-3-Clause |
| transformers | Apache-2.0 |
| safetensors | Apache-2.0 |
| Pillow | MIT-CMU |
| NumPy | BSD-3-Clause |
| OpenCV | Apache-2.0 |
| psutil | BSD-3-Clause |

## 数据集许可证

| 数据集 | 来源 | 许可证 |
|---|---|---|
| DS-0001 合成集 | `scripts/make_dataset.py` 程序化生成 | CC0-1.0（无第三方素材） |
| DS-0002 真实图补充集 | 本机 `C:\Windows\Web\Wallpaper` 本地复制 | 仅本地研究用，不随仓库分发，不入库 |
