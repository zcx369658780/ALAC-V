---
id: OPS-0001
title: 运行手册
status: 草案
date: 2026-11-14
owner: lead
task: TASK-0001
---

# 运行手册

## 1. 环境

```powershell
# 工作区
cd D:\ALAC_V

# 激活隔离环境（不修改系统全局依赖）
.\.venv\Scripts\Activate.ps1

# 关键版本核对
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 网络（本机特有）

系统代理为 `127.0.0.1:7897`，但 Windows schannel 栈在本环境下握手失败（`SEC_E_NO_CREDENTIALS`），因此：

- `curl`/`git` 的 HTTPS 可能失败；
- **Python（OpenSSL）与 uv 可用**，需显式设置环境变量：

```powershell
$env:HTTPS_PROXY='http://127.0.0.1:7897'
$env:HTTP_PROXY='http://127.0.0.1:7897'
# uv 缓存必须在工作区内（用户目录缓存不可写）
$env:UV_CACHE_DIR='D:\ALAC_V\.uv-cache'
```

## 2. 首次安装

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## 3. 模型下载

```powershell
.\.venv\Scripts\python.exe scripts\fetch_model.py
```

产物：`artifacts/models/dinov2_vits14/`（权重、LICENSE、SHA256SUMS）与 `data/manifests/model-dinov2_vits14.json`。

## 4. 建数据集

```powershell
.\.venv\Scripts\python.exe scripts\make_dataset.py --config configs\dataset_ds0001.yaml
```

## 5. 单图管线

```powershell
# 入库（归档 + 金字塔 + foveation + 特征 + 索引）
.\.venv\Scripts\python.exe -m alacv.cli ingest --input data\samples --variant C

# 检索
.\.venv\Scripts\python.exe -m alacv.cli query --image data\samples\q.png --variant C --topk 5

# 高清回溯（给定图库 image_id + ROI）
.\.venv\Scripts\python.exe -m alacv.cli backtrack --image-id <id> --roi 0 --out out.png
```

## 6. 实验

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py --config configs\experiment_abc.yaml
```

结果：`artifacts/runs/<run_id>/metrics.json` + 汇总表 `artifacts/experiments/EXP-0001-summary.json`。

## 7. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 8. 完整性校验

```powershell
.\.venv\Scripts\python.exe scripts\verify_raw_integrity.py
```

## 9. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `拒绝访问` 写 `D:\ALAC_V` | 工作区 ACL 缺 WRITE_OWNER | 运行 DSH 的 ACL 修复脚本，见 `D:\ALAC_V_acl_recovery` |
| `Failed to initialize cache at ...uv\cache` | uv 默认缓存在用户目录不可写 | `$env:UV_CACHE_DIR='D:\ALAC_V\.uv-cache'` |
| `SSL: CERTIFICATE_VERIFY_FAILED` | 代理证书 | 设置 `HTTPS_PROXY` 后重试；不要关闭校验 |
| CUDA OOM | 桌面占用 + batch 过大 | `--batch 1`，`torch.cuda.empty_cache()` |
| AVIF 无法读取 | Pillow 未编译 libavif | 安装 `pillow-avif-plugin` 或标记不支持；见审计报告 |
