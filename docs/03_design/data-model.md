---
id: DATA-0001
title: 数据模型与目录布局
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0001
---

# 数据模型与目录布局

## 1. 磁盘布局

```
D:\ALAC_V\
├── .venv/                        # 隔离 Python 环境（不入库）
├── configs/                      # 可复现配置
│   ├── default.yaml
│   ├── experiment_abc.yaml
│   └── model_dinov2_vits14.yaml
├── src/alacv/                    # 源码包
├── scripts/                      # 一次性/运维脚本（审计、下载、数据集、实验）
├── tests/                        # pytest
├── docs/                         # 文档系统
├── data/
│   ├── raw/<aa>/<sha256>.<ext>    # 原始归档：内容寻址、只读
│   ├── processed/<image_id>/
│   │   ├── pyramid/L0_1x.webp … L3_8x.webp
│   │   ├── foveated/foveated.webp
│   │   └── roi/roi_00.webp + roi.json
│   ├── features/<image_id>/<variant>__<source>.safetensors
│   ├── index/alacv.sqlite
│   └── manifests/
│       ├── dataset-<id>.json      # 数据集清单（含逐文件 SHA-256）
│       └── model-<id>.json        # 模型指纹与许可证
└── artifacts/
    ├── runs/<run_id>/             # 每次运行的日志、指标、配置快照
    ├── models/dinov2_vits14/      # 权重 + LICENSE
    └── figures/                   # 报告用图
```

### 只读约定

- `data/raw/**` 在归档后设为只读（Windows 去掉 `Write` 权限位；Linux chmod 444）。
- 任何代码路径不得对 `data/raw` 做 `write/rename/delete`；`store` 层只读取。
- 校验：`scripts/verify_raw_integrity.py` 重算全部 SHA-256 并与清单比对。

## 2. 标识符

| 标识 | 生成规则 | 用途 |
|---|---|---|
| `sha256` | 原始文件字节的 SHA-256（hex，64 字符） | 内容寻址、不可变性证明 |
| `image_id` | `sha256[:16]`（hex） | 短 ID，目录名与主键 |
| `run_id` | `<UTC ISO8601 紧凑>_<exp>_<variant>` | 运行追溯 |
| `config_sha256` | 规范化 YAML（键排序、去注释）+ 模型 ID 的 SHA-256 | 判断派生数据是否过期 |

## 3. SQLite Schema（`data/index/alacv.sqlite`）

WAL 模式；`foreign_keys=ON`；所有写入走显式事务。版本号存 `PRAGMA user_version`。

```sql
CREATE TABLE IF NOT EXISTS models (
  model_id      TEXT PRIMARY KEY,          -- 例: dinov2_vits14_hf_<revision 前8位>
  display_name  TEXT NOT NULL,
  arch          TEXT NOT NULL,             -- dinov2_vits14
  params        INTEGER,
  source        TEXT NOT NULL,             -- huggingface:facebook/dinov2-small
  revision      TEXT NOT NULL,             -- commit hash
  weights_sha256 TEXT NOT NULL,
  license       TEXT NOT NULL,             -- Apache-2.0
  license_file  TEXT,
  input_size    INTEGER NOT NULL,          -- 518
  patch_size    INTEGER NOT NULL,          -- 14
  cls_dim       INTEGER NOT NULL,          -- 384
  norm_mean    TEXT NOT NULL,              -- JSON [0.485,0.456,0.406]
  norm_std     TEXT NOT NULL,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
  image_id      TEXT PRIMARY KEY,
  sha256        TEXT NOT NULL UNIQUE,
  rel_path      TEXT NOT NULL,             -- 相对 data/raw
  source_name   TEXT,                      -- 原始文件名（仅供参考）
  ext           TEXT NOT NULL,             -- jpg/png/webp/avif
  mime          TEXT NOT NULL,
  width         INTEGER NOT NULL,
  height        INTEGER NOT NULL,
  channels      INTEGER NOT NULL,
  pil_mode      TEXT NOT NULL,             -- RGB/RGBA/L/P…
  color_space   TEXT NOT NULL,             -- sRGB / Display P3 / Gray / 未知
  icc_sha256    TEXT,
  file_size     INTEGER NOT NULL,
  exif_json     TEXT,
  dataset_id    TEXT,
  status        TEXT NOT NULL DEFAULT 'ok',-- ok | rejected
  reject_reason TEXT,
  ingested_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (            -- 任一派生文件（图/裁片）
  asset_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id      TEXT NOT NULL REFERENCES images(image_id),
  kind          TEXT NOT NULL,             -- pyramid | foveated | roi_crop | preview
  level         INTEGER,                   -- 金字塔层级 0..3
  label         TEXT,                      -- 例 roi_00
  rel_path      TEXT NOT NULL,
  width         INTEGER NOT NULL,
  height        INTEGER NOT NULL,
  format        TEXT NOT NULL,             -- webp/png
  bytes         INTEGER NOT NULL,
  sha256        TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  UNIQUE (image_id, kind, level, label, config_sha256)
);

CREATE TABLE IF NOT EXISTS features (
  feature_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id      TEXT NOT NULL REFERENCES images(image_id),
  variant       TEXT NOT NULL,             -- A | B | C
  source        TEXT NOT NULL,             -- original | pyramid:L0…L3 | foveated | roi:roi_00
  level         INTEGER,
  model_id      TEXT NOT NULL REFERENCES models(model_id),
  precision     TEXT NOT NULL,             -- fp16
  n_patches     INTEGER NOT NULL,
  patch_grid    INTEGER NOT NULL,          -- 37 表示 37x37
  cls_dim       INTEGER NOT NULL,
  rel_path      TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  sha256        TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  UNIQUE (image_id, source, config_sha256)
);

CREATE TABLE IF NOT EXISTS rois (
  roi_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  image_id      TEXT NOT NULL REFERENCES images(image_id),
  roi_index     INTEGER NOT NULL,
  x INTEGER NOT NULL, y INTEGER NOT NULL,
  w INTEGER NOT NULL, h INTEGER NOT NULL,   -- 原图像素坐标，左上原点
  score         REAL NOT NULL,
  method        TEXT NOT NULL,              -- gradient_energy_v1 | manual
  crop_asset_id INTEGER REFERENCES assets(asset_id),
  created_at    TEXT NOT NULL,
  UNIQUE (image_id, roi_index, method)
);

CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  exp_id        TEXT,
  variant       TEXT,
  config_sha256 TEXT NOT NULL,
  git_commit    TEXT,
  git_dirty     INTEGER,
  host          TEXT,
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  status        TEXT NOT NULL,             -- running | ok | failed
  metrics_json  TEXT,
  notes         TEXT
);

CREATE TABLE IF NOT EXISTS run_items (       -- 逐图明细，便于定位慢样本
  run_id      TEXT NOT NULL REFERENCES runs(run_id),
  image_id    TEXT NOT NULL,
  variant     TEXT NOT NULL,
  t_ingest_ms REAL, t_pyramid_ms REAL, t_fovea_ms REAL, t_encode_ms REAL,
  t_total_ms  REAL,
  peak_vram_mb REAL,
  rss_delta_mb REAL,
  feature_bytes INTEGER,
  PRIMARY KEY (run_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_features_lookup ON features(image_id, variant);
CREATE INDEX IF NOT EXISTS idx_assets_lookup   ON assets(image_id, kind);
CREATE INDEX IF NOT EXISTS idx_rois_image      ON rois(image_id);
```

Parquet 仅用于导出（`scripts/export_parquet.py`），不作为主存储——本阶段避免过早引入复杂数据栈。

## 4. SafeTensors 特征格式

每个文件只放一张图、一种来源的特征：

```python
{
  "cls":        float16[384],            # CLS token（已 L2 归一前的原始输出）
  "patch":      float16[n_patches, 384], # 去掉 CLS 后的 patch tokens
  "grid_h":     int32[1], "grid_w": int32[1],
  "meta_json":  string[1],               # JSON: model_id, preprocess, source, size
}
```

- FP16 存储；检索时按需转 FP32 累加，避免半精度点积的累积误差。
- 文件头元数据冗余保存 `model_id` 与 `config_sha256`，脱离 SQLite 也可自证来源。

## 5. 运行目录

```
artifacts/runs/20261114T103000Z_EXP0001_C/
├── config.snapshot.yaml     # 生效配置（合并默认值后）
├── config.sha256
├── env.json                 # python/torch/cuda/gpu/驱动版本
├── model.json               # 模型指纹
├── run.log                  # 结构化日志（JSONL）
├── metrics.json             # 聚合指标
├── items.jsonl              # 逐图明细
└── queries/                 # 检索查询与结果（实验用）
```

## 6. 不可变性证明链

```
manifest(dataset).files[].sha256
        == images.sha256
        == 归档文件重算 SHA-256
        == 特征文件 meta_json.source_sha256（若记录）
```

`scripts/verify_raw_integrity.py` 每次实验前运行，任何失配立即中止实验。
