# DS-0002 Gate 1 证据包

本目录只包含**可公开的紧凑证据**；图像、教师特征缓存与模型权重均保留在本地，不入库。

| 文件 | 内容 |
|---|---|
| `gate1-audit.json` | 全部审计检查的结构化结果（计数、隔离、解码、重复、确定性抽样、白名单、缓存对齐、资源） |
| `gate1-audit.md` | 审计摘要（人读版） |
| `sha256-inventory.txt` | 22,000 个 view 的 `SHA-256  path` 全量清单 |

## 数据位置（本地，不入库）

```
data/ds0002/
├── images/{train,validation,test}/<scene>/v<view>.jpg
├── manifests/{train,validation,test}.json     # 逐 view 的 SHA-256、增强参数、角色
├── manifests/ds0002.json                      # 全局计数、seed 范围、生成器指纹
├── eval/labels_{train,validation,test}.json   # 评估专用 sidecar（训练禁止读取）
└── features/teacher/                          # 冻结 DINOv2 特征缓存
    ├── index_{split}.json
    ├── {split}/shard-XXXX.safetensors
    └── cache.json
```

## 复现

```powershell
python scripts/generate_ds0002.py --config configs/ds0002.yaml --workers 1
python scripts/build_teacher_cache.py
python scripts/audit_gate1.py
```

确定性：同一 `(split, index, view)` 在任意机器上产生相同字节；生成器版本、PCG64、
配置 SHA-256 与逐文件 SHA-256 均记录在 manifest。
