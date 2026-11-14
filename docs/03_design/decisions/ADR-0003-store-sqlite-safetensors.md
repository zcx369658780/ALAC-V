---
id: ADR-0003
title: 元信息 SQLite + 特征 SafeTensors，不引入向量库
status: 已接受
date: 2026-11-14
owner: lead
task: TASK-0001
---

# ADR-0003：元信息 SQLite + 特征 SafeTensors，不引入向量库

## 背景

任务单明确"可以采用 SQLite/Parquet 管理元信息，使用 Safetensors 或其他适合的格式存储特征"，且"不要在第一阶段过早引入复杂数据库"。

## 决策

- **元信息**：单个 SQLite 文件 `data/index/alacv.sqlite`，WAL 模式，schema 见 `docs/03_design/data-model.md`。
- **特征**：每图每来源一个 `.safetensors` 文件，FP16，含 `cls` / `patch` / `grid_h` / `grid_w` / `meta_json`。
- **检索**：内存内暴力余弦检索（`numpy` 矩阵乘），不建 ANN 索引。
- **导出**：需要分析时 `scripts/export_parquet.py` 一次性导出，Parquet 不作主存储。

## 理由

- Phase 1 规模为数百到数千张图：384 维 × 1000 张 = 1.5 MB，全量常驻内存毫无压力；暴力检索 <1 ms。
- SQLite 零运维、单文件可备份、支持事务与 SQL 查询；Python 标准库自带。
- SafeTensors 零拷贝、可 mmap、格式自描述、禁止 pickle 执行代码（安全）。
- 向量库（FAISS/Milvus）在本阶段只增加安装失败面与调参维度，收益为零。

## 决策阈值（何时重新评估）

当满足任一条件时，在 Phase 2 任务单中重新评估 ANN：

- 图库 > 50,000 张；或
- 单次检索 P95 > 50 ms；或
- 需要多模态/多字段联合过滤导致暴力扫描不可接受。

## 后果

- **正面**：依赖少、可移植、易审计；`sqlite3` + `safetensors` 两个稳定组件。
- **负面**：多进程并发写 SQLite 需谨慎 → 用 WAL + 单写者（批处理串行），读可并发。
- **负面**：每图一个特征文件会有小文件开销（4 KB 簇浪费）→ 单图特征 ≥ 40 KB（patch 1369×384×2 B ≈ 1.05 MB），可忽略。
