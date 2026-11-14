"""描述子检索与高清回溯。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .imaging import open_image


class RetrievalIndex:
    """内存内暴力余弦检索（Phase 1 规模下足够；阈值见 ADR-0003）。"""

    def __init__(self, store, image_ids: list[str], sources: list[str]):
        if not image_ids:
            raise ValueError("检索库为空")
        self.store = store
        self.image_ids = list(image_ids)
        self.sources = list(sources)
        mat = np.stack([store.descriptor(iid, sources) for iid in self.image_ids]).astype(np.float32)
        self.matrix = mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.matrix = mat / np.clip(norms, 1e-12, None)

    def descriptor_of(self, image_id: str) -> np.ndarray:
        return self.store.descriptor(image_id, self.sources)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32)
        q = q / max(1e-12, float(np.linalg.norm(q)))
        scores = self.matrix @ q
        k = min(k, len(self.image_ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.image_ids[i], float(scores[i])) for i in idx]

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])


def recall_at_k(ranked: list[str], gt: set[str], k: int) -> float:
    return 1.0 if any(r in gt for r in ranked[:k]) else 0.0


def average_precision(ranked: list[str], gt: set[str], k: int) -> float:
    hits = 0
    prec_sum = 0.0
    for i, r in enumerate(ranked[:k], start=1):
        if r in gt:
            hits += 1
            prec_sum += hits / i
    return prec_sum / max(1, min(len(gt), k))


def evaluate_queries(index: RetrievalIndex, queries: list[dict], topk: int = 10) -> dict:
    """queries: [{"image_id": ..., "gt": [gallery ids]}]"""
    recalls1, recalls5, aps = [], [], []
    per_query = []
    for q in queries:
        ranked = [i for i, _ in index.search(index.descriptor_of(q["image_id"]), topk)]
        gt = set(q["gt"])
        r1 = recall_at_k(ranked, gt, 1)
        r5 = recall_at_k(ranked, gt, 5)
        ap = average_precision(ranked, gt, topk)
        recalls1.append(r1)
        recalls5.append(r5)
        aps.append(ap)
        per_query.append({"image_id": q["image_id"], "top1": ranked[0], "hit@1": r1, "ap": ap})
    n = max(1, len(queries))
    return {
        "n_queries": len(queries),
        "recall@1": sum(recalls1) / n,
        "recall@5": sum(recalls5) / n,
        "mAP@10": sum(aps) / n,
        "per_query": per_query,
    }


def similarity_separation(index: RetrievalIndex, positives: list[tuple[str, str]], negatives: list[tuple[str, str]]) -> dict:
    """同源对 vs 异源对的余弦分布（AUC 用 Mann-Whitney 形式）。"""
    def cos(a: str, b: str) -> float:
        return float(index.descriptor_of(a) @ index.descriptor_of(b))

    pos = [cos(a, b) for a, b in positives]
    neg = [cos(a, b) for a, b in negatives]
    if not pos or not neg:
        return {"n_pos": len(pos), "n_neg": len(neg)}
    pos_a = np.asarray(pos)
    neg_a = np.asarray(neg)
    # AUC = P(pos > neg)
    auc = float(((pos_a[:, None] > neg_a[None, :]).sum() + 0.5 * (pos_a[:, None] == neg_a[None, :]).sum()) / (len(pos) * len(neg)))
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "pos_mean": float(pos_a.mean()),
        "neg_mean": float(neg_a.mean()),
        "auc": auc,
        "gap": float(pos_a.mean() - neg_a.mean()),
    }


def backtrack_crop(cfg, store, image_id: str, roi_index: int = 0) -> tuple[np.ndarray, dict]:
    """从原始归档图按 ROI 坐标裁出高分辨率局部（真·回溯，不经过派生图）。"""
    row = store.get_image(image_id)
    if row is None:
        raise KeyError(image_id)
    rois = store.rois_of(image_id)
    if not rois:
        raise ValueError(f"{image_id} 没有 ROI 记录")
    if roi_index >= len(rois):
        raise IndexError(f"roi_index={roi_index} 越界（共 {len(rois)}）")
    r = rois[roi_index]
    p = cfg.path("paths.raw_dir") / row["rel_path"]
    img = open_image(p).image
    box = (int(r["x"]), int(r["y"]), int(r["x"] + r["w"]), int(r["y"] + r["h"]))
    crop = img.crop(box)
    return np.asarray(crop), {"image_id": image_id, "roi_index": roi_index, "box": box, "source": str(p)}


def recovery_from_variant(cfg, store, image_id: str, variant: str, roi_index: int = 0):
    """按变体实际保存的表示还原 ROI 区域（用于回溯质量对比）。

    与 `scripts/run_experiments.py::recover` 共用同一坐标契约：只在与 518 中心裁剪
    窗口的共同可见区域上重建（见 `alacv.backtrack`），不再使用缩放后未减偏移的坐标。

    - A : 518 归一化输入的可见部分（低分上界）
    - B : 金字塔 L0（近原始分辨率）
    - C : 全分辨率 ROI 裁片；缺失时退回 foveated 马赛克 ROI 区
    """
    from .backtrack import reconstruct_from_visible, reconstruct_full_resolution, visible_region

    row = store.get_image(image_id)
    if row is None:
        raise KeyError(image_id)
    rois = store.rois_of(image_id)
    r = rois[roi_index]
    box = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
    img = open_image(cfg.path("paths.raw_dir") / row["rel_path"]).image
    region = visible_region(box, img.size, int(cfg.get("model.input_size", 518)))
    if not region.visible:
        raise ValueError("ROI 完全落在编码器可见区域之外，不可比较")

    if variant == "A":
        return reconstruct_from_visible(img, region)

    if variant == "B":
        assets = {a["level"]: a for a in store.assets(image_id, "pyramid")}
        if 0 in assets:
            p = cfg.path("paths.processed_dir").parent / assets[0]["rel_path"]
            return reconstruct_full_resolution(open_image(p).image, region)

    if variant == "C":
        assets = {a["label"]: a for a in store.assets(image_id, "roi_crop")}
        label = f"roi_{roi_index:02d}"
        if label in assets:
            p = cfg.path("paths.processed_dir").parent / assets[label]["rel_path"]
            im = open_image(p).image
            if im.size == (box[2], box[3]):
                x, y, w, h = region.box
                return np.asarray(im.crop((x - box[0], y - box[1], x - box[0] + w, y - box[1] + h)))
        fov = [a for a in store.assets(image_id, "foveated")]
        if fov:
            import json

            p = cfg.path("paths.processed_dir").parent / fov[0]["rel_path"]
            meta_path = p.with_suffix(".json")
            cs = 0.5
            if meta_path.exists():
                cs = float(json.loads(meta_path.read_text(encoding="utf-8"))["canvas_scale"])
            im = open_image(p).image
            x, y, w, h = region.box
            cb = (int(x * cs), int(y * cs), int((x + w) * cs), int((y + h) * cs))
            from PIL import Image

            return np.asarray(im.crop(cb).resize((w, h), Image.Resampling.BICUBIC))

    return reconstruct_full_resolution(img, region)
