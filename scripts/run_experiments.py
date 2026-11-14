"""EXP-0001：A/B/C 三组对照实验。

每个 (variant, repeat) 在独立 run 目录与独立知识库中执行，保证测量互不污染。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.config import load_config  # noqa: E402
from alacv.features import DinoEncoder  # noqa: E402
from alacv.foveation import select_rois  # noqa: E402
from alacv.imaging import open_image  # noqa: E402
from alacv.logging_utils import setup_logging  # noqa: E402
from alacv.metrics import ResourceSampler, percentile, psnr, ssim  # noqa: E402
from alacv.pipeline import ImagePipeline  # noqa: E402
from alacv.retrieval import (  # noqa: E402
    RetrievalIndex,
    evaluate_queries,
    similarity_separation,
)
from alacv.store import VisualStore, sha256_file  # noqa: E402

PIL = __import__("PIL.Image", fromlist=["Image"])


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def env_info() -> dict:
    import torch

    info = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": p.name, "vram_mb": round(p.total_memory / 1024**2, 1), "capability": [p.major, p.minor]}
    try:
        import subprocess

        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    return info


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def recover(cfg, store, image_id: str, variant: str, box):
    """按变体实际保存的表示还原 ROI，返回 (重建像素, 可见交集)。

    具体实现位于 `alacv.backtrack.recover_variant`，与 CLI 共用同一坐标契约。
    """
    from alacv.backtrack import recover_variant

    return recover_variant(cfg, store, image_id, variant, box)


def run_one(config_path: Path, exp_id: str, variant: str, rep: int, manifest: dict, seed: int) -> dict:
    stamp = utc_stamp()
    run_id = f"{stamp}_{exp_id}_{variant}_r{rep}"
    run_dir = ROOT / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ws = run_dir / "store"
    for sub in ("raw", "processed", "features", "index"):
        (ws / sub).mkdir(parents=True, exist_ok=True)

    overrides = {
        "paths": {
            "raw_dir": str(ws / "raw"),
            "processed_dir": str(ws / "processed"),
            "features_dir": str(ws / "features"),
            "index_db": str(ws / "index" / "alacv.sqlite"),
            "manifests_dir": str(ROOT / "data" / "manifests"),
        }
    }
    cfg = load_config(config_path, overrides=overrides)
    cfg.snapshot(run_dir)
    logger = setup_logging(cfg.get("logging.level", "INFO"), run_dir / "run.log")

    store = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))
    encoder = DinoEncoder(cfg)
    (run_dir / "model.json").write_text(json.dumps(encoder.describe(), ensure_ascii=False, indent=2), encoding="utf-8")
    env = env_info()
    (run_dir / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    store.start_run(
        {
            "run_id": run_id,
            "exp_id": exp_id,
            "variant": variant,
            "config_sha256": cfg.sha256,
            "git_commit": env.get("git_commit"),
            "git_dirty": 1 if env.get("git_dirty") else 0,
            "host": env.get("gpu", {}).get("name", "unknown"),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "notes": f"EXP {exp_id} variant {variant} rep {rep}",
        }
    )

    pipe = ImagePipeline(cfg, store, encoder, run_id=run_id)
    files = list(manifest["files"])
    random.Random(seed + rep * 7919).shuffle(files)

    sampler = ResourceSampler()
    sampler.start()
    t_loop = time.perf_counter()
    items = []
    for f in files:
        item = pipe.process(ROOT / f["path"], variant, manifest["dataset_id"])
        item["role"] = f["role"]
        item["base_id"] = f.get("base_id")
        items.append(item)
        if item["status"] != "ok":
            logger.warning(f"rejected {f['path']}: {item.get('reason')}")
    wall_s = time.perf_counter() - t_loop
    res = sampler.stop()

    # ---------------- 评估 ----------------
    by_sha = {}
    for f in manifest["files"]:
        row = store.image_by_sha(f["sha256"])
        if row:
            by_sha[f["sha256"]] = row["image_id"]

    gallery = [(f["base_id"], by_sha[f["sha256"]]) for f in manifest["files"] if f["role"] == "base"]
    queries = [(f["base_id"], by_sha[f["sha256"]]) for f in manifest["files"] if f["role"] == "query"]
    sources = list(cfg.require("encoding.variants")[variant])

    t0 = time.perf_counter()
    index = RetrievalIndex(store, [gid for _, gid in gallery], sources)
    index_build_ms = (time.perf_counter() - t0) * 1000.0

    base_of = {gid: bid for bid, gid in gallery}
    qlist = [{"image_id": gid, "gt": [gid2 for bid2, gid2 in gallery if bid2 == bid]} for bid, gid in queries]

    t0 = time.perf_counter()
    retrieval = evaluate_queries(index, qlist, topk=10)
    eval_ms = (time.perf_counter() - t0) * 1000.0
    retrieval["query_ms_median"] = eval_ms / max(1, len(qlist))
    retrieval["index_build_ms"] = index_build_ms
    retrieval.pop("per_query", None)

    positives = [(gid, gid2) for bid, gid in queries for bid2, gid2 in gallery if bid == bid2]
    negatives = [(gid, gid2) for bid, gid in queries[:24] for bid2, gid2 in gallery if bid != bid2][:200]
    sep = similarity_separation(index, positives, negatives)

    # ---------------- 派生表示可见像素保真度 + ROI ----------------
    from alacv.backtrack import visible_region

    backtrack_psnr, backtrack_ssim, roi_hits = [], [], []
    coverages: list[float] = []
    excluded = 0
    roi_method = float(cfg.get("foveation.roi_frac", 0.25))
    input_size = int(cfg.get("model.input_size", 518))
    for f in manifest["files"]:
        if f["role"] != "base":
            continue
        iid = by_sha[f["sha256"]]
        rois = store.rois_of(iid)
        src_img = open_image(ROOT / f["path"]).image
        if rois:
            box = (rois[0]["x"], rois[0]["y"], rois[0]["w"], rois[0]["h"])
        else:
            r = select_rois(src_img, roi_frac=roi_method, max_rois=1)[0]
            box = (r.x, r.y, r.w, r.h)
        if f.get("gt_bbox"):
            roi_hits.append(1.0 if iou(box, f["gt_bbox"]) >= 0.3 else 0.0)
        region = visible_region(box, src_img.size, input_size)
        if not region.visible:
            excluded += 1
            continue
        coverages.append(region.coverage)
        gt = np.asarray(src_img.crop((region.box[0], region.box[1], region.box[0] + region.box[2], region.box[1] + region.box[3])))
        rec, _ = recover(cfg, store, iid, variant, box)
        if rec is not None and rec.shape == gt.shape:
            backtrack_psnr.append(psnr(gt, rec))
            backtrack_ssim.append(ssim(gt, rec))

    # ---------------- 存储 ----------------
    feature_bytes = 0
    for f in manifest["files"]:
        iid = by_sha[f["sha256"]]
        for s in sources:
            try:
                row = store.conn.execute(
                    "SELECT bytes FROM features WHERE image_id=? AND source=?", (iid, s)
                ).fetchone()
                feature_bytes += row["bytes"] if row else 0
            except Exception:
                pass
    asset_kinds = {"A": [], "B": ["pyramid"], "C": ["pyramid", "foveated", "roi_crop"]}[variant]
    asset_bytes = 0
    if asset_kinds:
        q = "SELECT COALESCE(SUM(bytes),0) FROM assets WHERE kind IN (%s)" % ",".join("?" * len(asset_kinds))
        asset_bytes = int(store.conn.execute(q, asset_kinds).fetchone()[0])

    ok_items = [i for i in items if i["status"] == "ok"]
    n = max(1, len(ok_items))
    metrics = {
        "run_id": run_id,
        "exp_id": exp_id,
        "variant": variant,
        "rep": rep,
        "n_processed": len(ok_items),
        "n_failed": len(items) - len(ok_items),
        "sources": sources,
        "dim": index.dim,
        "timing": {
            "t_total_median_ms": percentile([i["t_total_ms"] for i in ok_items], 50),
            "t_total_p95_ms": percentile([i["t_total_ms"] for i in ok_items], 95),
            "t_encode_median_ms": percentile([i["t_encode_ms"] for i in ok_items], 50),
            "t_pyramid_median_ms": percentile([i["t_pyramid_ms"] for i in ok_items], 50),
            "t_fovea_median_ms": percentile([i["t_fovea_ms"] for i in ok_items], 50),
            "t_ingest_median_ms": percentile([i["t_ingest_ms"] for i in ok_items], 50),
            "t_store_median_ms": percentile([i.get("t_store_ms", 0.0) for i in ok_items], 50),
            "wall_s": wall_s,
            "images_per_s": len(ok_items) / wall_s if wall_s > 0 else None,
        },
        "vram": {
            "peak_alloc_median_mb": percentile([i["peak_vram_mb"] for i in ok_items], 50),
            "activation_peak_median_mb": percentile([i["peak_vram_activation_mb"] for i in ok_items], 50),
            "activation_peak_max_mb": max([i["peak_vram_activation_mb"] for i in ok_items], default=0.0),
        },
        "resources": res,
        "storage": {
            "feature_bytes_per_image": feature_bytes / n,
            "asset_bytes_per_image": asset_bytes / n,
            "total_bytes_per_image": (feature_bytes + asset_bytes) / n,
        },
        "retrieval": retrieval,
        "separation": sep,
        "backtrack": {
            "metric": "derived_representation_visible_pixel_fidelity",
            "psnr_mean": float(np.mean(backtrack_psnr)) if backtrack_psnr else None,
            "ssim_mean": float(np.mean(backtrack_ssim)) if backtrack_ssim else None,
            "n": len(backtrack_psnr),
            "coverage_mean": float(np.mean(coverages)) if coverages else None,
            "coverage_min": float(np.min(coverages)) if coverages else None,
            "excluded_invisible": excluded,
        },
        "roi_hit_rate": float(np.mean(roi_hits)) if roi_hits else None,
        "db_stats": store.stats(),
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    store.finish_run(run_id, "ok", json.dumps(metrics, ensure_ascii=False), dt.datetime.now(dt.timezone.utc).isoformat())
    logger.info(
        f"{run_id} done: recall@1={retrieval['recall@1']:.3f} "
        f"t_total_med={metrics['timing']['t_total_median_ms']:.1f}ms "
        f"feat={metrics['storage']['feature_bytes_per_image']/1024:.0f}KB"
    )
    store.close()
    return metrics


def aggregate(per_variant: dict[str, list[dict]]) -> dict:
    out = {}
    numeric_paths = [
        ("timing", "t_total_median_ms"), ("timing", "t_total_p95_ms"), ("timing", "t_encode_median_ms"),
        ("vram", "peak_alloc_median_mb"), ("vram", "activation_peak_median_mb"),
        ("storage", "feature_bytes_per_image"), ("storage", "asset_bytes_per_image"), ("storage", "total_bytes_per_image"),
        ("retrieval", "recall@1"), ("retrieval", "recall@5"), ("retrieval", "mAP@10"),
        ("retrieval", "query_ms_median"), ("retrieval", "index_build_ms"),
        ("backtrack", "psnr_mean"), ("backtrack", "ssim_mean"),
        ("separation", "auc"),
    ]
    for variant, runs in per_variant.items():
        agg = {"n_runs": len(runs), "reps": runs}
        for sect, key in numeric_paths:
            vals = [r[sect][key] for r in runs if r.get(sect, {}).get(key) is not None]
            if vals:
                agg.setdefault(sect, {})[key] = float(np.median(vals))
        agg["resources"] = {
            k: float(np.median([r["resources"][k] for r in runs]))
            for k in ("avg_cpu_cores", "peak_cpu_cores", "peak_rss_mb")
        }
        agg["roi_hit_rate"] = next((r["roi_hit_rate"] for r in runs if r["roi_hit_rate"] is not None), None)
        agg["dim"] = int(np.median([r["dim"] for r in runs]))
        out[variant] = agg
    return out


def fmt_bytes(v: float | None) -> str:
    if v is None:
        return "n/a"
    if v >= 1024 * 1024:
        return f"{v/1024/1024:.2f} MB"
    return f"{v/1024:.1f} KB"


def write_summary_md(agg: dict, exp_id: str, path: Path) -> None:
    header = (
        "| 变体 | 维度 | 单图总时间 中位/P95 (ms) | 编码 (ms) | 显存峰值 (MB) | 激活峰值 (MB) "
        "| 特征/图 | 派生像素/图 | 总派生/图 | Recall@1 | Recall@5 | mAP@10 | 回溯 PSNR | 回溯 SSIM |"
    )
    sep = "|" + "---|" * 14
    lines = [f"# {exp_id} 汇总", "", "> 由 `scripts/run_experiments.py` 自动生成；每个数字来自独立 run 目录，取 3 次重复的中位数。", "", header, sep]
    for v, a in agg.items():
        t = a.get("timing", {})
        s = a.get("storage", {})
        r = a.get("retrieval", {})
        b = a.get("backtrack", {})
        vr = a.get("vram", {})
        f = lambda d, k, fmt: (fmt.format(d[k]) if d.get(k) is not None else "n/a")
        row = [
            v,
            f(a, "dim", "{}") if "dim" in a else "n/a",
            f(t, "t_total_median_ms", "{:.1f}") + " / " + f(t, "t_total_p95_ms", "{:.1f}"),
            f(t, "t_encode_median_ms", "{:.1f}"),
            f(vr, "peak_alloc_median_mb", "{:.0f}"),
            f(vr, "activation_peak_median_mb", "{:.0f}"),
            fmt_bytes(s.get("feature_bytes_per_image")),
            fmt_bytes(s.get("asset_bytes_per_image")),
            fmt_bytes(s.get("total_bytes_per_image")),
            f(r, "recall@1", "{:.3f}"),
            f(r, "recall@5", "{:.3f}"),
            f(r, "mAP@10", "{:.3f}"),
            f(b, "psnr_mean", "{:.2f}"),
            f(b, "ssim_mean", "{:.4f}"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_abc.yaml")
    ap.add_argument("--variants", default=None, help="逗号分隔，默认取配置")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    config_path = ROOT / args.config
    cfg = load_config(config_path)
    exp_id = cfg.get("experiment.id", "EXP-0001")
    variants = args.variants.split(",") if args.variants else cfg.get("experiment.variants", ["A", "B", "C"])
    repeat = args.repeat or int(cfg.get("experiment.repeat", 3))
    ds_id = args.dataset or cfg.get("experiment.dataset")
    manifest_path = ROOT / "data" / "manifests" / f"{ds_id}.json"
    if not manifest_path.exists():
        print(f"缺少数据清单: {manifest_path}；先运行 scripts/make_dataset.py", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(cfg.get("project.seed", 1234))

    per_variant: dict[str, list[dict]] = {v: [] for v in variants}
    order = [(v, r) for r in range(repeat) for v in variants]
    random.Random(seed).shuffle(order)  # 轮转顺序，抵消热漂移
    for variant, rep in order:
        m = run_one(config_path, exp_id, variant, rep, manifest, seed)
        per_variant[variant].append(m)
        print(
            f"[{variant} r{rep}] recall@1={m['retrieval']['recall@1']:.3f} "
            f"t={m['timing']['t_total_median_ms']:.1f}ms feat={m['storage']['feature_bytes_per_image']/1024:.0f}KB",
            flush=True,
        )

    agg = aggregate(per_variant)
    out_dir = ROOT / "artifacts" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{exp_id}-summary.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(agg, exp_id, out_dir / f"{exp_id}-summary.md")
    print(json.dumps({v: {k: a.get(k) for k in ("timing", "storage", "retrieval", "backtrack")} for v, a in agg.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
