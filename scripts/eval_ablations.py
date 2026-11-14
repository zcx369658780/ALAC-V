"""EXP-0001 附加消融：复用已存特征组合描述子，不额外前向。

用法：python scripts/eval_ablations.py [--config configs/experiment_abc.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.config import load_config  # noqa: E402
from alacv.retrieval import RetrievalIndex, evaluate_queries  # noqa: E402
from alacv.store import VisualStore  # noqa: E402


def eval_run(run_dir: Path, sources: list[str], manifest: dict, dataset_id: str) -> dict | None:
    ws = run_dir / "store"
    db = ws / "index" / "alacv.sqlite"
    if not db.exists():
        return None
    cfg = load_config(
        ROOT / "configs" / "experiment_abc.yaml",
        overrides={
            "paths": {
                "raw_dir": str(ws / "raw"),
                "processed_dir": str(ws / "processed"),
                "features_dir": str(ws / "features"),
                "index_db": str(db),
                "manifests_dir": str(ROOT / "data" / "manifests"),
            }
        },
    )
    store = VisualStore(db, cfg.path("paths.features_dir"))
    have = {r["source"] for r in store.conn.execute("SELECT DISTINCT source FROM features").fetchall()}
    if not set(sources) <= have:
        store.close()
        return None

    by_sha = {}
    for f in manifest["files"]:
        row = store.image_by_sha(f["sha256"])
        if row:
            by_sha[f["sha256"]] = row["image_id"]
    gallery = [(f["base_id"], by_sha[f["sha256"]]) for f in manifest["files"] if f["role"] == "base"]
    queries = [(f["base_id"], by_sha[f["sha256"]]) for f in manifest["files"] if f["role"] == "query"]

    t0 = time.perf_counter()
    index = RetrievalIndex(store, [g for _, g in gallery], sources)
    build_ms = (time.perf_counter() - t0) * 1000
    qlist = [{"image_id": q, "gt": [g2 for b2, g2 in gallery if b2 == b]} for b, q in queries]
    t0 = time.perf_counter()
    res = evaluate_queries(index, qlist, topk=10)
    qms = (time.perf_counter() - t0) * 1000 / max(1, len(qlist))

    feat_bytes = 0
    for qid in by_sha.values():
        for s in sources:
            row = store.conn.execute("SELECT bytes FROM features WHERE image_id=? AND source=?", (qid, s)).fetchone()
            feat_bytes += row["bytes"] if row else 0
    n = len(by_sha)
    store.close()
    return {
        "dim": index.dim,
        "recall@1": res["recall@1"],
        "recall@5": res["recall@5"],
        "mAP@10": res["mAP@10"],
        "feature_bytes_per_image": feat_bytes / max(1, n),
        "index_build_ms": build_ms,
        "query_ms_median": qms,
        "n_queries": res["n_queries"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_abc.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    ablations = cfg.get("ablations", {})
    ds_id = cfg.get("experiment.dataset")
    manifest = json.loads((ROOT / "data" / "manifests" / f"{ds_id}.json").read_text(encoding="utf-8"))

    runs = sorted(p for p in (ROOT / "artifacts" / "runs").iterdir() if p.is_dir() and "EXP-0001" in p.name)
    out: dict[str, list] = {k: [] for k in ablations}
    for run_dir in runs:
        variant = run_dir.name.split("_")[-2]
        for name, sources in ablations.items():
            r = eval_run(run_dir, list(sources), manifest, ds_id)
            if r:
                r["variant_run"] = run_dir.name
                r["variant"] = variant
                out[name].append(r)
                print(f"{name:4s} {variant} {run_dir.name}: recall@1={r['recall@1']:.3f} dim={r['dim']} "
                      f"feat={r['feature_bytes_per_image']/1024:.0f}KB", flush=True)

    summary = {}
    for name, rows in out.items():
        if not rows:
            continue
        summary[name] = {
            "sources": list(ablations[name]),
            "n_runs": len(rows),
            "dim": rows[0]["dim"],
            "recall@1": float(np.median([r["recall@1"] for r in rows])),
            "recall@5": float(np.median([r["recall@5"] for r in rows])),
            "mAP@10": float(np.median([r["mAP@10"] for r in rows])),
            "feature_bytes_per_image": float(np.median([r["feature_bytes_per_image"] for r in rows])),
            "query_ms_median": float(np.median([r["query_ms_median"] for r in rows])),
        }
    outdir = ROOT / "artifacts" / "experiments"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "EXP-0001-ablations.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
