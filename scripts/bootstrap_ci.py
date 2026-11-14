"""对 EXP-0001 的 per-query 命中做配对 bootstrap，给出 Recall@1 差异的 95% CI。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.config import load_config  # noqa: E402
from alacv.retrieval import RetrievalIndex  # noqa: E402
from alacv.store import VisualStore  # noqa: E402


def run_hits(run_dir: Path, variant: str, manifest: dict) -> dict[str, int]:
    ws = run_dir / "store"
    cfg = load_config(
        ROOT / "configs" / "experiment_abc.yaml",
        overrides={
            "paths": {
                "raw_dir": str(ws / "raw"),
                "processed_dir": str(ws / "processed"),
                "features_dir": str(ws / "features"),
                "index_db": str(ws / "index" / "alacv.sqlite"),
                "manifests_dir": str(ROOT / "data" / "manifests"),
            }
        },
    )
    store = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))
    sources = list(cfg.require("encoding.variants")[variant])
    by_sha = {f["sha256"]: store.image_by_sha(f["sha256"])["image_id"] for f in manifest["files"]}
    gallery = [(f["base_id"], by_sha[f["sha256"]]) for f in manifest["files"] if f["role"] == "base"]
    queries = [(f["base_id"], by_sha[f["sha256"]]) for f in manifest["files"] if f["role"] == "query"]
    index = RetrievalIndex(store, [g for _, g in gallery], sources)
    hits = {}
    for bid, qid in queries:
        ranked = [i for i, _ in index.search(index.descriptor_of(qid), 1)]
        hits[qid] = 1 if ranked[0] in {g for b, g in gallery if b == bid} else 0
    store.close()
    return hits


def main() -> int:
    manifest = json.loads(
        (ROOT / "data" / "manifests" / "ds-0001-alacv-synthetic-v1.json").read_text(encoding="utf-8")
    )
    runs = [p for p in (ROOT / "artifacts" / "runs").iterdir() if p.is_dir() and "EXP-0001" in p.name]
    data: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    for rd in runs:
        variant = rd.name.split("_")[-2]
        if variant in data and (rd / "metrics.json").exists():
            data[variant].append(run_hits(rd, variant, manifest))

    rng = np.random.default_rng(0)
    out = {}
    for variant, hist in data.items():
        keys = sorted(hist[0].keys())
        mat = np.array([[h[k] for k in keys] for h in hist])  # (reps, queries)
        out[variant] = {k: float(mat[:, i].mean()) for i, k in enumerate(keys)}

    def boot_diff(a: str, b: str, n: int = 5000) -> dict:
        keys = sorted(data[a][0].keys())
        ma = np.array([[h[k] for k in keys] for h in data[a]])
        mb = np.array([[h[k] for k in keys] for h in data[b]])
        idx = rng.integers(0, len(keys), size=(n, len(keys)))
        diffs = []
        for i in range(n):
            diffs.append(ma[:, idx[i]].mean() - mb[:, idx[i]].mean())
        d = np.array(diffs)
        return {
            "delta": float(ma.mean() - mb.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        }

    pairs = {"B-A": boot_diff("B", "A"), "C-A": boot_diff("C", "A"), "C-B": boot_diff("C", "B")}
    result = {"recall@1": out, "paired_bootstrap": pairs}
    path = ROOT / "artifacts" / "experiments" / "EXP-0001-ci.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
