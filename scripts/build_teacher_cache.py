"""构建 DS-0002 冻结 DINOv2 教师特征缓存（Gate 1）。

只使用 Phase 1 已固定的本地权重；不联网、不训练。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from alacv.config import load_config  # noqa: E402
from alacv.features import DinoEncoder  # noqa: E402
from alacv.imaging import open_image  # noqa: E402
from alacv.teacher_cache import TeacherCacheWriter, sha256_file  # noqa: E402

DATA = ROOT / "data" / "ds0002"
CACHE = DATA / "features" / "teacher"
MODEL_MANIFEST = ROOT / "data" / "manifests" / "model-dinov2_vits14.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="冒烟：每 split 只编码前 N 个 view")
    args = ap.parse_args()

    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    ds_manifest_path = DATA / "manifests" / "ds0002.json"
    ds_manifest = json.loads(ds_manifest_path.read_text(encoding="utf-8"))
    dataset_id = ds_manifest["dataset_id"]
    cfg = load_config(ROOT / "configs" / "default.yaml")
    encoder = DinoEncoder(cfg)
    actual_sha = sha256_file(cfg.path("model.local_dir") / manifest["weights_file"])
    if actual_sha != manifest["weights_sha256"]:
        print(f"FATAL: 权重 SHA-256 不匹配: {actual_sha} != {manifest['weights_sha256']}", file=sys.stderr)
        return 2
    if encoder.model_id != manifest["model_id"]:
        print("FATAL: 模型 ID 不匹配", file=sys.stderr)
        return 2

    cache_meta = {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "weights_sha256": manifest["weights_sha256"],
        "weights_file": manifest["weights_file"],
        "license": manifest["license"],
        "input_size": int(cfg.get("model.input_size")),
        "patch_size": int(cfg.get("model.patch_size")),
        "num_params": encoder.num_params,
    }

    CACHE.mkdir(parents=True, exist_ok=True)
    summary = {"cache_version": "ds0002-teacher-v2", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "dataset_id": dataset_id, "dataset_manifest_sha256": sha256_file(ds_manifest_path), "model": cache_meta, "splits": {}, "command": "scripts/build_teacher_cache.py",
               "python": platform.python_version(), "numpy": np.__version__,
               "torch": __import__("torch").__version__}
    t_all = time.perf_counter()
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        mp = DATA / "manifests" / f"{split}.json"
        m = json.loads(mp.read_text(encoding="utf-8"))
        views = m["views"][: args.limit] if args.limit else m["views"]
        writer = TeacherCacheWriter(CACHE, split, cache_meta, sha256_file(mp), shard_size=args.shard_size)
        t0 = time.perf_counter()
        for i in range(0, len(views), args.batch):
            chunk = views[i : i + args.batch]
            imgs = [open_image(DATA / v["path"]).image for v in chunk]
            fb = encoder.encode_pil(imgs)
            for j, v in enumerate(chunk):
                writer.add(v["view_id"], fb.cls[j], fb.patch[j])
            if (i // args.batch + 1) % 20 == 0:
                done = min(i + args.batch, len(views))
                print(f"[{split}] {done}/{len(views)} views ({time.perf_counter()-t0:.0f}s)", flush=True)
        idx = writer.close()
        summary["splits"][split] = {
            "n_views": idx["n_views"], "n_shards": idx["n_shards"],
            "bytes": sum(s["bytes"] for s in idx["shards"]),
            "index_sha256": sha256_file(CACHE / f"index_{split}.json"),
            "manifest_sha256": idx["manifest_sha256"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        print(f"[{split}] cached {idx['n_views']} views in {idx['n_shards']} shards", flush=True)

    summary["total_elapsed_s"] = round(time.perf_counter() - t_all, 2)
    (CACHE / "cache.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "model"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
