"""生成 DS-0002（Gate 1）：确定性场景、分组切分、manifest 与评估标签 sidecar。

支持多进程（每个场景独立、确定性），以便在 20 线程 CPU 上快速生成 3,200 个场景。

用法：
    python scripts/generate_ds0002.py --config configs/ds0002.yaml [--workers 12]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from alacv.synth import (  # noqa: E402
    GENERATOR_VERSION,
    PRNG,
    WIDTH,
    HEIGHT,
    augment,
    base_seed,
    render_scene,
    view_rng,
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _gen_scene(task: tuple) -> tuple[dict, list[dict]]:
    split, index, out_root_s, n_views, n_gallery, subsampling = task
    out_root = Path(out_root_s)
    scene = render_scene(split, index)
    sdir = out_root / "images" / split / f"{index:05d}"
    sdir.mkdir(parents=True, exist_ok=True)
    view_rows, gallery_views, query_views = [], [], []
    for v in range(n_views):
        rng = view_rng(split, index, v)
        if split == "train":
            strong, role = v > 0, "train"
        else:
            strong = v >= n_gallery
            role = "gallery" if v < n_gallery else "query"
        img, params = augment(scene.image, rng, strong=strong)
        quality = int(params.pop("jpeg_quality", 90))
        p = sdir / f"v{v}.jpg"
        img.save(p, format="JPEG", quality=quality, subsampling=subsampling)
        sha = sha256_file(p)
        vid = f"{split}_{index:05d}_v{v}"
        view_rows.append(
            {
                "view_id": vid,
                "scene_id": scene.scene_id,
                "base_seed": scene.base_seed,
                "split": split,
                "view_index": v,
                "role": role,
                "path": p.relative_to(out_root).as_posix(),
                "sha256": sha,
                "bytes": p.stat().st_size,
                "width": img.width,
                "height": img.height,
                "augment": params,
            }
        )
        (gallery_views if role == "gallery" else query_views).append(vid)
    scene_row = {
        "scene_id": scene.scene_id,
        "base_seed": scene.base_seed,
        "split": split,
        "targets": scene.targets,
        "factors": scene.factors,
        "gallery_views": gallery_views,
        "query_views": query_views,
        "all_views": [r["view_id"] for r in view_rows],
    }
    return scene_row, view_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ds0002.yaml")
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only-scenes", type=int, default=None, help="冒烟")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    out_root = ROOT / cfg["dataset"]["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(exist_ok=True)
    (out_root / "eval").mkdir(exist_ok=True)

    config_sha = sha256_text(json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    generator_sha = sha256_file(Path(__file__).resolve())
    scene_code_sha = sha256_file(ROOT / "src" / "alacv" / "synth" / "scene.py")
    subsampling = int(cfg["image"]["jpeg_subsampling"])

    summaries = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        sc = cfg["splits"][split]
        n_scenes = args.only_scenes or int(sc["scenes"])
        n_views = int(sc["views"])
        n_gallery = int(sc.get("gallery_views", 0))
        t0 = time.perf_counter()
        tasks = [(split, i, str(out_root), n_views, n_gallery, subsampling) for i in range(n_scenes)]
        scenes, views = [], []
        if args.workers <= 1:
            for done, t in enumerate(tasks, 1):
                srow, vrows = _gen_scene(t)
                scenes.append(srow); views.extend(vrows)
                if done % 100 == 0 or done == n_scenes:
                    print(f"[{split}] {done}/{n_scenes} scenes ({time.perf_counter()-t0:.0f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_gen_scene, t): t[1] for t in tasks}
                done = 0
                for fut in as_completed(futs):
                    srow, vrows = fut.result()
                    scenes.append(srow)
                    views.extend(vrows)
                    done += 1
                    if done % 100 == 0 or done == n_scenes:
                        print(f"[{split}] {done}/{n_scenes} scenes ({time.perf_counter()-t0:.0f}s)", flush=True)
        scenes.sort(key=lambda s: s["base_seed"])
        views.sort(key=lambda v: v["view_id"])
        manifest = {
            "dataset_id": cfg["dataset"]["id"],
            "split": split,
            "generator_version": GENERATOR_VERSION,
            "prng": PRNG,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "config_sha256": config_sha,
            "generator_script_sha256": generator_sha,
            "scene_code_sha256": scene_code_sha,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "width": WIDTH,
            "height": HEIGHT,
            "n_scenes": len(scenes),
            "n_views": len(views),
            "seed_range": [base_seed(split, 0), base_seed(split, n_scenes - 1)],
            "scenes": scenes,
            "views": views,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        (out_root / "manifests" / f"{split}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        labels = {
            "dataset_id": cfg["dataset"]["id"],
            "split": split,
            "note": "evaluation-only sidecar; must never be read by training code",
            "scenes": [
                {
                    "scene_id": s["scene_id"],
                    "base_seed": s["base_seed"],
                    "targets": s["targets"],
                    "factors": s["factors"],
                    "gallery_views": s["gallery_views"],
                    "query_views": s["query_views"],
                }
                for s in scenes
            ],
        }
        (out_root / "eval" / f"labels_{split}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries[split] = {"n_scenes": len(scenes), "n_views": len(views),
                            "seed_range": manifest["seed_range"], "elapsed_s": manifest["elapsed_s"]}
        print(f"[{split}] done: {len(scenes)} scenes / {len(views)} views", flush=True)

    if not args.only_scenes:
        global_manifest = {
            "dataset_id": cfg["dataset"]["id"],
            "generator_version": GENERATOR_VERSION,
            "prng": PRNG,
            "config_sha256": config_sha,
            "generator_script_sha256": generator_sha,
            "scene_code_sha256": scene_code_sha,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "splits": summaries,
            "isolation": "split seed ranges are disjoint by construction; see audit",
        }
        (out_root / "manifests" / "ds0002.json").write_text(
            json.dumps(global_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"out": str(out_root), "splits": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
