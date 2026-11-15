"""DS-0002 v2 生成（Gate 1 整改）：确定性场景、分组切分、训练索引与评估标签分离。

产物：
  data/ds0002/images/...                       图像（本地，gitignored）
  data/ds0002/manifests/{split}.json           审计用完整 manifest（含场景标签）
  data/ds0002/index/{split}.json               训练索引（仅 view_id/path/augment/pair_key）
  data/ds0002/eval/labels_{split}.json         评估标签 sidecar
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from alacv.synth import (  # noqa: E402
    GENERATOR_VERSION,
    PRNG,
    SPLIT_SEED_BASE,
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


def pair_key(scene_id: str, salt: str) -> str:
    return "g" + sha256_text(f"{salt}|pair|{scene_id}")[:16]


def view_id_for(split: str, index: int, view: int, salt: str) -> str:
    return "v" + sha256_text(f"{salt}|{split}|{index}|{view}")[:16]


def _gen_scene(task: tuple) -> tuple[dict, list[dict]]:
    split, index, out_root_s, n_views, n_gallery, subsampling, salt, q_gallery, q_strong = task
    out_root = Path(out_root_s)
    scene = render_scene(split, index)
    vid0 = view_id_for(split, index, 0, salt)
    sdir = out_root / "images" / vid0[:2]
    sdir.mkdir(parents=True, exist_ok=True)
    view_rows, gallery_views, query_views = [], [], []
    for v in range(n_views):
        rng = view_rng(split, index, v)
        strong = (v > 0) if split == "train" else (v >= n_gallery)
        role = "train" if split == "train" else ("gallery" if v < n_gallery else "query")
        img, params = augment(scene.image, rng, strong=strong, q_gallery=q_gallery, q_strong=q_strong)
        quality = int(params["jpeg_quality"])
        vid = view_id_for(split, index, v, salt)
        p = out_root / "images" / vid[:2] / f"{vid}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p, format="JPEG", quality=quality, subsampling=subsampling)
        view_rows.append(
            {
                "view_id": vid,
                "scene_id": scene.scene_id,
                "base_seed": scene.base_seed,
                "split": split,
                "view_index": v,
                "role": role,
                "path": p.relative_to(out_root).as_posix(),
                "sha256": sha256_file(p),
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
        "pair_key": pair_key(scene.scene_id, salt),
        "targets": scene.targets,
        "distractors": scene.distractors,
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
    ap.add_argument("--only-scenes", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if int(cfg["dataset"]["width"]) != WIDTH or int(cfg["dataset"]["height"]) != HEIGHT:
        raise SystemExit("配置宽高与生成器常量不一致")
    for s, sc in cfg["splits"].items():
        if int(sc["seed_base"]) != SPLIT_SEED_BASE[s]:
            raise SystemExit(f"{s} 的 seed_base 与生成器分段不一致")

    out_root = ROOT / cfg["dataset"]["out_dir"]
    for sub in ("manifests", "index", "eval"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    config_sha = sha256_text(json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    salt_file = out_root / "eval" / "identity_salt.json"
    if salt_file.exists():
        salt = json.loads(salt_file.read_text(encoding="utf-8"))["salt"]
    else:
        import secrets
        salt = secrets.token_hex(8)
        salt_file.write_text(json.dumps({"salt": salt, "note": "eval-only; never expose to training"},
                                        indent=2), encoding="utf-8")
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
        scenes, views = [], []
        q_g = int(cfg["image"]["jpeg_quality_gallery"])
        q_s = (int(cfg["image"]["jpeg_quality_train_strong_min"]),
               int(cfg["image"]["jpeg_quality_train_strong_max"]))
        for i in range(n_scenes):
            scene_row, view_rows = _gen_scene(
                (split, i, str(out_root), n_views, n_gallery, subsampling, salt, q_g, q_s))
            scenes.append(scene_row)
            views.extend(view_rows)
            if (i + 1) % 100 == 0 or i + 1 == n_scenes:
                print(f"[{split}] {i+1}/{n_scenes} scenes ({time.perf_counter()-t0:.0f}s)", flush=True)

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

        # 训练索引：只投影允许字段；pair_key 为不可逆的匿名分组键
        index = {
            "dataset_id": cfg["dataset"]["id"],
            "split": split,
            "n_views": len(views),
            "views": [
                {
                    "view_id": v["view_id"],
                    "path": v["path"],
                    "augment": v["augment"],
                    "pair_key": pair_key(v["scene_id"], salt),
                }
                for v in views
            ],
        }
        (out_root / "index" / f"{split}.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        labels = {
            "dataset_id": cfg["dataset"]["id"],
            "split": split,
            "note": "evaluation-only sidecar; training code must not read this file",
            "scenes": [
                {
                    "scene_id": s["scene_id"],
                    "pair_key": s["pair_key"],
                    "base_seed": s["base_seed"],
                    "targets": s["targets"],
                    "distractors": s["distractors"],
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
