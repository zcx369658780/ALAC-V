"""Gate 1 审计：完整性、隔离、确定性、标签分离、缓存对齐、资源与隐私。

输出（可提交）：
  artifacts/evidence/DS-0002/gate1-audit.json
  artifacts/evidence/DS-0002/gate1-audit.md
  artifacts/evidence/DS-0002/sha256-inventory.txt
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
from PIL import Image  # noqa: E402

DATA = ROOT / "data" / "ds0002"
CACHE = DATA / "features" / "teacher"
OUT = ROOT / "artifacts" / "evidence" / "DS-0002"
MODEL_MANIFEST = ROOT / "data" / "manifests" / "model-dinov2_vits14.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-sample", type=int, default=20)
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    report: dict = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checks": {}}
    t0 = time.perf_counter()

    splits = ["train", "validation", "test"]
    manifests = {s: load(DATA / "manifests" / f"{s}.json") for s in splits}
    labels = {s: load(DATA / "eval" / f"labels_{s}.json") for s in splits}

    # 1) counts
    counts = {}
    for s, m in manifests.items():
        roles = {}
        for v in m["views"]:
            roles[v["role"]] = roles.get(v["role"], 0) + 1
        counts[s] = {"scenes": m["n_scenes"], "views": m["n_views"], "roles": roles,
                     "manifest_sha256": sha256_file(DATA / "manifests" / f"{s}.json")}
    report["checks"]["counts"] = counts
    expected = {"train": (2000, 16000), "validation": (300, 1500), "test": (900, 4500)}
    report["checks"]["counts_ok"] = all(
        counts[s]["scenes"] == expected[s][0] and counts[s]["views"] == expected[s][1] for s in splits
    )

    # 2) split isolation
    seed_sets = {s: {v["base_seed"] for v in manifests[s]["views"]} for s in splits}
    view_sets = {s: {v["view_id"] for v in manifests[s]["views"]} for s in splits}
    sha_sets = {s: {v["sha256"] for v in manifests[s]["views"]} for s in splits}
    overlap = {
        "seeds_train_val": len(seed_sets["train"] & seed_sets["validation"]),
        "seeds_train_test": len(seed_sets["train"] & seed_sets["test"]),
        "seeds_val_test": len(seed_sets["validation"] & seed_sets["test"]),
        "view_ids": len(view_sets["train"] & view_sets["validation"]) + len(view_sets["train"] & view_sets["test"])
        + len(view_sets["validation"] & view_sets["test"]),
        "image_sha256": len(sha_sets["train"] & sha_sets["validation"]) + len(sha_sets["train"] & sha_sets["test"])
        + len(sha_sets["validation"] & sha_sets["test"]),
        "seed_ranges": {s: manifests[s]["seed_range"] for s in splits},
    }
    report["checks"]["split_isolation"] = overlap
    report["checks"]["split_isolation_ok"] = sum(overlap[k] for k in
                                                 ("seeds_train_val", "seeds_train_test", "seeds_val_test",
                                                  "view_ids", "image_sha256")) == 0

    # 3) decode + dimensions + duplicates
    bad, dims, total_bytes = [], set(), 0
    for s, m in manifests.items():
        for v in m["views"]:
            p = DATA / v["path"]
            total_bytes += p.stat().st_size
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    dims.add(im.size)
                    if im.size != (v["width"], v["height"]):
                        bad.append({"view": v["view_id"], "issue": "size_mismatch"})
                    if im.mode != "RGB":
                        bad.append({"view": v["view_id"], "issue": f"mode={im.mode}"})
            except Exception as exc:
                bad.append({"view": v["view_id"], "issue": f"decode:{type(exc).__name__}"})
    dup_counts = {}
    for s in splits:
        for h in sha_sets[s]:
            dup_counts[h] = dup_counts.get(h, 0) + 1
    duplicates = {h: c for h, c in dup_counts.items() if c > 1}
    report["checks"]["images"] = {
        "files": sum(len(manifests[s]["views"]) for s in splits),
        "total_bytes": total_bytes,
        "dimensions": sorted(dims),
        "bad": bad[:20], "bad_count": len(bad),
        "duplicate_hashes": len(duplicates),
        "duplicate_examples": list(duplicates.items())[:5],
    }
    report["checks"]["images_ok"] = not bad and not duplicates and dims == {(1024, 768)}

    # 4) eval labels
    label_issues = []
    for s in splits:
        m = manifests[s]
        by_id = {v["view_id"]: v for v in m["views"]}
        scene_ids = {sc["scene_id"] for sc in m["scenes"]}
        lab_ids = {sc["scene_id"] for sc in labels[s]["scenes"]}
        if scene_ids != lab_ids:
            label_issues.append({"split": s, "issue": "scene_id_set_mismatch"})
        for sc in labels[s]["scenes"]:
            if not sc["targets"]:
                label_issues.append({"split": s, "scene": sc["scene_id"], "issue": "no_target"})
            for t in sc["targets"]:
                if t["x"] < 0 or t["y"] < 0 or t["x"] + t["w"] > 1024 or t["y"] + t["h"] > 768:
                    label_issues.append({"split": s, "scene": sc["scene_id"], "issue": "bbox_out_of_bounds"})
            if s != "train":
                if len(sc["gallery_views"]) != 1 or len(sc["query_views"]) != 4:
                    label_issues.append({"split": s, "scene": sc["scene_id"], "issue": "gallery_query_count"})
                for vid in sc["gallery_views"] + sc["query_views"]:
                    if vid not in by_id:
                        label_issues.append({"split": s, "scene": sc["scene_id"], "issue": f"unknown_view:{vid}"})
    report["checks"]["eval_labels"] = {"issues": label_issues[:20], "issue_count": len(label_issues)}
    report["checks"]["eval_labels_ok"] = not label_issues

    # 5) deterministic regeneration spot-check
    from alacv.synth import augment, render_scene, sha256_image, view_rng

    spot = []
    for s in splits:
        m = manifests[s]
        for idx in (0, m["n_scenes"] // 2, m["n_scenes"] - 1):
            sc = m["scenes"][idx]
            scene = render_scene(s, idx)
            vid = sc["all_views"][0]
            view = next(v for v in m["views"] if v["view_id"] == vid)
            img, params = augment(scene.image, view_rng(s, idx, 0), strong=False)
            q = int(params.get("jpeg_quality", 90))
            sha, size = sha256_image(img, q)
            spot.append({"split": s, "scene": idx, "sha_match": sha == view["sha256"], "regen_sha": sha,
                         "manifest_sha": view["sha256"]})
    report["checks"]["regeneration"] = {"samples": spot, "ok": all(x["sha_match"] for x in spot)}

    # 6) loader allowlist
    from alacv.ds0002 import ALLOWED_ITEM_KEYS, Ds0002Dataset

    ds = Ds0002Dataset(DATA / "manifests" / "train.json", DATA)
    item = ds[0]
    keys = set(item.keys())
    report["checks"]["loader_allowlist"] = {
        "allowed": sorted(ALLOWED_ITEM_KEYS), "actual": sorted(keys),
        "forbidden_present": sorted(keys - set(ALLOWED_ITEM_KEYS)),
        "ok": keys == set(ALLOWED_ITEM_KEYS),
    }

    # 7) teacher cache
    if not args.skip_cache and CACHE.exists() and (CACHE / "cache.json").exists():
        from alacv.teacher_cache import TeacherCacheReader

        model = load(MODEL_MANIFEST)
        cache = load(CACHE / "cache.json")
        cchecks = {"model_matches": cache["model"]["weights_sha256"] == model["weights_sha256"],
                   "revision_matches": cache["model"]["revision"] == model["revision"], "splits": {}}
        for s in splits:
            idx = load(CACHE / f"index_{s}.json")
            bad_shards = [sh["path"] for sh in idx["shards"] if sha256_file(CACHE / sh["path"]) != sh["sha256"]]
            cchecks["splits"][s] = {
                "n_views": idx["n_views"], "n_shards": idx["n_shards"],
                "manifest_sha256_match": idx["manifest_sha256"] == counts[s]["manifest_sha256"],
                "bad_shards": bad_shards,
                "views_match_manifest": idx["n_views"] == manifests[s]["n_views"],
            }
        # sample alignment: recompute a few views
        from alacv.config import load_config
        from alacv.features import DinoEncoder
        from alacv.imaging import open_image

        cfg = load_config(ROOT / "configs" / "default.yaml")
        enc = DinoEncoder(cfg)
        rng = np.random.default_rng(0)
        sample_ids = []
        for s in splits:
            ids = [v["view_id"] for v in manifests[s]["views"]]
            sample_ids += list(rng.choice(ids, size=min(args.cache_sample, len(ids)), replace=False))
        max_err = 0.0
        for vid in sample_ids:
            s = vid.split("_")[0]
            reader = TeacherCacheReader(CACHE, s)
            c_cached, p_cached = reader.get(vid)
            v = next(v for v in manifests[s]["views"] if v["view_id"] == vid)
            img = open_image(DATA / v["path"]).image
            c_now, p_now = enc.encode_one(img)
            max_err = max(max_err, float(np.abs(c_cached.astype(np.float32) - c_now.astype(np.float32)).max()))
            max_err = max(max_err, float(np.abs(p_cached.astype(np.float32) - p_now.astype(np.float32)).max()))
        cchecks["sample_alignment"] = {"n": len(sample_ids), "max_abs_err": max_err, "ok": max_err < 5e-3}
        cchecks["ok"] = all([cchecks["model_matches"], cchecks["revision_matches"],
                             cchecks["sample_alignment"]["ok"]]
                            + [v["bad_shards"] == [] and v["views_match_manifest"] and v["manifest_sha256_match"]
                               for v in cchecks["splits"].values()])
        report["checks"]["teacher_cache"] = cchecks
    else:
        report["checks"]["teacher_cache"] = {"status": "NOT RUN"}

    # 8) resources + inventory
    import psutil

    report["checks"]["resources"] = {
        "disk_free_gb": round(psutil.disk_usage(str(ROOT)).free / 1024**3, 2),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "elapsed_s": round(time.perf_counter() - t0, 2),
        "dataset_bytes": total_bytes,
        "cache_bytes": sum(
            (CACHE / sh["path"]).stat().st_size
            for s in splits
            for sh in load(CACHE / f"index_{s}.json")["shards"]
        ) if (CACHE / f"index_train.json").exists() else 0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report["checks"]["resources"]["gpu"] = {"name": p.name, "vram_mb": round(p.total_memory / 1024**2, 1)}
    except Exception:
        pass

    lines = []
    for s in splits:
        for v in manifests[s]["views"]:
            lines.append(f"{v['sha256']}  {v['path']}")
    (OUT / "sha256-inventory.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report["ok"] = all(v is True or (isinstance(v, dict) and v.get("ok", True)) for v in [
        report["checks"]["counts_ok"], report["checks"]["split_isolation_ok"], report["checks"]["images_ok"],
        report["checks"]["eval_labels_ok"], report["checks"]["regeneration"]["ok"],
        report["checks"]["loader_allowlist"]["ok"], report["checks"].get("teacher_cache", {"ok": True}),
    ])
    (OUT / "gate1-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# DS-0002 Gate 1 审计", "",
          f"- 生成时间：{report['generated_at_utc']}",
          f"- 总判定：**{'PASS' if report['ok'] else 'FAIL'}**",
          f"- 场景/视图：" + "；".join(f"{s} {counts[s]['scenes']}/{counts[s]['views']}" for s in splits),
          f"- 图像总量：{total_bytes/1e9:.2f} GB", "",
          "| 检查 | 结果 |", "|---|---|"]
    for k, v in report["checks"].items():
        if k in ("counts", "split_isolation", "images", "eval_labels", "teacher_cache", "resources"):
            continue
        md.append(f"| {k} | {v} |")
    md += ["", "明细见 `gate1-audit.json`。"]
    (OUT / "gate1-audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "out": str(OUT)}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
