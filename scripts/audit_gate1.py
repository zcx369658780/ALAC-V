"""Gate 1 审计 v2：全量真实 SHA-256、索引隔离、缓存完整性、确定性与证据。

整改对应：
- P1-2：训练索引与标签分离（白名单 + 无 manifest 属性）
- P1-3：22,000 张图像逐一重算 SHA-256 / 字节数，inventory 使用真实值
- P1-4：缓存完整 view 集合、每 shard keys/shape/dtype/计数、重复 ID、SHA
- P2-1：确定性抽样包含 strong augmentation 视图
- 缺缓存时 fail-closed（不得 PASS）
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
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
SPLITS = ["train", "validation", "test"]


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
    ap.add_argument("--cache-sample", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report: dict = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "dataset_id": load(DATA / "manifests" / "ds0002.json")["dataset_id"],
                    "checks": {}}

    manifests = {s: load(DATA / "manifests" / f"{s}.json") for s in SPLITS}
    labels = {s: load(DATA / "eval" / f"labels_{s}.json") for s in SPLITS}
    indexes = {s: load(DATA / "index" / f"{s}.json") for s in SPLITS}

    # 1) counts
    counts = {}
    for s, m in manifests.items():
        roles = {}
        for v in m["views"]:
            roles[v["role"]] = roles.get(v["role"], 0) + 1
        counts[s] = {"scenes": m["n_scenes"], "views": m["n_views"], "roles": roles,
                     "manifest_sha256": sha256_file(DATA / "manifests" / f"{s}.json"),
                     "index_sha256": sha256_file(DATA / "index" / f"{s}.json")}
    report["checks"]["counts"] = counts
    report["checks"]["counts_ok"] = all(
        counts[s]["scenes"] == {"train": 2000, "validation": 300, "test": 900}[s]
        and counts[s]["views"] == {"train": 16000, "validation": 1500, "test": 4500}[s]
        for s in SPLITS
    )

    # 2) split isolation
    seeds = {s: {v["base_seed"] for v in manifests[s]["views"]} for s in SPLITS}
    vids = {s: {v["view_id"] for v in manifests[s]["views"]} for s in SPLITS}
    report["checks"]["split_isolation"] = {
        "seeds": {f"{a}|{b}": len(seeds[a] & seeds[b]) for a in SPLITS for b in SPLITS if a < b},
        "view_ids": {f"{a}|{b}": len(vids[a] & vids[b]) for a in SPLITS for b in SPLITS if a < b},
        "seed_ranges": {s: manifests[s]["seed_range"] for s in SPLITS},
    }
    report["checks"]["split_isolation_ok"] = all(
        v == 0 for v in report["checks"]["split_isolation"]["seeds"].values()
    ) and all(v == 0 for v in report["checks"]["split_isolation"]["view_ids"].values())

    # 3) full recompute of image SHA-256 + bytes + decode
    actual_sha, bad, dims, size_mismatch, total_bytes = {}, [], set(), [], 0
    for s in SPLITS:
        for v in manifests[s]["views"]:
            p = DATA / v["path"]
            st = p.stat()
            total_bytes += st.st_size
            sha = sha256_file(p)
            actual_sha[v["path"]] = sha
            if sha != v["sha256"]:
                bad.append({"view": v["view_id"], "issue": "sha256_mismatch"})
            if st.st_size != v["bytes"]:
                size_mismatch.append({"view": v["view_id"], "manifest": v["bytes"], "actual": st.st_size})
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
    dup = {}
    for sha in actual_sha.values():
        dup[sha] = dup.get(sha, 0) + 1
    duplicates = {h: c for h, c in dup.items() if c > 1}
    report["checks"]["images"] = {
        "files": len(actual_sha), "total_bytes": total_bytes, "dimensions": sorted(dims),
        "sha_mismatch": len(bad), "bad_examples": bad[:10],
        "bytes_mismatch": len(size_mismatch), "duplicate_sha256": len(duplicates),
        "duplicate_examples": list(duplicates.items())[:5],
    }
    report["checks"]["images_ok"] = (not bad) and (not size_mismatch) and (not duplicates) and dims == {(1024, 768)}

    # 4) eval labels completeness + agreement with manifest
    issues = []
    for s in SPLITS:
        by_id = {v["view_id"]: v for v in manifests[s]["views"]}
        if {sc["scene_id"] for sc in labels[s]["scenes"]} != {sc["scene_id"] for sc in manifests[s]["scenes"]}:
            issues.append({"split": s, "issue": "scene_id_set_mismatch"})
        for sc in labels[s]["scenes"]:
            msc = next(m for m in manifests[s]["scenes"] if m["scene_id"] == sc["scene_id"])
            if sc["targets"] != msc["targets"] or sc["factors"] != msc["factors"]:
                issues.append({"split": s, "scene": sc["scene_id"], "issue": "label_manifest_mismatch"})
            for t in sc["targets"]:
                if t["x"] < 0 or t["y"] < 0 or t["x"] + t["w"] > 1024 or t["y"] + t["h"] > 768:
                    issues.append({"split": s, "scene": sc["scene_id"], "issue": "bbox_out_of_bounds"})
            if s != "train":
                if len(sc["gallery_views"]) != 1 or len(sc["query_views"]) != 4:
                    issues.append({"split": s, "scene": sc["scene_id"], "issue": "gallery_query_count"})
            for vid in sc["gallery_views"] + sc["query_views"]:
                if vid not in by_id:
                    issues.append({"split": s, "scene": sc["scene_id"], "issue": f"unknown_view:{vid}"})
    report["checks"]["eval_labels"] = {"issues": issues[:10], "issue_count": len(issues)}
    report["checks"]["eval_labels_ok"] = not issues

    # 5) deterministic regeneration incl. strong augmentation
    from alacv.synth import augment, render_scene, sha256_image, view_rng

    samples = []
    for s in SPLITS:
        m = manifests[s]
        nv = len(m["scenes"][0]["all_views"])
        for idx in (0, m["n_scenes"] // 2):
            sc = m["scenes"][idx]
            scene = render_scene(s, idx)
            for v in (0, 1 if nv > 1 else 0):
                vid = sc["all_views"][v]
                view = next(x for x in m["views"] if x["view_id"] == vid)
                strong = (v > 0) if s == "train" else (v >= 1)
                img, params = augment(scene.image, view_rng(s, idx, v), strong=strong)
                sha, size = sha256_image(img, int(params["jpeg_quality"]))
                samples.append({"split": s, "scene": idx, "view": v, "strong": strong,
                                "sha_match": sha == view["sha256"], "actual_sha": sha,
                                "manifest_sha": view["sha256"], "size_match": size == view["bytes"]})
    report["checks"]["regeneration"] = {"samples": samples, "ok": all(x["sha_match"] and x["size_match"] for x in samples)}

    # 6) training index / loader isolation
    from alacv.ds0002 import ALLOWED_INDEX_KEYS, Ds0002Dataset

    idx_violations = []
    for s in SPLITS:
        for v in indexes[s]["views"]:
            if set(v) - ALLOWED_INDEX_KEYS:
                idx_violations.append({"split": s, "view": v["view_id"], "extra": sorted(set(v) - ALLOWED_INDEX_KEYS)})
    ds = Ds0002Dataset(DATA / "index" / "train.json", DATA)
    item = ds[0]
    report["checks"]["loader_allowlist"] = {
        "index_violations": idx_violations[:10],
        "item_keys": sorted(item),
        "item_ok": set(item) == {"view_id", "image", "augment"},
        "has_manifest_attr": hasattr(ds, "manifest"),
        "ok": not idx_violations and set(item) == {"view_id", "image", "augment"} and not hasattr(ds, "manifest"),
    }

    # 7) teacher cache — fail-closed
    cache_ok = False
    if not (CACHE / "cache.json").exists():
        report["checks"]["teacher_cache"] = {"status": "MISSING", "ok": False}
    else:
        from alacv.config import load_config
        from alacv.features import DinoEncoder
        from alacv.imaging import open_image
        from alacv.teacher_cache import TeacherCacheReader

        model = load(MODEL_MANIFEST)
        cache = load(CACHE / "cache.json")
        cchecks = {
            "model_matches": cache["model"]["weights_sha256"] == model["weights_sha256"],
            "revision_matches": cache["model"]["revision"] == model["revision"],
            "splits": {},
        }
        for s in SPLITS:
            idx = load(CACHE / f"index_{s}.json")
            expected_ids = [v["view_id"] for v in manifests[s]["views"]]
            got_ids = [vid for sh in idx["shards"] for vid in sh["views"]]
            shard_bad = []
            tensor_bad = []
            for sh in idx["shards"]:
                p = CACHE / sh["path"]
                if sha256_file(p) != sh["sha256"]:
                    shard_bad.append(sh["path"])
                from safetensors.numpy import load_file

                data = load_file(str(p))
                if set(data) != {"cls", "patch"}:
                    tensor_bad.append({"shard": sh["path"], "issue": "keys"})
                    continue
                if data["cls"].dtype != np.float16 or data["patch"].dtype != np.float16:
                    tensor_bad.append({"shard": sh["path"], "issue": "dtype"})
                if data["cls"].shape != (sh["n"], 384) or data["patch"].shape != (sh["n"], 1369, 384):
                    tensor_bad.append({"shard": sh["path"], "issue": "shape"})
            cchecks["splits"][s] = {
                "n_views": idx["n_views"], "n_shards": idx["n_shards"],
                "view_set_equal": sorted(got_ids) == sorted(expected_ids),
                "n_views_equal": idx["n_views"] == len(expected_ids),
                "manifest_sha256_match": idx["manifest_sha256"] == counts[s]["manifest_sha256"],
                "bad_shards": shard_bad, "tensor_issues": tensor_bad,
            }
        # 抽样重算（含 strong 视图）
        cfg = load_config(ROOT / "configs" / "default.yaml")
        enc = DinoEncoder(cfg)
        rng = np.random.default_rng(0)
        sample_ids = []
        for s in SPLITS:
            ids = [v["view_id"] for v in manifests[s]["views"] if v["view_index"] > 0] or [
                v["view_id"] for v in manifests[s]["views"]]
            sample_ids += list(rng.choice(ids, size=min(args.cache_sample // 3 + 1, len(ids)), replace=False))
        max_err = 0.0
        for vid in sample_ids:
            s = vid.split("_")[0]
            reader = TeacherCacheReader(CACHE, s)
            c_cached, p_cached = reader.get(vid)
            v = next(x for x in manifests[s]["views"] if x["view_id"] == vid)
            c_now, p_now = enc.encode_one(open_image(DATA / v["path"]).image)
            max_err = max(max_err, float(np.abs(c_cached.astype(np.float32) - c_now.astype(np.float32)).max()))
            max_err = max(max_err, float(np.abs(p_cached.astype(np.float32) - p_now.astype(np.float32)).max()))
        cchecks["sample_alignment"] = {"n": len(sample_ids), "max_abs_err": max_err, "ok": max_err < 5e-3}
        cchecks["ok"] = (
            cchecks["model_matches"] and cchecks["revision_matches"] and cchecks["sample_alignment"]["ok"]
            and all(v["view_set_equal"] and v["n_views_equal"] and v["manifest_sha256_match"]
                    and not v["bad_shards"] and not v["tensor_issues"] for v in cchecks["splits"].values())
        )
        report["checks"]["teacher_cache"] = cchecks
        cache_ok = cchecks["ok"]
        # 发布脱敏索引元数据（不含特征本身）
        meta_dir = OUT / "cache-index"
        if meta_dir.exists():
            shutil.rmtree(meta_dir)
        meta_dir.mkdir(parents=True)
        for s in SPLITS:
            shutil.copyfile(CACHE / f"index_{s}.json", meta_dir / f"index_{s}.json")

    # 8) sample montage 4x4
    from PIL import ImageDraw

    picks = []
    for s in SPLITS:
        vs = manifests[s]["views"]
        step = max(1, len(vs) // 6)
        picks += [vs[i] for i in range(0, len(vs), step)][:6]
    grid = Image.new("RGB", (4 * 256, 4 * 192), (20, 20, 20))
    for k, v in enumerate(picks[:16]):
        im = Image.open(DATA / v["path"]).convert("RGB").resize((256, 192))
        grid.paste(im, ((k % 4) * 256, (k // 4) * 192))
    grid.save(OUT / "sample-grid.png", quality=88)

    # 9) resources + inventory (actual SHA)
    import psutil

    report["checks"]["resources"] = {
        "disk_free_gb": round(psutil.disk_usage(str(ROOT)).free / 1024**3, 2),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
        "dataset_bytes": total_bytes,
        "cache_bytes": sum((CACHE / sh["path"]).stat().st_size for s in SPLITS
                           for sh in load(CACHE / f"index_{s}.json")["shards"])
        if (CACHE / "index_train.json").exists() else 0,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report["checks"]["resources"]["gpu"] = {"name": p.name, "vram_mb": round(p.total_memory / 1024**2, 1)}
    except Exception:
        pass

    lines = [f"{v}  {path}" for path, v in sorted(actual_sha.items())]
    (OUT / "sha256-inventory.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report["ok"] = bool(
        report["checks"]["counts_ok"] and report["checks"]["split_isolation_ok"]
        and report["checks"]["images_ok"] and report["checks"]["eval_labels_ok"]
        and report["checks"]["regeneration"]["ok"] and report["checks"]["loader_allowlist"]["ok"]
        and cache_ok
    )
    (OUT / "gate1-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# DS-0002 Gate 1 审计（v2）", "",
          f"- 生成时间：{report['generated_at_utc']}",
          f"- 数据集：`{report['dataset_id']}`",
          f"- 总判定：**{'PASS' if report['ok'] else 'FAIL'}**",
          "| 检查 | 结果 |", "|---|---|"]
    for k in ("counts_ok", "split_isolation_ok", "images_ok", "eval_labels_ok", "regeneration", "loader_allowlist", "teacher_cache"):
        v = report["checks"].get(k)
        res = v.get("ok", "n/a") if isinstance(v, dict) else v
        md.append(f"| {k} | {res} |")
    md += ["", "明细见 `gate1-audit.json`、`sha256-inventory.txt`、`cache-index/`、`sample-grid.png`。"]
    (OUT / "gate1-audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "out": str(OUT)}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
