"""WP2 修订：按 EXP-0001 §5.9 预注册口径，在 **ground-truth bbox** 上重算回溯指标。

- 主指标（预注册）：GT bbox 区域的派生表示可见像素保真度。
- 次指标（探索性）：模型选中 ROI（attended ROI）上的同一保真度，单独命名，不替代主指标。
- 不重跑编码，不加载 DINOv2；只读历史 run 的 store。
- 使用每个 run 自己的 config.snapshot 恢复参数，并验证九次参数一致。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.backtrack import crop_region, recover_gt_region, recover_variant, visible_region  # noqa: E402
from alacv.config import Config  # noqa: E402
from alacv.foveation import select_rois  # noqa: E402
from alacv.imaging import open_image  # noqa: E402
from alacv.metrics import psnr, ssim  # noqa: E402
from alacv.store import VisualStore  # noqa: E402

RUNS = ROOT / "artifacts" / "runs"
OUT = ROOT / "artifacts" / "experiments" / "EXP-0001-backtrack-corrected.json"
MANIFEST = ROOT / "data" / "manifests" / "ds-0001-alacv-synthetic-v1.json"
SANITIZATION = ROOT / "data" / "manifests" / "ds-0001-manifest-sanitization.json"
MODEL_MANIFEST = ROOT / "data" / "manifests" / "model-dinov2_vits14.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def param_digest(data: dict) -> str:
    payload = {k: v for k, v in data.items() if k not in ("paths", "experiment", "logging", "dataset")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def build_cfg(run_dir: Path):
    ws = run_dir / "store"
    data = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    data["paths"] = {
        "raw_dir": str(ws / "raw"),
        "processed_dir": str(ws / "processed"),
        "features_dir": str(ws / "features"),
        "index_db": str(ws / "index" / "alacv.sqlite"),
        "manifests_dir": str(ROOT / "data" / "manifests"),
    }
    return Config(data, root=ROOT), param_digest(data)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def union_coverage(roi_boxes, gt_box) -> float:
    gx, gy, gw, gh = gt_box
    if gw <= 0 or gh <= 0:
        return 0.0
    mask = np.zeros((gh, gw), dtype=bool)
    for bx, by, bw, bh in roi_boxes:
        x0, y0 = max(bx, gx), max(by, gy)
        x1, y1 = min(bx + bw, gx + gw), min(by + bh, gy + gh)
        if x1 > x0 and y1 > y0:
            mask[y0 - gy : y1 - gy, x0 - gx : x1 - gx] = True
    return float(mask.sum()) / float(gw * gh)


def find_runs() -> dict[tuple[str, int], Path]:
    out = {}
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and "EXP-0001" in p.name):
        parts = d.name.split("_")
        out[(parts[-2], int(parts[-1][1:]))] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bases = [f for f in manifest["files"] if f["role"] == "base"]
    runs = find_runs()
    reps = sorted({r for (_, r) in runs})
    if len(runs) != 9:
        raise SystemExit(f"期望 9 个 run，实际 {len(runs)}")

    digests = {}
    per_variant_gt: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    per_variant_att: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    per_image: list[dict] = []

    for rep in reps:
        if not all((v, rep) in runs for v in "ABC"):
            continue
        cfgs, stores = {}, {}
        for v in "ABC":
            cfg, dg = build_cfg(runs[(v, rep)])
            cfgs[v] = cfg
            digests[(v, rep)] = dg
            stores[v] = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))

        gt_stats = {v: {"psnr": [], "ssim": [], "cov": [], "excl": 0} for v in "ABC"}
        att_stats = {v: {"psnr": [], "ssim": [], "cov": [], "excl": 0} for v in "ABC"}

        for f in bases:
            gt_box = tuple(int(x) for x in f["gt_bbox"])
            iid = stores["C"].image_by_sha(f["sha256"])["image_id"]
            src_c = open_image(cfgs["C"].path("paths.raw_dir") / stores["C"].get_image(iid)["rel_path"]).image
            rois = stores["C"].rois_of(iid)
            if rois:
                att_box = (rois[0]["x"], rois[0]["y"], rois[0]["w"], rois[0]["h"])
            else:
                r = select_rois(src_c, roi_frac=float(cfgs["C"].get("foveation.roi_frac", 0.25)), max_rois=1)[0]
                att_box = (r.x, r.y, r.w, r.h)

            roi_boxes = [(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in rois]
            row = {
                "rep": rep,
                "base_id": f["base_id"],
                "gt_box": gt_box,
                "attended_box": att_box,
                "roi_boxes": roi_boxes,
                "roi_hit_iou03": {
                    "roi_00": bool(roi_boxes) and iou(roi_boxes[0], gt_box) >= 0.3,
                    "any": any(iou(b, gt_box) >= 0.3 for b in roi_boxes),
                },
                "roi_union_coverage_of_gt": union_coverage(roi_boxes, gt_box),
            }
            for label, box, stats in (("gt", gt_box, gt_stats), ("attended", att_box, att_stats)):
                region = visible_region(box, src_c.size, int(cfgs["C"].get("model.input_size", 518)))
                row[f"{label}_coverage"] = region.coverage
                if not region.visible:
                    for v in "ABC":
                        stats[v]["excl"] += 1
                    row[label] = None
                    continue
                gtp = crop_region(src_c, region.box)
                row[label] = {}
                for v in "ABC":
                    if label == "gt":
                        rec, _ = recover_gt_region(cfgs[v], stores[v], iid, v, box)
                    else:
                        rec, _ = recover_variant(cfgs[v], stores[v], iid, v, box)
                    if rec is None or rec.shape != gtp.shape:
                        stats[v]["excl"] += 1
                        row[label][v] = None
                        continue
                    p, s = psnr(gtp, rec), ssim(gtp, rec)
                    stats[v]["psnr"].append(p)
                    stats[v]["ssim"].append(s)
                    stats[v]["cov"].append(region.coverage)
                    row[label][v] = {"psnr": p, "ssim": s}
            per_image.append(row)

        roi_rows = [r for r in per_image if r["rep"] == rep]
        roi_stats = {
            "roi_00_hit_rate_iou03": float(np.mean([r["roi_hit_iou03"]["roi_00"] for r in roi_rows])),
            "any_roi_hit_rate_iou03": float(np.mean([r["roi_hit_iou03"]["any"] for r in roi_rows])),
            "mean_union_coverage_of_gt": float(np.mean([r["roi_union_coverage_of_gt"] for r in roi_rows])),
            "n_images": len(roi_rows),
        }
        for target, stats in (("gt_bbox", gt_stats), ("attended_roi", att_stats)):
            for v in "ABC":
                st = stats[v]
                n = len(st["psnr"])
                rec = {
                    "rep": rep,
                    "n": n,
                    "excluded": st["excl"],
                    "psnr_mean": float(np.mean(st["psnr"])) if n else None,
                    "ssim_mean": float(np.mean(st["ssim"])) if n else None,
                    "coverage_mean": float(np.mean(st["cov"])) if n else None,
                }
                (per_variant_gt if target == "gt_bbox" else per_variant_att)[v].append(rec)
        if per_variant_gt["C"]:
            per_variant_gt["C"][-1].update(roi_stats)

        for s in stores.values():
            s.close()

    def summarize(per_variant: dict[str, list[dict]]) -> dict:
        out = {}
        for v, rows in per_variant.items():
            out[v] = {
                "n_reps": len(rows),
                "n_images": rows[0]["n"],
                "excluded": rows[0]["excluded"],
                "coverage_mean": float(np.median([r["coverage_mean"] for r in rows])),
                "psnr_mean_median": float(np.median([r["psnr_mean"] for r in rows])),
                "ssim_mean_median": float(np.median([r["ssim_mean"] for r in rows])),
                "reps": rows,
            }
            if v == "C" and rows and "roi_00_hit_rate_iou03" in rows[0]:
                out[v]["roi_mechanism"] = {
                    "roi_00_hit_rate_iou03": float(np.median([r["roi_00_hit_rate_iou03"] for r in rows])),
                    "any_roi_hit_rate_iou03": float(np.median([r["any_roi_hit_rate_iou03"] for r in rows])),
                    "mean_union_coverage_of_gt": float(np.median([r["mean_union_coverage_of_gt"] for r in rows])),
                    "note": "C 的回溯表示使用全部已存 ROI 裁片（编码特征源仅含 roi_00）；"
                            "命中率与并集覆盖率分别报告，58.3% 只是 roi_00 的命中率。",
                }
        return out

    gt_summary = summarize(per_variant_gt)
    att_summary = summarize(per_variant_att)

    uniq = sorted(set(digests.values()))
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    old_gt = {
        v: (old.get("primary_metric", {}).get("summary", {}).get(v, {}) if old else {}) for v in "ABC"
    }

    payload = {
        "analysis_id": "EXP-0001-AMD-0001",
        "experiment_id": "EXP-0001",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation": {
            "primary_metric": "gt_bbox_derived_representation_visible_pixel_fidelity",
            "secondary_metric": "attended_roi_derived_representation_visible_pixel_fidelity",
            "preregistration": "EXP-0001 §5 指标 9（使用目标区域 ground-truth bbox）",
            "note": "A/B/C 使用同一评价框下的同一可见交集；C 在 ROI 覆盖处使用全分辨率裁片、其余使用 foveated 马赛克合成。",
        },
        "primary_metric": {"region": "gt_bbox", "summary": gt_summary},
        "secondary_metric": {"region": "attended_roi", "summary": att_summary},
        "delta_gt": {v: gt_summary[v]["psnr_mean_median"] - gt_summary["A"]["psnr_mean_median"] for v in "ABC"},
        "c_roi_mechanism": {
            "definition": "GT 区域回溯时，C 将**全部已存 ROI 裁片**贴回 foveated 马赛克（编码特征源仍只含 roi_00）。",
            "roi_00_hit_rate_iou03": gt_summary["C"].get("roi_mechanism", {}).get("roi_00_hit_rate_iou03"),
            "any_roi_hit_rate_iou03": gt_summary["C"].get("roi_mechanism", {}).get("any_roi_hit_rate_iou03"),
            "mean_union_coverage_of_gt": gt_summary["C"].get("roi_mechanism", {}).get("mean_union_coverage_of_gt"),
            "interpretation": "C 的全分辨率覆盖有限，仍有过半目标区域依赖 1/2 画布马赛克上采样；"
                              "这与 GT 区域保真度下降一致，但 58.3% 不能单独作为完整因果证明。",
        },
        "previous_roi_based_run": old_gt,
        "per_image": per_image,
        "provenance": {
            "script": {"path": "scripts/recompute_backtrack.py", "sha256": sha256_file(Path(__file__))},
            "git": {
                "commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain")),
            },
            "command": "cd <REPO> && .\\.venv\\Scripts\\python.exe scripts\\recompute_backtrack.py",
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pillow": __import__("PIL").__version__,
            },
            "runs": [
                {
                    "run_id": runs[(v, r)].name,
                    "variant": v,
                    "rep": r,
                    "config_sha256": (runs[(v, r)] / "config.sha256").read_text(encoding="utf-8").strip(),
                    "config_param_digest": digests[(v, r)],
                }
                for (v, r) in sorted(digests)
            ],
            "config_consistency": {
                "params_equal_across_nine_runs": len(uniq) == 1,
                "param_digest": uniq[0] if len(uniq) == 1 else None,
                "note": "参数摘要剔除 paths/experiment/logging/dataset；九次 run 的历史参数一致。",
            },
            "dataset": {
                "dataset_id": manifest["dataset_id"],
                "manifest_sha256_lf": json.loads(SANITIZATION.read_text(encoding="utf-8"))["manifest_sha256_lf"],
                "manifest_sha256_runtime_original_bytes": json.loads(SANITIZATION.read_text(encoding="utf-8")).get(
                    "manifest_sha256_runtime_original_bytes"
                ),
                "hash_spec": "sha256 over LF-normalized bytes",
            },
            "model": json.loads(MODEL_MANIFEST.read_text(encoding="utf-8")),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "delta_gt": payload["delta_gt"],
        "gt": {v: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items() if k != "reps"} for v, s in gt_summary.items()},
        "attended": {v: (round(s["psnr_mean_median"], 3), round(s["ssim_mean_median"], 3)) for v, s in att_summary.items()},
        "dirty": payload["provenance"]["git"]["dirty"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
