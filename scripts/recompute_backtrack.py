"""WP2：用修正后的坐标契约重算 EXP-0001 的派生表示保真度（不重跑编码）。

- 只读取历史 run 的 store（raw / processed / features / index），不修改原始 run。
- A/B/C 使用同一 ROI 与同一可见交集；完全不可见的样本排除并计数。
- 不涉及 DINOv2 前向，不产生新的实验编号，不覆盖 EXP-0001 预注册结论。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.backtrack import crop_region, recover_variant, visible_region  # noqa: E402
from alacv.config import load_config  # noqa: E402
from alacv.foveation import select_rois  # noqa: E402
from alacv.imaging import open_image  # noqa: E402
from alacv.metrics import psnr, ssim  # noqa: E402
from alacv.store import VisualStore  # noqa: E402

RUNS = ROOT / "artifacts" / "runs"
OUT = ROOT / "artifacts" / "experiments" / "EXP-0001-backtrack-corrected.json"


def find_runs() -> dict[tuple[str, int], Path]:
    out = {}
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and "EXP-0001" in p.name):
        parts = d.name.split("_")
        variant = parts[-2]
        rep = int(parts[-1][1:])
        out[(variant, rep)] = d
    return out


def build_cfg(run_dir: Path):
    ws = run_dir / "store"
    return load_config(
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


def old_value(metrics_path: Path) -> dict:
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    b = m.get("backtrack", {})
    return {"psnr_mean": b.get("psnr_mean"), "ssim_mean": b.get("ssim_mean"), "n": b.get("n")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    manifest = json.loads(
        (ROOT / "data" / "manifests" / "ds-0001-alacv-synthetic-v1.json").read_text(encoding="utf-8")
    )
    bases = [f for f in manifest["files"] if f["role"] == "base"]
    runs = find_runs()
    reps = sorted({r for (_, r) in runs})
    print("runs:", {f"{v}_r{r}": d.name for (v, r), d in runs.items()})

    per_variant: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    per_image: list[dict] = []

    for rep in reps:
        if not all((v, rep) in runs for v in "ABC"):
            print(f"rep {rep} incomplete, skipped")
            continue
        cfg_c = build_cfg(runs[("C", rep)])
        store_c = VisualStore(cfg_c.path("paths.index_db"), cfg_c.path("paths.features_dir"))
        maybe = {
            "A": (build_cfg(runs[("A", rep)]), None),
            "B": (build_cfg(runs[("B", rep)]), None),
        }
        stores = {"C": store_c}
        for v in "AB":
            c, _ = maybe[v]
            stores[v] = VisualStore(c.path("paths.index_db"), c.path("paths.features_dir"))
        cfgs = {"C": cfg_c, **{v: maybe[v][0] for v in "AB"}}

        results = {v: {"psnr": [], "ssim": [], "cov": [], "excluded": 0} for v in "ABC"}
        for f in bases:
            iid = store_c.image_by_sha(f["sha256"])["image_id"]
            src_c = open_image(cfg_c.path("paths.raw_dir") / store_c.get_image(iid)["rel_path"]).image
            rois = store_c.rois_of(iid)
            if rois:
                box = (rois[0]["x"], rois[0]["y"], rois[0]["w"], rois[0]["h"])
            else:
                r = select_rois(src_c, roi_frac=float(cfg_c.get("foveation.roi_frac", 0.25)), max_rois=1)[0]
                box = (r.x, r.y, r.w, r.h)
            size = int(cfg_c.get("model.input_size", 518))
            region = visible_region(box, src_c.size, size)
            row = {"rep": rep, "base_id": f["base_id"], "box": box, "coverage": region.coverage}
            if not region.visible:
                for v in "ABC":
                    results[v]["excluded"] += 1
                per_image.append({**row, "excluded": True})
                continue
            gt = crop_region(src_c, region.box)
            row["visible_box"] = region.box
            for v in "ABC":
                rec, _ = recover_variant(cfgs[v], stores[v], iid, v, box)
                if rec is None or rec.shape != gt.shape:
                    results[v]["excluded"] += 1
                    row[v] = None
                    continue
                p, s = psnr(gt, rec), ssim(gt, rec)
                results[v]["psnr"].append(p)
                results[v]["ssim"].append(s)
                results[v]["cov"].append(region.coverage)
                row[v] = {"psnr": p, "ssim": s}
            per_image.append(row)

        for v in "ABC":
            r = results[v]
            n = len(r["psnr"])
            per_variant[v].append(
                {
                    "rep": rep,
                    "n": n,
                    "excluded": r["excluded"],
                    "psnr_mean": float(np.mean(r["psnr"])) if n else None,
                    "ssim_mean": float(np.mean(r["ssim"])) if n else None,
                    "coverage_mean": float(np.mean(r["cov"])) if n else None,
                }
            )
            print(f"rep {rep} {v}: n={n} excl={r['excluded']} psnr={per_variant[v][-1]['psnr_mean']}")
        for s in stores.values():
            s.close()

    summary = {}
    for v in "ABC":
        rows = per_variant[v]
        if not rows:
            continue
        old = [old_value(runs[(v, r["rep"])] / "metrics.json") for r in rows]
        summary[v] = {
            "n_reps": len(rows),
            "n_images": rows[0]["n"],
            "excluded": rows[0]["excluded"],
            "coverage_mean": float(np.median([r["coverage_mean"] for r in rows if r["coverage_mean"]])),
            "psnr_mean_median": float(np.median([r["psnr_mean"] for r in rows if r["psnr_mean"]])) if any(r["psnr_mean"] for r in rows) else None,
            "ssim_mean_median": float(np.median([r["ssim_mean"] for r in rows if r["ssim_mean"]])),
            "old_psnr_median": float(np.median([o["psnr_mean"] for o in old if o["psnr_mean"] is not None])),
            "old_ssim_median": float(np.median([o["ssim_mean"] for o in old if o["ssim_mean"] is not None])),
            "reps": rows,
        }
        summary[v]["psnr_delta"] = summary[v]["psnr_mean_median"] - summary[v]["old_psnr_median"]
        summary[v]["ssim_delta"] = summary[v]["ssim_mean_median"] - summary[v]["old_ssim_median"]

    payload = {
        "experiment_id": "EXP-0001",
        "analysis": "derived_representation_visible_pixel_fidelity",
        "amendment": "EXP-0001-AMD-0001",
        "note": "不重跑编码；仅用修正后的 fit_to_input 几何在共同可见区域上重算像素保真度",
        "n_base_images": len(bases),
        "summary": summary,
        "per_image": per_image,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({v: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items() if k != "reps"} for v, s in summary.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
