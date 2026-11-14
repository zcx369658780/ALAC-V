"""生成报告插图：变体对比柱状图 + Foveated 示例。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.config import load_config  # noqa: E402
from alacv.foveation import build_foveated, select_rois  # noqa: E402
from alacv.imaging import open_image  # noqa: E402

FIG = ROOT / "artifacts" / "figures"


def variant_bars(summary: dict, ablations: dict) -> None:
    names = ["A", "B", "C"]
    labels = ["A\noriginal", "B\nmulti-scale", "C\nmulti-scale+ROI"]
    t = [summary[v]["timing"]["t_total_median_ms"] for v in names]
    enc = [summary[v]["timing"]["t_encode_median_ms"] for v in names]
    mb = [summary[v]["storage"]["total_bytes_per_image"] / (1024 * 1024) for v in names]
    r1 = [summary[v]["retrieval"]["recall@1"] for v in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(3)
    axes[0].bar(x - 0.2, t, 0.4, label="total", color="#4C72B0")
    axes[0].bar(x + 0.2, enc, 0.4, label="encode", color="#DD8452")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("ms / image")
    axes[0].set_title("Per-image latency (median)")
    axes[0].legend(frameon=False)
    for i, v in enumerate(t):
        axes[0].text(i - 0.2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, mb, 0.6, color="#55A868")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("MB / image")
    axes[1].set_title("Derived bytes (features + pixels)")
    for i, v in enumerate(mb):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    axes[2].bar(x, [r * 100 for r in r1], 0.6, color="#C44E52")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(70, 100)
    axes[2].set_ylabel("Recall@1 (%)")
    axes[2].set_title("Retrieval accuracy (72 queries)")
    for i, v in enumerate(r1):
        axes[2].text(i, v * 100, f"{v*100:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "exp0001_variants.png", dpi=160)
    plt.close(fig)

    if ablations:
        order = ["A0", "B2", "C2", "C3"]
        have = [k for k in order if k in ablations]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for k in have:
            a = ablations[k]
            ax.scatter(a["feature_bytes_per_image"] / 1024, a["recall@1"] * 100, s=90, label=f"{k} ({a['dim']}d)")
            ax.annotate(k, (a["feature_bytes_per_image"] / 1024, a["recall@1"] * 100), textcoords="offset points", xytext=(6, 4))
        for v in names:
            a = summary[v]
            ax.scatter(a["storage"]["feature_bytes_per_image"] / 1024, a["retrieval"]["recall@1"] * 100, marker="x", s=70, color="black")
            ax.annotate(v, (a["storage"]["feature_bytes_per_image"] / 1024, a["retrieval"]["recall@1"] * 100), textcoords="offset points", xytext=(6, -10))
        ax.set_xlabel("feature KB / image")
        ax.set_ylabel("Recall@1 (%)")
        ax.set_title("Accuracy vs feature budget (x = full variants)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "exp0001_budget.png", dpi=160)
        plt.close(fig)


def foveated_example() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    man = json.loads((ROOT / "data" / "manifests" / "ds-0001-alacv-synthetic-v1.json").read_text(encoding="utf-8"))
    base = [f for f in man["files"] if f["role"] == "base"][3]
    img = open_image(ROOT / base["path"]).image
    roi = select_rois(img, roi_frac=float(cfg.get("foveation.roi_frac")), max_rois=1)[0]
    fov = build_foveated(img, roi, cfg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(img)
    axes[0].add_patch(plt.Rectangle((roi.x, roi.y), roi.w, roi.h, fill=False, edgecolor="red", linewidth=2))
    axes[0].set_title("original (raw, read-only)")
    axes[1].imshow(fov.mosaic)
    axes[1].set_title(f"foveated mosaic {fov.mosaic.size[0]}x{fov.mosaic.size[1]}")
    axes[2].imshow(fov.roi_crop)
    axes[2].set_title(f"ROI crop 1:1 {fov.roi_crop.size[0]}x{fov.roi_crop.size[1]}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "foveated_example.png", dpi=150)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "artifacts" / "experiments" / "EXP-0001-summary.json").read_text(encoding="utf-8"))
    abl_path = ROOT / "artifacts" / "experiments" / "EXP-0001-ablations.json"
    abl = json.loads(abl_path.read_text(encoding="utf-8")) if abl_path.exists() else {}
    variant_bars(summary, abl)
    foveated_example()
    print("figures written:", [p.name for p in FIG.glob("*.png")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
