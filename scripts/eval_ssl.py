"""Gate 3 评估脚本：在 validation/test 上计算 384-d CLS 检索指标。

默认 validation；test 仅用于最终一次性评估，不得用于任何选择。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from alacv.ssl.arms import build_arm  # noqa: E402
from alacv.ssl.checkpoint import mean_ap_at_k, recall_at_k  # noqa: E402
from alacv.ssl.data import SslData  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="F0")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--config", default="configs/exp0002.yaml")
    ap.add_argument("--limit-scenes", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = build_arm(args.arm).to(device).eval()
    data = SslData(ROOT / cfg["data"]["root"], cfg, device=device, splits=(args.split,))
    labels = json.loads((ROOT / cfg["data"]["root"] / "eval" / f"labels_{args.split}.json").read_text(encoding="utf-8"))
    scenes = labels["scenes"][: args.limit_scenes]
    gallery_ids = [s["gallery_views"][0] for s in scenes]
    query_ids = [q for s in scenes for q in s["query_views"]]
    target = torch.arange(len(scenes)).repeat_interleave(4)

    with torch.no_grad():
        def desc(vid: str) -> torch.Tensor:
            cls, patch = data.teacher(vid)
            return arm.eval_cls(cls[None], patch[None])[0]

        gal = torch.stack([desc(v) for v in gallery_ids])
        qry = torch.stack([desc(v) for v in query_ids])

    result = {
        "arm": args.arm.upper(),
        "split": args.split,
        "scenes": len(scenes),
        "queries": len(query_ids),
        "recall@1": round(recall_at_k(qry, gal, target, 1), 4),
        "recall@5": round(recall_at_k(qry, gal, target, 5), 4),
        "map@10": round(mean_ap_at_k(qry, gal, target, 10), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
