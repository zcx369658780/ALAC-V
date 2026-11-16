"""Gate 3 训练脚本骨架：默认只做 forward/backward 冒烟，**绝不** optimizer.step。

正式训练需要 Gate 4 授权；即使传入 --allow-training，也必须设置
ALACV_GATE4_AUTHORIZED=1，否则拒绝运行。当前阶段该变量不应存在。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from alacv.ssl.arms import build_arm  # noqa: E402
from alacv.ssl.data import SslData  # noqa: E402
from alacv.ssl.masking import contiguous_block_mask, mask_seed_for, random_mask  # noqa: E402


def peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A2")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", default="configs/exp0002.yaml")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--allow-training", action="store_true")
    args = ap.parse_args()

    if args.allow_training and os.environ.get("ALACV_GATE4_AUTHORIZED") != "1":
        print("REFUSED: Gate 4 未授权，禁止 optimizer.step / 正式训练", file=sys.stderr)
        return 3
    if args.allow_training:
        print("REFUSED: 本阶段只允许 dry-run；Gate 4 尚未批准", file=sys.stderr)
        return 3

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    arm_name = args.arm.upper()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = SslData(ROOT / cfg["data"]["root"], cfg, device=device)
    arm = build_arm(arm_name).to(device)
    for p in arm.parameters():
        p.requires_grad_(arm_name not in {"F0", "R0"})

    views = data.indexes["train"][: args.batch]
    ids = [v["view_id"] for v in views]
    if arm_name == "B2":
        pair = views[0]["pair_key"]
        same = [v for v in data.indexes["train"] if v["pair_key"] == pair][: args.batch]
        ids2 = [v["view_id"] for v in same]
        masks = [np.zeros(1369, dtype=bool) for _ in ids]
    else:
        ids2 = None
        masks = []
        for i, v in enumerate(views):
            seed = mask_seed_for(args.seed, args.step + i)
            if arm_name in {"A1", "A2"}:
                masks.append(contiguous_block_mask(seed))
            elif arm_name == "C2":
                masks.append(random_mask(seed))
            else:
                masks.append(np.zeros(1369, dtype=bool))

    t0 = time.perf_counter()
    batch = data.make_batch(arm_name, ids, masks, ids2)
    t_data = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if arm_name in {"F0", "R0"}:
        with torch.no_grad():
            _ = arm.eval_cls(batch["teacher_cls"], batch["teacher_patch"])
        loss = None
        t_fwd = time.perf_counter() - t0
        t_bwd = 0.0
    else:
        out = arm(batch)
        loss = out["loss"]
        t_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        loss.backward()
        t_bwd = time.perf_counter() - t0
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    report = {
        "arm": arm_name,
        "seed": args.seed,
        "batch": args.batch,
        "trainable_params": arm.trainable_parameters(),
        "total_params": arm.total_parameters(),
        "loss": None if loss is None else float(loss.detach().cpu()),
        "peak_vram_mb": round(peak_vram_mb(), 2),
        "data_s": round(t_data, 4),
        "forward_s": round(t_fwd, 4),
        "backward_s": round(t_bwd, 4),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_runs": 0,
        "device": device,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
