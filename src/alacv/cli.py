"""ALAC-V 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .features import DinoEncoder
from .imaging import SUPPORTED_EXTS
from .ingest import verify_raw
from .pipeline import ImagePipeline
from .retrieval import RetrievalIndex
from .store import VisualStore


def _ctx(args):
    cfg = load_config(args.config)
    store = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))
    encoder = DinoEncoder(cfg)
    return cfg, store, encoder


def _collect(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = [p for p in sorted(input_path.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTS]
    return files


def cmd_ingest(args) -> int:
    cfg, store, encoder = _ctx(args)
    pipe = ImagePipeline(cfg, store, encoder)
    files = _collect(Path(args.input))
    if not files:
        print(f"未找到可处理图像: {args.input}")
        return 2
    ok = rejected = 0
    for i, f in enumerate(files, 1):
        item = pipe.process(f, args.variant, args.dataset_id)
        status = item["status"]
        if status == "ok":
            ok += 1
            print(
                f"[{i}/{len(files)}] {f.name} -> {item['image_id']} "
                f"total={item['t_total_ms']:.1f}ms enc={item['t_encode_ms']:.1f}ms "
                f"feat={item['feature_bytes']/1024:.0f}KB vram={item['peak_vram_mb']:.0f}MB"
            )
        else:
            rejected += 1
            print(f"[{i}/{len(files)}] {f.name} -> REJECTED: {item.get('reason')}")
    stats = store.stats()
    print(json.dumps({"ok": ok, "rejected": rejected, **stats}, ensure_ascii=False, indent=2))
    return 0


def cmd_query(args) -> int:
    cfg, store, encoder = _ctx(args)
    sources = list(cfg.require("encoding.variants")[args.variant])
    pipe = ImagePipeline(cfg, store, encoder)
    q = pipe.process(args.image, args.variant, None)
    if q["status"] != "ok":
        print("查询图无法处理: " + str(q.get("reason")))
        return 1
    rows = [r["image_id"] for r in store.list_images() if r["image_id"] != q["image_id"]]
    if not rows:
        print("图库为空（除查询图外）")
        return 1
    index = RetrievalIndex(store, rows, sources)
    results = index.search(index.descriptor_of(q["image_id"]), args.topk)
    print(f"查询 {q['image_id']}  dim={index.dim}  库={len(rows)}")
    for rank, (iid, score) in enumerate(results, 1):
        row = store.get_image(iid)
        rois = store.rois_of(iid)
        roi_txt = ""
        if rois:
            r = rois[0]
            roi_txt = f" roi=({r['x']},{r['y']},{r['w']},{r['h']})"
        print(f"{rank:2d}. {iid}  cos={score:.4f}  {row['source_name']}{roi_txt}")
    return 0


def cmd_backtrack(args) -> int:
    from .retrieval import backtrack_crop

    cfg, store, _ = _ctx(args)
    crop, meta = backtrack_crop(cfg, store, args.image_id, args.roi)
    from PIL import Image

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(out)
    print(json.dumps({**meta, "out": str(out), "size": [crop.shape[1], crop.shape[0]]}, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args) -> int:
    cfg = load_config(args.config)
    store = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args) -> int:
    cfg = load_config(args.config)
    store = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))
    res = verify_raw(cfg, store)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alacv", description="ALAC-V Phase 1 静态图像视觉编码")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="归档 + 派生 + 特征 + 入库")
    pi.add_argument("--input", required=True)
    pi.add_argument("--variant", default="C", choices=["A", "B", "C"])
    pi.add_argument("--dataset-id", default=None)
    pi.set_defaults(func=cmd_ingest)

    pq = sub.add_parser("query", help="以图搜图")
    pq.add_argument("--image", required=True)
    pq.add_argument("--variant", default="C", choices=["A", "B", "C"])
    pq.add_argument("--topk", type=int, default=5)
    pq.set_defaults(func=cmd_query)

    pb = sub.add_parser("backtrack", help="按 image_id + ROI 回溯高清局部")
    pb.add_argument("--image-id", required=True)
    pb.add_argument("--roi", type=int, default=0)
    pb.add_argument("--out", required=True)
    pb.set_defaults(func=cmd_backtrack)

    ps = sub.add_parser("stats", help="知识库统计")
    ps.set_defaults(func=cmd_stats)

    pv = sub.add_parser("verify", help="校验原始归档完整性")
    pv.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
