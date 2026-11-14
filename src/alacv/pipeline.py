"""变体 A/B/C 的图像处理编排：归档 → 金字塔 → foveation → 编码 → 入库。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import DinoEncoder
from .foveation import build_foveated, select_rois
from .imaging import build_pyramid, open_image, save_image, to_rgb
from .ingest import ingest_file
from .logging_utils import utc_now
from .metrics import VramTracker
from .store import VisualStore, sha256_file


@dataclass
class VariantPlan:
    name: str
    sources: list[str]

    @property
    def needs_pyramid(self) -> bool:
        return any(s.startswith("pyramid:") for s in self.sources)

    @property
    def needs_foveation(self) -> bool:
        return any(s == "foveated" or s.startswith("roi:") for s in self.sources)


def level_of(source: str) -> int | None:
    if source.startswith("pyramid:L"):
        return int(source[9:])
    return None


class ImagePipeline:
    def __init__(self, cfg, store: VisualStore, encoder: DinoEncoder, run_id: str | None = None):
        self.cfg = cfg
        self.store = store
        self.encoder = encoder
        self.run_id = run_id
        self.raw_dir = cfg.path("paths.raw_dir")
        self.processed_dir = cfg.path("paths.processed_dir")
        self.data_dir = self.processed_dir.parent
        self.cfg_hash = cfg.stable_sha256
        self.vram = VramTracker()
        self.vram_base_mb = self.vram.allocated_mb() or 0.0
        self.model_row = self._ensure_model_row()

    # ------------------------------------------------------------------ setup
    def _ensure_model_row(self) -> dict:
        m = self.encoder.describe()
        manifest = m.get("manifest") or {}
        rec = {
            "model_id": m["model_id"],
            "display_name": manifest.get("display_name", "DINOv2 ViT-S/14"),
            "arch": m["arch"],
            "params": m["num_params"],
            "source": manifest.get("source", "huggingface:facebook/dinov2-small"),
            "revision": manifest.get("revision", "unknown"),
            "weights_sha256": manifest.get("weights_sha256", "unknown"),
            "license": manifest.get("license", "Apache-2.0"),
            "license_file": manifest.get("license_file"),
            "input_size": m["input_size"],
            "patch_size": m["patch_size"],
            "cls_dim": m["cls_dim"],
            "norm_mean": json.dumps(list((0.485, 0.456, 0.406))),
            "norm_std": json.dumps(list((0.229, 0.224, 0.225))),
            "created_at": utc_now(),
        }
        self.store.upsert_model(rec)
        return rec

    def plan(self, variant: str) -> VariantPlan:
        variants = self.cfg.require("encoding.variants")
        if variant not in variants:
            raise KeyError(f"未定义变体 {variant}；已有 {sorted(variants)}")
        return VariantPlan(variant, list(variants[variant]))

    # ------------------------------------------------------------------ assets
    def _save_pyramid(self, image_id: str, img) -> tuple[dict[int, object], float]:
        levels = self.cfg.require("pyramid.scales")
        quality = int(self.cfg.get("pyramid.quality", 95))
        fmt = self.cfg.get("pyramid.format", "webp")
        pyr = build_pyramid(img, levels, self.cfg.get("pyramid.method", "bicubic"))
        t0 = time.perf_counter()
        out: dict[int, object] = {}
        for idx, denom in enumerate(levels):
            p = self.processed_dir / image_id / "pyramid" / f"L{idx}_{denom}x.{fmt}"
            nbytes = save_image(pyr[denom], p, fmt, quality)
            self.store.add_asset(
                {
                    "image_id": image_id,
                    "kind": "pyramid",
                    "level": idx,
                    "label": f"L{idx}",
                    "rel_path": p.relative_to(self.data_dir).as_posix(),
                    "width": pyr[denom].width,
                    "height": pyr[denom].height,
                    "format": fmt,
                    "bytes": nbytes,
                    "sha256": sha256_file(p),
                    "config_sha256": self.cfg_hash,
                    "created_at": utc_now(),
                }
            )
            out[idx] = pyr[denom]
        return out, (time.perf_counter() - t0) * 1000.0

    def _save_foveated(self, image_id: str, img) -> tuple[object, object, list[dict], float]:
        t0 = time.perf_counter()
        rois = select_rois(
            img,
            roi_frac=float(self.cfg.get("foveation.roi_frac", 0.25)),
            max_rois=int(self.cfg.get("foveation.max_rois", 2)),
        )
        roi = rois[0]
        fov = build_foveated(img, roi, self.cfg)
        fmt = self.cfg.get("foveation.format", "webp")
        quality = int(self.cfg.get("foveation.quality", 92))

        mp = self.processed_dir / image_id / "foveated" / f"foveated.{fmt}"
        mb = save_image(fov.mosaic, mp, fmt, quality)
        mp.with_suffix(".json").write_text(json.dumps(fov.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.add_asset(
            {
                "image_id": image_id,
                "kind": "foveated",
                "level": None,
                "label": "foveated",
                "rel_path": mp.relative_to(self.data_dir).as_posix(),
                "width": fov.mosaic.width,
                "height": fov.mosaic.height,
                "format": fmt,
                "bytes": mb,
                "sha256": sha256_file(mp),
                "config_sha256": self.cfg_hash,
                "created_at": utc_now(),
            }
        )

        roi_assets: list[dict] = []
        crops: dict[str, object] = {}
        for i, r in enumerate(rois):
            crop = img.crop((r.x, r.y, r.x + r.w, r.y + r.h))
            crops[f"roi_{i:02d}"] = crop
            rp = self.processed_dir / image_id / "roi" / f"roi_{i:02d}.{fmt}"
            rb = save_image(crop, rp, fmt, int(self.cfg.get("foveation.roi_quality", 96)))
            asset_id = self.store.add_asset(
                {
                    "image_id": image_id,
                    "kind": "roi_crop",
                    "level": None,
                    "label": f"roi_{i:02d}",
                    "rel_path": rp.relative_to(self.data_dir).as_posix(),
                    "width": r.w,
                    "height": r.h,
                    "format": fmt,
                    "bytes": rb,
                    "sha256": sha256_file(rp),
                    "config_sha256": self.cfg_hash,
                    "created_at": utc_now(),
                }
            )
            self.store.add_roi(
                {
                    "image_id": image_id,
                    "roi_index": i,
                    "x": r.x,
                    "y": r.y,
                    "w": r.w,
                    "h": r.h,
                    "score": r.score,
                    "method": r.method,
                    "crop_asset_id": asset_id,
                    "created_at": utc_now(),
                }
            )
            roi_assets.append({"index": i, "asset_id": asset_id, **r.as_dict()})
        return {"mosaic": fov.mosaic, "roi_crop": fov.roi_crop, "rois": roi_assets, "crops": crops}, (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------ source
    def _resolve_source(self, image_id: str, source: str, img, pyramid: dict[int, object], fove: dict | None = None):
        if source == "original":
            return to_rgb(img)
        if source.startswith("pyramid:L"):
            lvl = level_of(source)
            if lvl not in pyramid:
                raise KeyError(f"金字塔层级 L{lvl} 不存在")
            return pyramid[lvl]
        if source == "foveated":
            if fove:
                return fove["mosaic"]
            from .imaging import open_image as _open

            p = self.processed_dir / image_id / "foveated" / f"foveated.{self.cfg.get('foveation.format', 'webp')}"
            return _open(p).image
        if source.startswith("roi:"):
            label = source[4:]
            if fove and label in fove["crops"]:
                return fove["crops"][label]
            p = self.processed_dir / image_id / "roi" / f"{label}.{self.cfg.get('foveation.format', 'webp')}"
            if not p.exists():
                raise FileNotFoundError(p)
            from .imaging import open_image as _open

            return _open(p).image
        raise ValueError(f"未知来源: {source}")

    # ------------------------------------------------------------------ main
    def process(self, path: str | Path, variant: str, dataset_id: str | None = None) -> dict:
        plan = self.plan(variant)
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        try:
            decoded = open_image(path, exif_transpose=bool(self.cfg.get("image.exif_transpose", True)))
        except Exception:
            decoded = None
        res = ingest_file(path, self.cfg, self.store, dataset_id, decoded=decoded)
        t_ingest = (time.perf_counter() - t0) * 1000.0
        if res["status"] != "ok":
            return {
                "image_id": res["image_id"],
                "status": res["status"],
                "variant": variant,
                "reason": res.get("reason"),
                "t_ingest_ms": t_ingest,
            }
        image_id = res["image_id"]
        img = decoded.image

        pyramid: dict[int, object] = {}
        fove: dict | None = None
        t_pyr = 0.0
        t_fov = 0.0
        roi_meta: list[dict] = []
        if plan.needs_pyramid:
            pyramid, t_pyr = self._save_pyramid(image_id, img)
        if plan.needs_foveation:
            fove, t_fov = self._save_foveated(image_id, img)
            roi_meta = fove["rois"]

        inputs = [self._resolve_source(image_id, s, img, pyramid, fove) for s in plan.sources]

        self.vram.reset()
        t0 = time.perf_counter()
        fb = self.encoder.encode_pil(inputs)
        t_enc = (time.perf_counter() - t0) * 1000.0
        peak_alloc = self.vram.peak_mb() or 0.0

        t_store0 = time.perf_counter()
        bytes_added = 0
        for i, source in enumerate(plan.sources):
            meta = {
                "source": source,
                "image_id": image_id,
                "raw_sha256": res["sha256"],
                "model_id": self.model_row["model_id"],
                "config_sha256": self.cfg_hash,
                "grid": [fb.grid[0], fb.grid[1]],
                "input_size": int(self.cfg.get("model.input_size", 518)),
                "precision": self.cfg.get("model.precision", "fp16"),
                "variant": variant,
                "created_at": utc_now(),
            }
            written = self.store.write_features(image_id, source, fb.cls[i], fb.patch[i], meta)
            added = self.store.add_feature(
                {
                    "image_id": image_id,
                    "variant": variant,
                    "source": source,
                    "level": level_of(source),
                    "model_id": self.model_row["model_id"],
                    "precision": meta["precision"],
                    "n_patches": written["n_patches"],
                    "patch_grid": written["patch_grid"],
                    "cls_dim": written["cls_dim"],
                    "rel_path": written["rel_path"],
                    "bytes": written["bytes"],
                    "sha256": written["sha256"],
                    "config_sha256": self.cfg_hash,
                    "created_at": utc_now(),
                }
            )
            if added:
                bytes_added += written["bytes"]
        t_store = (time.perf_counter() - t_store0) * 1000.0
        t_total = (time.perf_counter() - t_all) * 1000.0

        item = {
            "image_id": image_id,
            "status": "ok",
            "variant": variant,
            "sources": plan.sources,
            "t_ingest_ms": t_ingest,
            "t_pyramid_ms": t_pyr,
            "t_fovea_ms": t_fov,
            "t_encode_ms": t_enc,
            "t_store_ms": t_store,
            "t_total_ms": t_total,
            "peak_vram_mb": peak_alloc,
            "peak_vram_activation_mb": max(0.0, peak_alloc - self.vram_base_mb),
            "feature_bytes": bytes_added,
            "roi": roi_meta,
            "image_size": [img.width, img.height],
        }
        if self.run_id:
            self.store.add_run_item(
                {
                    "run_id": self.run_id,
                    "image_id": image_id,
                    "variant": variant,
                    "t_ingest_ms": t_ingest,
                    "t_pyramid_ms": t_pyr,
                    "t_fovea_ms": t_fov,
                    "t_encode_ms": t_enc,
                    "t_total_ms": t_total,
                    "peak_vram_mb": peak_alloc,
                    "rss_delta_mb": 0.0,
                    "feature_bytes": bytes_added,
                }
            )
        return item

