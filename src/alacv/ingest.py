"""原始图像归档：解码校验、SHA-256、元数据、只读、内容寻址。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

from .imaging import open_image
from .logging_utils import utc_now
from .store import VisualStore, sha256_file

MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "AVIF": "image/avif",
}

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP", "AVIF")


class UnsupportedFormat(Exception):
    pass


def _channels(mode: str) -> int:
    return len(mode) if mode in ("RGB", "RGBA", "CMYK", "YCbCr") else 1


def _color_space(icc: bytes | None, mode: str) -> str:
    if icc:
        try:
            from PIL import ImageCms

            profile = ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc))
            desc = ImageCms.getProfileDescription(profile) or ""
            desc = desc.strip()
            low = desc.lower()
            if "srgb" in low or "iec 61966" in low:
                return "sRGB"
            if "display p3" in low or "p3" in low:
                return "Display P3"
            if "adobe" in low:
                return "Adobe RGB"
            if desc:
                return desc[:64]
        except Exception:
            return "ICC(无法解析)"
    if mode in ("L", "1", "I", "I;16", "F"):
        return "Gray"
    return "sRGB(假定)"


def _set_readonly(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass
    try:
        import ctypes

        FILE_ATTRIBUTE_READONLY = 0x1
        ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_READONLY)
    except Exception:
        pass


def _exif_summary(img_exif) -> dict:
    try:
        exif = img_exif or {}
        orientation = None
        if exif:
            orientation = exif.get(274)
        return {"present": bool(exif), "entries": len(exif), "orientation": orientation}
    except Exception:
        return {"present": False, "entries": 0, "orientation": None}


def ingest_file(
    path: str | Path,
    cfg,
    store: VisualStore,
    dataset_id: str | None = None,
    *,
    force_reingest: bool = False,
    decoded=None,
) -> dict:
    """归档一张图。返回 dict(image_id, created, status, timings...)。"""
    src = Path(path)
    t_a = __import__("time").perf_counter()
    sha = sha256_file(src)
    size = src.stat().st_size
    image_id = sha[:16]
    existing = store.image_by_sha(sha)
    if existing and not force_reingest:
        return {
            "image_id": existing["image_id"],
            "created": False,
            "status": existing["status"],
            "sha256": sha,
            "decode_ms": 0.0,
            "archive_ms": 0.0,
        }

    def reject(reason: str) -> dict:
        store.add_image(
            {
                "image_id": image_id,
                "sha256": sha,
                "rel_path": "",
                "source_name": src.name,
                "ext": src.suffix.lower().lstrip("."),
                "mime": "application/octet-stream",
                "width": 0,
                "height": 0,
                "channels": 0,
                "pil_mode": "?",
                "color_space": "unknown",
                "icc_sha256": None,
                "file_size": size,
                "exif_json": json.dumps({"present": False}),
                "dataset_id": dataset_id,
                "status": "rejected",
                "reject_reason": reason,
                "ingested_at": utc_now(),
            }
        )
        return {"image_id": image_id, "created": False, "status": "rejected", "reason": reason, "sha256": sha}

    if size == 0:
        return reject("空文件")
    try:
        if decoded is None:
            decoded = open_image(src, exif_transpose=bool(cfg.get("image.exif_transpose", True)))
    except Exception as exc:  # 损坏 / 不支持
        return reject(f"解码失败: {type(exc).__name__}: {exc}")

    fmt = (decoded.format or "").upper()
    if fmt not in MIME_BY_FORMAT:
        return reject(f"不支持的格式: {fmt or src.suffix}")
    max_side = int(cfg.get("image.max_side", 8192))
    if max(decoded.width, decoded.height) > max_side:
        return reject(f"尺寸超限: {decoded.width}x{decoded.height} > {max_side}")
    decode_ms = (__import__("time").perf_counter() - t_a) * 1000.0

    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "AVIF": ".avif"}.get(fmt, src.suffix.lower())
    raw_root = cfg.path("paths.raw_dir")
    dest = raw_root / sha[:2] / f"{sha}{ext}"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    _set_readonly(dest)

    icc = decoded.icc_profile
    record = {
        "image_id": image_id,
        "sha256": sha,
        "rel_path": dest.relative_to(raw_root).as_posix(),
        "source_name": src.name,
        "ext": ext.lstrip("."),
        "mime": MIME_BY_FORMAT[fmt],
        "width": decoded.width,
        "height": decoded.height,
        "channels": _channels(decoded.original_mode),
        "pil_mode": decoded.original_mode,
        "color_space": _color_space(icc, decoded.original_mode),
        "icc_sha256": hashlib.sha256(icc).hexdigest() if icc else None,
        "file_size": size,
        "exif_json": json.dumps(
            {"present": bool(decoded.exif_present), "entries": 0, "note": "仅记录存在性，不保存原值（隐私）"},
            ensure_ascii=False,
        ),
        "dataset_id": dataset_id,
        "status": "ok",
        "reject_reason": None,
        "ingested_at": utc_now(),
    }
    image_id, created = store.add_image(record)
    archive_ms = (__import__("time").perf_counter() - t_a) * 1000.0 - decode_ms
    return {
        "image_id": image_id,
        "created": created,
        "status": "ok",
        "sha256": sha,
        "decode_ms": decode_ms,
        "archive_ms": archive_ms,
        "width": decoded.width,
        "height": decoded.height,
    }


def raw_path_of(cfg, row) -> Path:
    return cfg.path("paths.raw_dir") / row["rel_path"]


def verify_raw(cfg, store: VisualStore) -> dict:
    """重算所有归档文件的 SHA-256 并比对。"""
    rows = store.conn.execute("SELECT image_id, sha256, rel_path FROM images WHERE status='ok'").fetchall()
    bad = []
    for r in rows:
        p = cfg.path("paths.raw_dir") / r["rel_path"]
        if not p.exists():
            bad.append({"image_id": r["image_id"], "error": "missing"})
            continue
        if sha256_file(p) != r["sha256"]:
            bad.append({"image_id": r["image_id"], "error": "sha256_mismatch"})
    return {"checked": len(rows), "bad": bad, "ok": not bad}
