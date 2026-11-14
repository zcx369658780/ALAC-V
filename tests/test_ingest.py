import os
import stat

import pytest
from PIL import Image

from alacv.ingest import ingest_file, verify_raw
from alacv.store import sha256_file


def test_ingest_jpeg_records_metadata(cfg, store, image_factory):
    p = image_factory("one.jpg", 640, 480)
    res = ingest_file(p, cfg, store, "ds-test")
    assert res["status"] == "ok"
    row = store.get_image(res["image_id"])
    assert row["sha256"] == sha256_file(p)
    assert row["width"] == 640 and row["height"] == 480
    assert row["ext"] == "jpg"
    assert row["file_size"] == p.stat().st_size
    assert row["status"] == "ok"


def test_ingest_is_idempotent(cfg, store, image_factory):
    p = image_factory("dup.jpg")
    a = ingest_file(p, cfg, store, "ds-test")
    b = ingest_file(p, cfg, store, "ds-test")
    assert a["image_id"] == b["image_id"]
    assert b["created"] is False
    assert store.stats()["images"] == 1


def test_raw_archive_is_readonly(cfg, store, image_factory):
    p = image_factory("ro.jpg")
    res = ingest_file(p, cfg, store)
    raw = cfg.path("paths.raw_dir") / store.get_image(res["image_id"])["rel_path"]
    assert raw.exists()
    assert not os.access(raw, os.W_OK), "归档文件必须只读"


def test_archive_path_is_content_addressed(cfg, store, image_factory):
    p = image_factory("sha.jpg")
    res = ingest_file(p, cfg, store)
    row = store.get_image(res["image_id"])
    assert res["sha256"] in row["rel_path"]
    assert row["rel_path"].startswith(res["sha256"][:2] + "/")


def test_corrupt_file_rejected(cfg, store, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image at all")
    res = ingest_file(bad, cfg, store)
    assert res["status"] == "rejected"
    assert store.stats()["images"] == 0
    assert store.stats()["rejected"] == 1


def test_unsupported_format_rejected(cfg, store, tmp_path):
    gif = tmp_path / "x.gif"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(gif)
    res = ingest_file(gif, cfg, store)
    assert res["status"] == "rejected"
    assert "不支持" in res["reason"]


def test_verify_raw_detects_tampering(cfg, store, image_factory):
    p = image_factory("v.jpg")
    res = ingest_file(p, cfg, store)
    assert verify_raw(cfg, store)["ok"] is True
    raw = cfg.path("paths.raw_dir") / store.get_image(res["image_id"])["rel_path"]
    os.chmod(raw, stat.S_IWRITE)
    raw.write_bytes(raw.read_bytes() + b"x")
    out = verify_raw(cfg, store)
    assert out["ok"] is False
    assert out["bad"][0]["error"] == "sha256_mismatch"


def test_duplicate_content_different_names_single_archive(cfg, store, tmp_path, image_factory):
    p1 = image_factory("a.jpg")
    p2 = tmp_path / "b.jpg"
    p2.write_bytes(p1.read_bytes())
    r1 = ingest_file(p1, cfg, store)
    r2 = ingest_file(p2, cfg, store)
    assert r1["image_id"] == r2["image_id"]
    assert store.stats()["images"] == 1


@pytest.mark.parametrize("fmt,ext", [("PNG", ".png"), ("WEBP", ".webp")])
def test_ingest_formats(cfg, store, image_factory, fmt, ext):
    p = image_factory(f"f{ext}", 200, 150, fmt=fmt)
    res = ingest_file(p, cfg, store)
    assert res["status"] == "ok", res
    assert store.get_image(res["image_id"])["ext"] == ext.lstrip(".")


def test_ingest_avif(cfg, store, image_factory, tmp_path):
    p = tmp_path / "a.avif"
    try:
        Image.new("RGB", (64, 64), (10, 20, 30)).save(p, format="AVIF")
    except Exception:
        pytest.skip("当前 Pillow 不支持 AVIF 写入")
    try:
        res = ingest_file(p, cfg, store)
    except Exception as exc:  # 读取侧不支持
        pytest.skip(f"当前 Pillow 不支持 AVIF 读取: {exc}")
    assert res["status"] == "ok", res
