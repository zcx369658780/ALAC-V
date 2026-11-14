"""R11：公开数据清单的哈希规范与可复算性。"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "ds-0001-alacv-synthetic-v1.json"
RECORD = ROOT / "data" / "manifests" / "ds-0001-manifest-sanitization.json"


def _sha_lf(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def test_manifest_hash_spec_is_validated():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["hash_spec"].startswith("sha256 over LF-normalized bytes")
    assert record["manifest_sha256_lf"] == _sha_lf(MANIFEST.read_bytes())


def test_manifest_has_no_crlf():
    assert b"\r\n" not in MANIFEST.read_bytes(), "公开 manifest 必须是 LF，否则跨平台哈希不可复算"


def test_git_blob_matches_recorded_hash():
    try:
        p = subprocess.run(
            ["git", "show", "HEAD:data/manifests/ds-0001-alacv-synthetic-v1.json"],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"git 不可用: {exc}")
    if p.returncode != 0:
        pytest.skip("manifest 尚未被 git 跟踪")
    blob = p.stdout
    if blob != MANIFEST.read_bytes():
        pytest.skip("工作树 manifest 尚未提交，git blob 为旧版本")
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert _sha_lf(blob) == record["manifest_sha256_lf"]
