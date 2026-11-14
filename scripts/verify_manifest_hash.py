"""公开 manifest 哈希校验（P1-3）。

哈希规范：**SHA-256 over LF-normalized bytes**，即 Git blob 内容 / GitHub 公开文件字节。
历史上 `ds-0001-manifest-sanitization.json` 记录的是 Windows CRLF 工作树字节，已作废并改名。

用法：
    python scripts/verify_manifest_hash.py            # 本地文件 vs 记录 vs git blob
    python scripts/verify_manifest_hash.py --github   # 额外对比 GitHub raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "ds-0001-alacv-synthetic-v1.json"
RECORD = ROOT / "data" / "manifests" / "ds-0001-manifest-sanitization.json"
RAW_URL = "https://raw.githubusercontent.com/zcx369658780/ALAC-V/main/data/manifests/ds-0001-alacv-synthetic-v1.json"


def lf_bytes(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def sha256_lf(b: bytes) -> str:
    return hashlib.sha256(lf_bytes(b)).hexdigest()


def git_blob(rev: str, path: str) -> bytes | None:
    try:
        p = subprocess.run(
            ["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, timeout=30
        )
        return p.stdout if p.returncode == 0 else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", action="store_true")
    args = ap.parse_args()

    worktree = MANIFEST.read_bytes()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    expected = record["manifest_sha256_lf"]

    checks = {
        "worktree_lf": sha256_lf(worktree),
        "recorded": expected,
    }
    blob = git_blob("HEAD", "data/manifests/ds-0001-alacv-synthetic-v1.json")
    if blob is not None:
        checks["git_blob_lf"] = sha256_lf(blob)
    if args.github:
        req = urllib.request.Request(RAW_URL, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=30) as r:
            checks["github_raw_lf"] = sha256_lf(r.read())

    ok = True
    for name, value in checks.items():
        match = value == expected
        ok = ok and match
        print(f"{name:16s} {value}  {'OK' if match else 'MISMATCH'}")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
