"""下载 DINOv2 ViT-S/14（HuggingFace），记录指纹与许可证。

本机代理对长连接不稳定，故对每个文件实现可续传的分块下载（Range 请求）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "facebook/dinov2-small"
LOCAL_DIR = ROOT / "artifacts" / "models" / "dinov2_vits14"
MANIFEST = ROOT / "data" / "manifests" / "model-dinov2_vits14.json"
APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"

WANT_SUFFIXES = (".json", ".safetensors", ".bin", ".txt", ".md")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_download(url: str, dest: Path, expected_size: int | None, retries: int = 12) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected_size is not None and have == expected_size:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, dest.open("ab") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    have += len(chunk)
            size = dest.stat().st_size
            if expected_size is None or size == expected_size:
                print(f"[fetch] {dest.name} {size/1e6:.1f} MB")
                return
            print(f"[fetch] {dest.name} 不完整 ({size}/{expected_size})，重试 {attempt}/{retries}")
        except Exception as exc:
            print(f"[fetch] {dest.name} 第 {attempt} 次失败: {type(exc).__name__}: {str(exc)[:80]}")
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"下载失败: {url}")


def main() -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(REPO_ID, files_metadata=True)
    revision = info.sha
    print(f"[fetch] {REPO_ID} @ {revision}")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for sib in info.siblings:
        name = sib.rfilename
        if not name.endswith(WANT_SUFFIXES):
            continue
        size = getattr(sib, "size", None)
        url = f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{name}"
        robust_download(url, LOCAL_DIR / name, size)
        names.append(name)

    weights = [p for p in LOCAL_DIR.glob("*.safetensors")] + [p for p in LOCAL_DIR.glob("*.bin")]
    if not weights:
        print("未找到权重文件", file=sys.stderr)
        return 1
    w = weights[0]
    weights_sha = sha256_file(w)

    lic = LOCAL_DIR / "LICENSE"
    if not lic.exists():
        try:
            robust_download(APACHE_URL, lic, None, retries=5)
            print("[fetch] LICENSE 未随仓库提供，已下载 Apache-2.0 全文")
        except Exception as exc:
            lic.write_text(
                "facebook/dinov2-small is released under the Apache License 2.0.\n"
                f"See https://www.apache.org/licenses/LICENSE-2.0\n(download failed: {exc})\n",
                encoding="utf-8",
            )

    cfg_json = LOCAL_DIR / "config.json"
    manifest = {
        "model_id": "facebook/dinov2-small",
        "display_name": "DINOv2 ViT-S/14 (dinov2-small)",
        "source": f"huggingface:{REPO_ID}",
        "revision": revision,
        "license": "Apache-2.0",
        "license_url": "https://github.com/facebookresearch/dinov2/blob/main/LICENSE",
        "license_file": str(lic.relative_to(ROOT)).replace("\\", "/"),
        "weights_file": w.name,
        "weights_bytes": w.stat().st_size,
        "weights_sha256": weights_sha,
        "config_sha256": sha256_file(cfg_json) if cfg_json.exists() else None,
        "files": names,
        "arch": "dinov2_vits14",
        "input_size": 518,
        "patch_size": 14,
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_dir": str(LOCAL_DIR.relative_to(ROOT)).replace("\\", "/"),
        "note": "权重冻结，未做任何训练/微调；SHA-256 用于自证来源",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOCAL_DIR / "SHA256SUMS").write_text(f"{weights_sha}  {w.name}\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
