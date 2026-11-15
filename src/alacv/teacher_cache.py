"""DS-0002 冻结 DINOv2 教师特征缓存（Gate 1 整改）。

格式：每个 split 分 shard，每个 shard 为 SafeTensors：
  cls   : float16 [N, 384]
  patch : float16 [N, 1369, 384]
索引 JSON 记录完整 view_ids 顺序、shard SHA-256、模型指纹、manifest SHA-256。

读取端默认 **fail-closed**：verify_sha=True，重复/缺失 view 抛错，shard 采用有界 LRU 缓存。
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

CACHE_VERSION = "ds0002-teacher-v2"
PATCH_TOKENS = 1369
CLS_DIM = 384


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TeacherCacheWriter:
    def __init__(self, root: Path, split: str, model_meta: dict, manifest_sha256: str,
                 shard_size: int = 1000):
        self.root = Path(root)
        self.split = split
        self.model_meta = model_meta
        self.manifest_sha256 = manifest_sha256
        self.shard_size = int(shard_size)
        self.dir = self.root / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cls: list[np.ndarray] = []
        self._patch: list[np.ndarray] = []
        self._views: list[str] = []
        self.shards: list[dict] = []
        self._shard_idx = 0

    def add(self, view_id: str, cls: np.ndarray, patch: np.ndarray) -> None:
        assert cls.shape == (CLS_DIM,), cls.shape
        assert patch.shape == (PATCH_TOKENS, CLS_DIM), patch.shape
        self._cls.append(cls.astype(np.float16))
        self._patch.append(patch.astype(np.float16))
        self._views.append(view_id)
        if len(self._views) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._views:
            return
        from safetensors.numpy import save_file

        name = f"shard-{self._shard_idx:04d}.safetensors"
        path = self.dir / name
        arrays = {
            "cls": np.stack(self._cls).astype(np.float16),
            "patch": np.stack(self._patch).astype(np.float16),
        }
        tmp = path.with_suffix(".safetensors.tmp")
        save_file(arrays, str(tmp))
        tmp.replace(path)
        self.shards.append(
            {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": sha256_file(path),
                "n": len(self._views),
                "bytes": path.stat().st_size,
                "tensor_spec": {"cls": [len(self._views), CLS_DIM], "patch": [len(self._views), PATCH_TOKENS, CLS_DIM],
                                "dtype": "float16"},
                "views": list(self._views),
            }
        )
        self._cls, self._patch, self._views = [], [], []
        self._shard_idx += 1

    def close(self) -> dict:
        self._flush()
        index = {
            "cache_version": CACHE_VERSION,
            "split": self.split,
            "dtype": "float16",
            "patch_tokens": PATCH_TOKENS,
            "cls_dim": CLS_DIM,
            "model": self.model_meta,
            "manifest_sha256": self.manifest_sha256,
            "n_views": sum(s["n"] for s in self.shards),
            "n_shards": len(self.shards),
            "shards": self.shards,
        }
        (self.root / f"index_{self.split}.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return index


class TeacherCacheReader:
    def __init__(self, root: Path, split: str, verify_sha: bool = True, max_cached_shards: int = 2):
        self.root = Path(root)
        self.index = json.loads((self.root / f"index_{split}.json").read_text(encoding="utf-8"))
        self.split = split
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._max = max(1, int(max_cached_shards))
        if verify_sha:
            for s in self.index["shards"]:
                if sha256_file(self.root / s["path"]) != s["sha256"]:
                    raise RuntimeError(f"shard 校验失败: {s['path']}")
        self.view_to_pos: dict[str, tuple[int, int]] = {}
        for si, s in enumerate(self.index["shards"]):
            for row, vid in enumerate(s["views"]):
                if vid in self.view_to_pos:
                    raise RuntimeError(f"缓存索引存在重复 view_id: {vid}")
                self.view_to_pos[vid] = (si, row)
        if len(self.view_to_pos) != self.index["n_views"]:
            raise RuntimeError("缓存索引 n_views 与 view 映射数量不一致")

    def _load(self, si: int) -> dict:
        if si in self._cache:
            self._cache.move_to_end(si)
            return self._cache[si]
        from safetensors.numpy import load_file

        data = load_file(str(self.root / self.index["shards"][si]["path"]))
        if set(data) != {"cls", "patch"}:
            raise RuntimeError(f"shard keys 不符: {sorted(data)}")
        if data["cls"].dtype != np.float16 or data["patch"].dtype != np.float16:
            raise RuntimeError("shard dtype 不是 float16")
        n = self.index["shards"][si]["n"]
        if data["cls"].shape != (n, CLS_DIM) or data["patch"].shape != (n, PATCH_TOKENS, CLS_DIM):
            raise RuntimeError("shard tensor shape 与索引不一致")
        self._cache[si] = data
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return data

    def get(self, view_id: str) -> tuple[np.ndarray, np.ndarray]:
        if view_id not in self.view_to_pos:
            raise KeyError(f"view 不在缓存索引中: {view_id}")
        si, row = self.view_to_pos[view_id]
        data = self._load(si)
        return data["cls"][row], data["patch"][row]
