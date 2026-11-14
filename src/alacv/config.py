"""配置加载 / 合并 / 校验 / 哈希 / 快照。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Config:
    """轻量配置对象：点号访问 + 稳定哈希。"""

    def __init__(self, data: dict, source: Path | None = None, root: Path | None = None):
        self._data = data
        self.source = source
        self.root = root or Path.cwd()

    # ---------------- 访问 ----------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def require(self, dotted: str) -> Any:
        sentinel = object()
        val = self.get(dotted, sentinel)
        if val is sentinel:
            raise ConfigError(f"缺少必需配置项: {dotted}")
        return val

    def to_dict(self) -> dict:
        return json.loads(_canonical(self._data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # ---------------- 路径 ----------------
    def path(self, dotted: str) -> Path:
        raw = self.require(dotted)
        p = Path(os.path.expandvars(str(raw)))
        return p if p.is_absolute() else (self.root / p)

    # ---------------- 哈希 / 快照 ----------------
    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self._data).encode("utf-8")).hexdigest()

    @property
    def stable_sha256(self) -> str:
        """内容哈希：剔除与"表示"无关的易变项（路径、实验编排、日志级别）。

        用于判断派生数据是否过期，保证同一表示在不同运行目录下哈希一致。
        """
        volatile = {"paths", "experiment", "logging", "dataset"}
        payload = {k: v for k, v in self._data.items() if k not in volatile}
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def snapshot(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "config.snapshot.yaml"
        dst.write_text(yaml.safe_dump(self._data, sort_keys=True, allow_unicode=True), encoding="utf-8")
        (out_dir / "config.sha256").write_text(self.sha256 + "\n", encoding="utf-8")
        return dst


def _validate(cfg: Config) -> None:
    scales = cfg.require("pyramid.scales")
    if not (isinstance(scales, list) and scales and all(isinstance(s, int) and s >= 1 for s in scales)):
        raise ConfigError("pyramid.scales 必须是正整数列表")
    if scales != sorted(scales):
        raise ConfigError("pyramid.scales 必须升序")
    isize = cfg.require("model.input_size")
    psize = cfg.require("model.patch_size")
    if isize % psize != 0:
        raise ConfigError(f"model.input_size={isize} 必须是 patch_size={psize} 的整数倍")
    variants = cfg.require("encoding.variants")
    if not isinstance(variants, dict) or not variants:
        raise ConfigError("encoding.variants 不能为空")
    for name, sources in variants.items():
        if not isinstance(sources, list) or not sources:
            raise ConfigError(f"变体 {name} 的源列表为空")
        for src in sources:
            if not _valid_source(src):
                raise ConfigError(f"变体 {name} 含非法源: {src!r}")
    prec = cfg.require("model.precision")
    if prec not in ("fp16", "fp32"):
        raise ConfigError("model.precision 只能是 fp16 / fp32")
    frac = cfg.require("foveation.roi_frac")
    if not (0 < float(frac) <= 1):
        raise ConfigError("foveation.roi_frac 必须在 (0, 1]")
    cs = float(cfg.require("foveation.canvas_scale"))
    if not (0 < cs <= 1):
        raise ConfigError("foveation.canvas_scale 必须在 (0, 1]")


def _valid_source(src: str) -> bool:
    if src == "original" or src == "foveated" or src == "mosaic_only":
        return True
    if src.startswith("pyramid:L"):
        return src[9:].isdigit()
    if src.startswith("roi:"):
        return len(src) > 4
    return False


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """加载 YAML，与 default.yaml 浅层合并（recursive），校验并返回 Config。"""
    base_data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    src_path: Path | None = None
    data = base_data
    if path is not None:
        src_path = Path(path)
        if not src_path.is_absolute():
            src_path = Path.cwd() / src_path
        override = yaml.safe_load(src_path.read_text(encoding="utf-8")) or {}
        data = _deep_merge(base_data, override)
    if overrides:
        data = _deep_merge(data, overrides)

    base_dir = src_path.parent if src_path else Path.cwd()
    root = base_dir.parent if base_dir.name == "configs" else Path.cwd()
    cfg = Config(data, source=src_path, root=root)
    _validate(cfg)
    return cfg


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)
