"""pytest 公共夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alacv.config import load_config  # noqa: E402
from alacv.store import VisualStore  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    overrides = {
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "features_dir": str(tmp_path / "features"),
            "index_db": str(tmp_path / "index" / "alacv.sqlite"),
            "manifests_dir": str(tmp_path / "manifests"),
        }
    }
    return load_config(ROOT / "configs" / "default.yaml", overrides=overrides)


@pytest.fixture()
def store(cfg):
    s = VisualStore(cfg.path("paths.index_db"), cfg.path("paths.features_dir"))
    yield s
    s.close()


def make_textured_image(w: int = 320, h: int = 240, seed: int = 0, patch: bool = True) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack(
        [
            120 + 40 * np.sin(2 * np.pi * xx / 97.0) + rng.normal(0, 3, (h, w)),
            110 + 50 * np.cos(2 * np.pi * yy / 61.0) + rng.normal(0, 3, (h, w)),
            100 + 30 * np.sin(2 * np.pi * (xx + yy) / 143.0) + rng.normal(0, 3, (h, w)),
        ],
        axis=-1,
    )
    if patch:
        ps = min(w, h) // 4
        px, py = w // 3, h // 3
        checker = ((np.mgrid[0:ps, 0:ps][0] // 6 + np.mgrid[0:ps, 0:ps][1] // 6) % 2) * 255
        arr[py : py + ps, px : px + ps, 0] = checker
        arr[py : py + ps, px : px + ps, 1] = 255 - checker
        arr[py : py + ps, px : px + ps, 2] = 30
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


@pytest.fixture()
def image_factory(tmp_path):
    def _make(name: str = "img.jpg", w: int = 320, h: int = 240, seed: int = 0, fmt: str | None = None, quality: int = 95) -> Path:
        p = tmp_path / name
        img = make_textured_image(w, h, seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        if fmt:
            img.save(p, format=fmt, quality=quality)
        else:
            img.save(p, quality=quality)
        return p

    return _make
