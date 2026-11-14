"""需要本地 DINOv2 权重；缺失时整体跳过。"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models" / "dinov2_vits14"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="本地无 DINOv2 权重")


@pytest.fixture(scope="module")
def encoder():
    from alacv.config import load_config
    from alacv.features import DinoEncoder

    cfg = load_config(ROOT / "configs" / "default.yaml")
    return DinoEncoder(cfg, device="cpu")


def test_shapes_and_params(encoder):
    assert encoder.cls_dim == 384
    assert encoder.num_params > 20_000_000
    assert encoder.num_params < 25_000_000


def test_forward_shapes(encoder, image_factory):
    p = image_factory("m.jpg", 400, 300)
    from alacv.imaging import open_image

    img = open_image(p).image
    cls, patch = encoder.encode_one(img)
    assert cls.shape == (384,)
    assert patch.shape == (37 * 37, 384)
    assert cls.dtype == np.float16


def test_deterministic(encoder, image_factory):
    from alacv.imaging import open_image

    img = open_image(image_factory("d.jpg", 320, 240)).image
    a, _ = encoder.encode_one(img)
    b, _ = encoder.encode_one(img)
    assert np.array_equal(a, b)


def test_different_images_differ(encoder, image_factory):
    from alacv.imaging import open_image

    i1 = open_image(image_factory("x.jpg", 320, 240, seed=1)).image
    i2 = open_image(image_factory("y.jpg", 320, 240, seed=2)).image
    c1, _ = encoder.encode_one(i1)
    c2, _ = encoder.encode_one(i2)
    c1f, c2f = c1.astype(np.float64), c2.astype(np.float64)
    n1, n2 = np.linalg.norm(c1f), np.linalg.norm(c2f)
    cos12 = float(np.dot(c1f, c2f) / (n1 * n2))
    self_cos = float(np.dot(c1f, c1f) / (n1 * n1))
    assert self_cos > cos12, "自身相似度必须高于不同图相似度"
    assert cos12 < 0.9999


def test_fp16_vs_fp32_close(image_factory):
    """FP16 与 FP32 前向的余弦一致性（ADR-0006 的数值校验）。"""
    import torch
    from transformers import Dinov2Model

    from alacv.config import load_config
    from alacv.features import IMAGENET_MEAN, IMAGENET_STD
    from alacv.imaging import fit_to_input, open_image

    cfg = load_config(ROOT / "configs" / "default.yaml")
    img = open_image(image_factory("z.jpg", 448, 448)).image

    x = fit_to_input(img, 518)
    arr = np.asarray(x, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    m = Dinov2Model.from_pretrained(str(MODEL_DIR), local_files_only=True)
    m.eval()
    with torch.inference_mode():
        c32 = m(pixel_values=t).last_hidden_state[:, 0].float().numpy()[0]
    m16 = Dinov2Model.from_pretrained(str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float16)
    m16.eval()
    with torch.inference_mode():
        c16 = m16(pixel_values=t.half()).last_hidden_state[:, 0].float().numpy()[0]
    cos = float(np.dot(c32, c16) / (np.linalg.norm(c32) * np.linalg.norm(c16)))
    assert cos > 0.998
