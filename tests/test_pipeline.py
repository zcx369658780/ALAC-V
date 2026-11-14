import numpy as np

from alacv.features import FeatureBatch
from alacv.pipeline import ImagePipeline


class FakeEncoder:
    """确定性假编码器：不加载模型，仅用于验证编排/存储逻辑。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cls_dim = 8

    def describe(self):
        return {
            "model_id": "fake",
            "arch": "fake",
            "num_params": 0,
            "cls_dim": self.cls_dim,
            "n_register_tokens": 0,
            "input_size": 518,
            "patch_size": 14,
            "grid": 37,
            "precision": "fp16",
            "device": "cpu",
            "manifest": {},
        }

    def encode_pil(self, images):
        n = len(images)
        rng = np.random.default_rng(n)
        cls = rng.normal(size=(n, self.cls_dim)).astype(np.float16)
        patch = rng.normal(size=(n, 4, 4, self.cls_dim)).astype(np.float16)
        return FeatureBatch(cls=cls, patch=patch, grid=(2, 2))


def test_variant_c_full_pipeline(cfg, store, image_factory):
    p = image_factory("scene.jpg", 480, 360)
    pipe = ImagePipeline(cfg, store, FakeEncoder(cfg))
    item = pipe.process(p, "C", "ds-test")
    assert item["status"] == "ok"
    assert item["sources"] == [
        "pyramid:L0", "pyramid:L1", "pyramid:L2", "pyramid:L3", "foveated", "roi:roi_00",
    ]
    iid = item["image_id"]
    assert len(store.assets(iid, "pyramid")) == 4
    assert len(store.assets(iid, "foveated")) == 1
    assert len(store.assets(iid, "roi_crop")) >= 1
    assert len(store.rois_of(iid)) >= 1
    assert store.stats()["features"] == 6
    assert item["t_total_ms"] > 0 and item["t_encode_ms"] >= 0
    assert item["feature_bytes"] > 0


def test_variant_a_only_original(cfg, store, image_factory):
    p = image_factory("a.jpg", 320, 240)
    pipe = ImagePipeline(cfg, store, FakeEncoder(cfg))
    item = pipe.process(p, "A", "ds-test")
    assert item["status"] == "ok"
    assert item["sources"] == ["original"]
    assert store.assets(item["image_id"], "pyramid") == []
    assert store.stats()["features"] == 1


def test_pipeline_idempotent(cfg, store, image_factory):
    p = image_factory("i.jpg", 320, 240)
    pipe = ImagePipeline(cfg, store, FakeEncoder(cfg))
    a = pipe.process(p, "B", "ds-test")
    b = pipe.process(p, "B", "ds-test")
    assert a["image_id"] == b["image_id"]
    assert store.stats()["features"] == 4
    assert store.stats()["images"] == 1


def test_rejected_image_returns_status(cfg, store, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"xx")
    pipe = ImagePipeline(cfg, store, FakeEncoder(cfg))
    item = pipe.process(bad, "A")
    assert item["status"] == "rejected"
