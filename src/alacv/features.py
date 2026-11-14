"""DINOv2 ViT-S/14 冻结编码器：CLS 全局特征 + Patch 局部特征（FP16）。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import fit_to_input

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FeatureBatch:
    cls: np.ndarray        # (N, D) float16
    patch: np.ndarray      # (N, P, D) float16
    grid: tuple[int, int]  # (gh, gw)


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class DinoEncoder:
    """封装模型加载、预处理、前向与特征后处理。"""

    def __init__(self, cfg, device: str | None = None, manifest: dict | None = None):
        import torch

        self.cfg = cfg
        self.device = _resolve_device(device or cfg.get("model.device", "auto"))
        self.input_size = int(cfg.get("model.input_size", 518))
        self.patch_size = int(cfg.get("model.patch_size", 14))
        self.precision = cfg.get("model.precision", "fp16")
        self.dtype = torch.float16 if self.precision == "fp16" and self.device != "cpu" else torch.float32
        self.batch_size = int(cfg.get("model.batch_size", 1))
        self.model_id = cfg.get("model.id", "facebook/dinov2-small")
        self.local_dir = cfg.path("model.local_dir")
        self.manifest = manifest or self._read_manifest()

        from transformers import Dinov2Model

        kwargs = {"local_files_only": True}
        try:
            model = Dinov2Model.from_pretrained(str(self.local_dir), torch_dtype=self.dtype, **kwargs)
        except TypeError:
            model = Dinov2Model.from_pretrained(str(self.local_dir), dtype=self.dtype, **kwargs)
        model.eval()
        model.to(self.device)
        self.model = model
        self.n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
        self.cls_dim = int(model.config.hidden_size)
        self.num_params = int(sum(p.numel() for p in model.parameters()))

    def _read_manifest(self) -> dict:
        p = self.cfg.path("paths.manifests_dir") / "model-dinov2_vits14.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    # ------------------------------------------------------------------
    def preprocess(self, img: Image.Image):
        import torch

        x = fit_to_input(img.convert("RGB"), self.input_size, self.cfg.get("pyramid.method", "bicubic"))
        arr = np.asarray(x, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
        return tensor

    def _forward(self, batch):
        import torch

        with torch.inference_mode():
            x = torch.stack(batch).to(self.device, dtype=self.dtype, non_blocking=True)
            out = self.model(pixel_values=x).last_hidden_state
            out = out.float()
        cls = out[:, 0]
        patch = out[:, 1 + self.n_reg :]
        return cls, patch

    def encode_pil(self, images: list[Image.Image]) -> FeatureBatch:
        import torch

        if not images:
            raise ValueError("encode_pil 收到空列表")
        all_cls, all_patch = [], []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            cls, patch = self._forward([self.preprocess(im) for im in chunk])
            all_cls.append(cls.cpu().numpy())
            all_patch.append(patch.cpu().numpy())
            if self.device == "cuda":
                del cls, patch
        cls = np.concatenate(all_cls, axis=0).astype(np.float16)
        patch = np.concatenate(all_patch, axis=0).astype(np.float16)
        n_patch = patch.shape[1]
        grid = int(round(math.sqrt(n_patch)))
        if grid * grid != n_patch:
            raise RuntimeError(f"patch 数 {n_patch} 不是完全平方数，输入尺寸 {self.input_size} 与 patch {self.patch_size} 不匹配")
        return FeatureBatch(cls=cls, patch=patch, grid=(grid, grid))

    def encode_one(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        fb = self.encode_pil([img])
        return fb.cls[0], fb.patch[0]

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        import torch

        return {
            "model_id": self.model_id,
            "arch": self.cfg.get("model.arch"),
            "num_params": self.num_params,
            "cls_dim": self.cls_dim,
            "n_register_tokens": self.n_reg,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "grid": self.input_size // self.patch_size,
            "precision": self.precision,
            "device": self.device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "manifest": self.manifest,
            "preprocess": {
                "resize": "short_side_to_input_size",
                "crop": "center",
                "interpolation": self.cfg.get("pyramid.method", "bicubic"),
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "color_order": "RGB",
                "value_range": [0.0, 1.0],
            },
        }
