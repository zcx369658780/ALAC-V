"""DS-0002 + teacher cache 的模型输入装配（Gate 3）。

学生路径：像素层 masked 图像 → 冻结 DINOv2 前向 → 可见 patch 特征。
教师路径：完整图像缓存特征（stop-gradient 常量）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from alacv.features import DinoEncoder
from alacv.imaging import open_image
from alacv.teacher_cache import TeacherCacheReader

from .masking import patch_pixels


class SslData:
    def __init__(self, root: Path, cfg: dict, device: str = "cuda", splits=("train", "validation")):
        self.root = Path(root)
        self.device = device
        self.cfg = cfg
        self.indexes = {}
        self.cache = {}
        for s in splits:
            self.indexes[s] = json.loads((self.root / "index" / f"{s}.json").read_text(encoding="utf-8"))["views"]
            self.cache[s] = TeacherCacheReader(self.root / "features" / "teacher", s, verify_sha=False)
        self._encoder = None
        self._by_id = {s: {v["view_id"]: v for v in self.indexes[s]} for s in splits}
        self._split_of = {v["view_id"]: s for s in splits for v in self.indexes[s]}

    # ---- teacher (cached, constant) ----
    def teacher(self, view_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._split_of[view_id]
        cls, patch = self.cache[s].get(view_id)
        return (torch.from_numpy(np.asarray(cls, dtype=np.float32)).to(self.device),
                torch.from_numpy(np.asarray(patch, dtype=np.float32)).to(self.device))

    # ---- student (masked pixel forward, frozen encoder) ----
    @property
    def encoder(self) -> DinoEncoder:
        if self._encoder is None:
            self._encoder = DinoEncoder(self._cfg_obj())
        return self._encoder

    def _cfg_obj(self):
        from alacv.config import load_config

        repo = Path(__file__).resolve().parents[3]
        return load_config(repo / "configs" / "default.yaml")

    def student_masked(self, view_id: str, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._split_of[view_id]
        v = self._by_id[s][view_id]
        img = open_image(self.root / v["path"]).image
        enc = self.encoder
        px = patch_pixels(mask, image_size=enc.input_size, patch=enc.patch_size)
        t = enc.preprocess(img).clone()
        t[:, torch.from_numpy(px)] = 0.0
        x = t[None].to(enc.device, dtype=enc.dtype)
        with torch.no_grad():
            out = enc.model(pixel_values=x).last_hidden_state.float()
        return out[0, 0].to(self.device), out[0, 1 + enc.n_reg:].to(self.device)

    # ---- batch ----
    def make_batch(self, arm: str, view_ids: list[str], masks: list[np.ndarray],
                   view_ids2: list[str] | None = None) -> dict:
        arm = arm.upper()
        batch = {
            "teacher_cls": torch.stack([self.teacher(v)[0] for v in view_ids]).to(self.device),
            "teacher_patch": torch.stack([self.teacher(v)[1] for v in view_ids]).to(self.device),
            "mask": torch.from_numpy(np.stack(masks)).to(self.device),
        }
        batch["teacher_cls2"] = batch["teacher_cls"]
        batch["teacher_patch2"] = batch["teacher_patch"]
        if view_ids2 is not None:
            batch["teacher_cls2"] = torch.stack([self.teacher(v)[0] for v in view_ids2]).to(self.device)
            batch["teacher_patch2"] = torch.stack([self.teacher(v)[1] for v in view_ids2]).to(self.device)
        if arm in {"A1", "A2", "C2"}:
            cls, patch = zip(*[self.student_masked(v, m) for v, m in zip(view_ids, masks)])
            batch["student_cls"] = torch.stack(cls).to(self.device)
            batch["student_patch"] = torch.stack(patch).to(self.device)
        else:
            batch["student_cls"] = batch["teacher_cls"]
            batch["student_patch"] = batch["teacher_patch"]
        return batch
