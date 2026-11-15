"""DS-0002 训练读取层（P1-2 整改）。

- 只加载**投影后的训练索引** `index/{split}.json`：`view_id / path / augment / pair_key`。
- `pair_key` 是不可逆的匿名分组键（`sha256("ds0002-pair:"+scene_id)[:12]`），用于 B2 同场景配对；
  它不包含也不可还原 scene_id。
- 对象内部不保存完整 audit manifest、targets、factors 或任何评估标签。
- 评估标签只能通过 `load_eval_labels()` 读取，训练代码禁止调用（见 tests）。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ALLOWED_INDEX_KEYS = frozenset({"view_id", "path", "augment", "pair_key"})
ALLOWED_ITEM_KEYS = frozenset({"view_id", "image", "augment"})


class Ds0002Dataset:
    def __init__(self, index_path: str | Path, image_root: str | Path, transform=None):
        self.index_path = Path(index_path)
        self.image_root = Path(image_root)
        self.transform = transform
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        views = payload["views"]
        for v in views:
            extra = set(v) - ALLOWED_INDEX_KEYS
            if extra:
                raise ValueError(f"训练索引包含未授权字段: {sorted(extra)}")
        self._views = views
        self._groups: dict[str, list[int]] = {}
        for i, v in enumerate(views):
            self._groups.setdefault(v["pair_key"], []).append(i)

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> dict:
        v = self._views[idx]
        img = Image.open(self.image_root / v["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"view_id": v["view_id"], "image": img, "augment": v["augment"]}

    def group_indices(self) -> list[list[int]]:
        """供训练采样器使用；返回同 pair_key 的索引组，不暴露 pair_key 本身。"""
        return [idxs for idxs in self._groups.values() if len(idxs) > 1]


class PairBatchSampler:
    """B2 用：每个 batch 由若干同场景两视图对组成；不向模型暴露分组信息。"""

    def __init__(self, dataset: Ds0002Dataset, batch_pairs: int, seed: int):
        self.pairs = dataset.group_indices()
        self.batch_pairs = batch_pairs
        import random

        self.rng = random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.pairs)))
        self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_pairs):
            batch = []
            for k in order[start : start + self.batch_pairs]:
                idxs = self.pairs[k]
                a, b = self.rng.sample(idxs, 2)
                batch.append((a, b))
            yield batch

    def __len__(self) -> int:
        return (len(self.pairs) + self.batch_pairs - 1) // self.batch_pairs


def load_eval_labels(path: str | Path) -> dict:
    """评估专用；训练代码禁止导入/调用。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
