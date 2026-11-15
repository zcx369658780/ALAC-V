"""DS-0002 读取层：训练 DataLoader 的字段白名单隔离。

训练代码只能通过 `Ds0002Dataset.__getitem__` 取得 `view_id`、`image`、`augment` 三个字段；
`scene_id`、`base_seed`、`targets`、`factors` 以及评估标签 sidecar 均不在返回结构中。
同场景配对（B2）由 `PairBatchSampler` 在**采样层**完成，不把 scene_id 传给模型。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ALLOWED_ITEM_KEYS = frozenset({"view_id", "image", "augment"})


class Ds0002Dataset:
    def __init__(self, manifest_path: str | Path, image_root: str | Path, transform=None):
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.views = self.manifest["views"]
        self.transform = transform
        # 采样层内部使用；不进入 __getitem__ 返回值
        self._groups: dict[str, list[int]] = {}
        for i, v in enumerate(self.views):
            self._groups.setdefault(v["scene_id"], []).append(i)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> dict:
        v = self.views[idx]
        img = Image.open(self.image_root / v["path"]).convert("RGB")
        item = {"view_id": v["view_id"], "image": img, "augment": v["augment"]}
        if self.transform is not None:
            item["image"] = self.transform(item["image"])
        return item

    # ---- 仅供训练采样器使用；调用方不得把 group 信息写入模型输入 ----
    def group_indices(self) -> list[list[int]]:
        return [idxs for idxs in self._groups.values() if len(idxs) > 1]


class PairBatchSampler:
    """B2 用：每个 batch 由若干 (同场景两视图) 对组成。

    只使用"同一源图像"这一关系；不向模型暴露场景 ID。
    """

    def __init__(self, dataset: Ds0002Dataset, batch_pairs: int, seed: int):
        self.pairs = dataset.group_indices()
        self.batch_pairs = batch_pairs
        self.rng = __import__("random").Random(seed)

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
