"""确定性 mask 采样：A2 连续块（821）与 C2 随机（821）。

坐标约定：37×37 patch 网格，扁平索引 `r*37 + c`。
"""

from __future__ import annotations

import hashlib

import numpy as np

GRID = 37
N_PATCHES = GRID * GRID
MASK_PATCHES = 821
BLOCKS = 4


def mask_sha256(mask: np.ndarray) -> str:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return hashlib.sha256(idx.astype(np.int32).tobytes()).hexdigest()


def contiguous_block_mask(seed: int, total: int = MASK_PATCHES) -> np.ndarray:
    """4 个严格互不重叠的连续块，合计恰好 total 个 patch。

    4 个块分别放在 4 个 19×19 象限内（14×14 随机偏移 0–4），保证互不重叠；
    剩余 37 个 patch 按块 0→3、先右列后底行扩展，且不越过各自象限边界。
    """
    rng = np.random.default_rng(int(seed))
    base = total // BLOCKS
    side = int(round(base ** 0.5))
    occupied = np.zeros((GRID, GRID), dtype=bool)
    boxes: list[list[int]] = []
    quad = [(0, 0), (0, 19), (19, 0), (19, 19)]
    for (qr, qc) in quad:
        off = GRID - 19 - side  # 4
        r = qr + int(rng.integers(0, off + 1))
        c = qc + int(rng.integers(0, off + 1))
        occupied[r:r + side, c:c + side] = True
        boxes.append([r, c, side, side, qr, qc])

    extra = total - int(occupied.sum())
    guard = 0
    while extra > 0 and guard < 200:
        guard += 1
        for i, (r, c, h, w, qr, qc) in enumerate(boxes):
            rmax, cmax = qr + 17, qc + 17
            if c + w <= cmax:
                for rr in range(r, r + h):
                    if not occupied[rr, c + w]:
                        occupied[rr, c + w] = True
                        extra -= 1
                        if extra == 0:
                            break
                boxes[i][3] += 1
            if extra == 0:
                break
            if r + h <= rmax:
                for cc in range(c, c + w):
                    if not occupied[r + h, cc]:
                        occupied[r + h, cc] = True
                        extra -= 1
                        if extra == 0:
                            break
                boxes[i][2] += 1
            if extra == 0:
                break
    if extra > 0:
        free = np.flatnonzero(~occupied.ravel())
        occupied.ravel()[free[:extra]] = True
        extra = 0
    assert occupied.sum() == total, int(occupied.sum())
    return occupied.ravel()


def random_mask(seed: int, total: int = MASK_PATCHES) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(N_PATCHES, size=total, replace=False)
    mask = np.zeros(N_PATCHES, dtype=bool)
    mask[idx] = True
    return mask


def mask_seed_for(base_seed: int, step: int) -> int:
    return int(base_seed) * 1000 + int(step)


def patch_pixels(mask: np.ndarray, image_size: int = 518, patch: int = 14) -> np.ndarray:
    """把 patch mask 展开成像素级布尔掩码（H×W）。"""
    grid = int(round(image_size / patch))
    m = np.asarray(mask, dtype=bool).reshape(grid, grid)
    return np.repeat(np.repeat(m, patch, axis=0), patch, axis=1)
