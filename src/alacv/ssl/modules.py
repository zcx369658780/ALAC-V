"""Adapter / Predictor 模块（384 宽，2 blocks）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

GRID = 37
N_PATCHES = GRID * GRID
D = 384


def sinusoidal_2d(n: int = N_PATCHES, d: int = D, grid: int = GRID) -> torch.Tensor:
    table = torch.zeros(n, d)
    pos = torch.arange(grid, dtype=torch.float32)
    yy, xx = torch.meshgrid(pos, pos, indexing="ij")
    for i in range(d // 4):
        freq = 1.0 / (10000 ** (2 * i / d))
        table[:, 4 * i + 0] = torch.sin(xx.reshape(-1) * freq)
        table[:, 4 * i + 1] = torch.cos(xx.reshape(-1) * freq)
        table[:, 4 * i + 2] = torch.sin(yy.reshape(-1) * freq)
        table[:, 4 * i + 3] = torch.cos(yy.reshape(-1) * freq)
    return table


class Block(nn.Module):
    def __init__(self, d: int = D, heads: int = 6, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Linear(d * mlp_ratio, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class Adapter(nn.Module):
    """输入 [CLS; visible patches; mask tokens]，输出同序；CLS 位于 index 0。"""

    def __init__(self, d: int = D, depth: int = 2, heads: int = 6, mlp_ratio: int = 4):
        super().__init__()
        self.register_buffer("pos_table", sinusoidal_2d(), persistent=False)
        self.cls_pos = nn.Parameter(torch.zeros(d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, cls_tok: torch.Tensor, vis_feats: torch.Tensor,
                vis_pos: torch.Tensor, mask_pos: torch.Tensor):
        b, nv, _ = vis_feats.shape
        nm = mask_pos.shape[1]
        mask_tok = self.mask_token.expand(b, nm, -1)
        seq = torch.cat([cls_tok[:, None, :], vis_feats, mask_tok], dim=1)
        pos = torch.cat([self.cls_pos.expand(b, 1, -1), self.pos_table[vis_pos],
                         self.pos_table[mask_pos]], dim=1)
        x = seq + pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), slice(1 + nv, 1 + nv + nm)


class Predictor(nn.Module):
    """对 adapter 输出序列再做 2 层 self-attention，取 mask 位置输出。"""

    def __init__(self, d: int = D, depth: int = 2, heads: int = 6, mlp_ratio: int = 4):
        super().__init__()
        self.register_buffer("pos_table", sinusoidal_2d(), persistent=False)
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, seq: torch.Tensor, pos_idx: torch.Tensor, mask_slice: slice) -> torch.Tensor:
        x = seq + self.pos_table[pos_idx]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, mask_slice]


class B2Predictor(nn.Module):
    """CLS(384) → 768 → 768 → 768。

    设计文档写 "MLP 768→768→768 ≈1.18M"，但输入 CLS 是 384 维，需要一个 384→768 投影；
    因此实际参数约 1.47M（详见 Gate 3 报告的 design-ambiguity 说明）。
    """

    def __init__(self, in_dim: int = D, d: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
