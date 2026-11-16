"""检索、checkpoint 选择、早停与 VICReg 触发规则（预注册逻辑）。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch


def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def descriptor_sha256(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


def recall_at_k(query: torch.Tensor, gallery: torch.Tensor, target: torch.Tensor, k: int) -> float:
    q = l2norm(query)
    g = l2norm(gallery)
    sim = q @ g.T
    top = sim.topk(k, dim=1).indices
    hit = (top == target[:, None]).any(dim=1).float().mean()
    return float(hit)


def mean_ap_at_k(query: torch.Tensor, gallery: torch.Tensor, target: torch.Tensor, k: int = 10) -> float:
    q = l2norm(query)
    g = l2norm(gallery)
    sim = q @ g.T
    order = sim.topk(min(k, g.shape[0]), dim=1).indices
    aps = []
    for i in range(q.shape[0]):
        rank = (order[i] == target[i]).nonzero(as_tuple=False)
        aps.append(0.0 if rank.numel() == 0 else 1.0 / (int(rank[0]) + 1))
    return float(np.mean(aps))


def kendall_tau(a: torch.Tensor, b: torch.Tensor) -> float:
    a = np.asarray(a.detach().cpu()).reshape(-1)
    b = np.asarray(b.detach().cpu()).reshape(-1)
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da * db > 0:
                conc += 1
            elif da * db < 0:
                disc += 1
    denom = n * (n - 1) / 2
    return (conc - disc) / denom if denom else 1.0


@dataclass
class CheckpointRecord:
    epoch: int
    val_recall1: float
    val_loss: float


def select_checkpoint(records: list[CheckpointRecord]) -> CheckpointRecord:
    """H1 checkpoint：val Recall@1 最优；并列看 val loss 更低；再并列取更早 epoch。"""
    if not records:
        raise ValueError("no checkpoint records")
    return sorted(records, key=lambda r: (-r.val_recall1, r.val_loss, r.epoch))[0]


class EarlyStopping:
    """按 validation prediction loss，patience=3。"""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad = 0
        self.stopped = False

    def update(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = float(val_loss)
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.stopped = True
        return self.stopped


@dataclass
class VicRegState:
    """B2 坍塌规则：有效秩 < F0 的 60% 连续 2 个评估 epoch → 下一 epoch 启用 variance，标签 B2+VR。"""

    active: bool = False
    consecutive: int = 0
    activation_epoch: int | None = None
    history: list = field(default_factory=list)

    def update(self, epoch: int, val_rank_ratio: float) -> bool:
        self.history.append({"epoch": epoch, "val_rank_ratio": float(val_rank_ratio)})
        if val_rank_ratio < 0.60:
            self.consecutive += 1
        else:
            self.consecutive = 0
        if not self.active and self.consecutive >= 2:
            self.active = True
            self.activation_epoch = epoch + 1
        return self.active


def effective_rank(x: torch.Tensor) -> float:
    s = torch.linalg.svdvals(x.detach().to(torch.float32))
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def participation_ratio(x: torch.Tensor) -> float:
    s = torch.linalg.svdvals(x.detach().to(torch.float32))
    return float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12))
