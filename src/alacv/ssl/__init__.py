"""Phase 2A.1 自监督模型组件（Gate 3）。"""

from .arms import A1, A2, B2, C2, F0, R0, build_arm
from .checkpoint import (
    CheckpointRecord,
    EarlyStopping,
    VicRegState,
    descriptor_sha256,
    effective_rank,
    kendall_tau,
    mean_ap_at_k,
    recall_at_k,
    select_checkpoint,
)
from .masking import (
    BLOCKS,
    GRID,
    MASK_PATCHES,
    N_PATCHES,
    contiguous_block_mask,
    mask_seed_for,
    mask_sha256,
    patch_pixels,
    random_mask,
)
from .modules import Adapter, B2Predictor, Predictor

__all__ = [
    "A1", "A2", "B2", "C2", "F0", "R0", "build_arm",
    "Adapter", "Predictor", "B2Predictor",
    "contiguous_block_mask", "random_mask", "mask_sha256", "mask_seed_for", "patch_pixels",
    "GRID", "N_PATCHES", "MASK_PATCHES", "BLOCKS",
    "CheckpointRecord", "select_checkpoint", "EarlyStopping", "VicRegState",
    "recall_at_k", "mean_ap_at_k", "kendall_tau", "descriptor_sha256", "effective_rank",
]
