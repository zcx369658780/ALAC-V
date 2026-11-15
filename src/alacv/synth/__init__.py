"""DS-0002 程序化合成数据（Gate 1）。"""

from .scene import (
    GENERATOR_VERSION,
    PRNG,
    SPLIT_COUNTS,
    SPLIT_SEED_BASE,
    VIEW_COUNTS,
    WIDTH,
    HEIGHT,
    Scene,
    augment,
    base_seed,
    render_scene,
    sha256_image,
    view_rng,
)

__all__ = [
    "GENERATOR_VERSION", "PRNG", "SPLIT_COUNTS", "SPLIT_SEED_BASE", "VIEW_COUNTS",
    "WIDTH", "HEIGHT", "Scene", "augment", "base_seed", "render_scene",
    "sha256_image", "view_rng",
]
