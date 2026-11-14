import numpy as np
import pytest

from alacv.metrics import Timer, psnr, ssim


def test_timer_measures_positive_time():
    import time

    with Timer() as t:
        time.sleep(0.02)
    assert t.elapsed_ms >= 15


def test_psnr_identical_is_infinite():
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    assert psnr(a, a) == float("inf")


def test_psnr_decreases_with_noise():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.normal(0, 5, a.shape), 0, 255).astype(np.uint8)
    c = np.clip(a.astype(int) + rng.normal(0, 30, a.shape), 0, 255).astype(np.uint8)
    assert psnr(a, b) > psnr(a, c) > 0


def test_ssim_identical_is_one():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (48, 48), dtype=np.uint8)
    assert pytest.approx(1.0, abs=1e-4) == ssim(a, a)


def test_ssim_ordering():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    mild = np.clip(a.astype(int) + rng.normal(0, 3, a.shape), 0, 255).astype(np.uint8)
    heavy = np.clip(a.astype(int) + rng.normal(0, 40, a.shape), 0, 255).astype(np.uint8)
    assert ssim(a, mild) > ssim(a, heavy)


def test_ssim_small_images():
    a = np.zeros((5, 5), dtype=np.uint8)
    assert pytest.approx(1.0, abs=1e-4) == ssim(a, a)
