"""测量工具：计时、显存、CPU/RAM、PSNR/SSIM。"""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager

import numpy as np


class Timer:
    def __init__(self) -> None:
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


@contextmanager
def cuda_synced():
    """确保计时包含 GPU 实际完成时间。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class VramTracker:
    def __init__(self) -> None:
        self.available = False
        try:
            import torch

            self.available = torch.cuda.is_available()
        except Exception:
            self.available = False

    def reset(self) -> None:
        if self.available:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def peak_mb(self) -> float | None:
        if not self.available:
            return None
        import torch

        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def allocated_mb(self) -> float | None:
        if not self.available:
            return None
        import torch

        return torch.cuda.memory_allocated() / (1024 * 1024)


class ResourceSampler:
    """后台线程采样进程 CPU% 与 RSS。"""

    def __init__(self, interval_s: float = 0.02):
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb = 0.0
        self.peak_cpu_pct = 0.0
        self._proc = None
        self._base_cpu = 0.0

    def start(self) -> None:
        import psutil

        self._proc = psutil.Process(os.getpid())
        self._proc.cpu_percent(None)
        self.peak_rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        self._base_cpu = sum(self._proc.cpu_times()[:2])
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                rss = self._proc.memory_info().rss / (1024 * 1024)
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
                cpu = self._proc.cpu_percent(None)
                self.peak_cpu_pct = max(self.peak_cpu_pct, cpu)
            except Exception:
                return

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        wall = max(1e-6, time.perf_counter() - self._t0)
        if self._proc is not None:
            cpu_now = sum(self._proc.cpu_times()[:2])
            cores = (cpu_now - self._base_cpu) / wall
            self.peak_rss_mb = max(self.peak_rss_mb, self._proc.memory_info().rss / (1024 * 1024))
        else:
            cores = 0.0
        return {
            "wall_s": wall,
            "avg_cpu_cores": float(cores),
            "peak_cpu_pct": float(self.peak_cpu_pct),
            "peak_rss_mb": float(self.peak_rss_mb),
        }


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"PSNR 形状不一致: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """11x11 高斯窗标准 SSIM（单通道或多通道分别平均）。"""
    import torch

    if a.shape != b.shape:
        raise ValueError(f"SSIM 形状不一致: {a.shape} vs {b.shape}")
    x = torch.from_numpy(a.astype(np.float32))
    y = torch.from_numpy(b.astype(np.float32))
    if x.ndim == 2:
        x, y = x[None, None], y[None, None]
    elif x.ndim == 3:
        x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    else:
        raise ValueError("仅支持 HW 或 HWC")
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    win = min(11, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    sigma = 1.5
    coords = torch.arange(win, dtype=torch.float32) - (win - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    kernel = (g.t() @ g).expand(c, 1, win, win).contiguous()
    pad = win // 2

    def filt(t):
        return torch.nn.functional.conv2d(t, kernel, padding=pad, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x2 = filt(x * x) - mu_x2
    sigma_y2 = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return float(s.mean().item())


def area_bytes(paths) -> int:
    total = 0
    for p in paths:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total


def percentile(values, q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
