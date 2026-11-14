"""Phase 0 环境审计：硬件、驱动、CUDA、Python、依赖、磁盘、GPU 自检。

输出：
  artifacts/runs/<ts>_AUDIT/environment.json
  artifacts/runs/<ts>_AUDIT/gpu_selfcheck.json
  docs/02_audit/environment-audit.json
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def run(cmd: list[str], timeout: int = 60) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return {"cmd": " ".join(cmd), "code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": " ".join(cmd), "error": f"{type(exc).__name__}: {exc}"}


def redact(obj):
    """把当前用户主目录替换为 %USERPROFILE%，避免公开仓库泄露账户路径。"""
    home = str(Path.home())
    variants = sorted({home, home.replace("\\", "/"), home.replace("/", "\\")}, key=len, reverse=True)
    if isinstance(obj, str):
        for v in variants:
            obj = obj.replace(v, "%USERPROFILE%")
        return obj
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    return obj


def cpu_name() -> str | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return None


def os_info() -> dict:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as k:
            def q(name):
                try:
                    return winreg.QueryValueEx(k, name)[0]
                except OSError:
                    return None

            info.update(
                {
                    "product_name": q("ProductName"),
                    "display_version": q("DisplayVersion"),
                    "build": q("CurrentBuild"),
                    "ubr": q("UBR"),
                    "edition": q("EditionID"),
                }
            )
    except Exception:
        pass
    return info


def hardware() -> dict:
    import psutil

    out = {
        "cpu_name": cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": getattr(psutil.cpu_freq(), "max", None),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "disks": [
            {
                "mount": p.mountpoint,
                "total_gb": round(psutil.disk_usage(p.mountpoint).total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage(p.mountpoint).free / (1024**3), 2),
            }
            for p in psutil.disk_partitions(all=False)
            if p.mountpoint
        ],
    }
    return out


def gpu_info() -> dict:
    q = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap,pcie.link.gen.max,pcie.link.width.max,power.limit",
            "--format=csv,noheader",
        ]
    )
    out = {"nvidia_smi_query": q}
    if q.get("code") == 0 and q.get("stdout"):
        cols = [c.strip() for c in q["stdout"].splitlines()[0].split(",")]
        out["parsed"] = dict(zip(["name", "driver", "memory_total", "memory_used", "compute_cap", "pcie_gen", "pcie_width", "power_limit"], cols))
    out["full"] = run(["nvidia-smi"])
    return out


def python_info() -> dict:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    try:
        from importlib.metadata import distributions

        pkgs = {}
        for d in distributions():
            name = (d.metadata["Name"] or "").lower()
            if name:
                pkgs[name] = d.version
    except Exception:
        pkgs = {}
    nvcc = run(["nvcc", "--version"])
    return {
        "executable": sys.executable,
        "version": sys.version,
        "venv_active": in_venv,
        "sys_prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", None),
        "packages": pkgs,
        "nvcc": nvcc.get("stdout", nvcc.get("error", "")),
    }


def gpu_selfcheck() -> dict:
    result: dict = {"timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        import torch
    except Exception as exc:
        result["error"] = f"torch 导入失败: {exc}"
        return result

    result["torch"] = torch.__version__
    result["torch_cuda"] = torch.version.cuda
    result["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    result["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        result["verdict"] = "GPU 不可用"
        return result

    props = torch.cuda.get_device_properties(0)
    result["device"] = {
        "name": props.name,
        "total_memory_mb": round(props.total_memory / (1024**2), 1),
        "capability": [props.major, props.minor],
        "multi_processor_count": props.multi_processor_count,
        "arch_list": list(torch.cuda.get_arch_list()),
    }

    torch.manual_seed(0)
    a32 = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    b32 = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    ref = a32 @ b32
    a16, b16 = a32.half(), b32.half()
    got = (a16 @ b16).float()
    diff = (got - ref).abs()
    result["fp16_matmul"] = {
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "relative_to_fp32": float((diff / ref.abs().clamp_min(1e-3)).mean().item()),
        "nan": bool(torch.isnan(got).any().item()),
        "inf": bool(torch.isinf(got).any().item()),
    }

    def bench(dtype, n=50):
        x = torch.randn(2048, 2048, device="cuda", dtype=dtype)
        y = torch.randn(2048, 2048, device="cuda", dtype=dtype)
        for _ in range(5):
            x @ y
        torch.cuda.synchronize()
        t0 = __import__("time").perf_counter()
        for _ in range(n):
            x @ y
        torch.cuda.synchronize()
        ms = (__import__("time").perf_counter() - t0) * 1000 / n
        tflops = (2 * 2048**3) / (ms / 1000) / 1e12
        return {"ms_per_matmul_2048": round(ms, 3), "tflops": round(tflops, 2)}

    result["fp16_bench"] = bench(torch.float16)
    result["fp32_bench"] = bench(torch.float32)
    torch.cuda.reset_peak_memory_stats()
    result["peak_mem_during_bench_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)

    model_dir = ROOT / "artifacts" / "models" / "dinov2_vits14"
    if model_dir.exists():
        try:
            from transformers import Dinov2Model

            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            model = Dinov2Model.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.float16)
            model.eval().to("cuda")
            x = torch.randn(1, 3, 518, 518, device="cuda", dtype=torch.float16)
            with torch.inference_mode():
                for _ in range(3):
                    model(pixel_values=x)
                torch.cuda.synchronize()
                t0 = __import__("time").perf_counter()
                n = 20
                for _ in range(n):
                    out = model(pixel_values=x)
                torch.cuda.synchronize()
            ms = (__import__("time").perf_counter() - t0) * 1000 / n
            result["dinov2_fp16"] = {
                "forward_ms": round(ms, 2),
                "hidden_shape": list(out.last_hidden_state.shape),
                "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "weights_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2), 1),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                "activation_peak_mb": round((torch.cuda.max_memory_allocated() - base) / (1024**2), 1),
            }
        except Exception as exc:
            result["dinov2_fp16"] = {"error": f"{type(exc).__name__}: {exc}"}

    result["verdict"] = "通过" if (result["fp16_matmul"]["nan"] is False) else "失败"
    return result


def main() -> int:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "artifacts" / "runs" / f"{ts}_AUDIT"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "os": os_info(),
        "hardware": hardware(),
        "gpu": gpu_info(),
        "python": python_info(),
        "cwd": str(Path.cwd()),
        "project_root": str(ROOT),
        "git": run(["git", "rev-parse", "HEAD"]),
        "git_status": run(["git", "status", "--porcelain"]),
    }
    env = redact(env)
    (out_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    check = gpu_selfcheck()
    (out_dir / "gpu_selfcheck.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_dir = ROOT / "docs" / "02_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    combined = {"environment": env, "gpu_selfcheck": check}
    (audit_dir / "environment-audit.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "verdict": check.get("verdict"), "device": check.get("device")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
