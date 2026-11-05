from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _cuda_version(encoded: int) -> str:
    return f"{encoded // 1000}.{(encoded % 1000) // 10}"


def _safe_command() -> list[str]:
    return [Path(argument).name if Path(argument).is_absolute() else argument for argument in sys.argv]


def _gpu_info() -> dict[str, Any] | None:
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "name": _decode(properties["name"]),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "memory_bytes": int(properties["totalGlobalMem"]),
            "driver_version": _cuda_version(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": _cuda_version(cp.cuda.runtime.runtimeGetVersion()),
        }
    except Exception as exc:  # GPU details are best-effort metadata.
        return {"unavailable": type(exc).__name__}


def _git_revision() -> str | None:
    project_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def runtime_provenance() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": _distribution_version("numpy"),
            "mach_beamform": _distribution_version("mach-beamform"),
            "cupy_cuda12x": _distribution_version("cupy-cuda12x"),
            "pymust": _distribution_version("pymust"),
        },
        "gpu": _gpu_info(),
        "git_revision": _git_revision(),
        "command": _safe_command(),
    }
