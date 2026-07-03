from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a project dependency
    torch = None  # type: ignore[assignment]


def now_perf() -> float:
    return time.perf_counter()


def elapsed_ms(start: float, end: float | None = None) -> float:
    return ((end if end is not None else now_perf()) - start) * 1000.0


def synchronize_device(device: Any) -> None:
    if torch is None:
        return
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def cuda_memory_snapshot(device: Any | None = None) -> dict[str, int | None]:
    empty_snapshot = {
        "gpu_memory_allocated_bytes": None,
        "gpu_memory_reserved_bytes": None,
        "gpu_max_memory_allocated_bytes": None,
    }
    if torch is None or not torch.cuda.is_available():
        return empty_snapshot
    resolved = device if device is not None else torch.device("cuda")
    device_type = getattr(resolved, "type", str(resolved).split(":", maxsplit=1)[0])
    if device_type != "cuda":
        return empty_snapshot
    return {
        "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(resolved)),
        "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(resolved)),
        "gpu_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
    }


def cuda_hardware_metadata() -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {
            "gpu_name": None,
            "cuda_version": getattr(torch.version, "cuda", None) if torch is not None else None,
            "cuda_device_count": 0,
            "cuda_driver": None,
        }

    driver = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            driver = completed.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, TimeoutError):
        driver = None

    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_driver": driver,
        "platform": platform.platform(),
    }


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def model_dump_compat(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"Cannot dump object of type {type(obj)!r}")
