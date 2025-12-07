from __future__ import annotations

import platform
from datetime import datetime
from typing import Any, Dict, List

# optional deps
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

try:
    import torch
except Exception:
    torch = None


def _bytes_to_gib(x: int) -> float:
    return float(x) / (1024.0 ** 3)


def _cuda_devices() -> List[int]:
    if (torch is None) or (not hasattr(torch, "cuda")) or (not torch.cuda.is_available()):
        return []
    try:
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _reset_and_mark_cuda_peaks() -> Dict[str, Any]:
    """
    评测前调用：重置峰值统计并记录设备名。
    """
    ctx: Dict[str, Any] = {"cuda_available": bool(_cuda_devices()), "devices": []}
    for idx in _cuda_devices():
        try:
            torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = f"cuda:{idx}"
        ctx["devices"].append({"index": idx, "name": name})
    return ctx


def _collect_cuda_peaks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    评测后调用：读取峰值显存（allocated/reserved）。
    """
    out: Dict[str, Any] = {
        "cuda_available": bool(ctx.get("cuda_available", False)),
        "per_device": [],
        "total_peak_allocated_gib": 0.0,
        "total_peak_reserved_gib": 0.0,
    }
    if not out["cuda_available"] or (torch is None):
        return out

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    total_alloc = 0
    total_res = 0
    for d in ctx.get("devices", []):
        idx = int(d["index"])
        name = d.get("name", f"cuda:{idx}")
        try:
            peak_alloc = int(torch.cuda.max_memory_allocated(idx))
        except Exception:
            peak_alloc = 0
        try:
            peak_res = int(torch.cuda.max_memory_reserved(idx))
        except Exception:
            peak_res = 0

        out["per_device"].append({
            "device": f"cuda:{idx}",
            "name": name,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_res,
            "peak_allocated_gib": _bytes_to_gib(peak_alloc),
            "peak_reserved_gib": _bytes_to_gib(peak_res),
        })
        total_alloc += peak_alloc
        total_res += peak_res

    out["total_peak_allocated_gib"] = _bytes_to_gib(total_alloc)
    out["total_peak_reserved_gib"] = _bytes_to_gib(total_res)
    return out


def _env_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import numpy as _np
        info["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import transformers as _tf
        info["transformers"] = _tf.__version__
    except Exception:
        pass
    try:
        import torch as _t
        info["torch"] = _t.__version__
        info["cuda_available"] = bool(_t.cuda.is_available())
        if _t.cuda.is_available():
            info["cuda_device_count"] = _t.cuda.device_count()
            info["cuda_devices"] = [_t.cuda.get_device_name(i) for i in range(_t.cuda.device_count())]
    except Exception:
        info["torch"] = None
    return info


def _proc_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    if not _HAS_PSUTIL:
        return snap
    p = psutil.Process()
    with p.oneshot():
        mem = p.memory_info()
        snap["rss_bytes"] = int(getattr(mem, "rss", 0))
        snap["vms_bytes"] = int(getattr(mem, "vms", 0))
        try:
            snap["cpu_percent"] = psutil.cpu_percent(interval=0.05)
        except Exception:
            snap["cpu_percent"] = None
    return snap
