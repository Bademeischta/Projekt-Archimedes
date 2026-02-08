"""Hardware monitoring for thermal throttling and metrics. Optional pynvml/psutil."""

import time
from typing import Any, Dict, Optional

_pynvml = None
_psutil = None

def _load_pynvml():
    global _pynvml
    if _pynvml is not None:
        return _pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
        return _pynvml
    except Exception:
        return None

def _load_psutil():
    global _psutil
    if _psutil is not None:
        return _psutil
    try:
        import psutil
        _psutil = psutil
        return _psutil
    except Exception:
        return None


def get_gpu_temp_c() -> Optional[float]:
    """Return GPU temperature in °C for device 0, or None if unavailable."""
    pynvml = _load_pynvml()
    if pynvml is None:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            return float(t)
        except Exception:
            return None
    except Exception:
        return None


def get_hardware_snapshot() -> Dict[str, Any]:
    """Return dict with gpu_utilization_pct, vram_mb, gpu_temp_c, cpu_load_pct, ram_mb, etc."""
    out = {
        "gpu_utilization_pct": None,
        "vram_mb": None,
        "gpu_temp_c": None,
        "cpu_load_pct": None,
        "ram_mb": None,
        "disk_io_read": None,
        "disk_io_write": None,
        "positions_per_watt": None,
        "timestamp": time.time(),
    }
    pynvml = _load_pynvml()
    if pynvml is not None:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                out["gpu_temp_c"] = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                out["gpu_utilization_pct"] = float(util.gpu)
            except Exception:
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                out["vram_mb"] = mem.used / (1024 * 1024)
            except Exception:
                pass
        except Exception:
            pass
    psutil = _load_psutil()
    if psutil is not None:
        try:
            out["cpu_load_pct"] = psutil.cpu_percent(interval=0.1)
        except Exception:
            pass
        try:
            v = psutil.virtual_memory()
            out["ram_mb"] = v.used / (1024 * 1024)
        except Exception:
            pass
        try:
            io = psutil.disk_io_counters()
            if io:
                out["disk_io_read"] = io.read_bytes
                out["disk_io_write"] = io.write_bytes
        except Exception:
            pass
    return out


def check_thermal_throttle(max_gpu_temp: float) -> bool:
    """
    If GPU temp >= max_gpu_temp, return True (caller should pause).
    Returns False otherwise.
    """
    t = get_gpu_temp_c()
    if t is None:
        return False
    return t >= max_gpu_temp
