"""Small GPU helper utilities."""

import torch


def print_gpu_status():
    if not torch.cuda.is_available():
        print("CUDA not available — running on CPU (will be significantly slower).")
        return
    name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {name}  |  VRAM: {total_mem:.1f} GB")


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def current_vram_usage_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)
