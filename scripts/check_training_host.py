#!/usr/bin/env python3

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path

import torch
from booster_assets import BOOSTER_ASSETS_DIR


def version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def check_distributed_cuda() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"CUDA GPUs:   {torch.cuda.device_count()}")
    print(f"NCCL:        {torch.cuda.nccl.version()}")
    if world_size == 1:
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    probe = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    torch.distributed.all_reduce(probe)
    expected = world_size * (world_size + 1) / 2
    if probe.item() != expected:
        raise RuntimeError(f"NCCL all-reduce returned {probe.item()}, expected {expected}")
    print(f"NCCL rank:   {rank}/{world_size} on cuda:{local_rank}, all-reduce={probe.item():g}")
    torch.distributed.destroy_process_group()


def main() -> None:
    errors: list[str] = []
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 is required, found {platform.python_version()}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        errors.append(f"Isaac Sim training requires Linux x86_64, found {platform.system()} {platform.machine()}")
    if not torch.cuda.is_available():
        errors.append("PyTorch cannot see a CUDA GPU")

    print(f"Python:     {platform.python_version()}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")
    print(f"Isaac Lab:  {version('isaaclab')}")
    print(f"Isaac Sim:  {version('isaacsim')}")
    print(f"Booster:    {version('booster-train')}")

    k1_urdf = Path(BOOSTER_ASSETS_DIR) / "robots" / "K1" / "K1_22dof.urdf"
    print(f"K1 URDF:    {k1_urdf}")
    if not k1_urdf.is_file():
        errors.append(f"K1 URDF does not exist: {k1_urdf}")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        memory_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        print(f"GPU:         {name}")
        print(f"Capability:  sm_{capability[0]}{capability[1]}")
        print(f"VRAM:        {memory_gib:.1f} GiB")
        if capability < (12, 0):
            print("Note: this is not a Blackwell GPU; training still works if Isaac Sim supports it.")
        check_distributed_cuda()

    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))


if __name__ == "__main__":
    main()
