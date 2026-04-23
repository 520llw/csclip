"""
biomedclip_zeroshot_cell_classify — Core data structures and device utilities
for BiomedCLIP-based cell classification in BALF samples.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class InstanceInfo:
    """Represents a single cell instance with its spatial information."""
    instance_id: int
    class_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords
    mask: np.ndarray                  # boolean mask (H, W)


def resolve_device(device_arg: str = "auto") -> str:
    """Resolve device string to an actual torch device identifier.

    'auto' -> 'cuda' if available, else 'cpu'.
    """
    if device_arg in ("cuda", "cpu"):
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_local_biomedclip_dir(local_dir) -> Path:
    """Verify that the local BiomedCLIP weight directory exists and contains
    the expected config files. Returns the resolved Path."""
    p = Path(local_dir) if not isinstance(local_dir, Path) else local_dir
    if not p.is_dir():
        raise FileNotFoundError(
            f"BiomedCLIP local directory not found: {p}. "
            "Please download the weights or set MEDSAM_ROOT correctly."
        )
    config = p / "open_clip_config.json"
    if not config.exists():
        for f in p.iterdir():
            if f.suffix == ".json":
                break
        else:
            raise FileNotFoundError(
                f"BiomedCLIP directory {p} contains no config files."
            )
    return p


__all__ = ["InstanceInfo", "ensure_local_biomedclip_dir", "resolve_device"]
