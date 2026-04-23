"""
Bundled weight locations for the labeling tool.

Priority:
  1. labeling_tool/weights/  (portable copy or symlink next to this package)
  2. MedSAM project root: assets/sam3.pt, biomedclip_local/

Project root follows MEDSAM_ROOT when set, else parent of labeling_tool/.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def labeling_tool_root() -> Path:
    return Path(__file__).resolve().parent


def medsam_project_root() -> Path:
    env = os.environ.get("MEDSAM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return labeling_tool_root().parent


def biomedclip_local_dir() -> Path:
    bundled = labeling_tool_root() / "weights" / "biomedclip"
    if bundled.is_dir():
        return bundled
    return medsam_project_root() / "biomedclip_local"


def sam3_checkpoint_path() -> Optional[Path]:
    for candidate in (
        labeling_tool_root() / "weights" / "sam3.pt",
        medsam_project_root() / "assets" / "sam3.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def sam3_package_dir_for_sys_path() -> Path:
    """Directory that must be on sys.path so `import sam3` resolves (sam3/sam3)."""
    return medsam_project_root() / "sam3" / "sam3"
