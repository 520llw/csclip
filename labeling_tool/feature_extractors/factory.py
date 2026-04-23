"""
Factory for creating feature extractors from configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base import BaseFeatureExtractor, ExtractorConfig


_REGISTRY = {
    "biomedclip": "labeling_tool.feature_extractors.biomedclip_extractor.BiomedCLIPExtractor",
    "dinov2": "labeling_tool.feature_extractors.dinov2_extractor.DINOv2Extractor",
}


def list_available_extractors() -> List[str]:
    """Return names of all registered extractors."""
    return list(_REGISTRY.keys())


def create_extractor(
    name: str = "biomedclip",
    device: str = "auto",
    config_dict: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> BaseFeatureExtractor:
    """Create a feature extractor by name.

    Args:
        name: extractor name (e.g. "biomedclip", "dinov2")
        device: target device
        config_dict: optional config overrides
        config_path: optional path to YAML config file

    Returns:
        Initialized BaseFeatureExtractor instance
    """
    if config_path:
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        name = file_cfg.get("model_name", name)
        device = file_cfg.get("device", device)
        config_dict = {**(config_dict or {}), **file_cfg}

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown extractor '{name}'. Available: {list(_REGISTRY.keys())}"
        )

    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    cfg = ExtractorConfig(model_name=name, device=device)
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                cfg.extra[key] = value

    return cls(cfg)


def register_extractor(name: str, full_class_path: str):
    """Register a custom extractor class."""
    _REGISTRY[name] = full_class_path
