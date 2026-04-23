"""
biomedclip_fewshot_support_experiment — Multi-scale feature encoding for
BiomedCLIP-based few-shot cell classification.

Extracts cell-level and context-level crops from an image using the instance
bounding box and mask, encodes each with BiomedCLIP, and returns a weighted
fusion of the two feature vectors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from biomedclip_zeroshot_cell_classify import InstanceInfo


def normalize_feature(feat: np.ndarray) -> np.ndarray:
    """L2-normalize a feature vector (or batch)."""
    feat = np.asarray(feat, dtype=np.float32)
    norm = np.linalg.norm(feat)
    if norm < 1e-12:
        return feat
    return feat / norm


def _crop_cell(
    image: np.ndarray,
    instance: InstanceInfo,
    margin_ratio: float,
    background_value: int,
) -> np.ndarray:
    """Crop cell region from image with margin, masking background outside the
    cell mask to a constant value."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = instance.bbox
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    crop = image[cy1:cy2, cx1:cx2].copy()

    mask_crop = instance.mask[cy1:cy2, cx1:cx2]
    bg = np.full_like(crop, background_value)
    crop = np.where(mask_crop[..., None] if crop.ndim == 3 else mask_crop, crop, bg)
    return crop


def _crop_context(
    image: np.ndarray,
    instance: InstanceInfo,
    margin_ratio: float,
) -> np.ndarray:
    """Crop a larger context region around the cell (no masking)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = instance.bbox
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    return image[cy1:cy2, cx1:cx2].copy()


def _encode_single_crop(
    model: Any,
    preprocess: Any,
    crop: np.ndarray,
    device: str,
) -> np.ndarray:
    """Encode a single crop through BiomedCLIP and return L2-normalised feature."""
    pil_img = Image.fromarray(crop)
    tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def encode_multiscale_feature(
    *,
    model: Any,
    preprocess: Any,
    image: np.ndarray,
    instance: InstanceInfo,
    device: str,
    cell_margin_ratio: float = 0.15,
    context_margin_ratio: float = 0.30,
    background_value: int = 128,
    cell_scale_weight: float = 0.90,
    context_scale_weight: float = 0.10,
) -> np.ndarray:
    """Encode a cell instance at two scales and return the weighted fusion.

    Scale 1 (cell): tight crop with margin, background masked.
    Scale 2 (context): wider crop without masking.
    """
    cell_crop = _crop_cell(image, instance, cell_margin_ratio, background_value)
    context_crop = _crop_context(image, instance, context_margin_ratio)

    cell_feat = _encode_single_crop(model, preprocess, cell_crop, device)
    context_feat = _encode_single_crop(model, preprocess, context_crop, device)

    fused = cell_scale_weight * cell_feat + context_scale_weight * context_feat
    return normalize_feature(fused)


__all__ = ["encode_multiscale_feature", "normalize_feature"]
