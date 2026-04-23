"""
DINOv2 feature extractor implementation.

DINOv2 (Meta) provides strong visual features without text alignment.
Useful when text prompts are ineffective (as shown in our baseline).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from biomedclip_zeroshot_cell_classify import InstanceInfo, resolve_device
from biomedclip_fewshot_support_experiment import normalize_feature

from .base import BaseFeatureExtractor, ExtractorConfig


class DINOv2Extractor(BaseFeatureExtractor):
    """Feature extractor using DINOv2 (vision-only, no text)."""

    MODELS = {
        "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14", 384),
        "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14", 768),
        "dinov2_vitl14": ("facebookresearch/dinov2", "dinov2_vitl14", 1024),
    }

    def __init__(self, config: ExtractorConfig):
        super().__init__(config)
        self._model = None
        self._transform = None
        self._device = resolve_device(config.device)
        variant = config.extra.get("variant", "dinov2_vits14")
        if variant not in self.MODELS:
            variant = "dinov2_vits14"
        self._variant = variant
        self._repo, self._model_name, self._dim = self.MODELS[variant]
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from torchvision import transforms

        self._model = torch.hub.load(self._repo, self._model_name)
        self._model.to(self._device).eval()

        self._transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._loaded = True

    @property
    def feature_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"DINOv2-{self._variant}"

    def _crop_cell(self, image: np.ndarray, instance: InstanceInfo) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = instance.bbox
        bw, bh = x2 - x1, y2 - y1
        margin = self.config.cell_margin_ratio
        mx, my = int(bw * margin), int(bh * margin)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        crop = image[cy1:cy2, cx1:cx2].copy()
        mask_crop = instance.mask[cy1:cy2, cx1:cx2]
        bg = np.full_like(crop, self.config.background_value)
        if crop.ndim == 3:
            crop = np.where(mask_crop[..., None], crop, bg)
        else:
            crop = np.where(mask_crop, crop, bg)
        return crop

    def encode_cell(self, image: np.ndarray, instance: InstanceInfo) -> np.ndarray:
        self._ensure_loaded()
        crop = self._crop_cell(image, instance)
        pil_img = Image.fromarray(crop)
        tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """DINOv2 has no text encoder. Returns zeros."""
        return np.zeros(self._dim, dtype=np.float32)
