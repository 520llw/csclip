"""
BiomedCLIP feature extractor implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from biomedclip_zeroshot_cell_classify import InstanceInfo, ensure_local_biomedclip_dir, resolve_device
from biomedclip_fewshot_support_experiment import encode_multiscale_feature, normalize_feature

from .base import BaseFeatureExtractor, ExtractorConfig


class BiomedCLIPExtractor(BaseFeatureExtractor):
    """Feature extractor using BiomedCLIP (OpenCLIP local weights)."""

    def __init__(self, config: ExtractorConfig):
        super().__init__(config)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = resolve_device(config.device)
        self._weights_dir = config.local_weights_dir
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from open_clip import create_model_from_pretrained, get_tokenizer
        from labeling_tool.paths import biomedclip_local_dir

        weights = self._weights_dir or str(biomedclip_local_dir())
        local_dir = ensure_local_biomedclip_dir(weights)
        model_id = f"local-dir:{local_dir}"

        self._model, self._preprocess = create_model_from_pretrained(
            model_id, device=self._device
        )
        self._model.to(self._device).eval()
        self._tokenizer = get_tokenizer(model_id)
        self._loaded = True

    @property
    def feature_dim(self) -> int:
        return 512

    @property
    def name(self) -> str:
        return "BiomedCLIP"

    def encode_cell(self, image: np.ndarray, instance: InstanceInfo) -> np.ndarray:
        self._ensure_loaded()
        return encode_multiscale_feature(
            model=self._model,
            preprocess=self._preprocess,
            image=image,
            instance=instance,
            device=self._device,
            cell_margin_ratio=self.config.cell_margin_ratio,
            context_margin_ratio=self.config.context_margin_ratio,
            background_value=self.config.background_value,
            cell_scale_weight=self.config.cell_scale_weight,
            context_scale_weight=self.config.context_scale_weight,
        )

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def get_raw_model(self):
        """Access the underlying OpenCLIP model (for prompt tuning etc)."""
        self._ensure_loaded()
        return self._model, self._preprocess, self._tokenizer
