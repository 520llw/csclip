"""
Base class for all feature extractors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from biomedclip_zeroshot_cell_classify import InstanceInfo


@dataclass
class ExtractorConfig:
    """Configuration for a feature extractor."""
    model_name: str = "biomedclip"
    device: str = "auto"
    cell_margin_ratio: float = 0.15
    context_margin_ratio: float = 0.30
    background_value: int = 128
    cell_scale_weight: float = 0.90
    context_scale_weight: float = 0.10
    local_weights_dir: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseFeatureExtractor(ABC):
    """Abstract base for all feature extractors.

    Implementations must provide:
      - encode_cell(): extract a feature vector from a cell instance in an image
      - encode_text(): encode text description to a feature vector
      - feature_dim: the dimensionality of output features
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of the output feature vectors."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the extractor."""
        ...

    @abstractmethod
    def encode_cell(
        self,
        image: np.ndarray,
        instance: InstanceInfo,
    ) -> np.ndarray:
        """Encode a cell instance from an image into a feature vector.

        Args:
            image: full RGB image (H, W, 3)
            instance: cell instance with bbox and mask

        Returns:
            L2-normalised feature vector of shape (feature_dim,)
        """
        ...

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text description into a feature vector.

        Args:
            text: description string

        Returns:
            L2-normalised feature vector of shape (feature_dim,)
        """
        ...

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts. Default: loop over encode_text.

        Returns:
            (n_texts, feature_dim) array
        """
        return np.stack([self.encode_text(t) for t in texts], axis=0)

    def encode_cells(
        self,
        image: np.ndarray,
        instances: List[InstanceInfo],
    ) -> np.ndarray:
        """Encode multiple cells from the same image. Default: loop.

        Returns:
            (n_instances, feature_dim) array
        """
        return np.stack([self.encode_cell(image, inst) for inst in instances], axis=0)
