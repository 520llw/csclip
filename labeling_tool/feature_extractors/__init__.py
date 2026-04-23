"""
Pluggable Feature Extractor Architecture.

Provides a unified interface for swapping between different vision-language
models (BiomedCLIP, DINOv2, Vision APIs) without changing the classification
pipeline code.

Usage:
    from labeling_tool.feature_extractors import create_extractor

    extractor = create_extractor("biomedclip", device="cuda")
    feature = extractor.encode_cell(image, instance)
    text_proto = extractor.encode_text("a lymphocyte cell")
"""
from .base import BaseFeatureExtractor, ExtractorConfig
from .factory import create_extractor, list_available_extractors

__all__ = [
    "BaseFeatureExtractor",
    "ExtractorConfig",
    "create_extractor",
    "list_available_extractors",
]
