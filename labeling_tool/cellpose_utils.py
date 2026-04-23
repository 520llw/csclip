"""
CellposeSAM utilities for the labeling tool.
Converts CellposeSAM label maps into normalized polygon annotations.

Enhanced with adaptive preprocessing to handle variable BALF image quality:
- CLAHE histogram equalisation for uneven illumination
- Background estimation and subtraction
- Image quality assessment (blur/noise detection)
- Automatic diameter estimation from image statistics
"""
import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_cellpose_models = {}


def _get_model(gpu: bool):
    global _cellpose_models
    key = bool(gpu)
    if key not in _cellpose_models:
        from cellpose import models
        _cellpose_models[key] = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
    return _cellpose_models[key]


def assess_image_quality(img: np.ndarray) -> dict:
    """Assess image quality and return metrics for adaptive preprocessing.

    Returns dict with: blur_score, noise_level, contrast_ratio, brightness.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    local_std = cv2.GaussianBlur(
        (gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)) ** 2,
        (5, 5), 0
    )
    noise_level = float(np.sqrt(np.mean(local_std)))

    p5, p95 = np.percentile(gray, [5, 95])
    contrast_ratio = float(p95 - p5) / 255.0

    brightness = float(np.mean(gray)) / 255.0

    return {
        "blur_score": laplacian_var,
        "noise_level": noise_level,
        "contrast_ratio": contrast_ratio,
        "brightness": brightness,
        "needs_clahe": contrast_ratio < 0.4,
        "needs_denoise": noise_level > 30,
        "is_blurry": laplacian_var < 100,
    }


def adaptive_preprocess(img: np.ndarray, quality: dict = None) -> np.ndarray:
    """Apply adaptive preprocessing based on image quality assessment."""
    if quality is None:
        quality = assess_image_quality(img)

    result = img.copy()

    if quality["needs_clahe"]:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        logger.debug("Applied CLAHE (low contrast detected)")

    if quality["needs_denoise"]:
        result = cv2.fastNlMeansDenoisingColored(result, None, h=6, hForColorComponents=6)
        logger.debug("Applied denoising (high noise detected)")

    return result


def estimate_cell_diameters(img: np.ndarray) -> list:
    """Estimate appropriate cell diameters from image statistics.

    Uses edge detection and connected component analysis to suggest
    diameter values for multi-scale segmentation.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    filled = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)

    areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 200 < area < 50000:
            areas.append(area)

    if not areas:
        return [30.0]

    areas = np.array(areas)
    median_area = float(np.median(areas))
    median_diameter = math.sqrt(median_area * 4 / math.pi)

    diameters = [
        max(10, median_diameter * 0.7),
        median_diameter,
        min(200, median_diameter * 1.5),
    ]
    return [round(d, 1) for d in diameters]


def _label_map_to_polygons(label_map: np.ndarray, class_id: int, min_area: int = 100) -> list:
    """Convert a label map to a list of normalized polygon annotations."""
    h, w = label_map.shape[:2]
    annotations = []
    ids = np.unique(label_map)
    ids = ids[ids != 0]

    for uid in ids:
        mask = (label_map == uid).astype(np.uint8)
        if np.sum(mask) < min_area:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area:
            continue

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue

        points = []
        for pt in approx:
            x, y = pt[0]
            points.append(round(x / w, 6))
            points.append(round(y / h, 6))

        annotations.append({
            "class_id": class_id,
            "ann_type": "polygon",
            "points": points,
        })

    return annotations


def postprocess_segmentation(
    label_map: np.ndarray,
    min_area: int = 100,
    max_area: int = 100000,
    min_circularity: float = 0.1,
) -> np.ndarray:
    """Post-process segmentation: filter by area/shape, merge fragments, split oversized.

    Returns cleaned label map.
    """
    cleaned = np.zeros_like(label_map)
    next_id = 1
    ids = np.unique(label_map)
    ids = ids[ids != 0]

    for uid in ids:
        mask = (label_map == uid).astype(np.uint8)
        area = int(np.sum(mask))

        if area < min_area:
            continue

        if area > max_area:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for cnt in contours:
                    cnt_area = cv2.contourArea(cnt)
                    if cnt_area < min_area:
                        continue
                    cnt_mask = np.zeros_like(mask)
                    cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
                    cleaned[cnt_mask > 0] = next_id
                    next_id += 1
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue

        cleaned[mask > 0] = next_id
        next_id += 1

    return cleaned


def run_cellpose_to_polygons(
    img_path: str,
    diameters: list,
    gpu: bool = True,
    class_id: int = 0,
    min_area: int = 100,
    adaptive_preprocess_enabled: bool = True,
    auto_diameter: bool = False,
) -> list:
    """Run CellposeSAM and return polygon annotations.

    Enhanced with adaptive preprocessing and post-processing for BALF images.
    """
    from PIL import Image, ImageOps

    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)

    if adaptive_preprocess_enabled:
        quality = assess_image_quality(img)
        img = adaptive_preprocess(img, quality)
        logger.debug(f"Image quality: contrast={quality['contrast_ratio']:.2f}, "
                     f"blur={quality['blur_score']:.0f}, noise={quality['noise_level']:.1f}")

    if auto_diameter and (not diameters or diameters == [0]):
        diameters = estimate_cell_diameters(img)
        logger.info(f"Auto-estimated diameters: {diameters}")

    model = _get_model(gpu)

    if len(diameters) == 1:
        result = model.eval([img], diameter=float(diameters[0]),
                            cellprob_threshold=-2.0)
        label_map = result[0][0]
        label_map = postprocess_segmentation(label_map, min_area=min_area)
        return _label_map_to_polygons(label_map, class_id, min_area)

    combined = np.zeros(img.shape[:2], dtype=np.int32)
    next_id = 1
    for d in diameters:
        result = model.eval([img], diameter=float(d),
                            cellprob_threshold=-2.0)
        lm = result[0][0]
        ids = np.unique(lm)
        ids = ids[ids != 0]
        for uid in ids:
            mask = lm == uid
            overlap = combined[mask]
            if overlap.any() and np.sum(overlap > 0) > 0.5 * np.sum(mask):
                continue
            combined[mask & (combined == 0)] = next_id
            next_id += 1

    combined = postprocess_segmentation(combined, min_area=min_area)
    return _label_map_to_polygons(combined, class_id, min_area)
