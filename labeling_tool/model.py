import os
import sys
import math
import logging
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps

from labeling_tool.paths import sam3_package_dir_for_sys_path, medsam_project_root

logger = logging.getLogger(__name__)

# SAM3 repo root must be on sys.path BEFORE D:\VM_share to avoid
# the repo directory being picked up as a namespace package.
_sam3_repo_root = str(medsam_project_root() / "sam3")
if os.path.isfile(os.path.join(_sam3_repo_root, "sam3", "__init__.py")):
    if _sam3_repo_root in sys.path:
        sys.path.remove(_sam3_repo_root)
    sys.path.insert(0, _sam3_repo_root)

try:
    import sam3
    if sam3.__file__ is None:
        raise ImportError("sam3 loaded as namespace package, not real package")
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    logger.warning(f"SAM3 not found or failed to import: {e}")
    build_sam3_image_model = None
    Sam3Processor = None

class SAM3Model:
    def __init__(self, checkpoint_path, device='cuda', confidence_threshold=0.3):
        self.processor = None
        self.device = device
        if build_sam3_image_model is None:
            return
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        if not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA not available, using CPU.")
        self.device = device
        sam3_pkg_dir = os.path.dirname(sam3.__file__)
        sam3_root = os.path.join(sam3_pkg_dir, "..")
        bpe_candidates = [
            os.path.join(sam3_pkg_dir, "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
        ]
        bpe_path = next((p for p in bpe_candidates if os.path.exists(p)), bpe_candidates[0])
        try:
            self.model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=checkpoint_path,
                device=device,
                eval_mode=True,
                load_from_HF=False,
                enable_inst_interactivity=True,  # 启用 SAM2 交互预测器，支持十三点、纯框
            )
            self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold, device=device)
        except Exception as e:
            err_str = str(e).lower()
            if ("out of memory" in err_str or ("cuda" in err_str and "memory" in err_str)) and device == "cuda":
                logger.warning(f"GPU OOM, fallback to CPU: {e}")
                try:
                    torch.cuda.empty_cache()
                    self.device = "cpu"
                    self.model = build_sam3_image_model(
                        bpe_path=bpe_path,
                        checkpoint_path=checkpoint_path,
                        device="cpu",
                        eval_mode=True,
                        load_from_HF=False,
                        enable_inst_interactivity=True,
                    )
                    self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold, device="cpu")
                    logger.info("SAM3 loaded on CPU (GPU OOM fallback)")
                except Exception as e2:
                    logger.error(f"Failed to load SAM3 on CPU: {e2}")
            else:
                logger.error(f"Error loading SAM3 model: {e}")

    def _release_vram(self):
        if torch.cuda.is_available() and self.device != 'cpu':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def predict_text(self, image_path, text_prompt, confidence=0.23):
        """Text-only prompt — supports comma-separated multi-text.
        Returns list of polygons (possibly multiple objects)."""
        if self.processor is None:
            raise RuntimeError("SAM3 model not loaded.")

        prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
        if not prompts:
            return []

        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = image.size
        logger.info(f"predict_text: prompts={prompts} image={os.path.basename(image_path)} "
                     f"size={width}x{height} confidence={confidence}")
        state = self.processor.set_image(image)

        old_threshold = self.processor.confidence_threshold
        self.processor.confidence_threshold = confidence
        try:
            all_polygons = []
            for prompt in prompts:
                try:
                    self.processor.reset_all_prompts(state)
                except Exception:
                    pass
                state = self.processor.set_text_prompt(prompt=prompt, state=state)

                masks = state.get('masks', None)
                n_masks = 0
                if masks is not None:
                    try:
                        n_masks = len(masks)
                    except TypeError:
                        n_masks = 0

                scores = state.get('scores', None)
                score_list = []
                if scores is not None and isinstance(scores, torch.Tensor):
                    score_list = scores.detach().cpu().tolist()
                logger.info(f"  prompt='{prompt}': {n_masks} masks, scores={score_list}")

                if n_masks == 0:
                    continue

                for i, mask in enumerate(state['masks']):
                    if isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy()
                    mask = mask.squeeze()
                    if mask.shape[0] != height or mask.shape[1] != width:
                        mask = cv2.resize(mask.astype(np.float32), (width, height),
                                         interpolation=cv2.INTER_LINEAR)
                    bin_mask = (mask > 0.5).astype(np.uint8) if mask.dtype != bool else mask.astype(np.uint8)
                    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 100:
                            continue
                        epsilon = 0.001 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        if len(approx) < 3:
                            continue
                        polygon = []
                        for pt in approx:
                            x, y = pt[0]
                            polygon.append(x / width)
                            polygon.append(y / height)
                        all_polygons.append(polygon)
                    logger.info(f"    mask[{i}]: contours={len(contours)} total_polys={len(all_polygons)}")

            logger.info(f"  total polygons returned: {len(all_polygons)}")
            return all_polygons
        finally:
            self.processor.confidence_threshold = old_threshold
            try:
                self.processor.reset_all_prompts(state)
            except Exception:
                pass
            del state, image
            self._release_vram()

    @staticmethod
    def _generate_13_points(box_xyxy):
        """Generate 13 foreground points from a bounding box [x1,y1,x2,y2] in pixel coords.

        Layout:
          1 center
          4 middle ring (40% radius, axis-aligned)
          4 outer ring main (80% radius, axis-aligned)
          4 outer ring diagonal (80% radius, 45/135/225/315 deg)
        """
        x1, y1, x2, y2 = box_xyxy
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh / 2

        mid_rx = bw * 0.2
        mid_ry = bh * 0.2
        out_rx = bw * 0.4
        out_ry = bh * 0.4

        pts = [[cx, cy]]
        # middle ring
        pts += [[cx, cy - mid_ry], [cx, cy + mid_ry],
                [cx - mid_rx, cy], [cx + mid_rx, cy]]
        # outer ring main
        pts += [[cx, cy - out_ry], [cx, cy + out_ry],
                [cx - out_rx, cy], [cx + out_rx, cy]]
        # outer ring diagonals
        for angle in [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]:
            pts.append([cx + out_rx * math.cos(angle),
                        cy - out_ry * math.sin(angle)])

        return np.array(pts, dtype=np.float32)

    def _mask_to_polygon(self, mask, width, height):
        """Convert a binary mask to a normalized polygon (flat list of x,y)."""
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        polygon = []
        for pt in approx:
            x, y = pt[0]
            polygon.append(x / width)
            polygon.append(y / height)
        return polygon

    def predict_box_inst(self, image_path, box):
        """Pure box prompt via predict_inst (SAM2-compatible, no text, no points)."""
        if self.processor is None:
            raise RuntimeError("SAM3 model not loaded.")
        if getattr(self.model, 'inst_interactive_predictor', None) is None:
            raise RuntimeError("SAM3 inst_interactive_predictor 不可用，当前模型不支持此策略")
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = image.size
        state = self.processor.set_image(image)

        if "sam2_backbone_out" not in (state.get("backbone_out") or {}):
            raise RuntimeError("SAM3 backbone 未返回 sam2_backbone_out，无法使用 predict_inst")

        try:
            masks, scores, _ = self.model.predict_inst(
                state,
                box=np.array(box, dtype=np.float32),
                multimask_output=False,
                normalize_coords=True,
            )
            if len(masks) == 0:
                return []
            mask = masks[0].squeeze()
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            return self._mask_to_polygon(mask, width, height)
        finally:
            del state
            del image
            self._release_vram()

    def predict_13points(self, image_path, box):
        """Box → 13-point prompt via predict_inst (SAM2-compatible pathway).

        Args:
            box: [x1, y1, x2, y2] in pixel coords
        Returns:
            polygon as flat list of normalized [x, y, ...] pairs
        """
        if self.processor is None:
            raise RuntimeError("SAM3 model not loaded.")
        if getattr(self.model, 'inst_interactive_predictor', None) is None:
            raise RuntimeError("SAM3 inst_interactive_predictor 不可用，当前模型不支持此策略")
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = image.size
        state = self.processor.set_image(image)

        if "sam2_backbone_out" not in (state.get("backbone_out") or {}):
            raise RuntimeError("SAM3 backbone 未返回 sam2_backbone_out，无法使用点提示")

        try:
            points = self._generate_13_points(box)
            labels = np.ones(len(points), dtype=np.int32)
            logger.info(f"predict_13points: box={box} -> {len(points)} points, shape={points.shape}")

            masks, scores, _ = self.model.predict_inst(
                state,
                point_coords=points,
                point_labels=labels,
                box=np.array(box, dtype=np.float32),
                multimask_output=False,
                normalize_coords=True,
            )
            if len(masks) == 0:
                return []
            mask = masks[0].squeeze()
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            mask = self._postprocess_mask_13pts(mask)
            return self._mask_to_polygon(mask, width, height)
        finally:
            del state
            del image
            self._release_vram()

    @staticmethod
    def _postprocess_mask_13pts(mask):
        """Enhanced 13-point post-processing with cell morphology priors.

        Pipeline:
          1. Morphological close (5x5 ellipse, 1 iteration)
          2. Convex hull defect filling (fill concavities < 50% of mask area)
          3. Fragment removal (discard disconnected regions < 10% of main)
          4. Boundary smoothing (small Gaussian blur + re-threshold)
        """
        if mask is None or np.sum(mask) == 0:
            return mask
        mask_uint8 = (mask > 0.5).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)

        mask_uint8 = SAM3Model._apply_convex_hull_defect_filling(mask_uint8, defect_area_threshold=0.5)

        mask_uint8 = SAM3Model._remove_small_fragments(mask_uint8, min_ratio=0.1)

        blurred = cv2.GaussianBlur(mask_uint8.astype(np.float32), (3, 3), 0.5)
        mask_uint8 = (blurred > 0.4).astype(np.uint8)

        return mask_uint8.astype(np.float32)

    @staticmethod
    def _remove_small_fragments(mask, min_ratio=0.1):
        """Remove disconnected fragments smaller than min_ratio of the largest component."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 2:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = areas.max()
        threshold = max_area * min_ratio
        result = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= threshold:
                result[labels == i] = 1
        return result

    @staticmethod
    def _apply_convex_hull_defect_filling(mask, defect_area_threshold=0.5):
        """凸包缺陷填充：填充凸包内、原 mask 外且面积 < mask 面积 50% 的连通缺陷"""
        if np.sum(mask) == 0:
            return mask
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        main_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_cnt)
        hull_mask = np.zeros_like(mask)
        cv2.fillPoly(hull_mask, [hull], 1)
        defects = np.logical_and(hull_mask > 0, mask == 0).astype(np.uint8)
        mask_area = np.sum(mask)
        if mask_area == 0:
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defects, connectivity=8)
        fill_mask = np.zeros_like(defects)
        for i in range(1, num_labels):
            defect_area = stats[i, cv2.CC_STAT_AREA]
            if defect_area < mask_area * defect_area_threshold:
                fill_mask[labels == i] = 1
        return np.maximum(mask, fill_mask).astype(np.uint8)

    @staticmethod
    def _iou_xyxy(a, b):
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _select_best_mask(self, state, input_box_xyxy):
        """From grounding API results, select the mask best matching input_box."""
        masks = state.get('masks')
        if masks is None or len(masks) == 0:
            return None
        if len(masks) == 1:
            m = masks[0]
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            return m.squeeze()

        pred_boxes = state.get('boxes')
        best_idx, best_iou = 0, -1.0
        if pred_boxes is not None and len(pred_boxes) > 0:
            for i, pb in enumerate(pred_boxes):
                if isinstance(pb, torch.Tensor):
                    pb = pb.detach().cpu().numpy().flatten()
                iou = self._iou_xyxy(input_box_xyxy, pb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

        m = masks[best_idx]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        return m.squeeze()

    def _predict_grounding(self, state, box, width, height, text_prompt=None):
        """Run grounding API for a single box on an already-set image state.
        Returns normalized polygon or []."""
        self.processor.reset_all_prompts(state)
        if text_prompt:
            state = self.processor.set_text_prompt(state=state, prompt=text_prompt)
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        bw = max(1.0, x2 - x1) / width
        bh = max(1.0, y2 - y1) / height
        state = self.processor.add_geometric_prompt(state=state, box=[cx, cy, bw, bh], label=True)

        mask = self._select_best_mask(state, box)
        if mask is None:
            return []
        return self._mask_to_polygon(mask, width, height)

    def predict(self, image_path, box, text_prompt=None):
        if self.processor is None:
            raise RuntimeError("SAM3 model not loaded.")
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = image.size
        state = self.processor.set_image(image)
        try:
            return self._predict_grounding(state, box, width, height, text_prompt)
        finally:
            self.processor.reset_all_prompts(state)
            del state, image
            self._release_vram()

    def _predict_inst_single(self, state, box, points=None, labels=None, width=0, height=0):
        """Run predict_inst for a single box (optionally with points).
        Returns polygon (normalized flat list) or [].
        """
        if getattr(self.model, 'inst_interactive_predictor', None) is None:
            raise RuntimeError("inst_interactive_predictor 不可用")
        kwargs = dict(
            box=np.array(box, dtype=np.float32),
            multimask_output=False,
            normalize_coords=True,
        )
        if points is not None and labels is not None:
            kwargs["point_coords"] = points
            kwargs["point_labels"] = labels

        masks, scores, _ = self.model.predict_inst(state, **kwargs)
        if masks is None or len(masks) == 0:
            return []
        m = masks[0].squeeze()
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        mask_sum = int(m.sum()) if m.dtype == bool else int((m > 0.5).sum())
        if mask_sum == 0:
            return []
        if points is not None:
            m = self._postprocess_mask_13pts(m)
        return self._mask_to_polygon(m, width, height)

    def _run_strategy(self, strategy, state, box, width, height, text=None):
        """Run one prediction strategy on one box, return polygon or []."""
        if strategy == "grounding":
            return self._predict_grounding(state, box, width, height, text)
        elif strategy == "box_inst":
            return self._predict_inst_single(state, box, width=width, height=height)
        elif strategy == "13points":
            pts = self._generate_13_points(box)
            lbl = np.ones(len(pts), dtype=np.int32)
            logger.info(f"_run_strategy 13points: {len(pts)} points -> predict_inst")
            return self._predict_inst_single(state, box, points=pts, labels=lbl,
                                             width=width, height=height)
        else:
            return self._predict_grounding(state, box, width, height, text)

    def predict_batch(self, image_path, boxes_with_class, prompt_types=None,
                      prompt_type="box", text_prompt=None, names=None):
        """Batch predict: set image once, iterate boxes with one or more strategies.

        Args:
            prompt_types: list of strategies to try, e.g. ["grounding", "13points"].
                         If provided, each box is tested with all strategies and the
                         best result (by mask coverage relative to bbox area) is kept.
            prompt_type:  legacy single prompt_type (used when prompt_types is None).
        Returns: [{class_id, points, ok, strategy, errors}]
        """
        if self.processor is None:
            raise RuntimeError("SAM3 model not loaded.")
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = image.size
        state = self.processor.set_image(image)
        names = names or {}

        has_inst = ("sam2_backbone_out" in (state.get("backbone_out") or {})
                    and getattr(self.model, 'inst_interactive_predictor', None) is not None)
        logger.info(f"predict_batch: image={os.path.basename(image_path)} "
                     f"size={width}x{height} has_inst={has_inst} "
                     f"boxes={len(boxes_with_class)}")

        strategies = self._resolve_strategies(prompt_types, prompt_type, has_inst)
        logger.info(f"  strategies={strategies}")

        results = []
        try:
            for idx, item in enumerate(boxes_with_class):
                cid = item["class_id"]
                box = item["box"]
                text = text_prompt or names.get(cid, "cell")
                bbox_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))

                best_poly = []
                best_strategy = None
                best_score = -1
                errors = []

                for strat in strategies:
                    try:
                        poly = self._run_strategy(strat, state, box, width, height, text)
                        if poly and len(poly) >= 6:
                            score = self._poly_bbox_coverage(poly, box, width, height, bbox_area)
                            logger.info(f"  box[{idx}] strat={strat}: poly_pts={len(poly)//2} score={score:.3f}")
                            if score > best_score:
                                best_score = score
                                best_poly = poly
                                best_strategy = strat
                        else:
                            logger.info(f"  box[{idx}] strat={strat}: empty polygon")
                    except Exception as e:
                        logger.warning(f"  box[{idx}] strat={strat}: error={e}")
                        errors.append(f"{strat}: {e}")

                result = {
                    "class_id": cid,
                    "points": best_poly,
                    "ok": bool(best_poly),
                    "strategy": best_strategy,
                }
                if errors:
                    result["errors"] = errors
                results.append(result)
        finally:
            self.processor.reset_all_prompts(state)
            del state, image
            self._release_vram()
        return results

    def _resolve_strategies(self, prompt_types, prompt_type, has_inst):
        """Convert user-facing prompt type(s) to internal strategy list."""
        if prompt_types:
            strategies = []
            mapping = {
                "box": "grounding",
                "box+text": "grounding",
                "box_inst": "box_inst",
                "13points": "13points",
                "grounding": "grounding",
            }
            for pt in prompt_types:
                s = mapping.get(pt, pt)
                if s in ("box_inst", "13points") and not has_inst:
                    logger.warning(f"  strategy {s} skipped: no inst_interactive_predictor")
                    continue
                if s not in strategies:
                    strategies.append(s)
            return strategies or ["grounding"]

        mapping = {
            "box": ["grounding"],
            "box+text": ["grounding"],
            "box_inst": ["box_inst"] if has_inst else ["grounding"],
            "13points": ["13points"] if has_inst else ["grounding"],
            "multi": (["grounding"] +
                      (["13points"] if has_inst else []) +
                      (["box_inst"] if has_inst else [])),
        }
        return mapping.get(prompt_type, ["grounding"])

    @staticmethod
    def _poly_bbox_coverage(poly, box, width, height, bbox_area):
        """Score a polygon by how well it covers the input bbox (0~1).
        Uses intersection area of polygon bbox with input bbox.
        """
        xs = [poly[i] * width for i in range(0, len(poly), 2)]
        ys = [poly[i] * height for i in range(1, len(poly), 2)]
        px1, py1, px2, py2 = min(xs), min(ys), max(xs), max(ys)
        bx1, by1, bx2, by2 = box
        ix1 = max(px1, bx1); iy1 = max(py1, by1)
        ix2 = min(px2, bx2); iy2 = min(py2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        poly_area = (px2 - px1) * (py2 - py1)
        union = bbox_area + poly_area - inter
        return inter / union if union > 0 else 0
