import os
import re
import glob
import cv2
import numpy as np
import yaml
import shutil
import tempfile
import logging
import traceback
import threading
import time
import json as _json
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, Response, JSONResponse

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import sys

sys.path.append(os.getcwd())

from labeling_tool.fewshot_biomedclip import (
    evaluate_dataset as fewshot_evaluate_dataset,
    predict_annotations as fewshot_predict_annotations,
    prepare_classifier as fewshot_prepare_classifier,
)
from labeling_tool.hybrid_classifier import (
    evaluate_dataset as hybrid_evaluate_dataset,
    predict_annotations as hybrid_predict_annotations,
    prepare_classifier as hybrid_prepare_classifier,
)
from labeling_tool.paths import sam3_checkpoint_path
from labeling_tool.database import (
    init_db,
    log_action,
    get_audit_log,
    save_annotation_version,
    get_annotation_versions,
    get_annotation_version_data,
    create_or_update_session,
    get_session,
    increment_session_stats,
    record_daily_stat,
    get_daily_stats,
    get_stats_summary,
    set_image_flag,
    remove_image_flag,
    get_image_flags,
    get_user_preferences,
    save_user_preferences,
    record_export,
    get_export_history,
    create_project,
    list_projects,
    get_project,
    update_project,
    add_dataset_to_project,
    remove_dataset_from_project,
    add_dataset_tag,
    remove_dataset_tag,
    get_dataset_tags,
    get_all_tags,
)

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from PIL import Image as PILImage, ImageOps as PILImageOps


def _imread_exif(path: str) -> np.ndarray:
    """Read image with EXIF orientation applied, matching browser display.

    cv2.imread ignores EXIF orientation tags, causing width/height mismatch
    between backend and browser for images with orientation != 1.
    """
    pil_img = PILImage.open(path)
    pil_img = PILImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _image_size_exif(path: str) -> tuple:
    """Return (width, height) with EXIF orientation applied."""
    pil_img = PILImage.open(path)
    pil_img = PILImageOps.exif_transpose(pil_img)
    return pil_img.size


# Utility functions



def _natural_sort_key(s: str):
    """Natural sort key for proper numeric ordering (1, 2, 10 instead of 1, 10, 2)."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(s))]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



PROJECT_ROOT = os.environ.get(
    "MEDSAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# Static Files Setup


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


import mimetypes as _mimetypes

@app.get("/static/{file_path:path}")
def serve_static_file(file_path: str):
    """Serve static files with no-cache headers to prevent stale JS/CSS."""
    full_path = os.path.normpath(os.path.join(STATIC_DIR, file_path))
    if not full_path.startswith(os.path.normpath(STATIC_DIR)):
        raise HTTPException(403, "Forbidden")
    if not os.path.isfile(full_path):
        raise HTTPException(404, "Not found")
    content_type = _mimetypes.guess_type(full_path)[0] or "application/octet-stream"
    with open(full_path, "rb") as f:
        content = f.read()
    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/")
def root():
    """Serve the main index.html page."""
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


PALETTE = [
    "#3b82f6",
    "#f97316",
    "#a855f7",
    "#22c55e",
    "#ef4444",
    "#06b6d4",
    "#eab308",
    "#ec4899",
    "#14b8a6",
    "#f43f5e",
    "#8b5cf6",
    "#84cc16",
    "#64748b",
    "#d946ef",
    "#0ea5e9",
    "#facc15",
]


# Data structures

#
# IMAGE_GROUPS: unique image directories
#   { group_id, group_name, train_images, val_images,
#     yaml_path, nc, names,
#     label_sets: [ {set_id, set_name, train_labels, val_labels, label_format} ] }
#

IMAGE_GROUPS: List[Dict[str, Any]] = []

# ?? Dirty-flag pattern: avoid redundant build_image_groups() calls ??
_groups_dirty = True  # Start dirty so first read triggers build
_groups_lock = threading.Lock()
_stats_cache: Dict[
    str, tuple
] = {}  # (group_id, label_set, subset) -> (timestamp, stats)
_STATS_CACHE_TTL = 30  # seconds

# mtime-keyed cache for label-dir summaries (labeled count, total annotations, class
# distribution). Avoids re-walking every .txt on every Datasets / Stats tab open.
# key: lbl_dir absolute path -> (dir_mtime, summary_dict)
_LABEL_DIR_CACHE: Dict[str, tuple] = {}
_LABEL_DIR_CACHE_LOCK = threading.Lock()

# mtime-keyed cache for _list_image_files. Image directories rarely change between
# requests; the existing implementation re-scandirs + sorts on every call.
# key: imgs_dir absolute path -> (dir_mtime, sorted_paths)
_IMAGE_DIR_CACHE: Dict[str, tuple] = {}
_IMAGE_DIR_CACHE_LOCK = threading.Lock()


def _mark_groups_dirty():
    """Mark IMAGE_GROUPS as stale. Next read endpoint will rebuild."""
    global _groups_dirty, _stats_cache
    _groups_dirty = True
    _stats_cache.clear()
    with _LABEL_DIR_CACHE_LOCK:
        _LABEL_DIR_CACHE.clear()
    with _IMAGE_DIR_CACHE_LOCK:
        _IMAGE_DIR_CACHE.clear()


def _ensure_groups_fresh():
    """Rebuild IMAGE_GROUPS only if dirty. Thread-safe."""
    global IMAGE_GROUPS, _groups_dirty
    if not _groups_dirty:
        return
    with _groups_lock:
        if not _groups_dirty:
            return
        IMAGE_GROUPS = build_image_groups()
        _groups_dirty = False


# ?? User datasets persistence ??????????????????????????????????
_DATASETS_JSON = os.path.join(os.path.dirname(__file__), "datasets.json")


def _load_user_datasets() -> List[Dict]:
    if not os.path.isfile(_DATASETS_JSON):
        return []
    try:
        with open(_DATASETS_JSON, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return data.get("datasets", [])
    except Exception as e:
        print(f"[WARN] Failed to load datasets.json: {e}")
        return []


def _load_excluded_groups() -> List[str]:
    if not os.path.isfile(_DATASETS_JSON):
        return []
    try:
        with open(_DATASETS_JSON, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return data.get("excluded_groups", [])
    except Exception:
        return []


def _save_user_datasets(datasets: List[Dict]):
    try:
        existing = {}
        if os.path.isfile(_DATASETS_JSON):
            with open(_DATASETS_JSON, "r", encoding="utf-8") as f:
                existing = _json.load(f)
        existing["datasets"] = datasets
        with open(_DATASETS_JSON, "w", encoding="utf-8") as f:
            _json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save datasets.json: {e}")


def _save_excluded_groups(excluded: List[str]):
    try:
        existing = {}
        if os.path.isfile(_DATASETS_JSON):
            with open(_DATASETS_JSON, "r", encoding="utf-8") as f:
                existing = _json.load(f)
        existing["excluded_groups"] = excluded
        with open(_DATASETS_JSON, "w", encoding="utf-8") as f:
            _json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save excluded_groups: {e}")


def _detect_dataset_structure(path: str) -> Dict[str, Any]:
    """Detect the directory structure of a potential dataset."""
    path = os.path.normpath(path)
    result = {
        "structure": "unknown",
        "train_images": "",
        "val_images": "",
        "img_count": 0,
        "has_labels": False,
        "label_dirs": [],
        "has_val": False,
    }

    # Check YOLO structure: path/images/train, path/images/val
    yolo_train = os.path.join(path, "images", "train")
    yolo_val = os.path.join(path, "images", "val")
    if os.path.isdir(yolo_train):
        result["structure"] = "yolo"
        result["train_images"] = yolo_train
        if os.path.isdir(yolo_val) and _count_images(yolo_val) > 0:
            result["val_images"] = yolo_val
            result["has_val"] = True
        else:
            result["val_images"] = ""
        result["img_count"] = _count_images(yolo_train) + _count_images(yolo_val)
        # Check for labels
        labels_train = os.path.join(path, "labels", "train")
        if os.path.isdir(labels_train):
            result["has_labels"] = True
        return result

    # Check simple train/val: path/train, path/val containing images
    simple_train = os.path.join(path, "train")
    simple_val = os.path.join(path, "val")
    if os.path.isdir(simple_train) and _count_images(simple_train) > 0:
        result["structure"] = "simple_split"
        result["train_images"] = simple_train
        if os.path.isdir(simple_val) and _count_images(simple_val) > 0:
            result["val_images"] = simple_val
            result["has_val"] = True
        else:
            result["val_images"] = ""
        result["img_count"] = _count_images(simple_train) + _count_images(simple_val)
        return result

    # Flat: directory itself contains images (????)
    count = _count_images(path)
    if count > 0:
        result["structure"] = "flat"
        result["train_images"] = path
        result["val_images"] = ""
        result["has_val"] = False
        result["img_count"] = count
        # Check for sibling labels dir
        parent = os.path.dirname(path)
        base = os.path.basename(path)
        labels_sibling = os.path.join(parent, "labels_" + base)
        if os.path.isdir(labels_sibling):
            result["has_labels"] = True
        return result

    return result


def build_image_groups():
    """Scan project and build image groups with their label sets."""
    groups_map = {}  # key = train_images path -> group dict
    seen_label_dirs = set()
    excluded = set(_load_excluded_groups())

    _PRUNE_DIRS = {
        ".git",
        "__pycache__",
        "weights",
        "node_modules",
        ".backup",
        ".venv",
        "venv",
    }
    yaml_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
        for f in files:
            if f.startswith("data") and f.endswith(".yaml"):
                yaml_files.append(os.path.join(root, f))
    yaml_files.sort()

    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8", errors="replace") as f:
                cfg = yaml.safe_load(f)
            if not cfg or "path" not in cfg:
                continue

            ds_path = cfg["path"]
            if not os.path.isabs(ds_path):
                ds_path = os.path.join(os.path.dirname(yf), ds_path)
            ds_path = os.path.normpath(ds_path)

            train_rel = cfg.get("train", "images/train")
            val_rel = cfg.get("val", "images/val")
            nc = cfg.get("nc", 0)

            raw_names = cfg.get("names", [])
            if isinstance(raw_names, dict):
                names = {int(k): v for k, v in raw_names.items()}
            elif isinstance(raw_names, list):
                names = {i: v for i, v in enumerate(raw_names)}
            else:
                names = {}

            train_images = os.path.normpath(os.path.join(ds_path, train_rel))
            val_images = os.path.normpath(os.path.join(ds_path, val_rel))

            if not os.path.isdir(train_images):
                continue

            gkey = train_images
            if gkey in excluded:
                continue
            if gkey not in groups_map:
                # Build a readable group name from the images path
                rel = os.path.relpath(train_images, PROJECT_ROOT).replace("\\", "/")
                # e.g. "cell_re_data/images/train" -> "cell_re_data"
                parts = rel.split("/images/")
                gname = parts[0].replace("/", " / ") if len(parts) >= 2 else rel

                has_val = os.path.isdir(val_images) and val_images != train_images
                actual_val_images = val_images if has_val else ""

                groups_map[gkey] = {
                    "group_id": gkey,
                    "group_name": gname,
                    "train_images": train_images,
                    "val_images": actual_val_images,
                    "has_val": has_val,
                    "yaml_path": yf,
                    "nc": nc,
                    "names": names,
                    "label_sets": [],
                    # Cache image counts here so /api/groups doesn't rescan on every request
                    "train_count": _count_images(train_images),
                    "val_count": _count_images(actual_val_images) if has_val else 0,
                }

            group = groups_map[gkey]
            # Update names if this yaml has more classes
            if nc > group["nc"]:
                group["nc"] = nc
                group["names"] = names
                group["yaml_path"] = yf

            # Resolve label dirs
            label_dirs_list = _resolve_label_dirs(train_images, val_images)
            for tl, vl, tag in label_dirs_list:
                lkey = tl
                if lkey in seen_label_dirs:
                    continue
                seen_label_dirs.add(lkey)

                fmt = _detect_label_format(tl) or _detect_label_format(vl) or "polygon"
                set_name = tag if tag else "labels"
                group["label_sets"].append(
                    {
                        "set_id": tag or "default",
                        "set_name": set_name,
                        "train_labels": tl,
                        "val_labels": vl,
                        "label_format": fmt,
                    }
                )

        except Exception as e:
            print(f"[WARN] Failed to parse {yf}: {e}")

    # ?? Merge user-added datasets from datasets.json ???????????
    for ds in _load_user_datasets():
        train_img = os.path.normpath(ds.get("train_images", ""))
        if not train_img or not os.path.isdir(train_img):
            continue
        gkey = train_img
        if gkey in excluded:
            continue
        if gkey in groups_map:
            # Already discovered via YAML; tag it and update name if user provided one
            groups_map[gkey]["_user_dataset_id"] = ds.get("id", "")
            if ds.get("name"):
                groups_map[gkey]["group_name"] = ds["name"]
            continue


        val_img_raw = ds.get("val_images", "")
        val_img = os.path.normpath(val_img_raw) if val_img_raw else ""
        has_val = val_img and val_img != train_img and os.path.isdir(val_img)
        actual_val_img = val_img if has_val else ""

        raw_names = ds.get("names", {"0": "object"})
        names = {int(k): v for k, v in raw_names.items()}
        group = {
            "group_id": gkey,
            "group_name": ds.get("name", os.path.basename(train_img)),
            "train_images": train_img,
            "val_images": actual_val_img,
            "has_val": has_val,
            "yaml_path": ds.get("yaml_path", ""),
            "nc": ds.get("nc", max(names.keys(), default=0) + 1),
            "names": names,
            "label_sets": [],
            "train_count": _count_images(train_img),
            "val_count": _count_images(actual_val_img) if has_val else 0,
            "_user_dataset_id": ds.get("id", ""),
        }
        label_dirs_list = _resolve_label_dirs(train_img, val_img)
        for tl, vl, tag in label_dirs_list:
            lkey = tl
            if lkey in seen_label_dirs:
                continue
            seen_label_dirs.add(lkey)
            fmt = _detect_label_format(tl) or _detect_label_format(vl) or "polygon"
            set_name = tag if tag else "labels"
            group["label_sets"].append(
                {
                    "set_id": tag or "default",
                    "set_name": set_name,
                    "train_labels": tl,
                    "val_labels": vl,
                    "label_format": fmt,
                }
            )
        if not group["label_sets"]:
            # Create a default label set
            parent = os.path.dirname(train_img)
            base = (
                os.path.basename(os.path.dirname(train_img))
                if "/images/" in train_img.replace("\\", "/")
                else os.path.basename(train_img)
            )
            labels_dir = (
                os.path.join(parent, "labels_" + base)
                if "/images/" not in train_img.replace("\\", "/")
                else train_img.replace("/images/", "/labels/", 1)
            )
            group["label_sets"].append(
                {
                    "set_id": "default",
                    "set_name": "default",
                    "train_labels": labels_dir,
                    "val_labels": labels_dir,
                    "label_format": "polygon",
                }
            )
        groups_map[gkey] = group

    for group in groups_map.values():
        group["label_sets"].sort(key=lambda ls: _natural_sort_key(ls["set_id"]))

    return sorted(groups_map.values(), key=lambda g: g["group_name"])


def _resolve_label_dirs(train_images_dir, val_images_dir):
    """Utility function."""
    results = []
    seen = set()
    # Normalize to forward slashes for cross-platform string ops, then re-normalize at end
    _train = train_images_dir.replace("\\", "/")
    _val = val_images_dir.replace("\\", "/") if val_images_dir else ""

    std_train = os.path.normpath(_train.replace("/images/", "/labels/", 1))
    std_val = os.path.normpath(_val.replace("/images/", "/labels/", 1)) if _val else ""

    if _dir_exists_or_has_txt(std_train) or (
        std_val and _dir_exists_or_has_txt(std_val)
    ):
        results.append((std_train, std_val, ""))
        seen.add(std_train)

    train_parts = _train.split("/images/")
    if len(train_parts) == 2:
        parent = os.path.normpath(train_parts[0])
        sub = train_parts[1]

        if _val:
            val_parts = _val.split("/images/")
            val_sub = val_parts[1] if len(val_parts) == 2 else "val"
        else:
            val_sub = "val"
        skip = {"images", ".git", "__pycache__", ".venv", ".venv_yolo", "assets"}
        if os.path.isdir(parent):
            for entry in sorted(os.listdir(parent)):
                if entry in skip or entry.startswith("."):
                    continue
                full = os.path.join(parent, entry)
                if not os.path.isdir(full):
                    continue
                ct = os.path.join(full, sub)
                cv = os.path.join(full, val_sub)
                if ct in seen:
                    continue
                if not (_dir_exists_or_has_txt(ct) or _dir_exists_or_has_txt(cv)):
                    continue
                if entry.startswith("labels_"):
                    tag = entry[len("labels_") :]
                elif entry == "labels":
                    tag = "default"
                else:
                    tag = entry
                results.append((ct, cv, tag))
                seen.add(ct)

    if not results:
        results.append((std_train, std_val, ""))
    return results


def _dir_exists_or_has_txt(d):
    """Return True if directory exists (even if empty)."""
    return os.path.isdir(d)


def _detect_label_format(labels_dir):
    if not os.path.isdir(labels_dir):
        return None
    for txt in glob.glob(os.path.join(labels_dir, "*.txt"))[:5]:
        try:
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        return "bbox"
                    elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                        return "polygon"
        except Exception:
            pass
    return None


def _extract_yaml_comment(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("#"):
                    return s.lstrip("# ").strip()
                if s and not s.startswith("#"):
                    break
    except Exception:
        pass
    return ""


_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _count_images(d):
    if not os.path.isdir(d):
        return 0
    count = 0
    try:
        with os.scandir(d) as it:
            for entry in it:
                if (
                    entry.is_file(follow_symlinks=False)
                    and os.path.splitext(entry.name)[1].lower() in _IMAGE_EXTS
                ):
                    count += 1
    except OSError:
        return 0
    return count


# ?? Build registry ?????????????????????????????????????????????
_ensure_groups_fresh()
print(f"[INFO] Discovered {len(IMAGE_GROUPS)} image groups:")
for g in IMAGE_GROUPS:
    sets = ", ".join(f"{s['set_name']}({s['label_format']})" for s in g["label_sets"])
    print(f"  - {g['group_name']} ({g['nc']}cls) ??[{sets}]")



# Pydantic models



class Annotation(BaseModel):
    class_id: int
    ann_type: str = "polygon"
    points: List[float]
    annotation_uid: Optional[str] = None


class SaveRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    filename: str
    annotations: List[Annotation]
    expected_mtime: Optional[float] = None

class ConflictCheckResponse(BaseModel):
    has_conflict: bool
    server_mtime: Optional[float] = None
    server_annotations: Optional[List[dict]] = None
    message: Optional[str] = None


class AddClassRequest(BaseModel):
    group_id: str
    class_name: str


class PredictRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    filename: str
    box: Optional[List[float]] = None
    class_id: Optional[int] = None
    text_prompt: Optional[str] = None
    prompt_type: str = "box"  # "box", "box_inst", "text", "box+text", "13points"


class BatchPredictRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    filename: str
    boxes: List[dict]  # [{class_id, box: [x1,y1,x2,y2] normalized}]
    prompt_type: str = "box"
    prompt_types: Optional[List[str]] = None
    text_prompt: Optional[str] = None


class CellposeRequest(BaseModel):
    group_id: str
    subset: str
    filename: str
    diameters: List[float] = [30.0]
    gpu: bool = True
    class_id: int = 0
    min_area: int = 100


class CellposeBatchRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    start_index: int = 0
    end_index: Optional[int] = None
    diameters: List[float] = [30.0]
    gpu: bool = True
    class_id: int = 0
    min_area: int = 100
    skip_existing: bool = True
    overwrite_existing: bool = False


class BatchJobActionRequest(BaseModel):
    job_id: str


class FewShotSupportItem(BaseModel):
    filename: str
    subset: str
    class_id: int
    ann_type: str = "polygon"
    points: List[float]
    support_key: Optional[str] = None
    annotation_uid: Optional[str] = None


class FewShotPredictRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    filename: str
    support_items: List[FewShotSupportItem]
    query_annotations: Optional[List[Annotation]] = None
    temperature: float = 1.0
    device: str = "auto"
    use_prompts: bool = False
    prompt_mode: str = "auto"
    prompt_ensembles: Optional[Dict[str, List[str]]] = None
    image_proto_weight: float = 0.5
    text_proto_weight: float = 0.5
    primary_prompt_weight: float = 0.75


class FewShotEvaluateRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    support_items: List[FewShotSupportItem]
    temperature: float = 1.0
    device: str = "auto"
    max_images: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    use_prompts: bool = False
    prompt_mode: str = "auto"
    prompt_ensembles: Optional[Dict[str, List[str]]] = None
    image_proto_weight: float = 0.5
    text_proto_weight: float = 0.5
    primary_prompt_weight: float = 0.75


class HybridPredictRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    filename: str
    support_items: List[FewShotSupportItem]
    query_annotations: Optional[List[Annotation]] = None
    device: str = "auto"
    enable_size_refiner: bool = True
    size_refiner_trigger_margin: float = 0.12
    size_refiner_min_separation: float = 1.0
    size_refiner_score_scale: float = 0.06
    size_refiner_max_adjust: float = 0.08
    text_prompt_names: Optional[Dict[str, str]] = None
    prompt_template: Optional[str] = None


class HybridEvaluateRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    support_items: List[FewShotSupportItem]
    device: str = "auto"
    max_images: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    enable_size_refiner: bool = True
    size_refiner_trigger_margin: float = 0.12
    size_refiner_min_separation: float = 1.0
    size_refiner_score_scale: float = 0.06
    size_refiner_max_adjust: float = 0.08
    text_prompt_names: Optional[Dict[str, str]] = None
    prompt_template: Optional[str] = None


class HybridGoldEvaluateRequest(HybridEvaluateRequest):
    gold_label_set_id: str


class SegmentGoldEvaluateRequest(BaseModel):
    group_id: str
    subset: str
    gold_label_set_id: str
    method: str = "cellpose"  # "cellpose" | "sam3"
    prompt_label_set_id: Optional[str] = None
    max_images: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    diameters: List[float] = [30.0]
    gpu: bool = True
    class_id: int = 0
    min_area: int = 100
    prompt_type: str = "box"
    prompt_types: Optional[List[str]] = None
    text_prompt: Optional[str] = None


class SimpleSegmentEvalRequest(BaseModel):
    """Segment evaluation with gold standard."""

    pred_labels_dir: str
    gold_labels_dir: str
    images_dir: str
    max_images: Optional[int] = None


class SimpleClassifyEvalRequest(BaseModel):
    """Segment evaluation with gold standard."""

    pred_labels_dir: str
    gold_labels_dir: str
    images_dir: str
    class_names: Optional[Dict[int, str]] = None
    max_images: Optional[int] = None


# ????????????????????????????????????????????????????????????????# Core Helpers


def _get_group(group_id: str) -> Dict[str, Any]:
    _ensure_groups_fresh()
    for g in IMAGE_GROUPS:
        if g["group_id"] == group_id:
            return g
    raise HTTPException(404, f"Image group not found")


def _get_label_set(g: Dict, label_set_id: str) -> Dict[str, Any]:
    for ls in g.get("label_sets", []):
        if ls["set_id"] == label_set_id:
            return ls
    raise HTTPException(404, f"Label set not found: {label_set_id}")


def _get_dirs(g: Dict, ls: Dict, subset: str) -> tuple:
    if subset == "val":
        imgs_dir = g.get("val_images", "")
        lbls_dir = ls.get("val_labels", "")
    else:
        imgs_dir = g["train_images"]
        lbls_dir = ls["train_labels"]
    return imgs_dir, lbls_dir


def _list_image_files(directory: str) -> List[str]:
    if not directory or not os.path.isdir(directory):
        return []
    try:
        mtime = os.path.getmtime(directory)
    except OSError:
        return []
    cached = _IMAGE_DIR_CACHE.get(directory)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    files: List[str] = []
    try:
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
                files.append(os.path.join(directory, f))
    except OSError:
        return []
    files.sort(key=lambda x: _natural_sort_key(os.path.basename(x)))
    with _IMAGE_DIR_CACHE_LOCK:
        _IMAGE_DIR_CACHE[directory] = (mtime, files)
    return files


def _label_dir_summary(lbl_dir: str, names: Dict[Any, str] | None = None) -> Dict[str, Any]:
    """Walk a label directory once and summarize: labeled count, total annotations,
    per-class distribution. Cached by directory mtime — repeat calls are O(1).

    `names` (class id -> display name) is applied at read time so distribution is
    keyed by human-readable class names. Stats are keyed by class_id internally so
    a single cache hit can be re-projected if names change.
    """
    if not lbl_dir or not os.path.isdir(lbl_dir):
        return {"labeled": 0, "total_annotations": 0, "by_class_id": {}, "by_class": {}}
    try:
        mtime = os.path.getmtime(lbl_dir)
    except OSError:
        return {"labeled": 0, "total_annotations": 0, "by_class_id": {}, "by_class": {}}

    cached = _LABEL_DIR_CACHE.get(lbl_dir)
    if cached is None or cached[0] != mtime:
        labeled = 0
        total_anns = 0
        by_class_id: Dict[int, int] = {}
        try:
            for f in os.listdir(lbl_dir):
                if not f.endswith(".txt"):
                    continue
                fp = os.path.join(lbl_dir, f)
                try:
                    if os.path.getsize(fp) <= 0:
                        continue
                    has_any = False
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                cid = int(parts[0])
                            except ValueError:
                                continue
                            total_anns += 1
                            by_class_id[cid] = by_class_id.get(cid, 0) + 1
                            has_any = True
                    if has_any:
                        labeled += 1
                except OSError:
                    continue
        except OSError:
            pass
        summary = {
            "labeled": labeled,
            "total_annotations": total_anns,
            "by_class_id": by_class_id,
        }
        with _LABEL_DIR_CACHE_LOCK:
            _LABEL_DIR_CACHE[lbl_dir] = (mtime, summary)
    else:
        summary = cached[1]

    # Project class-id histogram through the supplied names map for callers that
    # want a human-readable distribution.
    if names:
        by_class: Dict[str, int] = {}
        for cid, count in summary["by_class_id"].items():
            cname = names.get(cid, names.get(str(cid), f"class_{cid}"))
            by_class[cname] = by_class.get(cname, 0) + count
        return {**summary, "by_class": by_class}
    return {**summary, "by_class": {}}


def _label_path_for(lbls_dir: str, filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(lbls_dir, base + ".txt")


def _image_path_for(imgs_dir: str, filename: str) -> str:
    return os.path.join(imgs_dir, os.path.basename(filename))


def _read_annotations(label_path: str) -> List[Dict[str, Any]]:
    if not label_path or not os.path.isfile(label_path):
        return []
    annotations = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                ann = {
                    "class_id": class_id,
                    "ann_type": "bbox" if len(coords) == 4 else "polygon",
                    "points": coords,
                    "annotation_uid": f"L{line_num}",
                }
                annotations.append(ann)
    except Exception:
        pass
    return annotations


def _annotation_payload(annotations) -> List[Dict[str, Any]]:
    result = []
    for ann in annotations:
        if isinstance(ann, dict):
            item = {
                "class_id": int(ann.get("class_id", 0)),
                "ann_type": ann.get("ann_type", "polygon"),
                "points": [float(p) for p in ann.get("points", [])],
            }
            uid = ann.get("annotation_uid")
            if uid:
                item["annotation_uid"] = str(uid)
        else:
            item = {
                "class_id": int(ann.class_id),
                "ann_type": ann.ann_type,
                "points": [float(p) for p in ann.points],
            }
            if ann.annotation_uid:
                item["annotation_uid"] = str(ann.annotation_uid)
        result.append(item)
    return result


def _annotation_uid(ann) -> str:
    if isinstance(ann, dict):
        return str(ann.get("annotation_uid", "")).strip()
    return str(getattr(ann, "annotation_uid", "") or "").strip()


def _annotation_signature(ann) -> str:
    if isinstance(ann, dict):
        cid = ann.get("class_id", 0)
        pts = ann.get("points", [])
    else:
        cid = getattr(ann, "class_id", 0)
        pts = getattr(ann, "points", [])
    rounded = tuple(round(float(p), 4) for p in pts[:8])
    return f"{cid}:{rounded}"


def _make_support_key(filename: str, subset: str, ann) -> str:
    uid = _annotation_uid(ann)
    if uid:
        return f"{filename}:{subset}:{uid}"
    sig = _annotation_signature(ann)
    return f"{filename}:{subset}:{sig}"


def _rebind_support_against_annotations(item, anns, filename, subset):
    for ann in anns:
        if _annotation_matches_support(ann, item):
            return {
                "filename": filename,
                "subset": subset,
                "class_id": int(ann.get("class_id", item.get("class_id", 0))),
                "ann_type": ann.get("ann_type", item.get("ann_type", "polygon")),
                "points": ann.get("points", item.get("points", [])),
                "annotation_uid": _annotation_uid(ann) or _annotation_uid(item),
            }
    return None


def _resolve_support_items(g, ls, support_items):
    result = []
    for item in support_items:
        si = item if isinstance(item, dict) else item.dict()
        subset = si.get("subset", "train")
        filename = si.get("filename", "")
        imgs_dir, _ = _get_dirs(g, ls, subset)
        image_path = _image_path_for(imgs_dir, filename)
        result.append(
            {
                **si,
                "image_path": image_path,
                "support_key": si.get("support_key")
                or _make_support_key(filename, subset, si),
            }
        )
    return result


def _class_name_map(g) -> Dict[int, str]:
    return {int(k): str(v) for k, v in g.get("names", {}).items()}


def _ann_bbox(ann) -> Optional[List[float]]:
    pts = (
        ann.get("points", []) if isinstance(ann, dict) else getattr(ann, "points", [])
    )
    ann_type = (
        ann.get("ann_type", "polygon")
        if isinstance(ann, dict)
        else getattr(ann, "ann_type", "polygon")
    )
    if not pts:
        return None
    if ann_type == "bbox" and len(pts) == 4:
        cx, cy, w, h = pts
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    elif len(pts) >= 4:
        xs = pts[0::2]
        ys = pts[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _slice_files_for_request(files, start_index, end_index, max_images):
    if start_index is not None or end_index is not None:
        si = 0 if start_index is None else max(0, int(start_index))
        ei = (
            len(files) - 1
            if end_index is None
            else min(len(files) - 1, int(end_index))
        )
        if ei >= si:
            files = files[si : ei + 1]
    if max_images is not None and max_images > 0:
        files = files[: int(max_images)]
    return files


def _annotation_to_mask(ann, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = (
        ann.get("points", []) if isinstance(ann, dict) else getattr(ann, "points", [])
    )
    ann_type = (
        ann.get("ann_type", "polygon")
        if isinstance(ann, dict)
        else getattr(ann, "ann_type", "polygon")
    )
    if ann_type == "bbox" and len(pts) == 4:
        cx, cy, bw, bh = pts
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        mask[y1:y2, x1:x2] = 1
    elif len(pts) >= 6:
        poly_pts = np.array(
            [(int(pts[i] * w), int(pts[i + 1] * h)) for i in range(0, len(pts), 2)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [poly_pts], 1)
    return mask.astype(bool)


def _evaluate_segmentation_annotations(pred_annotations, gold_annotations, img_shape):
    h, w = img_shape[:2]
    pred_masks = [_annotation_to_mask(ann, h, w) for ann in pred_annotations]
    gold_masks = [_annotation_to_mask(ann, h, w) for ann in gold_annotations]

    matched = 0
    matches = []
    used_gold = set()

    for pi, pmask in enumerate(pred_masks):
        best_iou = 0.0
        best_gi = -1
        for gi, gmask in enumerate(gold_masks):
            if gi in used_gold:
                continue
            intersection = np.logical_and(pmask, gmask).sum()
            union = np.logical_or(pmask, gmask).sum()
            iou = float(intersection) / max(float(union), 1)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou > 0.3 and best_gi >= 0:
            matched += 1
            used_gold.add(best_gi)
            dice_denom = float(pmask.sum() + gold_masks[best_gi].sum())
            dice = (
                2.0
                * float(np.logical_and(pmask, gold_masks[best_gi]).sum())
                / max(dice_denom, 1)
            )
            matches.append(
                {
                    "pred_idx": pi,
                    "gold_idx": best_gi,
                    "iou": best_iou,
                    "dice": dice,
                }
            )

    precision = matched / max(len(pred_masks), 1)
    recall = matched / max(len(gold_masks), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "pred_count": len(pred_masks),
        "gold_count": len(gold_masks),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def _list_backups(label_path):
    backup_dir = os.path.join(os.path.dirname(label_path), ".backup")
    if not os.path.isdir(backup_dir):
        return []
    base = os.path.splitext(os.path.basename(label_path))[0]
    backups = []
    for f in os.listdir(backup_dir):
        if f.startswith(base + ".txt.") and f.endswith(".bak"):
            ts = f[len(base + ".txt.") : -4]
            bp = os.path.join(backup_dir, f)
            backups.append(
                {"timestamp": ts, "path": bp, "size": os.path.getsize(bp)}
            )
    backups.sort(key=lambda x: x["timestamp"], reverse=True)
    return backups


def _cleanup_old_backups(backup_dir, filename, keep_count=10):
    if not os.path.isdir(backup_dir):
        return
    base = os.path.splitext(filename)[0]
    backups = []
    for f in os.listdir(backup_dir):
        if f.startswith(base + ".") and f.endswith(".bak"):
            backups.append(os.path.join(backup_dir, f))
    backups.sort(reverse=True)
    for old in backups[keep_count:]:
        try:
            os.remove(old)
        except Exception:
            pass


# ????????????????????????????????????????????????????????????????# Core Annotation APIs (images, annotations, save, classes)


@app.get("/api/images")
def list_images(group_id: str, label_set_id: str, subset: str = "train"):
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, subset)
    if not imgs_dir or not os.path.isdir(imgs_dir):
        return []
    files = _list_image_files(imgs_dir)
    result = []
    for fp in files:
        fn = os.path.basename(fp)
        base = os.path.splitext(fn)[0]
        label_path = os.path.join(lbls_dir, base + ".txt") if lbls_dir else ""
        has_label = (
            bool(label_path)
            and os.path.exists(label_path)
            and os.path.getsize(label_path) > 0
        )
        result.append({"filename": fn, "has_label": has_label})
    return result


@app.get("/api/annotations")
def get_annotations(group_id: str, label_set_id: str, subset: str, filename: str):
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)
    _, lbls_dir = _get_dirs(g, ls, subset)
    label_path = _label_path_for(lbls_dir, filename)
    annotations = _read_annotations(label_path)
    return _annotation_payload(annotations)


@app.post("/api/save")
def save_annotations_endpoint(req: SaveRequest):
    g = _get_group(req.group_id)
    ls = _get_label_set(g, req.label_set_id)
    _, lbls_dir = _get_dirs(g, ls, req.subset)
    os.makedirs(lbls_dir, exist_ok=True)
    label_path = _label_path_for(lbls_dir, req.filename)

    if os.path.exists(label_path):
        backup_dir = os.path.join(os.path.dirname(label_path), ".backup")
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            backup_dir, os.path.basename(label_path) + f".{ts}.bak"
        )
        shutil.copy2(label_path, backup_path)
        _cleanup_old_backups(backup_dir, os.path.basename(label_path))

    lines = []
    for ann in req.annotations:
        if ann.points and len(ann.points) >= 4:
            lines.append(
                f"{ann.class_id} " + " ".join(f"{p:.6f}" for p in ann.points)
            )

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    try:
        ann_dicts = [
            {"class_id": a.class_id, "ann_type": a.ann_type, "points": list(a.points)}
            for a in req.annotations
        ]
        save_annotation_version(
            req.group_id, req.label_set_id, req.subset, req.filename, ann_dicts
        )
        record_daily_stat(
            group_id=req.group_id,
            annotations_created=len(lines),
            images_annotated=1,
        )
        log_action(
            "save_annotations",
            "annotation",
            group_id=req.group_id,
            filename=req.filename,
            details={"count": len(lines), "subset": req.subset},
        )
    except Exception as e:
        logging.warning(f"Failed to record annotation version: {e}")

    return {"ok": True, "count": len(lines)}


@app.get("/api/classes")
def get_classes(group_id: str):
    g = _get_group(group_id)
    classes = []
    for cid, name in sorted(g.get("names", {}).items(), key=lambda x: int(x[0])):
        color = PALETTE[int(cid) % len(PALETTE)]
        classes.append({"ID": int(cid), "Name": str(name), "Color": color})
    return classes


@app.post("/api/add_class")
def add_class(req: AddClassRequest):
    g = _get_group(req.group_id)
    names = g.get("names", {})
    new_id = max((int(k) for k in names.keys()), default=-1) + 1
    names[new_id] = req.class_name
    g["names"] = names
    g["nc"] = len(names)

    yaml_path = g.get("yaml_path", "")
    if yaml_path and os.path.isfile(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if isinstance(cfg.get("names"), list):
                cfg["names"].append(req.class_name)
            elif isinstance(cfg.get("names"), dict):
                cfg["names"][str(new_id)] = req.class_name
            else:
                cfg["names"] = {str(k): v for k, v in names.items()}
            cfg["nc"] = len(names)
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            print(f"[WARN] Failed to update YAML: {e}")

    color = PALETTE[new_id % len(PALETTE)]
    return {"ID": new_id, "Name": req.class_name, "Color": color, "nc": g["nc"]}


@app.get("/api/image_jpeg")
def get_image_jpeg(group_id: str, subset: str, filename: str):
    g = _get_group(group_id)
    imgs_dir = g.get("val_images", "") if subset == "val" else g["train_images"]
    if not imgs_dir:
        raise HTTPException(404, "Image directory not found")
    fn = os.path.basename(filename)
    img_path = os.path.join(imgs_dir, fn)
    if not os.path.exists(img_path):
        raise HTTPException(404, "Image not found")
    ext = os.path.splitext(fn)[1].lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    return FileResponse(img_path, media_type=media_types.get(ext, "image/jpeg"))


@app.post("/api/images/warmup_jpeg")
def warmup_jpeg(
    group_id: str = Body(...),
    subset: str = Body("train"),
    filenames: List[str] = Body([]),
):
    return {"ok": True, "warmed": len(filenames)}


# ????????????????????????????????????????????????????????????????# Backup Management APIs


class RestoreBackupRequest(BaseModel):
    group_id: str
    label_set_id: str
    subset: str
    filename: str
    backup_timestamp: str  # "latest" or timestamp like "20250115_143022"


@app.get("/api/annotations/backups")
def list_annotation_backups(
    group_id: str, label_set_id: str, subset: str, filename: str
):
    """List available backups for an annotation file."""
    try:
        g = _get_group(group_id)
        ls = _get_label_set(g, label_set_id)
        _, lbls_dir = _get_dirs(g, ls, subset)
        fn = os.path.basename(filename)
        base, _ = os.path.splitext(fn)
        label_path = os.path.join(lbls_dir, base + ".txt")

        backups = _list_backups(label_path)
        return {"current_exists": os.path.exists(label_path), "backups": backups}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/annotations/restore")
def restore_annotation_backup(req: RestoreBackupRequest):
    """Restore an annotation file from backup."""
    try:
        g = _get_group(req.group_id)
        ls = _get_label_set(g, req.label_set_id)
        _, lbls_dir = _get_dirs(g, ls, req.subset)
        fn = os.path.basename(req.filename)
        base, _ = os.path.splitext(fn)
        label_path = os.path.join(lbls_dir, base + ".txt")

        if req.backup_timestamp == "latest":
            backup_path = label_path + ".bak"
        else:
            backup_dir = os.path.join(lbls_dir, ".backup")
            backup_path = os.path.join(
                backup_dir, f"{base}.txt.{req.backup_timestamp}.bak"
            )

        if not os.path.exists(backup_path):
            raise HTTPException(404, "Backup not found")

        # Create backup of current before restoring
        if os.path.exists(label_path):
            try:
                os.makedirs(os.path.join(lbls_dir, ".backup"), exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pre_restore_bak = os.path.join(
                    lbls_dir, ".backup", f"{base}.txt.pre_restore_{timestamp}.bak"
                )
                shutil.copy2(label_path, pre_restore_bak)
            except Exception:
                pass

        # Restore from backup
        shutil.copy2(backup_path, label_path)

        # Return restored annotations
        annotations = _read_annotations(label_path)
        return {
            "status": "restored",
            "path": label_path,
            "backup_used": req.backup_timestamp,
            "annotations": annotations,
            "mtime": os.path.getmtime(label_path),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/annotations/backups")
def cleanup_old_backups(
    group_id: str, label_set_id: str, subset: str, filename: str, keep_count: int = 5
):
    """Clean up old backups, keeping only the most recent N versions."""
    try:
        g = _get_group(group_id)
        ls = _get_label_set(g, label_set_id)
        _, lbls_dir = _get_dirs(g, ls, subset)
        fn = os.path.basename(filename)
        base, _ = os.path.splitext(fn)
        label_path = os.path.join(lbls_dir, base + ".txt")
        backup_dir = os.path.join(lbls_dir, ".backup")

        if os.path.isdir(backup_dir):
            _cleanup_old_backups(backup_dir, base + ".txt", keep_count)

        backups = _list_backups(label_path)
        return {"status": "cleaned", "remaining_backups": len(backups)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/fewshot/evaluate_subset")
def fewshot_evaluate_subset(req: FewShotEvaluateRequest):
    g = _get_group(req.group_id)
    ls = _get_label_set(g, req.label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, req.subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")
    if not req.support_items:
        raise HTTPException(400, "Missing support items")

    files = _list_image_files(imgs_dir)
    if req.start_index is not None or req.end_index is not None:
        if not files:
            raise HTTPException(400, "Invalid class configuration")
        start_index = 0 if req.start_index is None else max(0, int(req.start_index))
        end_index = (
            len(files) - 1
            if req.end_index is None
            else min(len(files) - 1, int(req.end_index))
        )
        if end_index < start_index:
            raise HTTPException(400, "Invalid parameters")
        files = files[start_index : end_index + 1]
    if req.max_images is not None and req.max_images > 0:
        files = files[: int(req.max_images)]

    support_items = _resolve_support_items(g, ls, req.support_items)
    support_keys = {item["support_key"] for item in support_items}
    dataset_items = []
    for fp in files:
        filename = os.path.basename(fp)
        annotations = _annotation_payload(
            _read_annotations(_label_path_for(lbls_dir, filename))
        )
        annotations = [
            ann
            for ann in annotations
            if _make_support_key(filename, req.subset, ann) not in support_keys
        ]
        if not annotations:
            continue
        dataset_items.append(
            {
                "filename": filename,
                "image_path": fp,
                "annotations": annotations,
            }
        )

    if not dataset_items:
        raise HTTPException(400, "No valid evaluation data")

    prompt_ens = None
    if req.prompt_ensembles:
        prompt_ens = {int(k): v for k, v in req.prompt_ensembles.items()}
    try:
        classifier, _ = fewshot_prepare_classifier(
            support_items=support_items,
            class_names=_class_name_map(g),
            temperature=req.temperature,
            device=req.device,
            use_prompts=req.use_prompts,
            prompt_mode=req.prompt_mode,
            prompt_ensembles=prompt_ens,
            image_proto_weight=req.image_proto_weight,
            text_proto_weight=req.text_proto_weight,
            primary_prompt_weight=req.primary_prompt_weight,
        )
        result = fewshot_evaluate_dataset(
            classifier=classifier,
            dataset_items=dataset_items,
            temperature=req.temperature,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Few-shot subset evaluation failed")
        raise HTTPException(500, f"Few-shot classification error: {e}")

    return {
        "subset": req.subset,
        "device": classifier.device,
        **result,
    }


@app.post("/api/fewshot/predict_current")
def fewshot_predict_current(req: FewShotPredictRequest):
    g = _get_group(req.group_id)
    ls = _get_label_set(g, req.label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, req.subset)
    image_path = _image_path_for(imgs_dir, req.filename)
    if not os.path.exists(image_path):
        raise HTTPException(404, "Image not found")

    query_annotations = _annotation_payload(
        req.query_annotations
        or _read_annotations(_label_path_for(lbls_dir, req.filename))
    )
    if not query_annotations:
        raise HTTPException(400, "Invalid evaluation data")
    if not req.support_items:
        raise HTTPException(400, "Missing support items")

    support_items = _resolve_support_items(g, ls, req.support_items)
    prompt_ens = None
    if req.prompt_ensembles:
        prompt_ens = {int(k): v for k, v in req.prompt_ensembles.items()}
    try:
        classifier, _ = fewshot_prepare_classifier(
            support_items=support_items,
            class_names=_class_name_map(g),
            temperature=req.temperature,
            device=req.device,
            use_prompts=req.use_prompts,
            prompt_mode=req.prompt_mode,
            prompt_ensembles=prompt_ens,
            image_proto_weight=req.image_proto_weight,
            text_proto_weight=req.text_proto_weight,
            primary_prompt_weight=req.primary_prompt_weight,
        )
        result = fewshot_predict_annotations(
            classifier=classifier,
            image_path=image_path,
            annotations=query_annotations,
            temperature=req.temperature,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Few-shot prediction failed")
        raise HTTPException(500, f"Few-shot classification error: {e}")

    return {
        "filename": os.path.basename(req.filename),
        "subset": req.subset,
        "device": classifier.device,
        **result,
    }



# Support persistence (load / save / validate / defaults)


_DEFAULT_SUPPORTS_PATH = os.path.join(
    os.path.dirname(__file__), "default_supports.json"
)


def _supports_path(group, label_set_id: str = "") -> str:
    parent = os.path.dirname(group["train_images"])
    if label_set_id and label_set_id != "default":
        return os.path.join(parent, f".fewshot_supports_{label_set_id}.json")
    return os.path.join(parent, ".fewshot_supports.json")


def _load_default_supports() -> list:
    if not os.path.exists(_DEFAULT_SUPPORTS_PATH):
        return []
    try:
        with open(_DEFAULT_SUPPORTS_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _annotation_matches_support(ann: Dict[str, Any], support: Dict[str, Any]) -> bool:
    """Check if a label-file annotation matches a support item."""
    ann_uid = _annotation_uid(ann)
    support_uid = str(support.get("annotation_uid") or "").strip()
    if ann_uid and support_uid and ann_uid == support_uid:
        return True
    return _annotation_signature(ann) == _annotation_signature(support)


def _validate_supports_against_labels(
    supports: list, group: dict, label_set_id: str
) -> dict:
    """Check each support still exists in its label file; rebind if geometry shifted slightly."""
    valid, invalid = [], []
    label_set = None
    for ls in group.get("label_sets", []):
        if ls["set_id"] == label_set_id:
            label_set = ls
            break
    if label_set is None:
        return {"valid": supports, "invalid": [], "removed_count": 0}

    for item in supports:
        subset = item.get("subset", "train")
        fname = item.get("filename", "")
        _, lbls_dir = _get_dirs(group, label_set, subset)
        anns = _read_annotations(_label_path_for(lbls_dir, fname))
        rebound = _rebind_support_against_annotations(item, anns, fname, subset)
        if rebound:
            rebound["support_key"] = _make_support_key(fname, subset, rebound)
            valid.append(rebound)
        else:
            invalid.append({**item, "_reason": "annotation not found in label file"})
    return {"valid": valid, "invalid": invalid, "removed_count": len(invalid)}


class SaveSupportsRequest(BaseModel):
    group_id: str
    label_set_id: str = ""
    supports: List[Dict[str, Any]]


@app.get("/api/supports")
def get_supports(group_id: str, label_set_id: str = "", validate: bool = False):
    g = _get_group(group_id)
    path = _supports_path(g, label_set_id)
    supports = []
    source = "none"

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            supports = data if isinstance(data, list) else []
            source = "saved"
        except Exception:
            supports = []

    if not supports:
        supports = _load_default_supports()
        if supports:
            source = "default"

    if validate and supports:
        result = _validate_supports_against_labels(supports, g, label_set_id)
        if result["removed_count"] > 0:
            supports = result["valid"]
        else:
            supports = result["valid"]
        if source == "saved":
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    _json.dump(supports, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return {
            "supports": supports,
            "source": source,
            "validation": {
                "removed_count": result["removed_count"],
                "invalid": [
                    {
                        "filename": inv.get("filename"),
                        "class_id": inv.get("class_id"),
                        "reason": inv.get("_reason", ""),
                    }
                    for inv in result["invalid"]
                ],
            },
        }

    return {"supports": supports, "source": source}


@app.post("/api/supports/save")
def save_supports(req: SaveSupportsRequest):
    g = _get_group(req.group_id)
    path = _supports_path(g, req.label_set_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(req.supports, f, ensure_ascii=False, indent=2)
    return {"ok": True, "path": path, "count": len(req.supports)}



# Hybrid Adaptive Classification (latest innovation pipeline)



def _build_size_refiner_config(req) -> Dict[str, float]:
    return {
        "trigger_margin_max": req.size_refiner_trigger_margin,
        "min_separation_z": req.size_refiner_min_separation,
        "score_scale": req.size_refiner_score_scale,
        "max_adjust": req.size_refiner_max_adjust,
    }


def _parse_text_prompt_names(raw: Optional[Dict[str, str]]) -> Optional[Dict[int, str]]:
    """Convert str-key dict from JSON to int-key for prepare_classifier."""
    if not raw:
        return None
    result: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            result[int(k)] = str(v).strip()
        except (ValueError, TypeError):
            pass
    return result or None


def _build_sam_eval_predictions(
    g: Dict[str, Any],
    imgs_dir: str,
    prompt_lbls_dir: str,
    filename: str,
    req: SegmentGoldEvaluateRequest,
) -> List[Dict[str, Any]]:
    global _sam_model
    if _sam_model is None:
        _sam_model = _load_sam_model()
    if _sam_model is None:
        raise HTTPException(503, "SAM3 model not available")

    image_path = _image_path_for(imgs_dir, filename)
    img = _imread_exif(image_path)
    if img is None:
        raise HTTPException(404, f"Image not found: {filename}")
    h, w = img.shape[:2]
    prompt_annotations = _annotation_payload(
        _read_annotations(_label_path_for(prompt_lbls_dir, filename))
    )
    boxes_pixel = []
    for ann in prompt_annotations:
        bbox = _ann_bbox(ann)
        if not bbox:
            continue
        boxes_pixel.append(
            {
                "class_id": int(ann.get("class_id", req.class_id)),
                "box": [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h],
            }
        )
    if not boxes_pixel:
        return []

    results = _sam_model.predict_batch(
        image_path,
        boxes_pixel,
        prompt_types=req.prompt_types,
        prompt_type=req.prompt_type,
        text_prompt=req.text_prompt,
        names=g.get("names", {}),
    )
    return [
        {
            "class_id": int(item["class_id"]),
            "ann_type": "polygon",
            "points": [float(v) for v in item.get("points", [])],
        }
        for item in results
        if item.get("ok") and item.get("points") and len(item["points"]) >= 6
    ]


def _evaluate_segmentation_method(req: SegmentGoldEvaluateRequest) -> Dict[str, Any]:
    g = _get_group(req.group_id)
    gold_ls = _get_label_set(g, req.gold_label_set_id)
    imgs_dir, gold_lbls_dir = _get_dirs(g, gold_ls, req.subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")

    files = _slice_files_for_request(
        _list_image_files(imgs_dir), req.start_index, req.end_index, req.max_images
    )
    if not files:
        raise HTTPException(400, "Invalid class configuration")

    prompt_lbls_dir = None
    if req.method == "sam3":
        if not req.prompt_label_set_id:
            raise HTTPException(400, "SAM3 requires prompt_label_set_id")
        prompt_ls = _get_label_set(g, req.prompt_label_set_id)
        _, prompt_lbls_dir = _get_dirs(g, prompt_ls, req.subset)

    image_results = []
    agg_precisions, agg_recalls, agg_f1s, agg_ious, agg_dices = [], [], [], [], []
    total_pred = total_gold = total_matched = 0

    for fp in files:
        filename = os.path.basename(fp)
        gold_annotations = _annotation_payload(
            _read_annotations(_label_path_for(gold_lbls_dir, filename))
        )
        if not gold_annotations:
            continue
        if req.method == "cellpose":
            from labeling_tool.cellpose_utils import run_cellpose_to_polygons

            pred_annotations = run_cellpose_to_polygons(
                img_path=fp,
                diameters=req.diameters,
                gpu=req.gpu,
                class_id=req.class_id,
                min_area=req.min_area,
            )
        elif req.method == "sam3":
            pred_annotations = _build_sam_eval_predictions(
                g, imgs_dir, prompt_lbls_dir, filename, req
            )
        else:
            raise HTTPException(400, f"Unsupported segmentation method: {req.method}")

        img = _imread_exif(fp)
        if img is None:
            continue
        metrics = _evaluate_segmentation_annotations(
            pred_annotations, gold_annotations, img.shape[:2]
        )
        total_pred += metrics["pred_count"]
        total_gold += metrics["gold_count"]
        total_matched += metrics["matched"]
        agg_precisions.append(metrics["precision"])
        agg_recalls.append(metrics["recall"])
        agg_f1s.append(metrics["f1"])
        if metrics["matches"]:
            agg_ious.extend(m["iou"] for m in metrics["matches"])
            agg_dices.extend(m["dice"] for m in metrics["matches"])
        image_results.append(
            {
                "filename": filename,
                **{k: v for k, v in metrics.items() if k != "matches"},
            }
        )

    if not image_results:
        raise HTTPException(400, "Invalid request")

    return {
        "method": req.method,
        "subset": req.subset,
        "gold_label_set_id": req.gold_label_set_id,
        "prompt_label_set_id": req.prompt_label_set_id,
        "image_count": len(image_results),
        "metrics": {
            "pred_count": total_pred,
            "gold_count": total_gold,
            "matched": total_matched,
            "precision": float(np.mean(agg_precisions)) if agg_precisions else 0.0,
            "recall": float(np.mean(agg_recalls)) if agg_recalls else 0.0,
            "f1": float(np.mean(agg_f1s)) if agg_f1s else 0.0,
            "mean_matched_iou": float(np.mean(agg_ious)) if agg_ious else 0.0,
            "mean_matched_dice": float(np.mean(agg_dices)) if agg_dices else 0.0,
        },
        "sample_images": image_results[:50],
    }


@app.post("/api/hybrid/predict_current")
def hybrid_predict_current(req: HybridPredictRequest):
    g = _get_group(req.group_id)
    ls = _get_label_set(g, req.label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, req.subset)
    image_path = _image_path_for(imgs_dir, req.filename)
    if not os.path.exists(image_path):
        raise HTTPException(404, "Image not found")

    query_annotations = _annotation_payload(
        req.query_annotations
        or _read_annotations(_label_path_for(lbls_dir, req.filename))
    )
    if not query_annotations:
        raise HTTPException(400, "Invalid evaluation data")
    if not req.support_items:
        raise HTTPException(400, "Missing support items")

    support_items = _resolve_support_items(g, ls, req.support_items)
    try:
        classifier = hybrid_prepare_classifier(
            support_items=support_items,
            class_names=_class_name_map(g),
            device=req.device,
            enable_size_refiner=req.enable_size_refiner,
            size_refiner_config=_build_size_refiner_config(req),
            prompt_template=req.prompt_template or None,
            text_prompt_names=_parse_text_prompt_names(req.text_prompt_names),
        )
        result = hybrid_predict_annotations(
            classifier=classifier,
            image_path=image_path,
            annotations=query_annotations,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Hybrid prediction failed")
        raise HTTPException(500, f"Hybrid classification error: {e}")

    return {
        "filename": os.path.basename(req.filename),
        "subset": req.subset,
        "device": classifier.device,
        **result,
    }


@app.post("/api/hybrid/evaluate_subset")
def hybrid_evaluate_subset(req: HybridEvaluateRequest):
    g = _get_group(req.group_id)
    ls = _get_label_set(g, req.label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, req.subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")
    if not req.support_items:
        raise HTTPException(400, "Missing support items")

    files = _list_image_files(imgs_dir)
    if req.start_index is not None or req.end_index is not None:
        if not files:
            raise HTTPException(400, "Invalid class configuration")
        start_index = 0 if req.start_index is None else max(0, int(req.start_index))
        end_index = (
            len(files) - 1
            if req.end_index is None
            else min(len(files) - 1, int(req.end_index))
        )
        if end_index < start_index:
            raise HTTPException(400, "Invalid parameters")
        files = files[start_index : end_index + 1]
    if req.max_images is not None and req.max_images > 0:
        files = files[: int(req.max_images)]

    support_items = _resolve_support_items(g, ls, req.support_items)
    support_keys = {item["support_key"] for item in support_items}
    dataset_items = []
    for fp in files:
        filename = os.path.basename(fp)
        annotations = _annotation_payload(
            _read_annotations(_label_path_for(lbls_dir, filename))
        )
        annotations = [
            ann
            for ann in annotations
            if _make_support_key(filename, req.subset, ann) not in support_keys
        ]
        if not annotations:
            continue
        dataset_items.append(
            {
                "filename": filename,
                "image_path": fp,
                "annotations": annotations,
            }
        )

    if not dataset_items:
        raise HTTPException(400, "Invalid request")

    try:
        classifier = hybrid_prepare_classifier(
            support_items=support_items,
            class_names=_class_name_map(g),
            device=req.device,
            enable_size_refiner=req.enable_size_refiner,
            size_refiner_config=_build_size_refiner_config(req),
            prompt_template=req.prompt_template or None,
            text_prompt_names=_parse_text_prompt_names(req.text_prompt_names),
        )
        result = hybrid_evaluate_dataset(
            classifier=classifier,
            dataset_items=dataset_items,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Hybrid evaluation failed")
        raise HTTPException(500, f"Hybrid classification error: {e}")

    return {
        "subset": req.subset,
        "device": classifier.device,
        **result,
    }


@app.post("/api/hybrid/evaluate_gold")
def hybrid_evaluate_gold(req: HybridGoldEvaluateRequest):
    g = _get_group(req.group_id)
    support_ls = _get_label_set(g, req.label_set_id)
    gold_ls = _get_label_set(g, req.gold_label_set_id)
    imgs_dir, _ = _get_dirs(g, support_ls, req.subset)
    _, gold_lbls_dir = _get_dirs(g, gold_ls, req.subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")
    if not req.support_items:
        raise HTTPException(400, "Missing support items")

    files = _slice_files_for_request(
        _list_image_files(imgs_dir), req.start_index, req.end_index, req.max_images
    )
    support_items = _resolve_support_items(g, support_ls, req.support_items)
    support_keys = (
        {item["support_key"] for item in support_items}
        if req.gold_label_set_id == req.label_set_id
        else set()
    )
    dataset_items = []
    for fp in files:
        filename = os.path.basename(fp)
        annotations = _annotation_payload(
            _read_annotations(_label_path_for(gold_lbls_dir, filename))
        )
        if support_keys:
            annotations = [
                ann
                for ann in annotations
                if _make_support_key(filename, req.subset, ann) not in support_keys
            ]
        if not annotations:
            continue
        dataset_items.append(
            {
                "filename": filename,
                "image_path": fp,
                "annotations": annotations,
            }
        )

    if not dataset_items:
        raise HTTPException(400, "Invalid request")

    try:
        classifier = hybrid_prepare_classifier(
            support_items=support_items,
            class_names=_class_name_map(g),
            device=req.device,
            enable_size_refiner=req.enable_size_refiner,
            size_refiner_config=_build_size_refiner_config(req),
            prompt_template=req.prompt_template or None,
            text_prompt_names=_parse_text_prompt_names(req.text_prompt_names),
        )
        result = hybrid_evaluate_dataset(
            classifier=classifier,
            dataset_items=dataset_items,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Hybrid gold evaluation failed")
        raise HTTPException(500, f"Hybrid batch error: {e}")

    return {
        "subset": req.subset,
        "gold_label_set_id": req.gold_label_set_id,
        "device": classifier.device,
        **result,
    }


@app.post("/api/segment/evaluate_gold")
def segment_evaluate_gold(req: SegmentGoldEvaluateRequest):
    try:
        return _evaluate_segmentation_method(req)
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(503, f"Model not available: {e}")
    except Exception as e:
        logging.exception("Segmentation gold evaluation failed")
        raise HTTPException(500, f"Model not available: {e}")


@app.post("/api/evaluate/simple_segment")
def simple_segment_evaluation(req: SimpleSegmentEvalRequest):
    """Evaluation endpoint."""
    if not os.path.isdir(req.pred_labels_dir):
        raise HTTPException(400, f"Pred dir not found: {req.pred_labels_dir}")
    if not os.path.isdir(req.gold_labels_dir):
        raise HTTPException(400, f"Gold dir not found: {req.gold_labels_dir}")
    if not os.path.isdir(req.images_dir):
        raise HTTPException(400, f"Images dir not found: {req.images_dir}")

    img_exts = {".jpg", ".jpeg", ".png"}
    image_files = [
        f
        for f in os.listdir(req.images_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ]

    if req.max_images and req.max_images > 0:
        image_files = image_files[: req.max_images]

    total_pred = total_gold = total_matched = 0
    agg_precisions, agg_recalls, agg_f1s, agg_ious, agg_dices = [], [], [], [], []
    image_results = []

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        pred_path = os.path.join(req.pred_labels_dir, stem + ".txt")
        gold_path = os.path.join(req.gold_labels_dir, stem + ".txt")
        img_path = os.path.join(req.images_dir, img_file)


        pred_anns = _read_annotations(pred_path) if os.path.exists(pred_path) else []
        gold_anns = _read_annotations(gold_path) if os.path.exists(gold_path) else []

        if not gold_anns:
            continue

        img = _imread_exif(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        metrics = _evaluate_segmentation_annotations(pred_anns, gold_anns, (h, w))

        total_pred += metrics["pred_count"]
        total_gold += metrics["gold_count"]
        total_matched += metrics["matched"]
        agg_precisions.append(metrics["precision"])
        agg_recalls.append(metrics["recall"])
        agg_f1s.append(metrics["f1"])

        if metrics["matches"]:
            agg_ious.extend(m["iou"] for m in metrics["matches"])
            agg_dices.extend(m["dice"] for m in metrics["matches"])

        image_results.append(
            {
                "filename": img_file,
                "pred_count": metrics["pred_count"],
                "gold_count": metrics["gold_count"],
                "matched": metrics["matched"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    if not image_results:
        raise HTTPException(400, "Invalid evaluation request")

    return {
        "pred_labels_dir": req.pred_labels_dir,
        "gold_labels_dir": req.gold_labels_dir,
        "image_count": len(image_results),
        "metrics": {
            "pred_count": total_pred,
            "gold_count": total_gold,
            "matched": total_matched,
            "precision": float(np.mean(agg_precisions)) if agg_precisions else 0.0,
            "recall": float(np.mean(agg_recalls)) if agg_recalls else 0.0,
            "f1": float(np.mean(agg_f1s)) if agg_f1s else 0.0,
            "mean_matched_iou": float(np.mean(agg_ious)) if agg_ious else 0.0,
            "mean_matched_dice": float(np.mean(agg_dices)) if agg_dices else 0.0,
        },
        "sample_images": image_results[:20],
    }


@app.post("/api/evaluate/simple_classify")
def simple_classify_evaluation(req: SimpleClassifyEvalRequest):
    """Evaluation endpoint."""
    if not os.path.isdir(req.pred_labels_dir):
        raise HTTPException(400, f"Pred dir not found: {req.pred_labels_dir}")
    if not os.path.isdir(req.gold_labels_dir):
        raise HTTPException(400, f"Gold dir not found: {req.gold_labels_dir}")
    if not os.path.isdir(req.images_dir):
        raise HTTPException(400, f"Images dir not found: {req.images_dir}")

    img_exts = {".jpg", ".jpeg", ".png"}
    image_files = [
        f
        for f in os.listdir(req.images_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    ]

    if req.max_images and req.max_images > 0:
        image_files = image_files[: req.max_images]


    class_correct: Dict[int, int] = {}
    class_pred: Dict[int, int] = {}
    class_gold: Dict[int, int] = {}
    total_correct = total_pred = total_gold = 0
    image_results = []

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        pred_path = os.path.join(req.pred_labels_dir, stem + ".txt")
        gold_path = os.path.join(req.gold_labels_dir, stem + ".txt")

        pred_anns = _read_annotations(pred_path) if os.path.exists(pred_path) else []
        gold_anns = _read_annotations(gold_path) if os.path.exists(gold_path) else []

        if not gold_anns:
            continue

        img_correct = 0
        for gold in gold_anns:
            gold_cid = gold["class_id"]
            class_gold[gold_cid] = class_gold.get(gold_cid, 0) + 1
            total_gold += 1

            for pred in pred_anns:
                if pred["class_id"] == gold_cid:
                    img_correct += 1
                    total_correct += 1
                    class_correct[gold_cid] = class_correct.get(gold_cid, 0) + 1
                    break

        for pred in pred_anns:
            pred_cid = pred["class_id"]
            class_pred[pred_cid] = class_pred.get(pred_cid, 0) + 1
            total_pred += 1

        image_results.append(
            {
                "filename": img_file,
                "correct": img_correct,
                "pred_count": len(pred_anns),
                "gold_count": len(gold_anns),
            }
        )

    if not image_results:
        raise HTTPException(400, "Invalid evaluation request")


    accuracy = total_correct / total_gold if total_gold > 0 else 0.0
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    class_metrics = []
    all_classes = set(class_gold.keys()) | set(class_pred.keys())
    for cid in sorted(all_classes):
        c_correct = class_correct.get(cid, 0)
        c_pred = class_pred.get(cid, 0)
        c_gold = class_gold.get(cid, 0)

        c_precision = c_correct / c_pred if c_pred > 0 else 0.0
        c_recall = c_correct / c_gold if c_gold > 0 else 0.0
        c_f1 = (
            2 * c_precision * c_recall / (c_precision + c_recall)
            if (c_precision + c_recall) > 0
            else 0.0
        )

        cname = (
            req.class_names.get(cid, f"class_{cid}")
            if req.class_names
            else f"class_{cid}"
        )
        class_metrics.append(
            {
                "class_id": cid,
                "class_name": cname,
                "correct": c_correct,
                "pred": c_pred,
                "gold": c_gold,
                "precision": c_precision,
                "recall": c_recall,
                "f1": c_f1,
            }
        )

    return {
        "pred_labels_dir": req.pred_labels_dir,
        "gold_labels_dir": req.gold_labels_dir,
        "image_count": len(image_results),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_correct": total_correct,
            "total_pred": total_pred,
            "total_gold": total_gold,
        },
        "class_metrics": class_metrics,
        "sample_images": image_results[:20],
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    global _sam_model
    if _sam_model is None:
        _sam_model = _load_sam_model()
    if _sam_model is None:
        raise HTTPException(503, "SAM3 model not available")

    g = _get_group(req.group_id)
    imgs_dir = g["val_images"] if req.subset == "val" else g["train_images"]
    fn = os.path.basename(req.filename)
    img_path = os.path.join(imgs_dir, fn)
    img = _imread_exif(img_path)
    if img is None:
        raise HTTPException(404, "Image not found")
    h, w = img.shape[:2]

    text = req.text_prompt
    if not text and req.class_id is not None:
        text = g["names"].get(req.class_id, "cell")

    box = None
    if req.box and len(req.box) >= 4:
        box = [req.box[0] * w, req.box[1] * h, req.box[2] * w, req.box[3] * h]

    ptype = req.prompt_type
    try:
        if ptype == "13points" and box:
            polygon = _sam_model.predict_13points(img_path, box)
            return {"points": polygon, "strategy": "13points"}
        elif ptype == "box_inst" and box:
            polygon = _sam_model.predict_box_inst(img_path, box)
            return {"points": polygon, "strategy": "box_inst"}
        elif ptype == "text" and text:
            polygons = _sam_model.predict_text(img_path, text)
            return {"points": polygons, "multi": True}
        elif ptype in ("box+text", "grounding") and box:
            polygon = _sam_model.predict(img_path, box, text_prompt=text)
            return {"points": polygon, "strategy": "grounding"}
        elif ptype == "box" and box:
            polygon = _sam_model.predict(img_path, box, text_prompt=text)
            return {"points": polygon, "strategy": "grounding"}
        else:
            raise HTTPException(400, "Invalid request")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/predict_batch")
def predict_batch(req: BatchPredictRequest):
    """Run SAM3 on multiple bboxes ??set image once, iterate boxes."""
    global _sam_model
    if _sam_model is None:
        _sam_model = _load_sam_model()
    if _sam_model is None:
        raise HTTPException(503, "SAM3 model not available")

    g = _get_group(req.group_id)
    imgs_dir = g["val_images"] if req.subset == "val" else g["train_images"]
    fn = os.path.basename(req.filename)
    img_path = os.path.join(imgs_dir, fn)
    img = _imread_exif(img_path)
    if img is None:
        raise HTTPException(404, "Image not found")
    h, w = img.shape[:2]

    boxes_pixel = []
    for item in req.boxes:
        cid = item.get("class_id", 0)
        nb = item.get("box", [])
        if len(nb) < 4:
            continue
        boxes_pixel.append(
            {"class_id": cid, "box": [nb[0] * w, nb[1] * h, nb[2] * w, nb[3] * h]}
        )

    try:
        results = _sam_model.predict_batch(
            img_path,
            boxes_pixel,
            prompt_types=req.prompt_types,
            prompt_type=req.prompt_type,
            text_prompt=req.text_prompt,
            names=g.get("names", {}),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    return {
        "results": results,
        "total": len(results),
        "success": sum(1 for r in results if r.get("ok")),
        "strategies_used": list(
            {r.get("strategy", "?") for r in results if r.get("ok")}
        ),
    }


_sam_model = None
_sam_logger = logging.getLogger("sam3_loader")


def _load_sam_model():
    try:
        from labeling_tool.model import SAM3Model

        ckpt_path = sam3_checkpoint_path()
        if ckpt_path is None:
            _sam_logger.warning(
                "SAM3 checkpoint not found. Place sam3.pt under labeling_tool/weights/ "
                "or project assets/sam3.pt"
            )
            return None
        ckpt = str(ckpt_path)
        _sam_logger.info(f"Loading SAM3 model from {ckpt}")
        m = SAM3Model(ckpt)
        if m.processor:
            _sam_logger.info("SAM3 model loaded successfully")
            return m
        _sam_logger.warning("SAM3 model loaded but processor is None")
        return None
    except Exception as e:
        _sam_logger.error(f"Failed to load SAM3 model: {e}", exc_info=True)
        return None


@app.post("/api/cellpose_segment")
def cellpose_segment(req: CellposeRequest):
    """Run CellposeSAM on an image, return polygon annotations."""
    g = _get_group(req.group_id)
    imgs_dir = g["val_images"] if req.subset == "val" else g["train_images"]
    fn = os.path.basename(req.filename)
    img_path = os.path.join(imgs_dir, fn)
    if not os.path.exists(img_path):
        raise HTTPException(404, "Image not found")

    try:
        from labeling_tool.cellpose_utils import run_cellpose_to_polygons

        annotations = run_cellpose_to_polygons(
            img_path=img_path,
            diameters=req.diameters,
            gpu=req.gpu,
            class_id=req.class_id,
            min_area=req.min_area,
        )
        return {"annotations": annotations, "count": len(annotations)}
    except ImportError as e:
        raise HTTPException(503, f"CellposeSAM not available: {e}")
    except Exception as e:
        raise HTTPException(500, f"CellposeSAM error: {e}")


# ?? Cellpose Batch Job Infrastructure ??????????????????????????

CELLPOSE_BATCH_JOBS: Dict[str, dict] = {}
CELLPOSE_BATCH_JOBS_LOCK = threading.Lock()

_BATCH_JOB_TTL = 3600  # auto-cleanup completed jobs after 1 hour

def _cleanup_old_batch_jobs():
    now = time.time()
    terminal = {"saved", "discarded", "completed", "cancelled", "failed"}
    with CELLPOSE_BATCH_JOBS_LOCK:
        to_remove = [
            jid for jid, j in CELLPOSE_BATCH_JOBS.items()
            if j.get("status") in terminal
            and j.get("finished_at")
            and (now - j["finished_at"]) > _BATCH_JOB_TTL
        ]
        for jid in to_remove:
            del CELLPOSE_BATCH_JOBS[jid]


def _create_cellpose_batch_job(req, selected_files) -> str:
    _cleanup_old_batch_jobs()
    import uuid

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "group_id": req.group_id,
        "label_set_id": req.label_set_id,
        "subset": req.subset,
        "total": len(selected_files),
        "processed": 0,
        "committed": 0,
        "current_filename": "",
        "message": "",
        "started_at": time.time(),
        "finished_at": None,
        "stop_requested": False,
        "results": [],
        "pending_outputs": {},
        "config": {
            "diameters": req.diameters,
            "gpu": req.gpu,
            "class_id": req.class_id,
            "min_area": req.min_area,
            "skip_existing": req.skip_existing,
            "overwrite_existing": req.overwrite_existing,
        },
    }
    with CELLPOSE_BATCH_JOBS_LOCK:
        CELLPOSE_BATCH_JOBS[job_id] = job
    return job_id


def _public_batch_job(job: dict) -> dict:
    results = job.get("results", [])
    saved_count = sum(1 for r in results if r.get("status") == "success")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    failed_count = sum(1 for r in results if r.get("status") in ("error", "save_failed"))
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "total": job["total"],
        "processed": job["processed"],
        "committed": job["committed"],
        "saved": saved_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "current_filename": job["current_filename"],
        "message": job["message"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "results": results[-20:],
        "pending_count": len(job.get("pending_outputs", {})),
    }


def _write_annotations(label_path: str, annotations: list):
    """Write annotation list to a YOLO-format .txt file."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    lines = []
    for ann in annotations:
        cid = int(ann.get("class_id", 0))
        pts = ann.get("points", [])
        coords = " ".join(f"{float(v):.6f}" for v in pts)
        lines.append(f"{cid} {coords}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" if lines else "")


def _run_cellpose_batch_job(job_id: str, req, selected_files: list):
    """Background thread: runs CellposeSAM on each image."""
    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"

    try:
        g = _get_group(req.group_id)
        ls = _get_label_set(g, req.label_set_id)
        _, lbls_dir = _get_dirs(g, ls, req.subset)
    except Exception as e:
        logging.error(f"Batch job {job_id} setup failed: {e}")
        with CELLPOSE_BATCH_JOBS_LOCK:
            job = CELLPOSE_BATCH_JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["message"] = f"Setup failed: {e}"
                job["finished_at"] = time.time()
        return

    for i, img_path in enumerate(selected_files):
        with CELLPOSE_BATCH_JOBS_LOCK:
            job = CELLPOSE_BATCH_JOBS.get(job_id)
            if not job or job.get("stop_requested"):
                if job:
                    pending = len(job.get("pending_outputs", {}))
                    if pending > 0:
                        job["status"] = "awaiting_save"
                        job["message"] = f"Cancelled after {i} images. {pending} ready to commit."
                    else:
                        job["status"] = "cancelled"
                        job["message"] = f"Cancelled after {i} images"
                    job["finished_at"] = time.time()
                return

        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        label_path = os.path.join(lbls_dir, base + ".txt")

        with CELLPOSE_BATCH_JOBS_LOCK:
            job["current_filename"] = filename
            job["message"] = f"Processing {i + 1}/{len(selected_files)}: {filename}"

        if req.skip_existing and os.path.isfile(label_path):
            existing = _read_annotations(label_path)
            if existing:
                with CELLPOSE_BATCH_JOBS_LOCK:
                    job["processed"] += 1
                    job["results"].append(
                        {"filename": filename, "status": "skipped", "count": len(existing)}
                    )
                continue

        try:
            from labeling_tool.cellpose_utils import run_cellpose_to_polygons

            annotations = run_cellpose_to_polygons(
                img_path,
                diameters=req.diameters,
                gpu=req.gpu,
                class_id=req.class_id,
                min_area=req.min_area,
            )
            with CELLPOSE_BATCH_JOBS_LOCK:
                job["processed"] += 1
                job["pending_outputs"][filename] = {
                    "label_path": label_path,
                    "annotations": annotations,
                }
                job["results"].append(
                    {"filename": filename, "status": "success", "count": len(annotations)}
                )
        except Exception as e:
            logging.error(f"CellposeSAM batch error on {filename}: {e}")
            with CELLPOSE_BATCH_JOBS_LOCK:
                job["processed"] += 1
                job["results"].append(
                    {"filename": filename, "status": "error", "error": str(e)}
                )

    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(job_id)
        if job and job["status"] in ("running", "cancel_requested"):
            pending = len(job.get("pending_outputs", {}))
            if pending > 0:
                job["status"] = "awaiting_save"
                job["message"] = f"Done. {pending} images ready to commit."
            else:
                job["status"] = "completed"
                job["message"] = "All images processed (none had new results)."
            job["current_filename"] = ""
            job["finished_at"] = time.time()


@app.post("/api/cellpose_batch/start")
def start_cellpose_batch(req: CellposeBatchRequest):
    g = _get_group(req.group_id)
    ls = _get_label_set(g, req.label_set_id)
    imgs_dir, _ = _get_dirs(g, ls, req.subset)
    if not req.skip_existing and not req.overwrite_existing:
        raise HTTPException(400, "Invalid request")
    if ls.get("label_format") == "bbox":
        raise HTTPException(
            400, "Target label set is bbox format. CellposeSAM requires polygon format."
        )
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")

    files = _list_image_files(imgs_dir)
    if not files:
        raise HTTPException(400, "Invalid request")

    start_index = max(0, int(req.start_index))
    end_index = len(files) - 1 if req.end_index is None else int(req.end_index)
    end_index = min(len(files) - 1, end_index)
    if start_index >= len(files):
        raise HTTPException(400, "Invalid class configuration")
    if end_index < start_index:
        raise HTTPException(400, "Invalid evaluation data")

    selected_files = files[start_index : end_index + 1]
    if not selected_files:
        raise HTTPException(400, "No images selected")

    job_id = _create_cellpose_batch_job(req, selected_files)
    thread = threading.Thread(
        target=_run_cellpose_batch_job,
        args=(job_id, req, selected_files),
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job_id,
        "status": "queued",
        "total": len(selected_files),
        "start_index": start_index,
        "end_index": end_index,
        "start_filename": os.path.basename(selected_files[0]),
        "end_filename": os.path.basename(selected_files[-1]),
    }


@app.get("/api/cellpose_batch/status")
def get_cellpose_batch_status(job_id: str):
    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Batch job not found")
        return _public_batch_job(job)


@app.post("/api/cellpose_batch/cancel")
def cancel_cellpose_batch(req: BatchJobActionRequest):
    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(req.job_id)
        if not job:
            raise HTTPException(404, "Batch job not found")
        if job["status"] not in {"queued", "running", "cancel_requested"}:
            raise HTTPException(400, "Invalid class configuration")
        job["stop_requested"] = True
        job["status"] = "cancel_requested"
        job["message"] = "Processing..."
    return {"status": "cancel_requested", "job_id": req.job_id}


@app.post("/api/cellpose_batch/commit")
def commit_cellpose_batch(req: BatchJobActionRequest):
    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(req.job_id)
        if not job:
            raise HTTPException(404, "Batch job not found")
        if job["status"] not in {"awaiting_save", "failed", "saved", "cancelled"}:
            raise HTTPException(400, "Invalid request")
        if job["status"] == "saved":
            return _public_batch_job(job)

        pending_outputs = {
            filename: {
                "label_path": item["label_path"],
                "annotations": item["annotations"],
            }
            for filename, item in job["pending_outputs"].items()
        }
        job["status"] = "saving"
        job["message"] = "Processing..."

    committed = 0
    save_errors = []
    remaining_outputs = {}
    for filename, item in pending_outputs.items():
        try:
            _write_annotations(item["label_path"], item["annotations"])
            committed += 1
        except Exception as e:
            save_errors.append(f"{filename}: {e}")
            remaining_outputs[filename] = item

    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(req.job_id)
        if not job:
            raise HTTPException(404, "Batch job not found")
        job["committed"] += committed
        job["pending_outputs"] = remaining_outputs
        job["finished_at"] = time.time()
        if save_errors:
            job["status"] = "failed"
            job["message"] = f"Save failed for {len(save_errors)} files"
            for err in save_errors[:10]:
                job["results"].append(
                    {
                        "filename": err.split(":", 1)[0],
                        "status": "save_failed",
                        "error": err,
                    }
                )
        else:
            job["status"] = "saved"
            job["message"] = "Processing..."
            job["current_filename"] = ""
    return _public_batch_job(job)


@app.get("/api/cellpose_batch/preview")
def preview_cellpose_batch(job_id: str, filename: str = ""):
    """Return pending annotations for a specific image (or list of all pending filenames)."""
    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Batch job not found")
        pending = job.get("pending_outputs", {})
        if not filename:
            return {
                "filenames": list(pending.keys()),
                "count": len(pending),
                "status": job["status"],
            }
        item = pending.get(filename)
        if not item:
            return {"filename": filename, "annotations": [], "found": False}
        anns = item.get("annotations", [])
        return {
            "filename": filename,
            "found": True,
            "count": len(anns),
            "annotations": [
                {"class_id": a.get("class_id", 0), "points": a.get("points", [])}
                for a in anns
            ],
        }


@app.post("/api/cellpose_batch/discard")
def discard_cellpose_batch(req: BatchJobActionRequest):
    with CELLPOSE_BATCH_JOBS_LOCK:
        job = CELLPOSE_BATCH_JOBS.get(req.job_id)
        if not job:
            raise HTTPException(404, "Batch job not found")
        if job["status"] not in {"awaiting_save", "failed", "discarded", "cancelled"}:
            raise HTTPException(400, "Invalid evaluation data")
        if job["status"] == "discarded":
            return _public_batch_job(job)

        dropped = len(job.get("pending_outputs", {}))
        job["pending_outputs"] = {}
        job["status"] = "discarded"
        job["message"] = "Processing..."
        job["current_filename"] = ""
        job["finished_at"] = time.time()
    return _public_batch_job(job)



# Dataset management API



@app.get("/api/datasets")
def list_user_datasets():
    """List all user-managed datasets from datasets.json."""
    return _load_user_datasets()


@app.get("/api/groups")
def list_groups():
    """List all image groups (same as IMAGE_GROUPS)."""
    _ensure_groups_fresh()
    return IMAGE_GROUPS


def _find_class_yaml(path: str) -> Optional[Dict]:
    """Search for a YAML file with class names in path and parent directories.
    Returns dict with 'names', 'nc', 'yaml_path' or None."""
    search_dirs = [path]
    parent = os.path.dirname(path)
    if parent and parent != path:
        search_dirs.append(parent)
    # Also try grandparent (e.g. path/images/train -> path)
    grandparent = os.path.dirname(parent) if parent else None
    if grandparent and grandparent != parent:
        search_dirs.append(grandparent)

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        # First try data*.yaml, then any *.yaml
        for pattern in ["data*.yaml", "*.yaml", "data*.yml", "*.yml"]:
            for yf in sorted(glob.glob(os.path.join(d, pattern))):
                try:
                    with open(yf, "r", encoding="utf-8", errors="replace") as f:
                        cfg = yaml.safe_load(f)
                    if not cfg or "names" not in cfg:
                        continue
                    raw = cfg["names"]
                    if isinstance(raw, list) and len(raw) > 0:
                        names = {int(i): v for i, v in enumerate(raw)}
                    elif isinstance(raw, dict) and len(raw) > 0:
                        names = {int(k): v for k, v in raw.items()}
                    else:
                        continue
                    return {
                        "names": names,
                        "nc": cfg.get("nc", len(names)),
                        "yaml_path": yf,
                    }
                except Exception:
                    continue
    return None


@app.post("/api/datasets/detect")
def detect_dataset(path: str = Body(..., embed=True)):
    """Detect dataset structure at given path."""
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isdir(path):
        raise HTTPException(400, f"Invalid path: {path}")
    info = _detect_dataset_structure(path)
    info["path"] = path
    # Search for YAML with class names
    yaml_info = _find_class_yaml(path)
    if yaml_info:
        info["names"] = yaml_info["names"]
        info["nc"] = yaml_info["nc"]
        info["yaml_found"] = yaml_info["yaml_path"]
    return info


@app.post("/api/datasets/add")
def add_dataset(
    path: str = Body(...),
    name: str = Body(""),
):
    """Add a dataset from a local directory path, with persistence."""
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isdir(path):
        raise HTTPException(400, f"Invalid path: {path}")

    # Detect structure
    info = _detect_dataset_structure(path)
    if info["structure"] == "unknown":
        raise HTTPException(400, "Invalid request")

    train_images = info["train_images"]
    val_images = info["val_images"]

    # Check for duplicates
    datasets = _load_user_datasets()
    for ds in datasets:
        if os.path.normpath(ds.get("train_images", "")) == train_images:
            # Already exists, just refresh
            global IMAGE_GROUPS
            _mark_groups_dirty()
            return {
                "group_id": train_images,
                "dataset_id": ds["id"],
                "status": "already_exists",
            }

    # Also check if YAML already discovered this ??still add to user datasets for name override
    for g in IMAGE_GROUPS:
        if g["group_id"] == train_images and not g.get("_user_dataset_id"):
            ds_id = str(uuid.uuid4())[:8]
            ds_name = name.strip() or os.path.basename(path)
            entry = {
                "id": ds_id,
                "name": ds_name,
                "path": path,
                "train_images": train_images,
                "val_images": val_images,
                "nc": g.get("nc", 1),
                "names": {str(k): v for k, v in g.get("names", {}).items()},
                "yaml_path": g.get("yaml_path", ""),
                "structure": info["structure"],
                "added_at": datetime.now().isoformat(timespec="seconds"),
            }
            datasets.append(entry)
            _save_user_datasets(datasets)
            _mark_groups_dirty()
            return {"group_id": train_images, "dataset_id": ds_id, "status": "added"}

    # Determine names from existing yaml if present
    nc = 1
    names = {"0": "object"}
    yaml_path = ""
    yaml_info = _find_class_yaml(path)
    if yaml_info:
        names = {str(k): v for k, v in yaml_info["names"].items()}
        nc = yaml_info["nc"]
        yaml_path = yaml_info["yaml_path"]

    ds_id = str(uuid.uuid4())[:8]
    ds_name = name.strip() or os.path.basename(path)

    new_ds = {
        "id": ds_id,
        "name": ds_name,
        "path": path,
        "train_images": train_images,
        "val_images": val_images,
        "nc": nc,
        "names": names,
        "yaml_path": yaml_path,
        "structure": info["structure"],
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    datasets.append(new_ds)
    _save_user_datasets(datasets)

    _mark_groups_dirty()

    return {"group_id": train_images, "dataset_id": ds_id, "status": "added"}


@app.get("/api/datasets/file_info")
def dataset_file_info(group_id: str):
    """Return filesystem-level info about a dataset: paths, label sets, file counts."""
    g = _get_group(group_id)
    train_dir = g.get("train_images", "")
    val_dir = g.get("val_images", "")
    train_count = len(_list_image_files(train_dir)) if train_dir else 0
    val_count = len(_list_image_files(val_dir)) if val_dir else 0

    base_parts = train_dir.replace("\\", "/").split("/images/")
    base_path = base_parts[0] if len(base_parts) >= 2 else os.path.dirname(train_dir)

    ls_info = []
    for ls in g.get("label_sets", []):
        train_labels_dir = ls.get("train_labels", "")
        val_labels_dir = ls.get("val_labels", "")
        train_label_count = 0
        val_label_count = 0
        if train_labels_dir and os.path.isdir(train_labels_dir):
            train_label_count = len(
                [f for f in os.listdir(train_labels_dir) if f.endswith(".txt")]
            )
        if val_labels_dir and os.path.isdir(val_labels_dir):
            val_label_count = len(
                [f for f in os.listdir(val_labels_dir) if f.endswith(".txt")]
            )
        ls_info.append(
            {
                "set_id": ls["set_id"],
                "set_name": ls.get("set_name", ls["set_id"]),
                "format": ls.get("label_format", "unknown"),
                "train_labels": train_label_count,
                "val_labels": val_label_count,
                "train_labels_dir": train_labels_dir,
                "val_labels_dir": val_labels_dir,
            }
        )

    return {
        "group_id": group_id,
        "base_path": base_path,
        "train_images_dir": train_dir,
        "val_images_dir": val_dir,
        "train_images": train_count,
        "val_images": val_count,
        "label_sets": ls_info,
    }


@app.post("/api/datasets/init_annotations")
def init_annotations(
    group_id: str = Body(...),
    label_set_id: str = Body(...),
    subset: str = Body("train"),
):
    """Create empty .txt annotation files for all images that don't have one yet."""
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)

    created = 0
    for sub_key in ["train", "val"]:
        if sub_key == "train":
            imgs_dir = g.get("train_images", "")
            lbls_dir = ls.get("train_labels", "")
        else:
            imgs_dir = g.get("val_images", "")
            lbls_dir = ls.get("val_labels", "")
        if not imgs_dir or not lbls_dir:
            continue
        os.makedirs(lbls_dir, exist_ok=True)
        images = _list_image_files(imgs_dir)
        for img_path in images:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(lbls_dir, base + ".txt")
            if not os.path.exists(label_path):
                with open(label_path, "w") as f:
                    pass
                created += 1

    log_action(
        "init_annotations",
        "annotation",
        details={
            "group_id": group_id,
            "label_set_id": label_set_id,
            "created": created,
        },
    )
    return {"created": created, "group_id": group_id, "label_set_id": label_set_id}


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Remove a user-added dataset (does not delete files)."""
    datasets = _load_user_datasets()
    before = len(datasets)
    datasets = [ds for ds in datasets if ds.get("id") != dataset_id]
    if len(datasets) == before:
        raise HTTPException(404, f"Dataset not found: {dataset_id}")
    _save_user_datasets(datasets)

    global IMAGE_GROUPS
    _mark_groups_dirty()
    return {"status": "deleted", "remaining": len(datasets)}


@app.put("/api/datasets/{dataset_id}")
def update_dataset(
    dataset_id: str,
    name: str = Body(None),
    names: Dict[str, str] = Body(None),
    nc: int = Body(None),
):
    """Update a user-added dataset's name or class names."""
    datasets = _load_user_datasets()
    target = None
    for ds in datasets:
        if ds.get("id") == dataset_id:
            target = ds
            break
    if not target:
        raise HTTPException(404, f"Dataset not found: {dataset_id}")

    if name is not None:
        target["name"] = name.strip()
    if names is not None:
        target["names"] = names
    if nc is not None:
        target["nc"] = nc

    _save_user_datasets(datasets)

    global IMAGE_GROUPS
    _mark_groups_dirty()
    return {"status": "updated"}


@app.post("/api/datasets/exclude")
def exclude_dataset(group_id: str = Body(..., embed=True)):
    """Exclude a dataset (YAML or user-added) from the active list. Files are not deleted."""
    excluded = _load_excluded_groups()
    gid = os.path.normpath(group_id)
    if gid not in excluded:
        excluded.append(gid)
        _save_excluded_groups(excluded)
    global IMAGE_GROUPS
    _mark_groups_dirty()
    return {"status": "excluded", "group_id": gid}


@app.post("/api/datasets/restore")
def restore_dataset(group_id: str = Body(..., embed=True)):
    """Remove a dataset from the exclusion list, making it visible again."""
    excluded = _load_excluded_groups()
    gid = os.path.normpath(group_id)
    excluded = [e for e in excluded if e != gid]
    _save_excluded_groups(excluded)
    global IMAGE_GROUPS
    _mark_groups_dirty()
    return {"status": "restored", "group_id": gid}


@app.get("/api/datasets/excluded")
def list_excluded():
    """List all excluded group IDs."""
    return {"excluded_groups": _load_excluded_groups()}


@app.post("/api/datasets/browse")
def browse_directory(path: str = Body("", embed=True)):
    """List subdirectories at the given path for browsing."""
    if not path:
        # Return drive roots on Windows, or common paths
        roots = []
        if os.name == "nt":
            import string

            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.isdir(drive):
                    roots.append(
                        {
                            "name": drive,
                            "path": drive,
                            "has_children": True,
                            "img_count": 0,
                        }
                    )
        else:
            for p in ["/", os.path.expanduser("~"), PROJECT_ROOT]:
                if os.path.isdir(p):
                    roots.append(
                        {"name": p, "path": p, "has_children": True, "img_count": 0}
                    )
        return {"current": "", "items": roots}

    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isdir(path):
        raise HTTPException(400, f"Invalid path: {path}")

    items = []
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return {"current": path, "items": []}

    for entry in entries:
        if entry.startswith(".") or entry in ("__pycache__", "node_modules", ".git"):
            continue
        full = os.path.join(path, entry)
        if not os.path.isdir(full):
            continue
        try:
            has_children = any(
                os.path.isdir(os.path.join(full, sub))
                for sub in os.listdir(full)
                if not sub.startswith(".")
            )
        except PermissionError:
            has_children = False
        img_count = _count_images(full)
        items.append(
            {
                "name": entry,
                "path": full,
                "has_children": has_children,
                "img_count": img_count,
            }
        )
        if len(items) >= 100:
            break

    return {"current": path, "parent": os.path.dirname(path), "items": items}


@app.post("/api/datasets/resolve_paths")
def resolve_dataset_paths(
    group_id: str = Body(...),
    gold_label_set_id: str = Body(""),
    pred_label_set_id: str = Body(""),
    subset: str = Body("train"),
):
    """Resolve dataset paths for evaluation: images dir + label dirs."""
    g = _get_group(group_id)
    result = {}
    imgs_dir = g.get("val_images", "") if subset == "val" else g.get("train_images", "")
    result["images_path"] = imgs_dir or ""
    for key, ls_id in [("gold_path", gold_label_set_id), ("pred_path", pred_label_set_id)]:
        if ls_id:
            ls = _get_label_set(g, ls_id)
            _, lbls_dir = _get_dirs(g, ls, subset)
            result[key] = lbls_dir
        else:
            result[key] = ""
    return result


@app.get("/api/scan_label_dirs")
def scan_label_dirs(group_id: str, query: str = ""):
    """Scan filesystem for label-like directories under the current group."""
    g = _get_group(group_id)
    train_parts = g["train_images"].split("/images/")
    if len(train_parts) != 2:
        return []
    parent = train_parts[0]
    sub = train_parts[1]
    val_parts = g["val_images"].split("/images/")
    val_sub = val_parts[1] if len(val_parts) == 2 else "val"

    existing_ids = {ls["set_id"] for ls in g["label_sets"]}
    results = []
    q = query.lower()

    skip = {"images", ".git", "__pycache__", ".venv", ".venv_yolo", "assets"}
    if os.path.isdir(parent):
        for entry in sorted(os.listdir(parent)):
            if entry in skip or entry.startswith("."):
                continue
            full = os.path.join(parent, entry)
            if not os.path.isdir(full):
                continue
            if q and q not in entry.lower():
                continue
            train_dir = os.path.join(full, sub)
            val_dir = os.path.join(full, val_sub)
            has_sub = os.path.isdir(train_dir) or os.path.isdir(val_dir)
            has_txt_root = any(
                f.endswith(".txt")
                for f in os.listdir(full)
                if os.path.isfile(os.path.join(full, f))
            )
            if not has_sub and not has_txt_root:
                if not q:
                    continue
            if entry.startswith("labels_"):
                tag = entry[len("labels_") :]
            elif entry == "labels":
                tag = "default"
            else:
                tag = entry
            n_train = (
                len(glob.glob(os.path.join(train_dir, "*.txt")))
                if os.path.isdir(train_dir)
                else 0
            )
            n_val = (
                len(glob.glob(os.path.join(val_dir, "*.txt")))
                if os.path.isdir(val_dir)
                else 0
            )
            if not has_sub and has_txt_root:
                n_train = len([f for f in os.listdir(full) if f.endswith(".txt")])
            results.append(
                {
                    "dir_name": entry,
                    "set_id": tag,
                    "train_dir": train_dir if has_sub else full,
                    "val_dir": val_dir if has_sub else full,
                    "n_train": n_train,
                    "n_val": n_val,
                    "already_loaded": tag in existing_ids,
                }
            )
    return results


@app.get("/api/scan_image_dirs")
def scan_image_dirs(query: str = ""):
    """Scan project for directories containing images."""
    results = []
    q = query.lower()
    existing_ids = {g["group_id"] for g in IMAGE_GROUPS}

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d != "__pycache__"
            and d != "node_modules"
            and d != ".venv"
            and d != ".venv_yolo"
            and d != ".git"
            and d != "sam3"
            and d != "assets"
        ]
        rel = os.path.relpath(root, PROJECT_ROOT)
        if q and q not in rel.lower():
            continue
        img_count = sum(
            1 for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        if img_count == 0:
            continue
        yaml_nearby = (
            any(
                f.startswith("data") and f.endswith(".yaml")
                for f in os.listdir(os.path.dirname(root))
                if os.path.isfile(os.path.join(os.path.dirname(root), f))
            )
            if root != PROJECT_ROOT
            else False
        )
        results.append(
            {
                "path": root,
                "rel_path": rel,
                "img_count": img_count,
                "already_loaded": root in existing_ids,
                "has_yaml": yaml_nearby,
            }
        )
        if len(results) >= 50:
            break
    return results


@app.post("/api/load_image_dir")
def load_image_dir(path: str = Body(..., embed=True)):
    """Add an image directory as a selectable group ??now with persistence."""
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isdir(path):
        raise HTTPException(400, "Invalid request")
    # Check if already loaded
    for g in IMAGE_GROUPS:
        if g["group_id"] == path:
            return {"group_id": path}
    # Delegate to the persistent add_dataset logic
    result = add_dataset(path=path, name="")
    return {"group_id": result["group_id"]}


@app.post("/api/add_label_set")
def add_label_set(group_id: str = Body(...), dir_name: str = Body(...)):
    """Add a found directory as a label set, creating subdirs if needed.

    Supports both YOLO-style (images/train, images/val) and flat directory structures.
    """
    g = _get_group(group_id)
    train_img = g["train_images"].replace("\\", "/")
    val_img = (g.get("val_images") or "").replace("\\", "/")

    train_parts = train_img.split("/images/")
    if len(train_parts) == 2:
        parent = train_parts[0]
        sub = train_parts[1]
        val_parts = val_img.split("/images/") if val_img else []
        val_sub = val_parts[1] if len(val_parts) == 2 else "val"
        tl = os.path.join(parent, dir_name, sub)
        vl = os.path.join(parent, dir_name, val_sub)
    else:
        parent = os.path.dirname(train_img)
        tl = os.path.join(parent, dir_name)
        vl = ""

    os.makedirs(tl, exist_ok=True)
    created_paths = [tl]
    if vl:
        os.makedirs(vl, exist_ok=True)
        created_paths.append(vl)

    global IMAGE_GROUPS
    _mark_groups_dirty()

    tag = dir_name.replace("labels_", "").replace("labels", "") or "default"
    log_action(
        "label_set_created",
        "annotation",
        details={"group_id": group_id, "dir_name": dir_name, "paths": created_paths},
    )
    return {"set_id": tag, "set_name": tag, "paths": created_paths}


@app.post("/api/generate_bbox_set")
def generate_bbox_set(
    group_id: str = Body(...),
    source_set_id: str = Body(...),
    target_name: str = Body("auto_bbox"),
):
    """Generate a bbox label set from a polygon label set.
    Each polygon is converted to its bounding box [cx, cy, w, h] format.
    """
    g = _get_group(group_id)
    src = _get_label_set(g, source_set_id)

    train_parts = g["train_images"].split("/images/")
    if len(train_parts) != 2:
        raise HTTPException(400, "Invalid evaluation request")
    parent = train_parts[0]
    sub = train_parts[1]
    val_parts = g["val_images"].split("/images/")
    val_sub = val_parts[1] if len(val_parts) == 2 else "val"

    safe_name = target_name.strip().replace(" ", "_").replace("/", "_")
    if not safe_name:
        safe_name = f"{source_set_id}_bbox"
    folder = f"labels_{safe_name}" if not safe_name.startswith("labels") else safe_name

    total_files = 0
    total_boxes = 0
    for subset_key, src_key in [("train", "train_labels"), ("val", "val_labels")]:
        src_dir = src[src_key]
        tgt_dir = os.path.join(
            parent, folder, sub if subset_key == "train" else val_sub
        )
        os.makedirs(tgt_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            continue
        for txt_file in sorted(os.listdir(src_dir)):
            if not txt_file.endswith(".txt"):
                continue
            src_path = os.path.join(src_dir, txt_file)
            anns = _read_annotations(src_path)
            if not anns:
                continue
            bbox_lines = []
            for ann in anns:
                pts = ann["points"]
                if len(pts) < 4:
                    continue
                xs = [pts[i] for i in range(0, len(pts), 2)]
                ys = [pts[i] for i in range(1, len(pts), 2)]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min
                bbox_lines.append(
                    f"{ann['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                total_boxes += 1
            tgt_path = os.path.join(tgt_dir, txt_file)
            with open(tgt_path, "w") as f:
                f.write("\n".join(bbox_lines) + "\n")
            total_files += 1

    global IMAGE_GROUPS
    _mark_groups_dirty()

    tag = folder.replace("labels_", "")
    return {"set_id": tag, "set_name": tag, "files": total_files, "boxes": total_boxes}


@app.post("/api/refresh")
def refresh():
    global IMAGE_GROUPS
    _mark_groups_dirty()
    return {"count": len(IMAGE_GROUPS)}



# Dataset Statistics & Comparison



@app.get("/api/datasets/stats")
def dataset_stats(group_id: str, label_set_id: str = "default", subset: str = "train"):
    """Compute annotation statistics for a dataset."""
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")
    ls = next((l for l in group["label_sets"] if l["set_id"] == label_set_id), None)
    if not ls:
        ls = group["label_sets"][0] if group["label_sets"] else None
    if not ls:
        raise HTTPException(404, "Label set not found")

    img_dir = group["train_images"] if subset == "train" else group["val_images"]
    lbl_dir = (
        ls.get("train_labels", "") if subset == "train" else ls.get("val_labels", "")
    )

    # Count images
    img_exts = {".jpg", ".jpeg", ".png"}
    images = (
        [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts]
        if os.path.isdir(img_dir)
        else []
    )
    total_images = len(images)

    # Scan labels
    names = group.get("names", {})
    class_dist = {}
    labeled_count = 0
    total_anns = 0
    areas = []
    tiny_count = 0
    TINY_THRESHOLD = 0.0005  # normalized area

    for img_f in images:
        stem = os.path.splitext(img_f)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
        if not lbl_path or not os.path.exists(lbl_path):
            continue
        anns = _read_annotations(lbl_path)
        if anns:
            labeled_count += 1
        for a in anns:
            total_anns += 1
            cid = a["class_id"]
            cname = names.get(cid, names.get(str(cid), f"class_{cid}"))
            class_dist[cname] = class_dist.get(cname, 0) + 1
            # Compute area
            pts = a["points"]
            if a["ann_type"] == "bbox" and len(pts) >= 4:
                area = pts[2] * pts[3]  # w * h (normalized)
            elif len(pts) >= 6:
                # Shoelace formula for polygon area
                xs = pts[0::2]
                ys = pts[1::2]
                n = len(xs)
                area = (
                    abs(
                        sum(
                            xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                            for i in range(n)
                        )
                    )
                    / 2.0
                )
            else:
                area = 0
            areas.append(area)
            if area < TINY_THRESHOLD and area > 0:
                tiny_count += 1

    # Quality warnings
    warnings = []
    if tiny_count > 0:
        warnings.append(
            {
                "type": "tiny_annotation",
                "count": tiny_count,
                "threshold": TINY_THRESHOLD,
                "message": f"{tiny_count} tiny annotations (< {TINY_THRESHOLD})",
            }
        )
    if class_dist:
        max_c = max(class_dist.values())
        min_c = min(class_dist.values())
        min_name = [k for k, v in class_dist.items() if v == min_c][0]
        if max_c > 0 and min_c > 0 and max_c / min_c > 5:
            warnings.append(
                {
                    "type": "class_imbalance",
                    "ratio": round(max_c / min_c, 1),
                    "minority": min_name,
                    "message": f"Class imbalance: {min_name} has {min_c} (max {max_c}, ratio {max_c / min_c:.1f}:1)",
                }
            )
    unlabeled = total_images - labeled_count
    if unlabeled > 0:
        warnings.append(
            {
                "type": "unlabeled_images",
                "count": unlabeled,
                "message": f"{unlabeled} images missing labels",
            }
        )

    return {
        "group_id": group_id,
        "group_name": group.get("group_name", ""),
        "subset": subset,
        "total_images": total_images,
        "labeled_images": labeled_count,
        "label_progress": round(labeled_count / total_images, 4)
        if total_images > 0
        else 0,
        "total_annotations": total_anns,
        "avg_annotations_per_image": round(total_anns / labeled_count, 2)
        if labeled_count > 0
        else 0,
        "class_distribution": class_dist,
        "bbox_size_stats": {
            "min_area": round(min(areas), 6) if areas else 0,
            "max_area": round(max(areas), 6) if areas else 0,
            "mean_area": round(sum(areas) / len(areas), 6) if areas else 0,
        },
        "quality_warnings": warnings,
    }


@app.post("/api/datasets/stats_all")
def dataset_stats_all():
    """Compute summary statistics for ALL datasets."""
    results = []
    for g in IMAGE_GROUPS:
        label_sets = g["label_sets"]
        if not label_sets:
            results.append(
                {
                    "group_id": g["group_id"],
                    "group_name": g.get("group_name", ""),
                    "nc": g.get("nc", 0),
                    "train_count": g.get("train_count", 0),
                    "val_count": g.get("val_count", 0),
                    "total_annotations": 0,
                    "label_progress": 0,
                    "class_distribution": {},
                    "labeled_images": 0,
                    "user_dataset_id": g.get("_user_dataset_id", ""),
                    "label_sets": [],
                }
            )
            continue

        # Pick best label set: prefer polygon, then the one with most labels, else first
        def _ls_priority(ls):
            fmt = ls.get("label_format", "")
            return 0 if fmt == "polygon" else 1 if fmt == "bbox" else 2

        ls = sorted(label_sets, key=_ls_priority)[0]

        # Use _list_image_files for consistency with /api/images (mtime-cached).
        img_dir = g["train_images"]
        lbl_dir = ls.get("train_labels", "")
        images = _list_image_files(img_dir)  # already deduped
        names = g.get("names", {})

        # Single mtime-cached pass over the label directory. Repeat tab opens are
        # O(1) until a label file is written.
        summary = _label_dir_summary(lbl_dir, names) if lbl_dir else {
            "labeled": 0, "total_annotations": 0, "by_class": {}
        }
        labeled = summary["labeled"]
        total_anns = summary["total_annotations"]
        class_dist = summary["by_class"]

        results.append(
            {
                "group_id": g["group_id"],
                "group_name": g.get("group_name", ""),
                "nc": g.get("nc", 0),
                "names": {str(k): v for k, v in names.items()},
                "train_count": g.get("train_count", 0),
                "val_count": g.get("val_count", 0),
                "total_annotations": total_anns,
                "labeled_images": labeled,
                "label_progress": round(labeled / len(images), 4) if images else 0,
                "class_distribution": class_dist,
                "user_dataset_id": g.get("_user_dataset_id", ""),
                "label_sets": [
                    {
                        "set_id": l["set_id"],
                        "set_name": l["set_name"],
                        "label_format": l.get("label_format", ""),
                    }
                    for l in label_sets
                ],
            }
        )
    return results



# Export



@app.post("/api/export")
def export_dataset(
    group_id: str = Body(...),
    label_set_id: str = Body("default"),
    subset: str = Body("train"),
    format: str = Body("coco_json"),
):
    """Export annotations in various formats."""
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")
    ls = next((l for l in group["label_sets"] if l["set_id"] == label_set_id), None)
    if not ls:
        ls = group["label_sets"][0] if group["label_sets"] else None
    if not ls:
        raise HTTPException(404, "Label set not found")

    img_dir = group["train_images"] if subset == "train" else group["val_images"]
    lbl_dir = (
        ls.get("train_labels", "") if subset == "train" else ls.get("val_labels", "")
    )
    names = group.get("names", {})
    img_exts = {".jpg", ".jpeg", ".png"}
    images = (
        sorted(
            [
                f
                for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in img_exts
            ]
        )
        if os.path.isdir(img_dir)
        else []
    )

    if format == "coco_json":
        return _export_coco(
            images, img_dir, lbl_dir, names, group.get("group_name", "export")
        )
    elif format == "csv":
        return _export_csv(
            images, img_dir, lbl_dir, names, group.get("group_name", "export")
        )
    elif format == "yolo_zip":
        return _export_yolo_zip(
            images, img_dir, lbl_dir, group.get("group_name", "export")
        )
    else:
        raise HTTPException(400, f"Unknown format: {format}")


def _export_coco(images, img_dir, lbl_dir, names, ds_name):
    """Export as COCO JSON format."""
    from PIL import Image as PILImage

    coco = {
        "info": {"description": ds_name, "version": "1.0"},
        "images": [],
        "annotations": [],
        "categories": [{"id": int(k), "name": v} for k, v in names.items()],
    }
    ann_id = 1
    for img_idx, img_f in enumerate(images):
        img_path = os.path.join(img_dir, img_f)
        try:
            w, h = _image_size_exif(img_path)
        except Exception:
            w, h = 0, 0
        coco["images"].append(
            {"id": img_idx + 1, "file_name": img_f, "width": w, "height": h}
        )

        stem = os.path.splitext(img_f)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
        if not lbl_path or not os.path.exists(lbl_path):
            continue
        for a in _read_annotations(lbl_path):
            pts = a["points"]
            if a["ann_type"] == "bbox" and len(pts) >= 4:
                cx, cy, bw, bh = pts[0] * w, pts[1] * h, pts[2] * w, pts[3] * h
                x1, y1 = cx - bw / 2, cy - bh / 2
                bbox = [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)]
                seg = []
                area = bw * bh
            else:
                xs = [pts[i] * w for i in range(0, len(pts), 2)]
                ys = [pts[i] * h for i in range(1, len(pts), 2)]
                seg_flat = []
                for xi, yi in zip(xs, ys):
                    seg_flat.extend([round(xi, 2), round(yi, 2)])
                seg = [seg_flat]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                bbox = [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ]
                n = len(xs)
                area = (
                    abs(
                        sum(
                            xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                            for i in range(n)
                        )
                    )
                    / 2.0
                )

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_idx + 1,
                    "category_id": a["class_id"],
                    "bbox": bbox,
                    "segmentation": seg,
                    "area": round(area, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    import json as json_mod

    content = json_mod.dumps(coco, ensure_ascii=False, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{ds_name}_coco.json"'},
    )


def _export_csv(images, img_dir, lbl_dir, names, ds_name):
    """Export as CSV."""
    import csv, io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["image", "class_id", "class_name", "ann_type", "coordinates"])
    for img_f in images:
        stem = os.path.splitext(img_f)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
        if not lbl_path or not os.path.exists(lbl_path):
            continue
        for a in _read_annotations(lbl_path):
            cname = names.get(
                a["class_id"], names.get(str(a["class_id"]), f"class_{a['class_id']}")
            )
            coords = " ".join(f"{x:.6f}" for x in a["points"])
            writer.writerow([img_f, a["class_id"], cname, a["ann_type"], coords])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{ds_name}_annotations.csv"'
        },
    )


def _export_yolo_zip(images, img_dir, lbl_dir, ds_name):
    """Export YOLO labels as ZIP."""
    import zipfile, io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if lbl_path and os.path.exists(lbl_path):
                zf.write(lbl_path, f"labels/{stem}.txt")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{ds_name}_yolo_labels.zip"'
        },
    )



# Enhanced Dataset Management


# ?? Dataset metadata persistence ?????????????????????????????
_DATASET_METADATA_FILE = os.path.join(
    os.path.dirname(__file__), "dataset_metadata.json"
)


def _load_dataset_metadata() -> Dict[str, Any]:
    """Load dataset metadata including descriptions, tags, and custom fields."""
    if not os.path.isfile(_DATASET_METADATA_FILE):
        return {}
    try:
        with open(_DATASET_METADATA_FILE, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load dataset metadata: {e}")
        return {}


def _save_dataset_metadata(metadata: Dict[str, Any]):
    """Save dataset metadata."""
    try:
        with open(_DATASET_METADATA_FILE, "w", encoding="utf-8") as f:
            _json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save dataset metadata: {e}")


def _get_dataset_metadata(group_id: str) -> Dict[str, Any]:
    """Get metadata for a specific dataset."""
    metadata = _load_dataset_metadata()
    return metadata.get(
        group_id,
        {
            "description": "",
            "tags": [],
            "status": "active",
            "priority": "normal",
            "assigned_to": "",
            "notes": "",
            "starred": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _update_dataset_metadata(group_id: str, updates: Dict[str, Any]):
    """Update metadata for a specific dataset."""
    metadata = _load_dataset_metadata()
    if group_id not in metadata:
        metadata[group_id] = {
            "description": "",
            "tags": [],
            "status": "active",
            "priority": "normal",
            "assigned_to": "",
            "notes": "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    metadata[group_id].update(updates)
    metadata[group_id]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_dataset_metadata(metadata)


@app.get("/api/datasets/enhanced")
def list_datasets_enhanced(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    tag: Optional[str] = None,
    sort_by: str = "name",
    sort_order: str = "asc",
):
    """List all datasets with enhanced metadata, filtering and sorting."""
    _ensure_groups_fresh()
    metadata = _load_dataset_metadata()
    results = []

    for g in IMAGE_GROUPS:
        group_id = g["group_id"]
        meta = metadata.get(group_id, {})

        # Get annotation progress
        label_sets = g.get("label_sets", [])
        ls = label_sets[0] if label_sets else None

        progress = 0
        if ls:
            try:
                stats = _get_quick_stats(group_id, ls["set_id"], "train")
                progress = stats.get("label_progress", 0)
            except:
                pass

        item = {
            "group_id": group_id,
            "group_name": g.get("group_name", ""),
            "nc": g.get("nc", 0),
            "train_count": g.get("train_count", 0),
            "val_count": g.get("val_count", 0),
            "has_val": g.get("has_val", False),
            "label_progress": progress,
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "status": meta.get("status", "active"),
            "priority": meta.get("priority", "normal"),
            "assigned_to": meta.get("assigned_to", ""),
            "notes": meta.get("notes", ""),
            "starred": meta.get("starred", False),
            "created_at": meta.get(
                "created_at", datetime.now().isoformat(timespec="seconds")
            ),
            "updated_at": meta.get(
                "updated_at", datetime.now().isoformat(timespec="seconds")
            ),
            "label_sets": [
                {"set_id": l["set_id"], "set_name": l["set_name"]} for l in label_sets
            ],
            "names": g.get("names", {}),
        }

        # Apply filters
        if status and item["status"] != status:
            continue
        if priority and item["priority"] != priority:
            continue
        if tag and tag not in item["tags"]:
            continue

        results.append(item)

    # Sort results
    reverse = sort_order == "desc"
    if sort_by == "name":
        results.sort(key=lambda x: x["group_name"].lower(), reverse=reverse)
    elif sort_by == "progress":
        results.sort(key=lambda x: x["label_progress"], reverse=reverse)
    elif sort_by == "created":
        results.sort(key=lambda x: x["created_at"], reverse=reverse)
    elif sort_by == "updated":
        results.sort(key=lambda x: x["updated_at"], reverse=reverse)
    elif sort_by == "train_count":
        results.sort(key=lambda x: x["train_count"], reverse=reverse)

    return results


def _get_quick_stats(group_id: str, label_set_id: str, subset: str) -> Dict[str, Any]:
    """Get quick statistics for a dataset (with TTL cache)."""
    cache_key = f"{group_id}|{label_set_id}|{subset}"
    if cache_key in _stats_cache:
        cached_time, cached_data = _stats_cache[cache_key]
        if time.time() - cached_time < _STATS_CACHE_TTL:
            return cached_data

    group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
    if not group:
        return {}

    ls = next((l for l in group["label_sets"] if l["set_id"] == label_set_id), None)
    if not ls:
        return {}

    imgs_dir, lbls_dir = _get_dirs(group, ls, subset)
    if not imgs_dir or not os.path.isdir(imgs_dir):
        return {}

    files = _list_image_files(imgs_dir)
    total = len(files)

    labeled = 0
    total_anns = 0

    for fp in files:
        fn = os.path.basename(fp)
        base, _ = os.path.splitext(fn)
        lbl_path = os.path.join(lbls_dir, base + ".txt") if lbls_dir else ""
        if lbl_path and os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            anns = _read_annotations(lbl_path)
            if anns:
                labeled += 1
                total_anns += len(anns)

    result = {
        "total_images": total,
        "labeled_images": labeled,
        "label_progress": round(labeled / total, 4) if total > 0 else 0,
        "total_annotations": total_anns,
        "avg_annotations_per_image": round(total_anns / labeled, 2)
        if labeled > 0
        else 0,
    }
    _stats_cache[cache_key] = (time.time(), result)
    return result


@app.get("/api/datasets/{group_id}/metadata")
def get_dataset_metadata(group_id: str):
    """Get metadata for a specific dataset."""
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
    if not group:
        raise HTTPException(404, "Dataset not found")

    metadata = _get_dataset_metadata(group_id)
    return {"group_id": group_id, "group_name": group.get("group_name", ""), **metadata}


@app.put("/api/datasets/{group_id}/metadata")
def update_dataset_metadata(
    group_id: str,
    description: Optional[str] = Body(None),
    tags: Optional[List[str]] = Body(None),
    status: Optional[str] = Body(None),
    priority: Optional[str] = Body(None),
    assigned_to: Optional[str] = Body(None),
    notes: Optional[str] = Body(None),
    starred: Optional[bool] = Body(None),
):
    """Update metadata for a specific dataset."""
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
    if not group:
        raise HTTPException(404, "Dataset not found")

    updates = {}
    if description is not None:
        updates["description"] = description
    if tags is not None:
        updates["tags"] = tags
    if status is not None:
        updates["status"] = status
    if priority is not None:
        updates["priority"] = priority
    if assigned_to is not None:
        updates["assigned_to"] = assigned_to
    if notes is not None:
        updates["notes"] = notes
    if starred is not None:
        updates["starred"] = starred

    _update_dataset_metadata(group_id, updates)
    return {"status": "updated", "group_id": group_id}


@app.post("/api/datasets/bulk_action")
def bulk_dataset_action(
    action: str = Body(...),
    group_ids: List[str] = Body(...),
    tags: Optional[List[str]] = Body(None),
    status: Optional[str] = Body(None),
    priority: Optional[str] = Body(None),
):
    """Perform bulk actions on multiple datasets."""
    results = {"success": [], "failed": []}

    for group_id in group_ids:
        try:
            if action == "exclude":
                excluded = _load_excluded_groups()
                if group_id not in excluded:
                    excluded.append(group_id)
                    _save_excluded_groups(excluded)
                results["success"].append(group_id)

            elif action == "restore":
                excluded = _load_excluded_groups()
                excluded = [e for e in excluded if e != group_id]
                _save_excluded_groups(excluded)
                results["success"].append(group_id)

            elif action == "update_tags" and tags is not None:
                _update_dataset_metadata(group_id, {"tags": tags})
                results["success"].append(group_id)

            elif action == "update_status" and status is not None:
                _update_dataset_metadata(group_id, {"status": status})
                results["success"].append(group_id)

            elif action == "update_priority" and priority is not None:
                _update_dataset_metadata(group_id, {"priority": priority})
                results["success"].append(group_id)

            else:
                results["failed"].append({"id": group_id, "reason": "Unknown action"})

        except Exception as e:
            results["failed"].append({"id": group_id, "reason": str(e)})

    # Refresh image groups if needed
    if action in ["exclude", "restore"] and results["success"]:
        global IMAGE_GROUPS
        _mark_groups_dirty()

    return results


@app.get("/api/datasets/summary")
def get_datasets_summary():
    """Get summary statistics across all datasets."""
    total_datasets = len(IMAGE_GROUPS)
    total_images = sum(
        g.get("train_count", 0) + g.get("val_count", 0) for g in IMAGE_GROUPS
    )

    metadata = _load_dataset_metadata()
    status_counts = {"active": 0, "archived": 0, "pending": 0}
    priority_counts = {"high": 0, "normal": 0, "low": 0}
    all_tags = set()

    for group_id, meta in metadata.items():
        status = meta.get("status", "active")
        if status in status_counts:
            status_counts[status] += 1

        priority = meta.get("priority", "normal")
        if priority in priority_counts:
            priority_counts[priority] += 1

        all_tags.update(meta.get("tags", []))

    # Calculate overall progress
    total_progress = 0
    datasets_with_progress = 0

    for g in IMAGE_GROUPS:
        label_sets = g.get("label_sets", [])
        if label_sets:
            try:
                stats = _get_quick_stats(
                    g["group_id"], label_sets[0]["set_id"], "train"
                )
                if stats.get("total_images", 0) > 0:
                    total_progress += stats.get("label_progress", 0)
                    datasets_with_progress += 1
            except:
                pass

    avg_progress = (
        round(total_progress / datasets_with_progress, 4)
        if datasets_with_progress > 0
        else 0
    )

    return {
        "total_datasets": total_datasets,
        "total_images": total_images,
        "avg_label_progress": avg_progress,
        "status_distribution": status_counts,
        "priority_distribution": priority_counts,
        "all_tags": sorted(list(all_tags)),
        "datasets_with_metadata": len(metadata),
    }


# ????????????????????????????????????????????????????????????????# Data Quality Management APIs


class QualityCheckRequest(BaseModel):
    group_id: str
    label_set_id: str = "default"
    subset: str = "train"
    checks: List[str] = None  # List of check types to run


@app.post("/api/datasets/quality_check")
def run_quality_check(req: QualityCheckRequest):
    """
    Run comprehensive quality checks on a dataset.

    Check types:
    - duplicate_images: Find duplicate or near-duplicate images
    - corrupted_images: Detect corrupted or unreadable images
    - orphaned_labels: Label files without corresponding images
    - orphaned_images: Images without corresponding label files
    - invalid_annotations: Malformed or invalid annotation files
    - tiny_annotations: Annotations that are too small
    - oversized_annotations: Annotations that are too large
    - class_imbalance: Severe class imbalance issues
    - negative_samples: Images with no annotations (if not expected)
    - outlier_boxes: Annotations with extreme aspect ratios
    """
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == req.group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")

    ls = next((l for l in group["label_sets"] if l["set_id"] == req.label_set_id), None)
    if not ls:
        ls = group["label_sets"][0] if group["label_sets"] else None
    if not ls:
        raise HTTPException(404, "Label set not found")

    img_dir = group["train_images"] if req.subset == "train" else group["val_images"]
    lbl_dir = (
        ls.get("train_labels", "")
        if req.subset == "train"
        else ls.get("val_labels", "")
    )

    if not os.path.isdir(img_dir):
        raise HTTPException(404, "Image directory not found")

    checks_to_run = req.checks or ["all"]
    run_all = "all" in checks_to_run

    results = {
        "group_id": req.group_id,
        "group_name": group.get("group_name", ""),
        "subset": req.subset,
        "checks_performed": [],
        "issues": [],
        "summary": {
            "total_checked": 0,
            "issues_found": 0,
            "severity_counts": {"critical": 0, "warning": 0, "info": 0},
        },
    }

    # Get all images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    images = []
    for f in os.listdir(img_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in img_exts:
            images.append(f)

    results["summary"]["total_checked"] = len(images)

    # Check 1: Corrupted images
    if run_all or "corrupted_images" in checks_to_run:
        results["checks_performed"].append("corrupted_images")
        corrupted = []
        for img_f in images:
            img_path = os.path.join(img_dir, img_f)
            try:
                from PIL import Image as PILImage

                with PILImage.open(img_path) as im:
                    im.verify()
            except Exception as e:
                corrupted.append(
                    {"filename": img_f, "error": str(e), "severity": "critical"}
                )

        if corrupted:
            results["issues"].append(
                {
                    "type": "corrupted_images",
                                        "title": "Item",
                    "description": f"{len(corrupted)} corrupted annotation files",
                    "count": len(corrupted),
                    "severity": "critical",
                    "items": corrupted[:50],  # Limit to first 50
                }
            )
            results["summary"]["issues_found"] += len(corrupted)
            results["summary"]["severity_counts"]["critical"] += len(corrupted)

    # Check 2: Orphaned labels and images
    if (
        run_all
        or "orphaned_labels" in checks_to_run
        or "orphaned_images" in checks_to_run
    ):
        results["checks_performed"].extend(["orphaned_labels", "orphaned_images"])

        image_stems = {os.path.splitext(f)[0] for f in images}
        label_files = set()

        if lbl_dir and os.path.isdir(lbl_dir):
            for f in os.listdir(lbl_dir):
                if f.endswith(".txt"):
                    label_files.add(os.path.splitext(f)[0])

        # Orphaned labels (labels without images)
        orphaned_labels = label_files - image_stems
        if orphaned_labels:
            items = [
                {"filename": f + ".txt", "severity": "warning"}
                for f in list(orphaned_labels)[:50]
            ]
            results["issues"].append(
                {
                    "type": "orphaned_labels",
                    "title": "Invalid parameters",
                    "description": f"{len(orphaned_labels)} label files without matching images",
                    "count": len(orphaned_labels),
                    "severity": "warning",
                    "items": items,
                }
            )
            results["summary"]["issues_found"] += len(orphaned_labels)
            results["summary"]["severity_counts"]["warning"] += len(orphaned_labels)

        # Orphaned images (images without labels)
        orphaned_images = image_stems - label_files
        if orphaned_images:
            items = [
                {"filename": f + ".jpg", "severity": "info"}
                for f in list(orphaned_images)[:50]
            ]
            results["issues"].append(
                {
                    "type": "orphaned_images",
                                        "title": "Item",
                    "description": f"{len(orphaned_images)} images without label files",
                    "count": len(orphaned_images),
                    "severity": "info",
                    "items": items,
                }
            )
            results["summary"]["issues_found"] += len(orphaned_images)
            results["summary"]["severity_counts"]["info"] += len(orphaned_images)

    # Check 3: Invalid annotations
    if run_all or "invalid_annotations" in checks_to_run:
        results["checks_performed"].append("invalid_annotations")
        invalid = []

        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if not lbl_path or not os.path.exists(lbl_path):
                continue

            try:
                with open(lbl_path, "r") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        invalid.append(
                            {
                                "filename": img_f,
                                "line": line_num,
                                "reason": "Invalid evaluation data",
                                "severity": "warning",
                            }
                        )
                        continue

                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        # Check class_id range
                        if class_id < 0 or class_id >= group.get("nc", 999):
                            invalid.append(
                                {
                                    "filename": img_f,
                                    "line": line_num,
                                    "reason": f"Invalid class ID {class_id}",
                                    "severity": "warning",
                                }
                            )

                        # Check coordinate values
                        for c in coords:
                            if c < 0 or c > 1:
                                invalid.append(
                                    {
                                        "filename": img_f,
                                        "line": line_num,
                                        "reason": "Coordinates outside [0,1] range",
                                        "severity": "warning",
                                    }
                                )
                                break

                    except ValueError:
                        invalid.append(
                            {
                                "filename": img_f,
                                "line": line_num,
                                                                "reason": "reason",
                                "severity": "warning",
                            }
                        )

            except Exception as e:
                invalid.append(
                    {
                        "filename": img_f,
                        "reason": f"Parse error: {str(e)}",
                        "severity": "warning",
                    }
                )

        if invalid:
            results["issues"].append(
                {
                    "type": "invalid_annotations",
                    "title": "Warnings",
                    "description": "broken string",
                    "count": len(invalid),
                    "severity": "warning",
                    "items": invalid[:50],
                }
            )
            results["summary"]["issues_found"] += len(invalid)
            results["summary"]["severity_counts"]["warning"] += len(invalid)

    # Check 4: Tiny and oversized annotations
    if (
        run_all
        or "tiny_annotations" in checks_to_run
        or "oversized_annotations" in checks_to_run
    ):
        results["checks_performed"].extend(
            ["tiny_annotations", "oversized_annotations"]
        )
        tiny = []
        oversized = []

        TINY_THRESHOLD = 0.0001  # Normalized area (e.g., 0.01% of image)
        OVERSIZE_THRESHOLD = 0.9  # 90% of image

        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if not lbl_path or not os.path.exists(lbl_path):
                continue

            anns = _read_annotations(lbl_path)
            for ann in anns:
                pts = ann["points"]
                if ann["ann_type"] == "bbox" and len(pts) >= 4:
                    area = pts[2] * pts[3]  # w * h
                elif len(pts) >= 6:
                    xs = pts[0::2]
                    ys = pts[1::2]
                    n = len(xs)
                    area = (
                        abs(
                            sum(
                                xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                                for i in range(n)
                            )
                        )
                        / 2.0
                    )
                else:
                    continue

                if area < TINY_THRESHOLD:
                    tiny.append(
                        {
                            "filename": img_f,
                            "class_id": ann["class_id"],
                            "area": round(area, 6),
                            "severity": "warning",
                        }
                    )
                elif area > OVERSIZE_THRESHOLD:
                    oversized.append(
                        {
                            "filename": img_f,
                            "class_id": ann["class_id"],
                            "area": round(area, 6),
                            "severity": "info",
                        }
                    )

        if tiny:
            results["issues"].append(
                {
                    "type": "tiny_annotations",
                                        "title": "Item",
                    "description": "broken string",
                    "count": len(tiny),
                    "severity": "warning",
                    "items": tiny[:50],
                }
            )
            results["summary"]["issues_found"] += len(tiny)
            results["summary"]["severity_counts"]["warning"] += len(tiny)

        if oversized:
            results["issues"].append(
                {
                    "type": "oversized_annotations",
                                        "title": "Item",
                    "description": "broken string",
                    "count": len(oversized),
                    "severity": "info",
                    "items": oversized[:50],
                }
            )
            results["summary"]["issues_found"] += len(oversized)
            results["summary"]["severity_counts"]["info"] += len(oversized)

    # Check 5: Outlier boxes (extreme aspect ratios)
    if run_all or "outlier_boxes" in checks_to_run:
        results["checks_performed"].append("outlier_boxes")
        outliers = []
        MIN_ASPECT = 0.05  # 1:20
        MAX_ASPECT = 20.0  # 20:1

        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if not lbl_path or not os.path.exists(lbl_path):
                continue

            anns = _read_annotations(lbl_path)
            for ann in anns:
                pts = ann["points"]
                if ann["ann_type"] == "bbox" and len(pts) >= 4:
                    w, h = pts[2], pts[3]
                    if w > 0 and h > 0:
                        aspect = w / h
                        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                            outliers.append(
                                {
                                    "filename": img_f,
                                    "class_id": ann["class_id"],
                                    "aspect_ratio": round(aspect, 2),
                                    "severity": "info",
                                }
                            )

        if outliers:
            results["issues"].append(
                {
                    "type": "outlier_boxes",
                    "title": "Invalid class configuration",
                    "description": f"{len(outliers)} aspect ratio outliers (range {MIN_ASPECT}-{MAX_ASPECT})",
                    "count": len(outliers),
                    "severity": "info",
                    "items": outliers[:50],
                }
            )
            results["summary"]["issues_found"] += len(outliers)
            results["summary"]["severity_counts"]["info"] += len(outliers)

    return results


class QualityFixRequest(BaseModel):
    group_id: str
    label_set_id: str = "default"
    subset: str = "train"
    fix_type: str  # delete_orphaned_labels, cleanup_empty, remove_tiny, etc.
    dry_run: bool = True  # If True, only report what would be done


@app.post("/api/datasets/quality_fix")
def apply_quality_fix(req: QualityFixRequest):
    """
    Apply automatic fixes for data quality issues.

    Fix types:
    - delete_orphaned_labels: Remove label files without corresponding images
    - cleanup_empty: Remove empty annotation files
    - remove_tiny: Remove annotations smaller than threshold
    - fix_coordinates: Clip out-of-bounds coordinates to [0,1]
    """
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == req.group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")

    ls = next((l for l in group["label_sets"] if l["set_id"] == req.label_set_id), None)
    if not ls:
        ls = group["label_sets"][0] if group["label_sets"] else None
    if not ls:
        raise HTTPException(404, "Label set not found")

    img_dir = group["train_images"] if req.subset == "train" else group["val_images"]
    lbl_dir = (
        ls.get("train_labels", "")
        if req.subset == "train"
        else ls.get("val_labels", "")
    )

    if not os.path.isdir(img_dir) or not lbl_dir or not os.path.isdir(lbl_dir):
        raise HTTPException(404, "Directory not found")

    results = {
        "fix_type": req.fix_type,
        "dry_run": req.dry_run,
        "actions": [],
        "summary": {"processed": 0, "fixed": 0, "failed": 0},
    }

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    images = {
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    }

    if req.fix_type == "delete_orphaned_labels":
        label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

        for lbl_f in label_files:
            stem = os.path.splitext(lbl_f)[0]
            if stem not in images:
                lbl_path = os.path.join(lbl_dir, lbl_f)
                results["actions"].append(
                    {
                        "action": "delete",
                        "file": lbl_f,
                        "reason": "No corresponding image",
                    }
                )

                if not req.dry_run:
                    try:
                        os.remove(lbl_path)
                        results["summary"]["fixed"] += 1
                    except Exception as e:
                        results["actions"][-1]["error"] = str(e)
                        results["summary"]["failed"] += 1
                else:
                    results["summary"]["fixed"] += 1

                results["summary"]["processed"] += 1

    elif req.fix_type == "cleanup_empty":
        label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

        for lbl_f in label_files:
            lbl_path = os.path.join(lbl_dir, lbl_f)
            try:
                if os.path.getsize(lbl_path) == 0:
                    results["actions"].append(
                        {"action": "delete", "file": lbl_f, "reason": "Empty file"}
                    )

                    if not req.dry_run:
                        os.remove(lbl_path)

                    results["summary"]["fixed"] += 1
                    results["summary"]["processed"] += 1
            except Exception as e:
                results["actions"].append(
                    {"action": "error", "file": lbl_f, "error": str(e)}
                )
                results["summary"]["failed"] += 1
                results["summary"]["processed"] += 1

    elif req.fix_type == "remove_tiny":
        TINY_THRESHOLD = 0.0001
        label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

        for lbl_f in label_files:
            lbl_path = os.path.join(lbl_dir, lbl_f)
            try:
                anns = _read_annotations(lbl_path)
                if not anns:
                    continue

                filtered_anns = []
                removed_count = 0

                for ann in anns:
                    pts = ann["points"]
                    if ann["ann_type"] == "bbox" and len(pts) >= 4:
                        area = pts[2] * pts[3]
                    elif len(pts) >= 6:
                        xs = pts[0::2]
                        ys = pts[1::2]
                        n = len(xs)
                        area = (
                            abs(
                                sum(
                                    xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                                    for i in range(n)
                                )
                            )
                            / 2.0
                        )
                    else:
                        area = 0

                    if area < TINY_THRESHOLD:
                        removed_count += 1
                    else:
                        filtered_anns.append(ann)

                if removed_count > 0:
                    results["actions"].append(
                        {
                            "action": "filter",
                            "file": lbl_f,
                            "removed": removed_count,
                            "remaining": len(filtered_anns),
                        }
                    )

                    if not req.dry_run:
                        if filtered_anns:
                            _write_annotations(lbl_path, filtered_anns)
                        else:
                            os.remove(lbl_path)

                    results["summary"]["fixed"] += 1
                    results["summary"]["processed"] += 1

            except Exception as e:
                results["actions"].append(
                    {"action": "error", "file": lbl_f, "error": str(e)}
                )
                results["summary"]["failed"] += 1

    elif req.fix_type == "fix_coordinates":
        label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

        for lbl_f in label_files:
            lbl_path = os.path.join(lbl_dir, lbl_f)
            try:
                anns = _read_annotations(lbl_path)
                if not anns:
                    continue

                modified = False
                for ann in anns:
                    pts = ann["points"]
                    new_pts = []
                    for i, v in enumerate(pts):
                        if v < 0:
                            new_pts.append(0.0)
                            modified = True
                        elif v > 1:
                            new_pts.append(1.0)
                            modified = True
                        else:
                            new_pts.append(v)

                    if modified:
                        ann["points"] = new_pts

                if modified:
                    results["actions"].append({"action": "fix_coords", "file": lbl_f})

                    if not req.dry_run:
                        _write_annotations(lbl_path, anns)

                    results["summary"]["fixed"] += 1
                    results["summary"]["processed"] += 1

            except Exception as e:
                results["actions"].append(
                    {"action": "error", "file": lbl_f, "error": str(e)}
                )
                results["summary"]["failed"] += 1

    else:
        raise HTTPException(400, f"Unknown fix type: {req.fix_type}")

    return results


# ????????????????????????????????????????????????????????????????# Data Split Management APIs


class DataSplitRequest(BaseModel):
    group_id: str
    label_set_id: str = "default"
    strategy: str = "random"  # random, stratified, manual
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    dry_run: bool = True


class DataSplitApplyRequest(BaseModel):
    group_id: str
    label_set_id: str = "default"
    splits: Dict[str, List[str]]  # { "train": [...], "val": [...], "test": [...] }


@app.post("/api/datasets/split_preview")
def preview_data_split(req: DataSplitRequest):
    """
    Preview how a dataset would be split into train/val/test.

    Strategies:
    - random: Random split
    - stratified: Try to maintain class distribution (if labels exist)
    """
    if abs(req.train_ratio + req.val_ratio + req.test_ratio - 1.0) > 0.001:
        raise HTTPException(400, "Ratios must sum to 1.0")

    group = next((g for g in IMAGE_GROUPS if g["group_id"] == req.group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")

    ls = next((l for l in group["label_sets"] if l["set_id"] == req.label_set_id), None)
    if not ls:
        ls = group["label_sets"][0] if group["label_sets"] else None
    if not ls:
        raise HTTPException(404, "Label set not found")

    img_dir = group["train_images"]
    lbl_dir = ls.get("train_labels", "")

    if not os.path.isdir(img_dir):
        raise HTTPException(404, "Image directory not found")

    # Get all images
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    images = [
        f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts
    ]
    images.sort()

    if not images:
        return {"error": "No images found"}

    total = len(images)

    # For stratified split, try to read labels
    if req.strategy == "stratified" and lbl_dir and os.path.isdir(lbl_dir):
        # Group images by primary class
        class_groups = {}
        unlabeled = []

        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            if os.path.exists(lbl_path):
                anns = _read_annotations(lbl_path)
                if anns:
                    # Use first class as primary
                    primary_class = anns[0]["class_id"]
                    if primary_class not in class_groups:
                        class_groups[primary_class] = []
                    class_groups[primary_class].append(img_f)
                else:
                    unlabeled.append(img_f)
            else:
                unlabeled.append(img_f)

        # Split each class group
        train_files, val_files, test_files = [], [], []

        import random

        random.seed(req.seed)

        for class_id, files in class_groups.items():
            random.shuffle(files)
            n = len(files)
            n_train = int(n * req.train_ratio)
            n_val = int(n * req.val_ratio)

            train_files.extend(files[:n_train])
            val_files.extend(files[n_train : n_train + n_val])
            test_files.extend(files[n_train + n_val :])

        # Handle unlabeled images
        if unlabeled:
            random.shuffle(unlabeled)
            n = len(unlabeled)
            n_train = int(n * req.train_ratio)
            n_val = int(n * req.val_ratio)

            train_files.extend(unlabeled[:n_train])
            val_files.extend(unlabeled[n_train : n_train + n_val])
            test_files.extend(unlabeled[n_train + n_val :])

    else:
        # Random split
        import random

        random.seed(req.seed)

        shuffled = images.copy()
        random.shuffle(shuffled)

        n_train = int(total * req.train_ratio)
        n_val = int(total * req.val_ratio)

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train : n_train + n_val]
        test_files = shuffled[n_train + n_val :]

    # Compute class distribution for each split
    def get_class_dist(files):
        dist = {}
        for img_f in files:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if os.path.exists(lbl_path):
                anns = _read_annotations(lbl_path)
                for ann in anns:
                    cid = ann["class_id"]
                    dist[cid] = dist.get(cid, 0) + 1
        return dist

    return {
        "strategy": req.strategy,
        "ratios": {
            "train": req.train_ratio,
            "val": req.val_ratio,
            "test": req.test_ratio,
        },
        "seed": req.seed,
        "splits": {
            "train": {
                "count": len(train_files),
                "percentage": round(len(train_files) / total * 100, 1),
                "sample_files": train_files[:10],
                "class_distribution": get_class_dist(train_files)
                if req.strategy == "stratified"
                else None,
            },
            "val": {
                "count": len(val_files),
                "percentage": round(len(val_files) / total * 100, 1),
                "sample_files": val_files[:10],
                "class_distribution": get_class_dist(val_files)
                if req.strategy == "stratified"
                else None,
            },
            "test": {
                "count": len(test_files),
                "percentage": round(len(test_files) / total * 100, 1),
                "sample_files": test_files[:10],
                "class_distribution": get_class_dist(test_files)
                if req.strategy == "stratified"
                else None,
            },
        },
        "total_images": total,
    }


@app.post("/api/datasets/split_apply")
def apply_data_split(req: DataSplitApplyRequest):
    """
    Apply a data split by creating/updating data.yaml with split information.
    """
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == req.group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")

    # Find the data.yaml file
    group_path = group["group_id"]
    yaml_path = None

    # Look for data*.yaml in the group directory
    parent_dir = os.path.dirname(group_path)
    for f in os.listdir(parent_dir if parent_dir else "."):
        if f.startswith("data") and f.endswith(".yaml"):
            yaml_path = os.path.join(parent_dir, f)
            break

    if not yaml_path:
        raise HTTPException(404, "data.yaml not found")

    try:
        import yaml

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Update split information
        base_img_dir = os.path.dirname(group["train_images"])

        # Create split files or update config
        if req.splits.get("train"):
            config["train"] = os.path.join(base_img_dir, "train")
        if req.splits.get("val"):
            config["val"] = os.path.join(base_img_dir, "val")
        if req.splits.get("test"):
            config["test"] = os.path.join(base_img_dir, "test")

        # Write updated config
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # Actually move/copy files if needed (optional)
        # For now, we just update the config

        return {
            "status": "success",
            "yaml_path": yaml_path,
            "splits_applied": list(req.splits.keys()),
            "message": "Split configuration updated. Please refresh the dataset.",
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to apply split: {str(e)}")


# ????????????????????????????????????????????????????????????????# Visualization & Analytics APIs


@app.get("/api/datasets/visualization")
def get_dataset_visualization(
    group_id: str,
    label_set_id: str = "default",
    subset: str = "train",
    viz_type: str = "class_distribution",  # class_distribution, bbox_size, annotation_count
):
    """
    Get data for visualization charts.

    viz_type options:
    - class_distribution: Count per class
    - bbox_size_distribution: Size distribution of bounding boxes
    - annotation_count_per_image: Histogram of annotations per image
    - annotation_area_vs_count: Scatter plot data
    """
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")

    ls = next((l for l in group["label_sets"] if l["set_id"] == label_set_id), None)
    if not ls:
        ls = group["label_sets"][0] if group["label_sets"] else None
    if not ls:
        raise HTTPException(404, "Label set not found")

    img_dir = group["train_images"] if subset == "train" else group["val_images"]
    lbl_dir = (
        ls.get("train_labels", "") if subset == "train" else ls.get("val_labels", "")
    )

    if not os.path.isdir(img_dir):
        raise HTTPException(404, "Image directory not found")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    images = [
        f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts
    ]
    names = group.get("names", {})

    if viz_type == "class_distribution":
        class_counts = {}
        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if not lbl_path or not os.path.exists(lbl_path):
                continue
            anns = _read_annotations(lbl_path)
            for ann in anns:
                cid = ann["class_id"]
                cname = names.get(cid, names.get(str(cid), f"class_{cid}"))
                class_counts[cname] = class_counts.get(cname, 0) + 1

        return {
            "viz_type": viz_type,
            "chart_type": "bar",
            "data": {
                "labels": list(class_counts.keys()),
                "values": list(class_counts.values()),
                "colors": [
                    f"hsl({i * 360 / len(class_counts)}, 70%, 60%)"
                    for i in range(len(class_counts))
                ],
            },
            "title": "Warnings",
            "total_annotations": sum(class_counts.values()),
        }

    elif viz_type == "bbox_size_distribution":
        # Bucket sizes into ranges
        size_ranges = {
            "tiny (< 0.001)": 0,
            "small (0.001-0.01)": 0,
            "medium (0.01-0.1)": 0,
            "large (0.1-0.5)": 0,
            "huge (> 0.5)": 0,
        }

        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if not lbl_path or not os.path.exists(lbl_path):
                continue
            anns = _read_annotations(lbl_path)
            for ann in anns:
                pts = ann["points"]
                if ann["ann_type"] == "bbox" and len(pts) >= 4:
                    area = pts[2] * pts[3]
                elif len(pts) >= 6:
                    xs = pts[0::2]
                    ys = pts[1::2]
                    n = len(xs)
                    area = (
                        abs(
                            sum(
                                xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                                for i in range(n)
                            )
                        )
                        / 2.0
                    )
                else:
                    continue

                if area < 0.001:
                    size_ranges["tiny (< 0.001)"] += 1
                elif area < 0.01:
                    size_ranges["small (0.001-0.01)"] += 1
                elif area < 0.1:
                    size_ranges["medium (0.01-0.1)"] += 1
                elif area < 0.5:
                    size_ranges["large (0.1-0.5)"] += 1
                else:
                    size_ranges["huge (> 0.5)"] += 1

        return {
            "viz_type": viz_type,
            "chart_type": "pie",
            "data": {
                "labels": list(size_ranges.keys()),
                "values": list(size_ranges.values()),
                "colors": ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1", "#5f27cd"],
            },
            "title": "Invalid parameters",
            "total_annotations": sum(size_ranges.values()),
        }

    elif viz_type == "annotation_count_per_image":
        count_distribution = {}

        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if not lbl_path or not os.path.exists(lbl_path):
                count = 0
            else:
                anns = _read_annotations(lbl_path)
                count = len(anns)

            bucket = min(count, 10)  # Group 10+ together
            bucket_label = "10+" if bucket == 10 else str(bucket)
            count_distribution[bucket_label] = (
                count_distribution.get(bucket_label, 0) + 1
            )

        # Sort by numeric value
        sorted_items = sorted(
            count_distribution.items(),
            key=lambda x: int(x[0]) if x[0].isdigit() else 999,
        )

        return {
            "viz_type": viz_type,
            "chart_type": "bar",
            "data": {
                "labels": [item[0] for item in sorted_items],
                "values": [item[1] for item in sorted_items],
                "colors": "#7aa2ff",
            },
                        "title": "Item",
            "x_label": "Warnings",
            "y_label": "Warnings",
        }

    elif viz_type == "annotation_area_vs_count":
        # Scatter plot: total annotation area vs annotation count per image
        scatter_data = []

        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if not lbl_path or not os.path.exists(lbl_path):
                continue

            anns = _read_annotations(lbl_path)
            total_area = 0
            for ann in anns:
                pts = ann["points"]
                if ann["ann_type"] == "bbox" and len(pts) >= 4:
                    area = pts[2] * pts[3]
                elif len(pts) >= 6:
                    xs = pts[0::2]
                    ys = pts[1::2]
                    n = len(xs)
                    area = (
                        abs(
                            sum(
                                xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                                for i in range(n)
                            )
                        )
                        / 2.0
                    )
                else:
                    area = 0
                total_area += area

            scatter_data.append(
                {
                    "filename": img_f,
                    "count": len(anns),
                    "total_area": round(total_area, 4),
                }
            )

        return {
            "viz_type": viz_type,
            "chart_type": "scatter",
            "data": scatter_data,
            "title": "Images vs Annotations",
            "x_label": "Warnings",
            "y_label": "Invalid evaluation request",
        }

    else:
        raise HTTPException(400, f"Unknown visualization type: {viz_type}")


# ????????????????????????????????????????????????????????????????# Dataset Comparison & Recommendation APIs


class DatasetCompareRequest(BaseModel):
    group_ids: List[str]
    metrics: List[str] = None  # image_count, annotation_count, class_distribution, etc.


@app.post("/api/datasets/compare")
def compare_datasets(req: DatasetCompareRequest):
    """
    Compare multiple datasets across various metrics.
    """
    if len(req.group_ids) < 2:
        raise HTTPException(400, "At least 2 datasets required for comparison")

    metrics = req.metrics or ["basic_stats", "class_distribution"]
    results = {"datasets": [], "comparisons": {}}

    # Collect data for each dataset
    dataset_data = {}
    for group_id in req.group_ids:
        group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
        if not group:
            continue

        ls = group["label_sets"][0] if group["label_sets"] else None
        if not ls:
            continue

        img_dir = group["train_images"]
        lbl_dir = ls.get("train_labels", "") if ls else ""
        names = group.get("names", {})

        if not os.path.isdir(img_dir):
            continue

        img_exts = {".jpg", ".jpeg", ".png"}
        images = [
            f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts
        ]

        data = {
            "group_id": group_id,
            "group_name": group.get("group_name", ""),
            "image_count": len(images),
            "annotation_count": 0,
            "class_distribution": {},
            "avg_annotations_per_image": 0,
        }

        labeled_count = 0
        for img_f in images:
            stem = os.path.splitext(img_f)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
            if os.path.exists(lbl_path):
                anns = _read_annotations(lbl_path)
                if anns:
                    labeled_count += 1
                    data["annotation_count"] += len(anns)
                    for ann in anns:
                        cid = ann["class_id"]
                        cname = names.get(cid, f"class_{cid}")
                        data["class_distribution"][cname] = (
                            data["class_distribution"].get(cname, 0) + 1
                        )

        if labeled_count > 0:
            data["avg_annotations_per_image"] = round(
                data["annotation_count"] / labeled_count, 2
            )

        dataset_data[group_id] = data
        results["datasets"].append(data)

    # Generate comparisons
    if "basic_stats" in metrics:
        results["comparisons"]["basic_stats"] = {
            "image_count": {gid: d["image_count"] for gid, d in dataset_data.items()},
            "annotation_count": {
                gid: d["annotation_count"] for gid, d in dataset_data.items()
            },
            "avg_annotations_per_image": {
                gid: d["avg_annotations_per_image"] for gid, d in dataset_data.items()
            },
        }

    if "class_distribution" in metrics:
        # Merge all classes
        all_classes = set()
        for d in dataset_data.values():
            all_classes.update(d["class_distribution"].keys())

        class_comparison = {}
        for cls in all_classes:
            class_comparison[cls] = {
                gid: d["class_distribution"].get(cls, 0)
                for gid, d in dataset_data.items()
            }

        results["comparisons"]["class_distribution"] = class_comparison

    return results


@app.get("/api/datasets/recommendations")
def get_labeling_recommendations(
    group_id: str,
    label_set_id: str = "default",
    subset: str = "train",
    strategy: str = "least_confident",  # least_confident, diverse, random
):
    """
    Get smart recommendations for which images to label next.

    Strategies:
    - least_confident: Images that likely need more annotations
    - diverse: Images that are different from already labeled ones
    - random: Random selection
    """
    group = next((g for g in IMAGE_GROUPS if g["group_id"] == group_id), None)
    if not group:
        raise HTTPException(404, "Group not found")

    ls = next((l for l in group["label_sets"] if l["set_id"] == label_set_id), None)
    if not ls:
        ls = group["label_sets"][0] if group["label_sets"] else None
    if not ls:
        raise HTTPException(404, "Label set not found")

    img_dir = group["train_images"] if subset == "train" else group["val_images"]
    lbl_dir = (
        ls.get("train_labels", "") if subset == "train" else ls.get("val_labels", "")
    )

    if not os.path.isdir(img_dir):
        raise HTTPException(404, "Image directory not found")

    img_exts = {".jpg", ".jpeg", ".png"}
    images = [
        f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts
    ]

    # Get images without labels
    unlabeled = []
    labeled = []

    for img_f in images:
        stem = os.path.splitext(img_f)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt") if lbl_dir else ""
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            anns = _read_annotations(lbl_path)
            if anns:
                labeled.append({"filename": img_f, "annotations": len(anns)})
            else:
                unlabeled.append(img_f)
        else:
            unlabeled.append(img_f)

    recommendations = []

    if strategy == "random":
        import random

        random.shuffle(unlabeled)
        recommendations = unlabeled[:20]

    elif strategy == "least_confident":
        # Prioritize images near already labeled ones (they might be similar)
        # This is a simple heuristic - in practice, you might use model predictions
        if labeled:
            # Get indices of labeled images
            labeled_indices = {img["filename"]: i for i, img in enumerate(labeled)}
            # For now, just recommend first unlabeled ones
            recommendations = unlabeled[:20]
        else:
            recommendations = unlabeled[:20]

    elif strategy == "diverse":
        # Pick evenly spaced samples
        if len(unlabeled) <= 20:
            recommendations = unlabeled
        else:
            step = len(unlabeled) // 20
            recommendations = [unlabeled[i * step] for i in range(20)]

    return {
        "strategy": strategy,
        "total_unlabeled": len(unlabeled),
        "total_labeled": len(labeled),
        "recommendations": recommendations,
        "priority_scores": [
            {"filename": f, "score": 1.0 - (i / len(recommendations))}
            for i, f in enumerate(recommendations)
        ]
        if recommendations
        else [],
    }


@app.get("/api/images/paginated")
def list_images_paginated(
    group_id: str,
    label_set_id: str,
    subset: str = "train",
    page: int = 1,
    page_size: int = 100,
    filter_type: Optional[str] = None,  # "all", "labeled", "unlabeled"
    search: Optional[str] = None,
):
    """List images with pagination and filtering support."""
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, subset)

    if not imgs_dir or not os.path.isdir(imgs_dir):
        return {"total": 0, "page": page, "page_size": page_size, "images": []}

    files = _list_image_files(imgs_dir)

    # Build image list with label status
    images = []
    for fp in files:
        fn = os.path.basename(fp)
        base, _ = os.path.splitext(fn)
        label_path = os.path.join(lbls_dir, base + ".txt") if lbls_dir else ""
        has_label = (
            label_path
            and os.path.exists(label_path)
            and os.path.getsize(label_path) > 0
        )

        images.append(
            {
                "filename": fn,
                "has_label": has_label,
            }
        )

    # Apply filters
    if filter_type == "labeled":
        images = [img for img in images if img["has_label"]]
    elif filter_type == "unlabeled":
        images = [img for img in images if not img["has_label"]]

    if search:
        search_lower = search.lower()
        images = [img for img in images if search_lower in img["filename"].lower()]

    # Pagination
    total = len(images)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_images = images[start_idx:end_idx]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "images": paginated_images,
    }



# Enhanced Dataset Management - End




# Enterprise Data Management APIs



# ?? Project Management ?????????????????????????????????????????


@app.post("/api/projects")
def api_create_project(
    name: str = Body(...), description: str = Body(""), config: dict = Body(None)
):
    project = create_project(name, description, config)
    log_action("project_created", "project", details={"project_id": project["id"], "name": name})
    return project


@app.get("/api/projects")
def api_list_projects():
    return list_projects()


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    p = get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@app.put("/api/projects/{project_id}")
def api_update_project(project_id: str, updates: Dict = Body(...)):
    p = update_project(project_id, updates)
    if not p:
        raise HTTPException(404, "Project not found")
    log_action("project_updated", "project", details={"project_id": project_id})
    return p


@app.post("/api/projects/{project_id}/datasets")
def api_add_dataset_to_project(
    project_id: str, group_id: str = Body(...), role: str = Body("primary")
):
    add_dataset_to_project(project_id, group_id, role)
    log_action(
        "dataset_added_to_project",
        "project",
        details={"project_id": project_id, "group_id": group_id},
    )
    return {"ok": True}


@app.delete("/api/projects/{project_id}/datasets/{group_id}")
def api_remove_dataset_from_project(project_id: str, group_id: str):
    remove_dataset_from_project(project_id, group_id)
    return {"ok": True}


@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str):
    """Permanently delete a project and its dataset associations."""
    p = get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    from labeling_tool.database import get_db

    with get_db() as conn:
        conn.execute("DELETE FROM project_datasets WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    log_action(
        "project_deleted",
        "project",
        details={"project_id": project_id, "name": p.get("name", "")},
    )
    return {"ok": True, "deleted": project_id}


# ?? Session Management ?????????????????????????????????????????


@app.post("/api/session")
def api_create_or_update_session(
    session_id: str = Body(...),
    group_id: str = Body(""),
    label_set_id: str = Body(""),
    subset: str = Body("train"),
    current_image: str = Body(""),
    current_image_index: int = Body(-1),
):
    return create_or_update_session(
        session_id, group_id, label_set_id, subset, current_image, current_image_index
    )


@app.get("/api/session/{session_id}")
def api_get_session(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


# ?? Annotation Version Control ?????????????????????????????????


@app.get("/api/annotation_versions")
def api_get_annotation_versions(
    group_id: str, label_set_id: str, subset: str, filename: str
):
    return get_annotation_versions(group_id, label_set_id, subset, filename)


@app.get("/api/version_compare")
def api_compare_versions(version_a: int, version_b: int):
    """Compare two annotation versions and return the diff."""
    va = get_annotation_version_data(version_a)
    vb = get_annotation_version_data(version_b)
    if not va or not vb:
        raise HTTPException(404, "One or both versions not found")

    def parse_lines(data_str):
        if not data_str:
            return []
        return [l.strip() for l in data_str.strip().split("\n") if l.strip()]

    lines_a = set(parse_lines(va.get("data", "")))
    lines_b = set(parse_lines(vb.get("data", "")))
    added = list(lines_b - lines_a)
    removed = list(lines_a - lines_b)
    unchanged = list(lines_a & lines_b)

    return {
        "version_a": {"id": version_a, "created_at": va.get("created_at"), "annotation_count": len(lines_a)},
        "version_b": {"id": version_b, "created_at": vb.get("created_at"), "annotation_count": len(lines_b)},
        "added": len(added),
        "removed": len(removed),
        "unchanged": len(unchanged),
        "diff_lines_added": added[:50],
        "diff_lines_removed": removed[:50],
    }


@app.get("/api/annotation_versions/{version_id}")
def api_get_annotation_version(version_id: int):
    v = get_annotation_version_data(version_id)
    if not v:
        raise HTTPException(404, "Version not found")
    return v


@app.post("/api/annotation_versions/{version_id}/restore")
def api_restore_annotation_version(version_id: int):
    v = get_annotation_version_data(version_id)
    if not v:
        raise HTTPException(404, "Version not found")

    g = _get_group(v["group_id"])
    ls = _get_label_set(g, v["label_set_id"])
    _, lbls_dir = _get_dirs(g, ls, v["subset"])
    label_path = _label_path_for(lbls_dir, v["filename"])

    lines = []
    for ann in v["annotations"]:
        pts = ann.get("points", [])
        if pts and len(pts) >= 4:
            lines.append(f"{ann['class_id']} " + " ".join(f"{p:.6f}" for p in pts))

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    log_action(
        "version_restored",
        "annotation",
        details={"version_id": version_id, "filename": v["filename"]},
        group_id=v["group_id"],
        filename=v["filename"],
    )
    return {"ok": True, "restored_version": v["version_num"], "count": len(lines)}


# ?? Image Flags ????????????????????????????????????????????????


@app.post("/api/image_flags")
def api_set_image_flag(
    group_id: str = Body(...),
    label_set_id: str = Body(...),
    subset: str = Body(...),
    filename: str = Body(...),
    flag_type: str = Body(...),
    flag_value: str = Body(""),
):
    set_image_flag(group_id, label_set_id, subset, filename, flag_type, flag_value)
    log_action(
        "flag_set",
        "annotation",
        details={"flag_type": flag_type, "flag_value": flag_value},
        group_id=group_id,
        filename=filename,
    )
    return {"ok": True}


@app.delete("/api/image_flags")
def api_remove_image_flag(
    group_id: str, label_set_id: str, subset: str, filename: str, flag_type: str
):
    remove_image_flag(group_id, label_set_id, subset, filename, flag_type)
    return {"ok": True}


@app.get("/api/image_flags")
def api_get_image_flags(
    group_id: str, label_set_id: str, subset: str, filename: str = None
):
    return get_image_flags(group_id, label_set_id, subset, filename)


# ?? Audit Log ??????????????????????????????????????????????????


@app.get("/api/audit_log")
def api_get_audit_log(
    limit: int = 100,
    offset: int = 0,
    category: str = None,
    action: str = None,
    start_date: str = None,
    end_date: str = None,
):
    return get_audit_log(limit, offset, category, action, start_date, end_date)


# ?? Statistics ??????????????????????????????????????????????????


@app.get("/api/stats/daily")
def api_get_daily_stats(days: int = 30, group_id: str = None):
    return get_daily_stats(days, group_id)


@app.get("/api/stats/summary")
def api_get_stats_summary(days: int = 30, group_id: str = None):
    return get_stats_summary(days, group_id)


@app.get("/api/stats/overview")
def api_get_stats_overview():
    _ensure_groups_fresh()
    total_groups = len(IMAGE_GROUPS)
    total_images = sum(g.get("train_count", 0) + g.get("val_count", 0) for g in IMAGE_GROUPS)
    total_labeled = 0
    for g in IMAGE_GROUPS:
        first_ls = g.get("label_sets", [])
        if not first_ls:
            continue
        ls = first_ls[0]
        for subset in ("train", "val"):
            _, lbls_dir = _get_dirs(g, ls, subset)
            # mtime-cached; subsequent calls within the same TTL are O(1).
            total_labeled += _label_dir_summary(lbls_dir)["labeled"]

    summary = get_stats_summary(30)
    recent_exports = get_export_history(limit=5)

    return {
        "total_datasets": total_groups,
        "total_images": total_images,
        "total_labeled": total_labeled,
        "label_percentage": round(total_labeled / max(total_images, 1) * 100, 1),
        "recent_activity": summary,
        "recent_exports": recent_exports,
    }





# ?? User Preferences ??????????????????????????????????????????


@app.get("/api/preferences")
def api_get_preferences(user_id: str = "default"):
    return get_user_preferences(user_id)


@app.put("/api/preferences")
def api_save_preferences(
    user_id: str = Body("default"),
    preferences: dict = Body(None),
    keyboard_shortcuts: dict = Body(None),
    ui_state: dict = Body(None),
):
    save_user_preferences(user_id, preferences, keyboard_shortcuts, ui_state)
    return {"ok": True}


# ?? Dataset Tags ???????????????????????????????????????????????


@app.post("/api/dataset_tags")
def api_add_tag(group_id: str = Body(...), tag: str = Body(...)):
    add_dataset_tag(group_id, tag)
    return {"ok": True}


@app.delete("/api/dataset_tags")
def api_remove_tag(group_id: str, tag: str):
    remove_dataset_tag(group_id, tag)
    return {"ok": True}


@app.get("/api/dataset_tags/{group_id}")
def api_get_dataset_tags(group_id: str):
    return get_dataset_tags(group_id)


@app.get("/api/tags")
def api_get_all_tags():
    return get_all_tags()


# ?? Export History ?????????????????????????????????????????????


@app.get("/api/export_history")
def api_get_export_history(group_id: str = None, limit: int = 50):
    return get_export_history(group_id, limit)


# ?? Multi-Format Export ????????????????????????????????????????


@app.post("/api/export/coco")
def api_export_coco(
    group_id: str = Body(...),
    label_set_id: str = Body(...),
    subset: str = Body("train"),
):
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")

    files = _list_image_files(imgs_dir)
    names = _class_name_map(g)

    coco = {
        "info": {
            "description": f"BALF Annotation Export - {g['group_name']}",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "categories": [
            {"id": cid, "name": name, "supercategory": "cell"}
            for cid, name in sorted(names.items())
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    total_ann_count = 0
    for img_idx, fp in enumerate(files):
        fn = os.path.basename(fp)
        anns = _read_annotations(_label_path_for(lbls_dir, fn))
        if not anns:
            continue

        try:
            w, h = _image_size_exif(fp)
        except Exception:
            continue

        coco["images"].append(
            {"id": img_idx + 1, "file_name": fn, "width": w, "height": h}
        )

        for ann in anns:
            pts = ann["points"]
            if ann["ann_type"] == "bbox" and len(pts) == 4:
                cx, cy, bw, bh = pts
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                bw_px = bw * w
                bh_px = bh * h
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_idx + 1,
                        "category_id": ann["class_id"],
                        "bbox": [round(x, 1), round(y, 1), round(bw_px, 1), round(bh_px, 1)],
                        "area": round(bw_px * bh_px, 1),
                        "iscrowd": 0,
                    }
                )
            elif len(pts) >= 6:
                seg = [round(pts[i] * (w if i % 2 == 0 else h), 1) for i in range(len(pts))]
                xs = [seg[i] for i in range(0, len(seg), 2)]
                ys = [seg[i] for i in range(1, len(seg), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_idx + 1,
                        "category_id": ann["class_id"],
                        "segmentation": [seg],
                        "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                        "area": round((x2 - x1) * (y2 - y1), 1),
                        "iscrowd": 0,
                    }
                )
            ann_id += 1
            total_ann_count += 1

    export_dir = os.path.join(os.path.dirname(imgs_dir), "exports")
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, f"coco_{subset}_{ts}.json")

    with open(export_path, "w", encoding="utf-8") as f:
        _json.dump(coco, f, ensure_ascii=False, indent=2)

    record_export(group_id, label_set_id, "coco", export_path, len(coco["images"]), total_ann_count)
    log_action("export_coco", "export", group_id=group_id, details={"path": export_path, "images": len(coco["images"])})

    return {
        "ok": True,
        "format": "coco",
        "path": export_path,
        "images": len(coco["images"]),
        "annotations": total_ann_count,
    }


@app.post("/api/export/voc")
def api_export_voc(
    group_id: str = Body(...),
    label_set_id: str = Body(...),
    subset: str = Body("train"),
):
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")

    files = _list_image_files(imgs_dir)
    names = _class_name_map(g)

    export_dir = os.path.join(os.path.dirname(imgs_dir), "exports", "voc")
    ann_dir = os.path.join(export_dir, "Annotations")
    os.makedirs(ann_dir, exist_ok=True)

    exported = 0
    total_anns = 0
    for fp in files:
        fn = os.path.basename(fp)
        anns = _read_annotations(_label_path_for(lbls_dir, fn))
        if not anns:
            continue

        try:
            w, h = _image_size_exif(fp)
        except Exception:
            continue

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<annotation>",
            f"  <filename>{fn}</filename>",
            f"  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>",
        ]

        for ann in anns:
            bbox = _ann_bbox(ann)
            if not bbox:
                continue
            x1, y1, x2, y2 = (
                int(bbox[0] * w),
                int(bbox[1] * h),
                int(bbox[2] * w),
                int(bbox[3] * h),
            )
            class_name = names.get(ann["class_id"], f"class_{ann['class_id']}")
            xml_lines.extend(
                [
                    "  <object>",
                    f"    <name>{class_name}</name>",
                    "    <bndbox>",
                    f"      <xmin>{x1}</xmin><ymin>{y1}</ymin>",
                    f"      <xmax>{x2}</xmax><ymax>{y2}</ymax>",
                    "    </bndbox>",
                    "  </object>",
                ]
            )
            total_anns += 1

        xml_lines.append("</annotation>")
        base = os.path.splitext(fn)[0]
        with open(os.path.join(ann_dir, base + ".xml"), "w", encoding="utf-8") as f:
            f.write("\n".join(xml_lines))
        exported += 1

    record_export(group_id, label_set_id, "voc", export_dir, exported, total_anns)
    log_action("export_voc", "export", group_id=group_id)

    return {"ok": True, "format": "voc", "path": export_dir, "images": exported, "annotations": total_anns}


@app.post("/api/export/masks")
def api_export_masks(
    group_id: str = Body(...),
    label_set_id: str = Body(...),
    subset: str = Body("train"),
):
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")

    files = _list_image_files(imgs_dir)
    export_dir = os.path.join(os.path.dirname(imgs_dir), "exports", "masks")
    os.makedirs(export_dir, exist_ok=True)

    exported = 0
    for fp in files:
        fn = os.path.basename(fp)
        anns = _read_annotations(_label_path_for(lbls_dir, fn))
        if not anns:
            continue

        try:
            w, h = _image_size_exif(fp)
        except Exception:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            class_mask = _annotation_to_mask(ann, h, w)
            mask[class_mask] = ann["class_id"] + 1

        base = os.path.splitext(fn)[0]
        cv2.imwrite(os.path.join(export_dir, base + ".png"), mask)
        exported += 1

    record_export(group_id, label_set_id, "mask", export_dir, exported, 0)
    log_action("export_masks", "export", group_id=group_id)

    return {"ok": True, "format": "mask", "path": export_dir, "images": exported}


@app.post("/api/export/csv")
def api_export_csv_stats(
    group_id: str = Body(...),
    label_set_id: str = Body(...),
    subset: str = Body("train"),
):
    g = _get_group(group_id)
    ls = _get_label_set(g, label_set_id)
    imgs_dir, lbls_dir = _get_dirs(g, ls, subset)
    if not os.path.isdir(imgs_dir):
        raise HTTPException(404, "Image directory not found")

    files = _list_image_files(imgs_dir)
    names = _class_name_map(g)

    rows = ["filename,total_annotations," + ",".join(names.get(i, f"class_{i}") for i in range(g.get("nc", 0)))]
    for fp in files:
        fn = os.path.basename(fp)
        anns = _read_annotations(_label_path_for(lbls_dir, fn))
        counts = {i: 0 for i in range(g.get("nc", 0))}
        for ann in anns:
            cid = ann["class_id"]
            if cid in counts:
                counts[cid] += 1
        row = [fn, str(len(anns))] + [str(counts.get(i, 0)) for i in range(g.get("nc", 0))]
        rows.append(",".join(row))

    export_dir = os.path.join(os.path.dirname(imgs_dir), "exports")
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, f"stats_{subset}_{ts}.csv")

    with open(export_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(rows))

    record_export(group_id, label_set_id, "csv", export_path, len(files), 0)

    return {"ok": True, "format": "csv", "path": export_path, "rows": len(rows) - 1}
