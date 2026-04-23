#!/usr/bin/env python3
"""
Batch preprocess PBC_dataset_normal_DIB by extracting the center WBC mask
from each image using CellposeSAM.

Outputs:
- /home/xut/csclip/cell_datasets/PBC_dataset_normal_DIB_processed/masks/<class>/<stem>.png
- /home/xut/csclip/cell_datasets/PBC_dataset_normal_DIB_processed/meta/pbc_instances.jsonl
- /home/xut/csclip/cell_datasets/PBC_dataset_normal_DIB_processed/meta/pbc_summary.json

Rule:
- Run Cellpose cpsam on each image
- Select the instance whose centroid is closest to the image center
- Save binary mask + bbox + status metadata
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SOURCE_ROOT = Path('/home/xut/csclip/cell_datasets/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB')
OUTPUT_ROOT = Path('/home/xut/csclip/cell_datasets/PBC_dataset_normal_DIB_processed')
MASK_ROOT = OUTPUT_ROOT / 'masks'
META_ROOT = OUTPUT_ROOT / 'meta'
JSONL_PATH = META_ROOT / 'pbc_instances.jsonl'
SUMMARY_PATH = META_ROOT / 'pbc_summary.json'

CLASSES = [
    'basophil',
    'eosinophil',
    'erythroblast',
    'ig',
    'lymphocyte',
    'monocyte',
    'neutrophil',
    'platelet',
]


@dataclass
class Record:
    image_path: str
    class_name: str
    class_id: int
    width: int
    height: int
    mask_path: str
    bbox: list[int]
    selected_instance_id: int
    n_instances: int
    mask_area: int
    fallback: bool
    status: str


_model = None


def get_model():
    global _model
    if _model is None:
        from cellpose import models
        import torch

        _model = models.CellposeModel(
            gpu=torch.cuda.is_available(),
            pretrained_model='cpsam',
        )
    return _model


def segment_center_cell(img_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], int, int, bool]:
    h, w = img_rgb.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    model = get_model()
    masks, _, _ = model.eval(
        img_rgb,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=-2.0,
        channels=[0, 0],
    )

    ids = np.unique(masks)
    ids = ids[ids > 0]

    if len(ids) == 0:
        mask = np.ones((h, w), dtype=bool)
        bbox = (0, 0, w, h)
        return mask, bbox, -1, 0, True

    best_id = -1
    best_dist = float('inf')
    for iid in ids:
        inst_mask = masks == iid
        ys, xs = np.where(inst_mask)
        if len(ys) == 0:
            continue
        dist = ((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = int(iid)

    mask = masks == best_id
    ys, xs = np.where(mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bbox = (x1, y1, x2, y2)
    return mask, bbox, best_id, int(len(ids)), False


def save_mask(mask: np.ndarray, class_name: str, stem: str) -> Path:
    out_dir = MASK_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{stem}.png'
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)
    return out_path


def iter_images() -> list[tuple[str, int, Path]]:
    items: list[tuple[str, int, Path]] = []
    for class_id, class_name in enumerate(CLASSES):
        class_dir = SOURCE_ROOT / class_name
        for path in sorted(class_dir.glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            items.append((class_name, class_id, path))
    return items


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def load_processed_image_paths(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_path = obj.get('image_path')
            status = obj.get('status')
            if image_path and status in {'ok', 'fallback_full_image', 'error'}:
                processed.add(str(image_path))
    return processed


def main() -> None:
    META_ROOT.mkdir(parents=True, exist_ok=True)
    MASK_ROOT.mkdir(parents=True, exist_ok=True)

    items = iter_images()
    total = len(items)
    processed_paths = load_processed_image_paths(JSONL_PATH)
    print(f'[PBC preprocess] total images: {total}', flush=True)
    print(f'[Resume] existing records: {len(processed_paths)}', flush=True)
    print(f'[Output root] {OUTPUT_ROOT}', flush=True)

    processed = 0
    fallback_count = 0
    per_class = {c: {'total': 0, 'processed': 0, 'fallback': 0} for c in CLASSES}
    started = time.time()

    for idx, (class_name, class_id, img_path) in enumerate(items, start=1):
        per_class[class_name]['total'] += 1
        if str(img_path) in processed_paths:
            per_class[class_name]['processed'] += 1
            continue
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            h, w = img.shape[:2]
            mask, bbox, selected_id, n_instances, fallback = segment_center_cell(img)
            mask_path = save_mask(mask, class_name, img_path.stem)

            if fallback:
                fallback_count += 1
                per_class[class_name]['fallback'] += 1

            rec = Record(
                image_path=str(img_path),
                class_name=class_name,
                class_id=class_id,
                width=w,
                height=h,
                mask_path=str(mask_path),
                bbox=[int(v) for v in bbox],
                selected_instance_id=int(selected_id),
                n_instances=int(n_instances),
                mask_area=int(mask.sum()),
                fallback=bool(fallback),
                status='ok' if not fallback else 'fallback_full_image',
            )
            append_jsonl(JSONL_PATH, asdict(rec))
            processed += 1
            per_class[class_name]['processed'] += 1
        except Exception as exc:
            rec = {
                'image_path': str(img_path),
                'class_name': class_name,
                'class_id': class_id,
                'status': 'error',
                'error': repr(exc),
            }
            append_jsonl(JSONL_PATH, rec)

        if idx == 1 or idx % 50 == 0:
            elapsed = time.time() - started
            speed = idx / elapsed if elapsed > 0 else 0.0
            eta = (total - idx) / speed if speed > 0 else -1
            print(
                f'[{idx}/{total}] processed={processed} fallback={fallback_count} '
                f'speed={speed:.2f} img/s eta={eta/60:.1f} min',
                flush=True,
            )

    summary = {
        'source_root': str(SOURCE_ROOT),
        'output_root': str(OUTPUT_ROOT),
        'jsonl_path': str(JSONL_PATH),
        'total_images': total,
        'processed': processed,
        'fallback_count': fallback_count,
        'elapsed_sec': time.time() - started,
        'classes': per_class,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[Done] summary saved to {SUMMARY_PATH}', flush=True)


if __name__ == '__main__':
    main()
