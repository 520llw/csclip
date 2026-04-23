#!/usr/bin/env python3
"""
从与 hybrid adaptive 实验相同的流程中选出 40 个 support，
导出为 labeling_tool 使用的 default_supports.json 格式。
使用：cell_re_data/labels_polygon/train，seed=42，center_picked，10 per class。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from biomedclip_fewshot_prompted_cell_classify import (
    LOCAL_BIOMEDCLIP_DIR,
    collect_support_candidates,
    load_model_and_prompt_features,
)
from biomedclip_query_adaptive_classifier import (
    build_support_records,
    select_support_records,
)
from biomedclip_zeroshot_cell_classify import resolve_device


SHOTS_PER_CLASS = 10
SEED = 42
STRATEGY = "center_picked"
OUT_PATH = Path(__file__).resolve().parent / "default_supports.json"


def get_polygon_points_from_label(label_path: Path, instance_id: int) -> list[float] | None:
    """从 label 文件中读取指定 instance_id 对应行的归一化 polygon 坐标 (x1,y1,x2,y2,...)。"""
    if not label_path.exists():
        return None
    lines = label_path.read_text().splitlines()
    for i, line in enumerate(lines, start=1):
        if i != instance_id:
            continue
        parts = line.strip().split()
        if len(parts) < 7:
            return None
        coords = list(map(float, parts[1:]))
        return coords
    return None


def main():
    device = resolve_device("cuda")
    print("[INFO] Loading model...")
    model, preprocess, *_ = load_model_and_prompt_features(
        device=device,
        local_biomedclip_dir=Path(LOCAL_BIOMEDCLIP_DIR),
        prompt_preset="default",
    )
    print("[INFO] Collecting support candidates (train, labels_polygon)...")
    support_candidates = collect_support_candidates("train")
    print("[INFO] Building support records...")
    support_records = build_support_records(
        candidates=support_candidates,
        model=model,
        preprocess=preprocess,
        device=device,
        cell_margin_ratio=0.15,
        context_margin_ratio=0.30,
        background_value=128,
        cell_scale_weight=0.90,
        context_scale_weight=0.10,
        margin_extra_pixels=0,
    )
    print("[INFO] Selecting supports (center_picked, seed=42, 10 per class)...")
    selected_supports, raw_counts = select_support_records(
        support_records=support_records,
        shots_per_class=SHOTS_PER_CLASS,
        seed=SEED,
        strategy=STRATEGY,
    )
    # Flatten to list of SupportRecord
    selected_list = [
        record
        for records in selected_supports.values()
        for record in records
    ]
    print(f"[INFO] Selected {len(selected_list)} supports. Exporting to {OUT_PATH} ...")
    payload = []
    for record in selected_list:
        c = record.candidate
        label_path = Path(c.label_path)
        points = get_polygon_points_from_label(label_path, c.instance_id)
        if points is None:
            print(f"[WARN] No polygon for {c.image_name} instance_id={c.instance_id}, skip")
            continue
        payload.append({
            "filename": c.image_name,
            "subset": "train",
            "class_id": c.class_id,
            "ann_type": "polygon",
            "points": points,
        })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {len(payload)} supports to {OUT_PATH}")


if __name__ == "__main__":
    main()
