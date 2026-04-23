#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'sam3'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo
from biomedclip_fewshot_support_experiment import encode_multiscale_feature
from experiments.extract_features import compute_granule_morphology, crop_cell

SOURCE_JSONL = Path('/home/xut/csclip/cell_datasets/PBC_dataset_normal_DIB_processed/meta/pbc_instances.jsonl')
CACHE_DIR = Path('/home/xut/csclip/experiments/feature_cache')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GROUPS = {
    'all': None,
    'balf4': {'eosinophil', 'neutrophil', 'lymphocyte', 'monocyte'},
}


def load_records(group: str) -> list[dict]:
    allowed = GROUPS[group]
    records = []
    with SOURCE_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('status') not in {'ok', 'fallback_full_image'}:
                continue
            if allowed is not None and rec['class_name'] not in allowed:
                continue
            records.append(rec)
    return records


def record_to_instance(rec: dict) -> InstanceInfo:
    mask = np.array(Image.open(rec['mask_path']).convert('L')) > 0
    x1, y1, x2, y2 = [int(v) for v in rec['bbox']]
    return InstanceInfo(
        instance_id=1,
        class_id=int(rec['class_id']),
        bbox=(x1, y1, x2, y2),
        mask=mask,
    )


def encode_biomedclip(records: list[dict], group: str) -> None:
    from labeling_tool.fewshot_biomedclip import _load_model_bundle

    bundle = _load_model_bundle('auto')
    feats, morphs, labels = [], [], []
    t0 = time.time()
    for i, rec in enumerate(records, start=1):
        image = np.array(Image.open(rec['image_path']).convert('RGB'))
        inst = record_to_instance(rec)
        feat = encode_multiscale_feature(
            model=bundle['model'],
            preprocess=bundle['preprocess'],
            image=image,
            instance=inst,
            device=bundle['device'],
            cell_margin_ratio=0.10,
            context_margin_ratio=0.30,
            background_value=128,
            cell_scale_weight=0.85,
            context_scale_weight=0.15,
        )
        morph = compute_granule_morphology(image, inst)
        feats.append(feat.astype(np.float32))
        morphs.append(morph.astype(np.float32))
        labels.append(int(rec['class_id']))
        if i == 1 or i % 200 == 0:
            print(f'[biomedclip][{group}] {i}/{len(records)}  {time.time()-t0:.1f}s', flush=True)
    out = CACHE_DIR / f'pbc_{group}_biomedclip_all.npz'
    np.savez_compressed(out, feats=np.stack(feats), morphs=np.stack(morphs), labels=np.array(labels, dtype=np.int64))
    print(f'[saved] {out}', flush=True)
    del bundle
    gc.collect()
    torch.cuda.empty_cache()


def encode_phikon(records: list[dict], group: str) -> None:
    from transformers import AutoImageProcessor, AutoModel

    model_dir = Path('/home/xut/csclip/model_weights/phikon_v2')
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(DEVICE).eval()
    feats, labels = [], []
    t0 = time.time()
    for i, rec in enumerate(records, start=1):
        image = np.array(Image.open(rec['image_path']).convert('RGB'))
        inst = record_to_instance(rec)
        crop = crop_cell(image, inst, margin=0.15, mask_bg=False)
        inputs = processor(images=Image.fromarray(crop), return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            feat = outputs.last_hidden_state[:, 0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
        labels.append(int(rec['class_id']))
        if i == 1 or i % 200 == 0:
            print(f'[phikon_v2][{group}] {i}/{len(records)}  {time.time()-t0:.1f}s', flush=True)
    out = CACHE_DIR / f'pbc_{group}_phikon_v2_all.npz'
    bc = np.load(CACHE_DIR / f'pbc_{group}_biomedclip_all.npz')
    np.savez_compressed(out, feats=np.stack(feats), morphs=bc['morphs'], labels=np.array(labels, dtype=np.int64))
    print(f'[saved] {out}', flush=True)
    del model, processor, bc
    gc.collect()
    torch.cuda.empty_cache()


def encode_dinov2(records: list[dict], group: str) -> None:
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
    state = torch.load('/home/xut/csclip/model_weights/dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats, labels = [], []
    t0 = time.time()
    for i, rec in enumerate(records, start=1):
        image = np.array(Image.open(rec['image_path']).convert('RGB'))
        inst = record_to_instance(rec)
        crop = crop_cell(image, inst, margin=0.15, mask_bg=False)
        inp = transform(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model(inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
        labels.append(int(rec['class_id']))
        if i == 1 or i % 200 == 0:
            print(f'[dinov2_s][{group}] {i}/{len(records)}  {time.time()-t0:.1f}s', flush=True)
    out = CACHE_DIR / f'pbc_{group}_dinov2_s_all.npz'
    bc = np.load(CACHE_DIR / f'pbc_{group}_biomedclip_all.npz')
    np.savez_compressed(out, feats=np.stack(feats), morphs=bc['morphs'], labels=np.array(labels, dtype=np.int64))
    print(f'[saved] {out}', flush=True)
    del model, bc
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'biomedclip'
    group = sys.argv[2] if len(sys.argv) > 2 else 'balf4'
    if group not in GROUPS:
        raise ValueError(f'unknown group: {group}')
    records = load_records(group)
    print(f'[group] {group}  n={len(records)}  classes={Counter(r["class_name"] for r in records)}', flush=True)
    if model_name == 'biomedclip':
        encode_biomedclip(records, group)
    elif model_name == 'phikon_v2':
        encode_phikon(records, group)
    elif model_name == 'dinov2_s':
        encode_dinov2(records, group)
    else:
        raise ValueError(f'unknown model: {model_name}')


if __name__ == '__main__':
    main()
