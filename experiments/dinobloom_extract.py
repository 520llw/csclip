#!/usr/bin/env python3
"""
DinoBloom feature extraction for BALF cell crops.

DinoBloom is a hematology-specific DINOv2 model trained on 380k+ WBC images.
It should provide better features for blood cell classification than generic DINOv2.

Uses DinoBloom-B (ViT-B/14, 768d features) for good speed/quality tradeoff.
"""
import sys, gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(line_buffering=True)

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
DATA2_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
MC_ROOT = Path("/home/xut/csclip/cell_datasets/MultiCenter_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}


def load_dinobloom(variant="b"):
    """Load DinoBloom model weights."""
    print(f"Downloading DinoBloom-{variant} weights...", flush=True)
    ckpt_path = hf_hub_download(
        repo_id="MarrLab/DinoBloom",
        filename=f"pytorch_model_{variant}.bin"
    )
    print(f"Loading model from {ckpt_path}...", flush=True)

    if variant == "s":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
        feat_dim = 384
    elif variant == "b":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
        feat_dim = 768
    elif variant == "l":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False)
        feat_dim = 1024
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=False)
        feat_dim = 1536

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "teacher" in ckpt:
        state = {k.replace("teacher.backbone.", ""): v
                 for k, v in ckpt["teacher"].items()
                 if "backbone" in k}
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    model.eval()
    model.cuda()
    print(f"DinoBloom-{variant} loaded. Feature dim: {feat_dim}", flush=True)
    return model, feat_dim


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def extract_cell_crops(data_root, split, img_ext="png"):
    """Extract cell crops from images using polygon annotations."""
    from skimage.draw import polygon as sk_polygon

    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels_polygon" / split

    images = sorted(list(img_dir.glob(f"*.{img_ext}")) +
                    list(img_dir.glob("*.jpg")) +
                    list(img_dir.glob("*.jpeg")))
    images = list(dict.fromkeys(images))

    crops, labels = [], []
    for ip in images:
        lp = lbl_dir / (ip.stem + ".txt")
        if not lp.exists():
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]

        for line in open(lp):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cid = int(parts[0])
            if cid not in CLASS_NAMES:
                continue
            pts = [float(x) for x in parts[1:]]
            xs = [pts[i] * w for i in range(0, len(pts), 2)]
            ys = [pts[i] * h for i in range(1, len(pts), 2)]

            x1, x2 = max(0, int(min(xs)) - 5), min(w, int(max(xs)) + 5)
            y1, y2 = max(0, int(min(ys)) - 5), min(h, int(max(ys)) + 5)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = img[y1:y2, x1:x2]
            crops.append(Image.fromarray(crop))
            labels.append(cid)

    return crops, labels


@torch.no_grad()
def extract_features(model, crops, transform, batch_size=64):
    """Extract features from cell crops."""
    features = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i+batch_size]
        tensors = torch.stack([transform(c) for c in batch]).cuda()
        feats = model(tensors)
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
        features.append(feats.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i//batch_size + 1}/{(len(crops) + batch_size - 1)//batch_size}", flush=True)

    return np.concatenate(features, axis=0)


def process_dataset(model, transform, data_root, dataset_name, prefix="", img_ext="png"):
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}", flush=True)

    for split in ["train", "val"]:
        print(f"\n  Extracting {split} crops...", flush=True)
        crops, labels = extract_cell_crops(data_root, split, img_ext)
        print(f"  Got {len(crops)} crops", flush=True)

        if len(crops) == 0:
            print(f"  WARNING: No crops found!", flush=True)
            continue

        print(f"  Extracting features...", flush=True)
        feats = extract_features(model, crops, transform)
        labels = np.array(labels)

        existing = CACHE_DIR / f"{prefix}biomedclip_{split}.npz"
        if existing.exists():
            d = np.load(existing)
            morphs = d["morphs"]
            print(f"  Loaded morphs from {existing.name}: {morphs.shape}", flush=True)
        else:
            morphs = np.zeros((len(labels), 40))
            print(f"  No morph cache found, using zeros", flush=True)

        out_path = CACHE_DIR / f"{prefix}dinobloom_b_{split}.npz"
        np.savez_compressed(out_path, feats=feats, morphs=morphs, labels=labels)
        print(f"  Saved to {out_path.name}: feats={feats.shape}", flush=True)

        label_counts = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"  Labels: {label_counts}", flush=True)


def main():
    model, feat_dim = load_dinobloom("b")
    transform = get_transform()

    process_dataset(model, transform, DATA2_ROOT, "data2_organized",
                    prefix="", img_ext="png")

    process_dataset(model, transform, MC_ROOT, "MultiCenter_organized",
                    prefix="multicenter_", img_ext="jpg")

    print("\n\nDinoBloom extraction complete!", flush=True)
    print(f"Feature dimension: {feat_dim}", flush=True)

    for f in sorted(CACHE_DIR.glob("dinobloom_*.npz")):
        d = np.load(f)
        print(f"  {f.name}: feats={d['feats'].shape} labels={np.unique(d['labels'], return_counts=True)}")


if __name__ == "__main__":
    main()
