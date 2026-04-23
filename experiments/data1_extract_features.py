#!/usr/bin/env python3
"""
Extract features from BiomedCLIP, Phikon-v2, DINOv2-S for data1_organized.
Saves to feature_cache/data1_{model}_{split}.npz with keys: feats, morphs, labels.
data1 has 7 classes; images are 853x640 JPG, no EXIF rotation issues.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys, time, gc
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from biomedclip_zeroshot_cell_classify import InstanceInfo

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data1_organized")
WEIGHTS_DIR = Path("/home/xut/csclip/model_weights")
CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {0: "CCEC", 1: "RBC", 2: "SEC", 3: "Eosinophil",
               4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_yolo(lp):
    anns = []
    if not lp.exists():
        return anns
    for line in open(lp):
        p = line.strip().split()
        if len(p) < 7:
            continue
        c = int(p[0])
        if c in CLASS_NAMES:
            anns.append({"class_id": c, "points": [float(x) for x in p[1:]]})
    return anns


def build_cell_index(split):
    idir = DATA_ROOT / "images" / split
    ldir = DATA_ROOT / "labels_polygon" / split
    cells = []
    exts = (".jpg", ".jpeg", ".png")
    for ip in sorted(idir.iterdir()):
        if ip.suffix.lower() not in exts:
            continue
        anns = load_yolo(ldir / (ip.stem + ".txt"))
        for i, ann in enumerate(anns):
            cells.append({"image_path": str(ip), "ann": ann, "idx": i})
    return cells


def ann2inst(ann, h, w, iid):
    pts = ann["points"]
    xs = [pts[i] * w for i in range(0, len(pts), 2)]
    ys = [pts[i] * h for i in range(1, len(pts), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(h, w))
    if len(rr) == 0:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return InstanceInfo(instance_id=iid, class_id=ann["class_id"],
                        bbox=(max(0, int(np.min(cc))), max(0, int(np.min(rr))),
                              min(w, int(np.max(cc)) + 1), min(h, int(np.max(rr)) + 1)),
                        mask=mask)


def crop_cell(image, inst, margin=0.15, mask_bg=False, bg_val=128):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    crop = image[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)].copy()
    if mask_bg:
        mc = inst.mask[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        crop = np.where(mc[..., None], crop, np.full_like(crop, bg_val))
    return crop


def compute_granule_morphology(image, inst):
    from biomedclip_query_adaptive_classifier import compute_morphology_features
    base = compute_morphology_features(image=image, instance=inst)
    x1, y1, x2, y2 = inst.bbox
    cell = image[y1:y2, x1:x2].copy()
    mask = inst.mask[y1:y2, x1:x2]
    if cell.size == 0 or not mask.any():
        return np.concatenate([base, np.zeros(28, dtype=np.float32)])
    pixels = cell[mask]
    hsv = cv2.cvtColor(cell, cv2.COLOR_RGB2HSV)
    hp = hsv[mask]
    gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
    gm = gray[mask]
    r, g, b = pixels[:, 0].astype(float), pixels[:, 1].astype(float), pixels[:, 2].astype(float)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lm = lap[mask]

    gabor_responses = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for freq in [0.1, 0.2]:
            kern = cv2.getGaborKernel((9, 9), 2.0, theta, 1.0 / freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            gabor_responses.append(float(np.mean(np.abs(filtered[mask]))))

    pad_gray = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    lbp_val = np.zeros_like(gray, dtype=float)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        lbp_val += (pad_gray[1 + dy:gray.shape[0] + 1 + dy,
                    1 + dx:gray.shape[1] + 1 + dx] > gray).astype(float)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh & (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 2]
    n_granules = len(areas) / max(1, np.sum(mask) / 100)
    mean_gs = float(np.mean(areas)) / 100 if areas else 0
    std_gs = float(np.std(areas)) / 100 if len(areas) > 1 else 0

    hist = cv2.calcHist([gray], [0], mask.astype(np.uint8) * 255, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    m_g, s_g = float(np.mean(gm)), float(np.std(gm)) + 1e-6
    dark_thr = np.percentile(gm, 25)
    dark_mask = (gray < dark_thr) & mask
    dark_area = np.sum(dark_mask)
    n_lobes = 0
    if dark_area > 10:
        cnts_d, _ = cv2.findContours(dark_mask.astype(np.uint8) * 255,
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_lobes = len([c for c in cnts_d if cv2.contourArea(c) > 5])
    edges = cv2.Canny(gray, 50, 150)
    em = edges[mask]

    extra = np.array([
        float(np.mean(hp[:, 0])) / 180, float(np.std(hp[:, 0])) / 180,
        float(np.mean(hp[:, 1])) / 255, float(np.std(hp[:, 1])) / 255,
        float(np.mean(hp[:, 2])) / 255, float(np.std(hp[:, 2])) / 255,
        float(np.mean(r / (g + 1e-6))), float(np.mean((r - g) / (r + g + 1e-6))),
        float(np.mean((r - b) / (r + b + 1e-6))),
        float(np.var(lm)) / 1000 if len(lm) > 0 else 0,
        float(np.mean(np.abs(lm))) / 100 if len(lm) > 0 else 0,
        float(np.mean(gabor_responses)) / 100, float(np.std(gabor_responses)) / 100,
        float(np.mean(lbp_val[mask])) / 8, float(np.std(lbp_val[mask])) / 8,
        n_granules, mean_gs, std_gs,
        float(-np.sum(hist * np.log(hist + 1e-10))),
        float(np.mean(((gm.astype(float) - m_g) / s_g) ** 3)),
        float(dark_area) / (float(np.sum(mask)) + 1e-6), float(n_lobes) / 5,
        float(np.sum(em > 0) / len(em)) if len(em) > 0 else 0,
        float(np.sum(gm < dark_thr) / len(gm)),
        float(np.percentile(r, 75) - np.percentile(r, 25)) / 255,
        float(np.mean(r > g)),
        float(np.mean(r)) / 255 - float(np.mean(g)) / 255,
        float(np.std(r) - np.std(g)) / 255,
    ], dtype=np.float32)
    return np.concatenate([base, extra])


def extract_biomedclip(cells, split_name):
    print(f"\nExtracting BiomedCLIP for {split_name} ({len(cells)} cells)...")
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    b = _load_model_bundle("auto")
    model, preprocess = b["model"], b["preprocess"]

    feats, labels, morphs = [], [], []
    t0 = time.time()
    cur_path, cur_img = None, None
    for i, cell in enumerate(cells):
        if cell["image_path"] != cur_path:
            cur_path = cell["image_path"]
            cur_img = np.array(Image.open(cur_path).convert("RGB"))
        h, w = cur_img.shape[:2]
        inst = ann2inst(cell["ann"], h, w, cell["idx"] + 1)
        if inst is None:
            continue

        cc = crop_cell(cur_img, inst, margin=0.10, mask_bg=True)
        ctx = crop_cell(cur_img, inst, margin=0.30, mask_bg=False)
        ct = preprocess(Image.fromarray(cc)).unsqueeze(0).to(DEVICE)
        cxt = preprocess(Image.fromarray(ctx)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cf = model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
            xf = model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
        fused = 0.85 * cf + 0.15 * xf
        fused /= fused.norm(dim=-1, keepdim=True)
        feats.append(fused.squeeze(0).cpu().numpy().astype(np.float32))

        morph = compute_granule_morphology(cur_img, inst)
        morphs.append(morph)
        labels.append(cell["ann"]["class_id"])

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(cells)} ({time.time() - t0:.1f}s)")

    feats_arr = np.stack(feats)
    morphs_arr = np.stack(morphs)
    labels_arr = np.array(labels)
    print(f"  Done: {len(feats)} cells, feats={feats_arr.shape}, morph={morphs_arr.shape} in {time.time() - t0:.1f}s")

    out = CACHE_DIR / f"data1_biomedclip_{split_name}.npz"
    np.savez_compressed(out, feats=feats_arr, morphs=morphs_arr, labels=labels_arr)
    print(f"  Saved to {out}")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return feats_arr, morphs_arr, labels_arr


def extract_phikon(cells, split_name, morphs_from_bc=None, labels_from_bc=None):
    print(f"\nExtracting Phikon-v2 for {split_name} ({len(cells)} cells)...")
    from transformers import AutoModel, AutoImageProcessor
    model_dir = Path("/home/xut/csclip/model_weights/phikon_v2")
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(DEVICE).eval()

    feats, labels = [], []
    t0 = time.time()
    cur_path, cur_img = None, None
    for i, cell in enumerate(cells):
        if cell["image_path"] != cur_path:
            cur_path = cell["image_path"]
            cur_img = np.array(Image.open(cur_path).convert("RGB"))
        h, w = cur_img.shape[:2]
        inst = ann2inst(cell["ann"], h, w, cell["idx"] + 1)
        if inst is None:
            continue
        crop = crop_cell(cur_img, inst, margin=0.15)
        inputs = processor(images=Image.fromarray(crop), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
            feat = out.last_hidden_state[:, 0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
        labels.append(cell["ann"]["class_id"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(cells)} ({time.time() - t0:.1f}s)")

    feats_arr = np.stack(feats)
    labels_arr = np.array(labels)
    print(f"  Done: {len(feats)} cells, feats={feats_arr.shape} in {time.time() - t0:.1f}s")

    morphs = morphs_from_bc if morphs_from_bc is not None else np.zeros((len(feats), 40))
    out = CACHE_DIR / f"data1_phikon_v2_{split_name}.npz"
    np.savez_compressed(out, feats=feats_arr, morphs=morphs, labels=labels_arr)
    print(f"  Saved to {out}")
    del model; gc.collect(); torch.cuda.empty_cache()


def extract_dinov2s(cells, split_name, morphs_from_bc=None, labels_from_bc=None):
    print(f"\nExtracting DINOv2-S for {split_name} ({len(cells)} cells)...")
    m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                          num_classes=0, img_size=518)
    state = torch.load(WEIGHTS_DIR / 'dinov2_vits14_pretrain.pth',
                       map_location='cpu', weights_only=True)
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE).eval()
    t = transforms.Compose([transforms.Resize(518), transforms.CenterCrop(518),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    feats, labels = [], []
    t0 = time.time()
    cur_path, cur_img = None, None
    for i, cell in enumerate(cells):
        if cell["image_path"] != cur_path:
            cur_path = cell["image_path"]
            cur_img = np.array(Image.open(cur_path).convert("RGB"))
        h, w = cur_img.shape[:2]
        inst = ann2inst(cell["ann"], h, w, cell["idx"] + 1)
        if inst is None:
            continue
        crop = crop_cell(cur_img, inst, margin=0.15)
        inp = t(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = m(inp); f /= f.norm(dim=-1, keepdim=True)
        feats.append(f.squeeze(0).cpu().numpy().astype(np.float32))
        labels.append(cell["ann"]["class_id"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(cells)} ({time.time() - t0:.1f}s)")

    feats_arr = np.stack(feats)
    labels_arr = np.array(labels)
    print(f"  Done: {len(feats)} cells, feats={feats_arr.shape} in {time.time() - t0:.1f}s")

    morphs = morphs_from_bc if morphs_from_bc is not None else np.zeros((len(feats), 40))
    out = CACHE_DIR / f"data1_dinov2_s_{split_name}.npz"
    np.savez_compressed(out, feats=feats_arr, morphs=morphs, labels=labels_arr)
    print(f"  Saved to {out}")
    del m; gc.collect(); torch.cuda.empty_cache()


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Feature Extraction for data1_organized (7 classes)")
    print("=" * 80)

    for split in ["train", "val"]:
        cells = build_cell_index(split)
        print(f"\n{split}: {len(cells)} cells")

        bc_f, bc_m, bc_l = extract_biomedclip(cells, split)
        extract_phikon(cells, split, morphs_from_bc=bc_m, labels_from_bc=bc_l)
        extract_dinov2s(cells, split, morphs_from_bc=bc_m, labels_from_bc=bc_l)

    print("\n\nAll feature extraction complete!")
    for f in sorted(CACHE_DIR.glob("data1_*.npz")):
        d = np.load(f)
        print(f"  {f.name}: feats={d['feats'].shape} morphs={d['morphs'].shape} labels={d['labels'].shape}")


if __name__ == "__main__":
    main()
