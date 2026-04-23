#!/usr/bin/env python3
"""
Step 1: Pre-extract all features and save to disk.
Run once per model, then classification experiments can iterate without re-encoding.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import time
import gc
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from biomedclip_zeroshot_cell_classify import InstanceInfo

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
WEIGHTS_DIR = Path("/home/xut/csclip/model_weights")
CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
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
    for ip in sorted(idir.glob("*.png")):
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
                              min(w, int(np.max(cc))+1), min(h, int(np.max(rr))+1)), mask=mask)


def crop_cell(image, inst, margin=0.15, mask_bg=False, bg_val=128):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()
    if mask_bg:
        mc = inst.mask[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
        crop = np.where(mc[..., None], crop, np.full_like(crop, bg_val))
    return crop


def compute_granule_morphology(image, inst):
    """40-dim morphology with granule features."""
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
    r, g, b = pixels[:,0].astype(float), pixels[:,1].astype(float), pixels[:,2].astype(float)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lm = lap[mask]
    
    gabor_responses = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.2]:
            kern = cv2.getGaborKernel((9,9), 2.0, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            gabor_responses.append(float(np.mean(np.abs(filtered[mask]))))
    
    pad_gray = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    lbp_val = np.zeros_like(gray, dtype=float)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        lbp_val += (pad_gray[1+dy:gray.shape[0]+1+dy, 1+dx:gray.shape[1]+1+dx] > gray).astype(float)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh & (mask.astype(np.uint8)*255)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 2]
    n_granules = len(areas) / max(1, np.sum(mask)/100)
    mean_gs = float(np.mean(areas))/100 if areas else 0
    std_gs = float(np.std(areas))/100 if len(areas)>1 else 0
    
    hist = cv2.calcHist([gray], [0], mask.astype(np.uint8)*255, [16], [0,256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    m_g, s_g = float(np.mean(gm)), float(np.std(gm))+1e-6
    dark_thr = np.percentile(gm, 25)
    dark_mask = (gray < dark_thr) & mask
    dark_area = np.sum(dark_mask)
    n_lobes = 0
    if dark_area > 10:
        cnts_d, _ = cv2.findContours(dark_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_lobes = len([c for c in cnts_d if cv2.contourArea(c) > 5])
    edges = cv2.Canny(gray, 50, 150)
    em = edges[mask]
    
    extra = np.array([
        float(np.mean(hp[:,0]))/180, float(np.std(hp[:,0]))/180,
        float(np.mean(hp[:,1]))/255, float(np.std(hp[:,1]))/255,
        float(np.mean(hp[:,2]))/255, float(np.std(hp[:,2]))/255,
        float(np.mean(r/(g+1e-6))), float(np.mean((r-g)/(r+g+1e-6))), float(np.mean((r-b)/(r+b+1e-6))),
        float(np.var(lm))/1000 if len(lm)>0 else 0,
        float(np.mean(np.abs(lm)))/100 if len(lm)>0 else 0,
        float(np.mean(gabor_responses))/100, float(np.std(gabor_responses))/100,
        float(np.mean(lbp_val[mask]))/8, float(np.std(lbp_val[mask]))/8,
        n_granules, mean_gs, std_gs,
        float(-np.sum(hist*np.log(hist+1e-10))),
        float(np.mean(((gm.astype(float)-m_g)/s_g)**3)),
        float(dark_area)/(float(np.sum(mask))+1e-6), float(n_lobes)/5,
        float(np.sum(em>0)/len(em)) if len(em)>0 else 0,
        float(np.sum(gm<dark_thr)/len(gm)),
        float(np.percentile(r, 75)-np.percentile(r, 25))/255,
        float(np.mean(r>g)),
        float(np.mean(r))/255 - float(np.mean(g))/255,
        float(np.std(r)-np.std(g))/255,
    ], dtype=np.float32)
    return np.concatenate([base, extra])


def extract_model(model_name, cells, split_name):
    """Extract features for all cells using a single model."""
    print(f"\nExtracting {model_name} for {split_name} ({len(cells)} cells)...")
    
    if model_name == "biomedclip":
        from labeling_tool.fewshot_biomedclip import _load_model_bundle
        b = _load_model_bundle("auto")
        model, preprocess = b["model"], b["preprocess"]
        
        def encode(img, inst):
            cc = crop_cell(img, inst, margin=0.10, mask_bg=True)
            ctx = crop_cell(img, inst, margin=0.30, mask_bg=False)
            ct = preprocess(Image.fromarray(cc)).unsqueeze(0).to(DEVICE)
            cxt = preprocess(Image.fromarray(ctx)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                cf = model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
                xf = model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
            fused = 0.85*cf + 0.15*xf
            fused /= fused.norm(dim=-1, keepdim=True)
            return fused.squeeze(0).cpu().numpy().astype(np.float32)
    elif model_name == "dinov2_s":
        m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
        state = torch.load(WEIGHTS_DIR / 'dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
        m.load_state_dict(state, strict=False)
        m = m.to(DEVICE).eval()
        t = transforms.Compose([transforms.Resize(518), transforms.CenterCrop(518),
            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        def encode(img, inst):
            crop = crop_cell(img, inst, margin=0.15, mask_bg=False)
            inp = t(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                f = m(inp); f /= f.norm(dim=-1, keepdim=True)
            return f.squeeze(0).cpu().numpy().astype(np.float32)
    else:
        raise ValueError(model_name)
    
    feats = []
    labels = []
    morphs = []
    t0 = time.time()
    for i, cell in enumerate(cells):
        img = np.array(Image.open(cell["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
        if inst is None:
            feats.append(None)
            labels.append(cell["ann"]["class_id"])
            morphs.append(None)
            continue
        f = encode(img, inst)
        morph = compute_granule_morphology(img, inst)
        feats.append(f)
        labels.append(cell["ann"]["class_id"])
        morphs.append(morph)
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(cells)} ({time.time()-t0:.1f}s)")
    
    valid = [i for i in range(len(feats)) if feats[i] is not None]
    feats_arr = np.stack([feats[i] for i in valid])
    morphs_arr = np.stack([morphs[i] for i in valid])
    labels_arr = np.array([labels[i] for i in valid])
    
    print(f"  Done: {len(valid)} valid cells in {time.time()-t0:.1f}s")
    
    out = CACHE_DIR / f"{model_name}_{split_name}.npz"
    np.savez_compressed(out, feats=feats_arr, morphs=morphs_arr, labels=labels_arr)
    print(f"  Saved to {out}")
    
    # Free GPU
    if model_name == "biomedclip":
        del model
    elif model_name == "dinov2_s":
        del m
    gc.collect()
    torch.cuda.empty_cache()


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    train_cells = build_cell_index("train")
    val_cells = build_cell_index("val")
    print(f"Train: {len(train_cells)} | Val: {len(val_cells)}")
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "biomedclip"
    
    for split_name, cells in [("train", train_cells), ("val", val_cells)]:
        extract_model(model_name, cells, split_name)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
