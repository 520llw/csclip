"""
Raabin-WBC-top3 feature extraction.

Dataset format: each image is a single cropped WBC; YOLO label (one line) gives class.
Classes: 0=Eosinophil, 1=Lymphocyte, 2=Neutrophil

Pipeline:
  1. Load image; generate mask via Otsu (cell=foreground darker than background)
  2. Build InstanceInfo with bbox = full image, mask = Otsu mask
  3. Extract features: BiomedCLIP (512) + Phikon-v2 (1024) + DINOv2-S (384)
  4. Extract 40-dim morphology
  5. Save to feature_cache/raabin_{backbone}_all.npz
"""
from __future__ import annotations
import os, sys, time, gc
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

sys.path.insert(0, "/home/xut/csclip")
sys.path.insert(0, "/home/xut/csclip/experiments")

from biomedclip_zeroshot_cell_classify import InstanceInfo
import timm

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/Raabin-WBC-top3")
CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS = Path("/home/xut/csclip/model_weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = DATA_ROOT / "images"
LBL_DIR = DATA_ROOT / "labels"


def otsu_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Generate cell mask via Otsu on grayscale.
    Cell is darker than background in WBC stains, so foreground = (gray < threshold)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Keep largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.ones(image_rgb.shape[:2], dtype=bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == largest)
    # Sanity check: if cell fills <5% of image, fall back to full image
    if out.sum() < 0.05 * out.size:
        return np.ones(image_rgb.shape[:2], dtype=bool)
    return out


def build_instance(image_rgb: np.ndarray, class_id: int, iid: int) -> InstanceInfo:
    mask = otsu_mask(image_rgb)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = image_rgb.shape[:2]
        return InstanceInfo(instance_id=iid, class_id=class_id,
                            bbox=(0, 0, w, h), mask=np.ones((h, w), dtype=bool))
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return InstanceInfo(instance_id=iid, class_id=class_id,
                        bbox=(x1, y1, x2, y2), mask=mask)


def crop_cell(image: np.ndarray, inst: InstanceInfo, margin: float = 0.15,
              mask_bg: bool = False, bg_val: int = 128) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin); my = int(bh * margin)
    nx1 = max(0, x1 - mx); ny1 = max(0, y1 - my)
    nx2 = min(w, x2 + mx); ny2 = min(h, y2 + my)
    crop = image[ny1:ny2, nx1:nx2].copy()
    if mask_bg:
        m = inst.mask[ny1:ny2, nx1:nx2]
        crop[~m] = bg_val
    return crop


def compute_morph_40(image: np.ndarray, inst: InstanceInfo) -> np.ndarray:
    """40-dim morphology, same as extract_features.compute_granule_morphology."""
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
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.2]:
            kern = cv2.getGaborKernel((9, 9), 2.0, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F)
            filt = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            gabor_responses.append(float(np.mean(np.abs(filt[mask]))))
    pad_gray = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    lbp = np.zeros_like(gray, dtype=float)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        lbp += (pad_gray[1+dy:gray.shape[0]+1+dy, 1+dx:gray.shape[1]+1+dx] > gray).astype(float)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh & (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 2]
    n_granules = len(areas) / max(1, np.sum(mask)/100)
    mean_gs = float(np.mean(areas))/100 if areas else 0
    std_gs = float(np.std(areas))/100 if len(areas) > 1 else 0
    hist = cv2.calcHist([gray], [0], mask.astype(np.uint8)*255, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    m_g, s_g = float(np.mean(gm)), float(np.std(gm)) + 1e-6
    dark_thr = np.percentile(gm, 25)
    dark_mask = (gray < dark_thr) & mask
    dark_area = int(np.sum(dark_mask))
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
        float(np.mean(lbp[mask]))/8, float(np.std(lbp[mask]))/8,
        n_granules, mean_gs, std_gs,
        float(-np.sum(hist*np.log(hist+1e-10))),
        float(np.mean(((gm.astype(float)-m_g)/s_g)**3)),
        float(dark_area)/(float(np.sum(mask))+1e-6), float(n_lobes)/5,
        float(np.sum(em>0)/len(em)) if len(em) > 0 else 0,
        float(np.sum(gm<dark_thr)/len(gm)),
        float(np.percentile(r, 75)-np.percentile(r, 25))/255,
        float(np.mean(r>g)),
        float(np.mean(r))/255 - float(np.mean(g))/255,
        float(np.std(r)-np.std(g))/255,
    ], dtype=np.float32)
    return np.concatenate([base, extra])


def read_label(label_path: Path) -> int | None:
    """Read YOLO label file, return class id (first line)."""
    if not label_path.exists():
        return None
    with label_path.open() as f:
        line = f.readline().strip()
    if not line:
        return None
    return int(line.split()[0])


def build_index():
    """Return list of (image_path, class_id) for all images with valid labels."""
    cells = []
    for img in sorted(IMG_DIR.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = LBL_DIR / (img.stem + ".txt")
        cid = read_label(lbl)
        if cid is None:
            continue
        cells.append((img, cid))
    return cells


def extract_backbone(model_name, cells, out_path):
    if out_path.exists():
        print(f"  Skip {model_name}: {out_path} exists")
        return
    print(f"\n[{model_name}] extracting {len(cells)} cells -> {out_path.name}")

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
            fused = 0.85 * cf + 0.15 * xf
            fused /= fused.norm(dim=-1, keepdim=True)
            return fused.squeeze(0).cpu().numpy().astype(np.float32)

    elif model_name == "phikon_v2":
        from transformers import AutoModel, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(str(WEIGHTS / "phikon_v2"))
        model = AutoModel.from_pretrained(str(WEIGHTS / "phikon_v2")).to(DEVICE).eval()
        def encode(img, inst):
            crop = crop_cell(img, inst, margin=0.15)
            inputs = proc(images=Image.fromarray(crop), return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs)
                feat = out.last_hidden_state[:, 0]
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)

    elif model_name == "dinov2_s":
        m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                              num_classes=0, img_size=518)
        state = torch.load(WEIGHTS / 'dinov2_vits14_pretrain.pth',
                           map_location='cpu', weights_only=True)
        m.load_state_dict(state, strict=False)
        m = m.to(DEVICE).eval()
        t = transforms.Compose([
            transforms.Resize(518), transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        def encode(img, inst):
            crop = crop_cell(img, inst, margin=0.15, mask_bg=False)
            inp = t(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                f = m(inp); f /= f.norm(dim=-1, keepdim=True)
            return f.squeeze(0).cpu().numpy().astype(np.float32)
        model = m
    else:
        raise ValueError(model_name)

    feats, labels, morphs = [], [], []
    t0 = time.time()
    for i, (img_path, cid) in enumerate(cells):
        img = np.array(Image.open(img_path).convert("RGB"))
        inst = build_instance(img, cid, i + 1)
        try:
            f = encode(img, inst)
        except Exception as e:
            print(f"  encode error on {img_path.name}: {e}")
            continue
        feats.append(f)
        labels.append(cid)
        morphs.append(compute_morph_40(img, inst))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(cells)} ({time.time()-t0:.1f}s)", flush=True)

    feats_arr = np.stack(feats)
    morphs_arr = np.stack(morphs)
    labels_arr = np.array(labels, dtype=np.int64)
    print(f"  Done {len(feats)} in {time.time()-t0:.1f}s. Feats: {feats_arr.shape}")
    np.savez_compressed(out_path, feats=feats_arr, morphs=morphs_arr, labels=labels_arr)
    print(f"  Saved -> {out_path}")

    del model
    if model_name == "dinov2_s":
        del m
    gc.collect()
    torch.cuda.empty_cache()


def main():
    cells = build_index()
    print(f"Total cells indexed: {len(cells)}")
    from collections import Counter
    print("Class distribution:", Counter(c for _, c in cells))

    for name in ["biomedclip", "phikon_v2", "dinov2_s"]:
        out = CACHE_DIR / f"raabin_{name}_all.npz"
        extract_backbone(name, cells, out)

    print("\nAll feature caches ready.")


if __name__ == "__main__":
    main()
