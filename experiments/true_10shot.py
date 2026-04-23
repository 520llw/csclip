#!/usr/bin/env python3
"""
TRUE 10-shot classification: Zero data leakage.

Reality simulation:
  - User has a NEW cell type dataset with ONLY 10 labeled cells per class
  - No other labeled data exists
  - Feature extractor (BiomedCLIP/DINOv2) is frozen, used as-is
  - Morphology normalization uses ONLY the 40 support cells
  - Val set cells are encoded one-by-one at inference time (no batch normalization)

Models to test:
  1. BiomedCLIP (ViT-B/16, biomedical CLIP)
  2. DINOv2 (ViT-S/14, self-supervised)
  3. BiomedCLIP + Tip-Adapter style cache
  4. Dual-space (visual + morphology)

Tip-Adapter key idea:
  score = alpha * CLIP_logits + (1-alpha) * Cache_logits
  Cache_logits = softmax(-beta * ||f_query - f_support||^2) @ labels_onehot
  This is training-free and only needs the support set.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo
from PIL import Image
from skimage.draw import polygon as sk_polygon

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


# ========== Data loading ==========

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
    """Build index of all cells with their image paths and annotation info."""
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


def metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    pc, f1s = {}, []
    for c in cids:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        pp = sum(1 for p in pred if p == c)
        gp = sum(1 for g in gt if g == c)
        pr = tp / pp if pp else 0.0
        rc = tp / gp if gp else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        pc[c] = {"p": pr, "r": rc, "f1": f1, "n": gp}
        f1s.append(f1)
    return {"acc": correct/total if total else 0, "mf1": float(np.mean(f1s)), "pc": pc}


# ========== Feature encoding ==========

def encode_cell_biomedclip(model, preprocess, device, image, inst,
                            cell_margin=0.10, ctx_margin=0.30, bg=128, cw=0.85, ctxw=0.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw*cell_margin), int(bh*cell_margin)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()
    mc = inst.mask[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
    crop = np.where(mc[..., None], crop, np.full_like(crop, bg))
    cmx, cmy = int(bw*ctx_margin), int(bh*ctx_margin)
    ctx = image[max(0,y1-cmy):min(h,y2+cmy), max(0,x1-cmx):min(w,x2+cmx)].copy()
    ct = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    cxt = preprocess(Image.fromarray(ctx)).unsqueeze(0).to(device)
    with torch.no_grad():
        cf = model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
        xf = model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
    fused = cw*cf + ctxw*xf
    fused /= fused.norm(dim=-1, keepdim=True)
    return fused.squeeze(0).cpu().numpy().astype(np.float32)


def encode_cell_dinov2(dinov2_model, dinov2_transform, device, image, inst, cell_margin=0.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw*cell_margin), int(bh*cell_margin)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
    pil = Image.fromarray(crop)
    t = dinov2_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        f = dinov2_model(t)
        f /= f.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu().numpy().astype(np.float32)


def compute_morphology(image, inst):
    """30-dim enhanced morphology features."""
    from biomedclip_query_adaptive_classifier import compute_morphology_features
    base = compute_morphology_features(image=image, instance=inst)
    x1, y1, x2, y2 = inst.bbox
    cell_region = image[y1:y2, x1:x2].copy()
    mask_region = inst.mask[y1:y2, x1:x2]
    if cell_region.size == 0 or not mask_region.any():
        return np.concatenate([base, np.zeros(18, dtype=np.float32)])
    pixels = cell_region[mask_region]
    hsv_region = cv2.cvtColor(cell_region, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_region[mask_region]
    h_mean = float(np.mean(hsv_pixels[:, 0])) / 180.0
    h_std = float(np.std(hsv_pixels[:, 0])) / 180.0
    s_mean = float(np.mean(hsv_pixels[:, 1])) / 255.0
    s_std = float(np.std(hsv_pixels[:, 1])) / 255.0
    v_mean = float(np.mean(hsv_pixels[:, 2])) / 255.0
    v_std = float(np.std(hsv_pixels[:, 2])) / 255.0
    r, g, b = pixels[:, 0].astype(float), pixels[:, 1].astype(float), pixels[:, 2].astype(float)
    red_dominance = float(np.mean(r / (g + 1e-6)))
    rg_ratio = float(np.mean((r - g) / (r + g + 1e-6)))
    rb_ratio = float(np.mean((r - b) / (r + b + 1e-6)))
    gray = cv2.cvtColor(cell_region, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_masked = laplacian[mask_region]
    granule_intensity = float(np.var(lap_masked)) / 1000.0 if len(lap_masked) > 0 else 0.0
    granule_mean = float(np.mean(np.abs(lap_masked))) / 100.0 if len(lap_masked) > 0 else 0.0
    hist = cv2.calcHist([gray], [0], mask_region.astype(np.uint8)*255, [16], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-6)
    hist_entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
    gray_m = gray[mask_region]
    mean_g = float(np.mean(gray_m)); std_g = float(np.std(gray_m)) + 1e-6
    hist_skewness = float(np.mean(((gray_m.astype(float) - mean_g) / std_g) ** 3))
    dark_thr = np.percentile(gray_m, 25)
    dark_ratio = float(np.sum(gray_m < dark_thr) / len(gray_m))
    edges = cv2.Canny(gray, 50, 150)
    edge_m = edges[mask_region]
    edge_density = float(np.sum(edge_m > 0) / len(edge_m)) if len(edge_m) > 0 else 0.0
    dark_mask = (gray < dark_thr) & mask_region
    dark_area = np.sum(dark_mask)
    if dark_area > 10:
        dark_u8 = dark_mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(dark_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_dark = len([c for c in cnts if cv2.contourArea(c) > 5])
    else:
        n_dark = 0
    extra = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std,
                      red_dominance, rg_ratio, rb_ratio, granule_intensity, granule_mean,
                      hist_entropy, hist_skewness, dark_ratio, edge_density,
                      float(n_dark)/5.0, float(dark_area)/(float(np.sum(mask_region))+1e-6), 0.0],
                     dtype=np.float32)
    return np.concatenate([base, extra])


def encode_single_cell(image_path, ann, idx, model, preprocess, device, model_type="biomedclip",
                        dinov2_model=None, dinov2_transform=None):
    """Encode a single cell: load image, create instance, extract features."""
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    inst = ann2inst(ann, h, w, idx + 1)
    if inst is None:
        return None
    if model_type == "biomedclip":
        feat = encode_cell_biomedclip(model, preprocess, device, img, inst)
    elif model_type == "dinov2":
        feat = encode_cell_dinov2(dinov2_model, dinov2_transform, device, img, inst)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    morph = compute_morphology(img, inst)
    return {"gt": ann["class_id"], "feat": feat, "morph": morph}


# ========== TRUE 10-shot selection ==========

def select_true_10shot(train_cells, seed):
    """Select exactly 10 cells per class from the cell index.
    Returns the raw cell descriptors (image_path, ann, idx) — NOT features.
    """
    random.seed(seed)
    pc = defaultdict(list)
    for cell in train_cells:
        pc[cell["ann"]["class_id"]].append(cell)
    support_cells = {}
    for c in sorted(CLASS_NAMES.keys()):
        support_cells[c] = random.sample(pc[c], min(N_SHOT, len(pc[c])))
    return support_cells


# ========== Classification strategies ==========

def classify_prototype(query_recs, support_recs, cids):
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support_recs[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)
    gt, pred = [], []
    for r in query_recs:
        scores = [float(r["feat"] @ protos[c]) for c in cids]
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_tip_adapter(query_recs, support_recs, cids, beta=5.0, alpha=0.5):
    """Tip-Adapter: training-free cache model.
    Cache_logits = exp(-beta * ||f_q - f_s||^2) @ one_hot_labels
    Final = alpha * proto_logits + (1-alpha) * cache_logits
    """
    n_classes = len(cids)
    cid_to_idx = {c: i for i, c in enumerate(cids)}
    
    # Build cache from support (ONLY 40 cells)
    cache_keys = []
    cache_labels = []
    for c in cids:
        for s in support_recs[c]:
            cache_keys.append(s["feat"])
            lbl = np.zeros(n_classes, dtype=np.float32)
            lbl[cid_to_idx[c]] = 1.0
            cache_labels.append(lbl)
    cache_keys = np.stack(cache_keys)  # (40, D)
    cache_labels = np.stack(cache_labels)  # (40, C)
    
    # Prototypes for CLIP-like logits
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support_recs[c]])
        p = feats.mean(0)
        protos[c] = p / np.linalg.norm(p)
    
    gt, pred = [], []
    for r in query_recs:
        # Proto logits
        proto_logits = np.array([float(r["feat"] @ protos[c]) for c in cids])
        
        # Cache logits
        diffs = cache_keys - r["feat"]
        dists_sq = np.sum(diffs ** 2, axis=1)
        affinities = np.exp(-beta * dists_sq)
        cache_logits = affinities @ cache_labels
        
        final = alpha * proto_logits + (1 - alpha) * cache_logits
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(final))])
    return metrics(gt, pred, cids)


def classify_dual_space(query_recs, support_recs, cids, vis_w=0.65, morph_w=0.35, topk=7):
    """Dual-space kNN. Morph normalization uses ONLY support cells."""
    # Morph normalization from support ONLY
    all_support_morph = []
    for c in cids:
        for s in support_recs[c]:
            all_support_morph.append(s["morph"])
    all_support_morph = np.stack(all_support_morph)
    g_mean = all_support_morph.mean(0)
    g_std = all_support_morph.std(0) + 1e-8

    sf = {c: np.stack([s["feat"] for s in support_recs[c]]) for c in cids}
    sm = {c: (np.stack([s["morph"] for s in support_recs[c]]) - g_mean) / g_std for c in cids}

    gt, pred = [], []
    for r in query_recs:
        q_m = (r["morph"] - g_mean) / g_std
        scores = []
        for c in cids:
            vis_sims = sf[c] @ r["feat"]
            morph_dists = np.array([np.linalg.norm(q_m - sm[c][i]) for i in range(len(sm[c]))])
            morph_sims = 1.0 / (1.0 + morph_dists)
            combined = vis_w * vis_sims + morph_w * morph_sims
            top = np.sort(combined)[::-1][:topk]
            scores.append(float(top.mean()))
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def classify_tip_adapter_morph(query_recs, support_recs, cids, beta=5.0, alpha=0.3, morph_w=0.3):
    """Tip-Adapter + morphology fusion."""
    n_classes = len(cids)
    cid_to_idx = {c: i for i, c in enumerate(cids)}
    
    cache_keys, cache_labels = [], []
    for c in cids:
        for s in support_recs[c]:
            cache_keys.append(s["feat"])
            lbl = np.zeros(n_classes, dtype=np.float32)
            lbl[cid_to_idx[c]] = 1.0
            cache_labels.append(lbl)
    cache_keys = np.stack(cache_keys)
    cache_labels = np.stack(cache_labels)
    
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support_recs[c]])
        p = feats.mean(0); protos[c] = p / np.linalg.norm(p)
    
    all_sm = []
    for c in cids:
        for s in support_recs[c]:
            all_sm.append(s["morph"])
    all_sm = np.stack(all_sm)
    g_mean = all_sm.mean(0); g_std = all_sm.std(0) + 1e-8
    sm_norm = {c: (np.stack([s["morph"] for s in support_recs[c]]) - g_mean) / g_std for c in cids}
    
    gt, pred = [], []
    for r in query_recs:
        proto_logits = np.array([float(r["feat"] @ protos[c]) for c in cids])
        diffs = cache_keys - r["feat"]
        dists_sq = np.sum(diffs ** 2, axis=1)
        affinities = np.exp(-beta * dists_sq)
        cache_logits = affinities @ cache_labels
        
        q_m = (r["morph"] - g_mean) / g_std
        morph_scores = []
        for c in cids:
            dists = np.array([np.linalg.norm(q_m - sm_norm[c][i]) for i in range(len(sm_norm[c]))])
            morph_scores.append(float(np.mean(1.0 / (1.0 + np.sort(dists)[:5]))))
        morph_scores = np.array(morph_scores)
        
        final = alpha * proto_logits + (1 - alpha - morph_w) * cache_logits + morph_w * morph_scores
        gt.append(r["gt"])
        pred.append(cids[int(np.argmax(final))])
    return metrics(gt, pred, cids)


# ========== Main ==========

def main():
    print("=" * 80)
    print("TRUE 10-SHOT: Zero data leakage experiment")
    print("  Support: 10 cells/class (40 total), encoded individually")
    print("  Query: val set cells, encoded at inference time")
    print("  No global statistics from train set")
    print("=" * 80)

    cids = sorted(CLASS_NAMES.keys())
    train_cells = build_cell_index("train")
    val_cells = build_cell_index("val")
    print(f"Train cell index: {len(train_cells)} cells")
    print(f"Val cell index: {len(val_cells)} cells")

    # Load BiomedCLIP
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    bclip_model = bundle["model"]
    bclip_preprocess = bundle["preprocess"]
    device = bundle["device"]
    print(f"BiomedCLIP loaded on {device}")

    # Try loading DINOv2
    dinov2_model, dinov2_transform = None, None
    try:
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                       source='github', trust_repo=True)
        dinov2_model = dinov2_model.to(device).eval()
        from torchvision import transforms
        dinov2_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("DINOv2 (ViT-S/14) loaded")
    except Exception as e:
        print(f"DINOv2 load failed: {e}")
        print("  Will try local load or skip DINOv2")

    # Strategies to test
    model_configs = [("biomedclip", bclip_model, bclip_preprocess)]
    if dinov2_model is not None:
        model_configs.append(("dinov2", dinov2_model, dinov2_transform))

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "per_class": defaultdict(list)})

    for model_name, model_obj, preproc_obj in model_configs:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        
        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            support_cells = select_true_10shot(train_cells, seed)
            
            # Encode ONLY support cells (true 10-shot)
            t0 = time.time()
            support_recs = {}
            for c in cids:
                recs = []
                for cell in support_cells[c]:
                    r = encode_single_cell(cell["image_path"], cell["ann"], cell["idx"],
                                           model_obj if model_name == "biomedclip" else None,
                                           preproc_obj if model_name == "biomedclip" else None,
                                           device, model_type=model_name,
                                           dinov2_model=model_obj if model_name == "dinov2" else None,
                                           dinov2_transform=preproc_obj if model_name == "dinov2" else None)
                    if r:
                        recs.append(r)
                support_recs[c] = recs
            t_support = time.time() - t0
            print(f"    Support encoding: {t_support:.1f}s ({sum(len(v) for v in support_recs.values())} cells)")
            
            # Encode val cells (inference time)
            t0 = time.time()
            query_recs = []
            for cell in val_cells:
                r = encode_single_cell(cell["image_path"], cell["ann"], cell["idx"],
                                       model_obj if model_name == "biomedclip" else None,
                                       preproc_obj if model_name == "biomedclip" else None,
                                       device, model_type=model_name,
                                       dinov2_model=model_obj if model_name == "dinov2" else None,
                                       dinov2_transform=preproc_obj if model_name == "dinov2" else None)
                if r:
                    query_recs.append(r)
            t_query = time.time() - t0
            print(f"    Query encoding: {t_query:.1f}s ({len(query_recs)} cells)")

            # Test strategies
            strategies = {
                f"{model_name}:proto": lambda qr, sr: classify_prototype(qr, sr, cids),
                f"{model_name}:tip_b5_a0.5": lambda qr, sr: classify_tip_adapter(qr, sr, cids, 5.0, 0.5),
                f"{model_name}:tip_b10_a0.5": lambda qr, sr: classify_tip_adapter(qr, sr, cids, 10.0, 0.5),
                f"{model_name}:tip_b20_a0.5": lambda qr, sr: classify_tip_adapter(qr, sr, cids, 20.0, 0.5),
                f"{model_name}:tip_b5_a0.3": lambda qr, sr: classify_tip_adapter(qr, sr, cids, 5.0, 0.3),
                f"{model_name}:tip_b10_a0.3": lambda qr, sr: classify_tip_adapter(qr, sr, cids, 10.0, 0.3),
                f"{model_name}:dual_v65m35k7": lambda qr, sr: classify_dual_space(qr, sr, cids, 0.65, 0.35, 7),
                f"{model_name}:dual_v60m40k5": lambda qr, sr: classify_dual_space(qr, sr, cids, 0.60, 0.40, 5),
                f"{model_name}:dual_v55m45k5": lambda qr, sr: classify_dual_space(qr, sr, cids, 0.55, 0.45, 5),
                f"{model_name}:tip_morph_b5": lambda qr, sr: classify_tip_adapter_morph(qr, sr, cids, 5.0, 0.3, 0.3),
                f"{model_name}:tip_morph_b10": lambda qr, sr: classify_tip_adapter_morph(qr, sr, cids, 10.0, 0.3, 0.3),
                f"{model_name}:tip_morph_b10a4m2": lambda qr, sr: classify_tip_adapter_morph(qr, sr, cids, 10.0, 0.4, 0.2),
            }

            for sname, fn in strategies.items():
                m = fn(query_recs, support_recs)
                all_results[sname]["acc"].append(m["acc"])
                all_results[sname]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[sname]["per_class"][c].append(m["pc"][c]["f1"])

    # Print results
    print("\n" + "=" * 110)
    print("TRUE 10-SHOT RESULTS (5 seeds, zero data leakage)")
    print("=" * 110)
    header = f"{'Strategy':<35} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 105)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r:
        pc_str = " ".join(f"{np.mean(v['per_class'][c]):>7.4f}" for c in cids)
        print(f"{name:<35} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]}")
    print(f"  Acc = {np.mean(best[1]['acc']):.4f} ± {np.std(best[1]['acc']):.4f}")
    print(f"  mF1 = {np.mean(best[1]['mf1']):.4f} ± {np.std(best[1]['mf1']):.4f}")
    for c in cids:
        print(f"  {CLASS_NAMES[c]}: F1 = {np.mean(best[1]['per_class'][c]):.4f}")

    with open(Path(__file__).parent / "true_10shot_results.json", "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["per_class"][c])) for c in cids}}
                   for n, v in all_results.items()}, f, indent=2)
    print("\nSaved to experiments/true_10shot_results.json")


if __name__ == "__main__":
    main()
