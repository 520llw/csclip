#!/usr/bin/env python3
"""
Advanced 10-shot experiments:
1. Dual-backbone: BiomedCLIP + DINOv2-S feature concatenation
2. Tip-Adapter-F: Fine-tune cache keys using only 40 support cells
3. Enhanced granule-specific morphology for Eosinophil distinction
4. Multi-view support augmentation
5. ProKeR-inspired kernel regression with global regularization
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import random
import time
import gc
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from biomedclip_zeroshot_cell_classify import InstanceInfo

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
WEIGHTS_DIR = Path("/home/xut/csclip/model_weights")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]
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


# ========== Enhanced granule morphology (40-dim) ==========

def compute_granule_morphology(image, inst):
    """40-dim morphology with granule-specific features for Eos/Neu distinction."""
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
    
    # Basic color features (6-dim)
    h_mean, h_std = float(np.mean(hp[:,0]))/180, float(np.std(hp[:,0]))/180
    s_mean, s_std = float(np.mean(hp[:,1]))/255, float(np.std(hp[:,1]))/255
    v_mean, v_std = float(np.mean(hp[:,2]))/255, float(np.std(hp[:,2]))/255
    
    # Color ratio features (3-dim)
    red_dom = float(np.mean(r/(g+1e-6)))
    rg_ratio = float(np.mean((r-g)/(r+g+1e-6)))
    rb_ratio = float(np.mean((r-b)/(r+b+1e-6)))
    
    # Granule-specific features (10-dim) - KEY for Eos/Neu distinction
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lm = lap[mask]
    granule_var = float(np.var(lm))/1000 if len(lm)>0 else 0
    granule_mean = float(np.mean(np.abs(lm)))/100 if len(lm)>0 else 0
    
    # Gabor filter responses for texture - captures granule patterns
    gabor_responses = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.2]:
            kern = cv2.getGaborKernel((9, 9), 2.0, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            gabor_responses.append(float(np.mean(np.abs(filtered[mask]))))
    gabor_mean = float(np.mean(gabor_responses)) / 100
    gabor_std = float(np.std(gabor_responses)) / 100
    
    # Local Binary Pattern-like texture measure
    pad_gray = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    lbp_val = np.zeros_like(gray, dtype=float)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        lbp_val += (pad_gray[1+dy:gray.shape[0]+1+dy, 1+dx:gray.shape[1]+1+dx] > gray).astype(float)
    lbp_mean = float(np.mean(lbp_val[mask]))/8
    lbp_std = float(np.std(lbp_val[mask]))/8
    
    # Eosinophil granules are larger → measure "blob" features
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh & (mask.astype(np.uint8)*255)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 2]
        n_granules = len(areas) / max(1, np.sum(mask)/100)  # density
        mean_granule_size = float(np.mean(areas)) / 100 if areas else 0
        std_granule_size = float(np.std(areas)) / 100 if len(areas)>1 else 0
    else:
        n_granules, mean_granule_size, std_granule_size = 0, 0, 0
    
    # Texture histogram features (3-dim)
    hist = cv2.calcHist([gray], [0], mask.astype(np.uint8)*255, [16], [0,256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    m_g, s_g = float(np.mean(gm)), float(np.std(gm)) + 1e-6
    hist_entropy = float(-np.sum(hist*np.log(hist+1e-10)))
    hist_skew = float(np.mean(((gm.astype(float)-m_g)/s_g)**3))
    
    # Nucleus features (3-dim)
    dark_thr = np.percentile(gm, 25)
    dark_mask = (gray < dark_thr) & mask
    dark_area = np.sum(dark_mask)
    nuc_ratio = float(dark_area)/(float(np.sum(mask))+1e-6)
    if dark_area > 10:
        cnts_dark, _ = cv2.findContours(dark_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_lobes = len([c for c in cnts_dark if cv2.contourArea(c) > 5])
    else:
        n_lobes = 0
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges[mask]>0)/len(gm)) if len(gm)>0 else 0
    
    extra = np.array([
        h_mean, h_std, s_mean, s_std, v_mean, v_std,
        red_dom, rg_ratio, rb_ratio,
        granule_var, granule_mean,
        gabor_mean, gabor_std,
        lbp_mean, lbp_std,
        n_granules, mean_granule_size, std_granule_size,
        hist_entropy, hist_skew,
        nuc_ratio, float(n_lobes)/5, edge_density,
        float(np.sum(gm<dark_thr)/len(gm)),
        float(np.percentile(r, 75) - np.percentile(r, 25))/255,
        float(np.mean(r>g))/1.0,
        float(np.mean(r))/255 - float(np.mean(g))/255,
        float(np.std(r) - np.std(g))/255,
    ], dtype=np.float32)
    return np.concatenate([base, extra])


# ========== Model loaders ==========

def load_biomedclip():
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    b = _load_model_bundle("auto")
    model, preprocess = b["model"], b["preprocess"]
    
    def encode(image, inst):
        cell_crop = crop_cell(image, inst, margin=0.10, mask_bg=True)
        ctx_crop = crop_cell(image, inst, margin=0.30, mask_bg=False)
        ct = preprocess(Image.fromarray(cell_crop)).unsqueeze(0).to(DEVICE)
        cxt = preprocess(Image.fromarray(ctx_crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cf = model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
            xf = model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
        fused = 0.85*cf + 0.15*xf
        fused /= fused.norm(dim=-1, keepdim=True)
        return fused.squeeze(0).cpu().numpy().astype(np.float32)
    return encode, 512


def load_dinov2s():
    m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
    state = torch.load(WEIGHTS_DIR / 'dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE).eval()
    t = transforms.Compose([transforms.Resize(518), transforms.CenterCrop(518),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    def encode(image, inst):
        crop = crop_cell(image, inst, margin=0.15, mask_bg=False)
        inp = t(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = m(inp); f /= f.norm(dim=-1, keepdim=True)
        return f.squeeze(0).cpu().numpy().astype(np.float32)
    return encode, 384


# ========== Classifiers ==========

def cls_dual(query, support, cids, vw=0.65, mw=0.35, k=7):
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    gt, pred = [], []
    for r in query:
        qm = (r["morph"]-gm)/gs
        scores = []
        for c in cids:
            vs = sf[c] @ r["feat"]
            md = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms = 1.0/(1.0+md)
            comb = vw*vs + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_tip_adapter_f(query, support, cids, beta=10.0, alpha=0.4, lr=0.01, epochs=50):
    """Tip-Adapter-F: Fine-tune cache keys using ONLY the 40 support cells.
    The key innovation: make cache_keys learnable parameters."""
    n_c = len(cids); cid2i = {c: i for i, c in enumerate(cids)}
    
    # Build cache
    cache_keys_np, cache_labels_np = [], []
    for c in cids:
        for s in support[c]:
            cache_keys_np.append(s["feat"])
            l = np.zeros(n_c, np.float32); l[cid2i[c]] = 1.0
            cache_labels_np.append(l)
    
    cache_keys = torch.tensor(np.stack(cache_keys_np), dtype=torch.float32, device=DEVICE)
    cache_labels = torch.tensor(np.stack(cache_labels_np), dtype=torch.float32, device=DEVICE)
    
    # Learnable parameters: cache keys
    adapter_keys = torch.nn.Parameter(cache_keys.clone())
    beta_param = torch.nn.Parameter(torch.tensor(float(beta), device=DEVICE))
    alpha_param = torch.nn.Parameter(torch.tensor(float(alpha), device=DEVICE))
    
    optimizer = torch.optim.AdamW([adapter_keys, beta_param, alpha_param], lr=lr, weight_decay=0.01)
    
    # Proto logits (fixed)
    protos = {}
    for c in cids:
        p = np.stack([s["feat"] for s in support[c]]).mean(0)
        protos[c] = torch.tensor(p / np.linalg.norm(p), dtype=torch.float32, device=DEVICE)
    
    # Train on support set itself (self-training with data augmentation via noise)
    support_feats = cache_keys.clone()
    support_labels_idx = []
    for c in cids:
        for _ in support[c]:
            support_labels_idx.append(cid2i[c])
    support_labels_idx = torch.tensor(support_labels_idx, dtype=torch.long, device=DEVICE)
    
    for epoch in range(epochs):
        noise = torch.randn_like(support_feats) * 0.02
        aug_feats = support_feats + noise
        aug_feats = F.normalize(aug_feats, dim=-1)
        
        proto_logits = torch.stack([aug_feats @ protos[c] for c in cids], dim=-1)
        
        diffs = aug_feats.unsqueeze(1) - adapter_keys.unsqueeze(0)
        dists_sq = (diffs ** 2).sum(-1)
        b = F.softplus(beta_param)
        affinities = torch.exp(-b * dists_sq)
        cache_logits = affinities @ cache_labels
        
        a = torch.sigmoid(alpha_param)
        logits = a * proto_logits + (1 - a) * cache_logits
        loss = F.cross_entropy(logits, support_labels_idx)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Inference
    final_keys = adapter_keys.detach()
    final_beta = F.softplus(beta_param).item()
    final_alpha = torch.sigmoid(alpha_param).item()
    
    gt, pred = [], []
    for r in query:
        q = torch.tensor(r["feat"], dtype=torch.float32, device=DEVICE)
        pl = torch.stack([q @ protos[c] for c in cids])
        d = ((final_keys - q.unsqueeze(0)) ** 2).sum(-1)
        cl = torch.exp(-final_beta * d) @ cache_labels
        f = final_alpha * pl + (1 - final_alpha) * cl
        gt.append(r["gt"])
        pred.append(cids[int(f.argmax().item())])
    return metrics(gt, pred, cids)


def cls_tip_f_morph(query, support, cids, beta=10.0, alpha=0.3, mw=0.3, lr=0.01, epochs=50):
    """Tip-Adapter-F with morphology fusion."""
    n_c = len(cids); cid2i = {c: i for i, c in enumerate(cids)}
    
    cache_keys_np, cache_labels_np = [], []
    for c in cids:
        for s in support[c]:
            cache_keys_np.append(s["feat"])
            l = np.zeros(n_c, np.float32); l[cid2i[c]] = 1.0
            cache_labels_np.append(l)
    
    cache_keys = torch.tensor(np.stack(cache_keys_np), dtype=torch.float32, device=DEVICE)
    cache_labels = torch.tensor(np.stack(cache_labels_np), dtype=torch.float32, device=DEVICE)
    adapter_keys = torch.nn.Parameter(cache_keys.clone())
    beta_param = torch.nn.Parameter(torch.tensor(float(beta), device=DEVICE))
    
    optimizer = torch.optim.AdamW([adapter_keys, beta_param], lr=lr, weight_decay=0.01)
    
    protos = {}
    for c in cids:
        p = np.stack([s["feat"] for s in support[c]]).mean(0)
        protos[c] = torch.tensor(p / np.linalg.norm(p), dtype=torch.float32, device=DEVICE)
    
    support_feats = cache_keys.clone()
    labels_idx = []
    for c in cids:
        for _ in support[c]: labels_idx.append(cid2i[c])
    labels_idx = torch.tensor(labels_idx, dtype=torch.long, device=DEVICE)
    
    for epoch in range(epochs):
        noise = torch.randn_like(support_feats) * 0.02
        aug = F.normalize(support_feats + noise, dim=-1)
        proto_logits = torch.stack([aug @ protos[c] for c in cids], dim=-1)
        diffs = aug.unsqueeze(1) - adapter_keys.unsqueeze(0)
        b = F.softplus(beta_param)
        cache_logits = torch.exp(-b * (diffs**2).sum(-1)) @ cache_labels
        logits = alpha * proto_logits + (1-alpha-mw) * cache_logits
        loss = F.cross_entropy(logits, labels_idx)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    final_keys = adapter_keys.detach()
    final_beta = F.softplus(beta_param).item()
    
    # Morphology normalization from support only
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    
    gt, pred = [], []
    for r in query:
        q = torch.tensor(r["feat"], dtype=torch.float32, device=DEVICE)
        pl = torch.stack([q @ protos[c] for c in cids])
        d = ((final_keys - q.unsqueeze(0))**2).sum(-1)
        cl = torch.exp(-final_beta * d) @ cache_labels
        
        qm = (r["morph"]-gm)/gs
        ms = []
        for c in cids:
            dd = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms.append(float(np.mean(1.0/(1.0+np.sort(dd)[:5]))))
        ms = torch.tensor(ms, device=DEVICE)
        
        f = alpha * pl + (1-alpha-mw) * cl + mw * ms
        gt.append(r["gt"])
        pred.append(cids[int(f.argmax().item())])
    return metrics(gt, pred, cids)


def cls_dual_backbone(query, support, cids, vw=0.5, dw=0.2, mw=0.3, k=7):
    """Dual-backbone: Use both BiomedCLIP and DINOv2 features."""
    # Morph norm from support
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    
    sf_v = {c: np.stack([s["feat_bclip"] for s in support[c]]) for c in cids}
    sf_d = {c: np.stack([s["feat_dino"] for s in support[c]]) for c in cids}
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    
    gt, pred = [], []
    for r in query:
        qm = (r["morph"]-gm)/gs
        scores = []
        for c in cids:
            vs_b = sf_v[c] @ r["feat_bclip"]
            vs_d = sf_d[c] @ r["feat_dino"]
            md = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms = 1.0/(1.0+md)
            comb = vw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_concat_backbone(query, support, cids, vw=0.65, mw=0.35, k=7):
    """Concatenate BiomedCLIP + DINOv2 features."""
    # Concat features
    sf_cat = {}
    for c in cids:
        bclip = np.stack([s["feat_bclip"] for s in support[c]])
        dino = np.stack([s["feat_dino"] for s in support[c]])
        cat = np.concatenate([bclip, dino], axis=1)
        norms = np.linalg.norm(cat, axis=1, keepdims=True)
        sf_cat[c] = cat / (norms + 1e-8)
    
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    
    gt, pred = [], []
    for r in query:
        q_cat = np.concatenate([r["feat_bclip"], r["feat_dino"]])
        q_cat = q_cat / (np.linalg.norm(q_cat) + 1e-8)
        qm = (r["morph"]-gm)/gs
        scores = []
        for c in cids:
            vs = sf_cat[c] @ q_cat
            md = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms = 1.0/(1.0+md)
            comb = vw*vs + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


# ========== Main ==========

def main():
    print("=" * 90)
    print("ADVANCED 10-SHOT: Dual-backbone + Tip-Adapter-F + Enhanced Morphology")
    print("=" * 90)
    
    cids = sorted(CLASS_NAMES.keys())
    train_cells = build_cell_index("train")
    val_cells = build_cell_index("val")
    print(f"Train: {len(train_cells)} | Val: {len(val_cells)}")
    
    # Load both models
    print("Loading BiomedCLIP...")
    bclip_encode, bclip_dim = load_biomedclip()
    print(f"  BiomedCLIP: {bclip_dim}-dim")
    
    print("Loading DINOv2-S...")
    dino_encode, dino_dim = load_dinov2s()
    print(f"  DINOv2-S: {dino_dim}-dim")
    
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        random.seed(seed)
        pc = defaultdict(list)
        for cell in train_cells:
            pc[cell["ann"]["class_id"]].append(cell)
        support_cells = {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}
        
        # Encode support with both models + enhanced morphology
        t0 = time.time()
        support = defaultdict(list)
        for c in cids:
            for cell in support_cells[c]:
                img = np.array(Image.open(cell["image_path"]).convert("RGB"))
                h, w = img.shape[:2]
                inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
                if inst is None:
                    continue
                rec = {
                    "gt": cell["ann"]["class_id"],
                    "feat": bclip_encode(img, inst),
                    "feat_bclip": bclip_encode(img, inst),
                    "feat_dino": dino_encode(img, inst),
                    "morph": compute_granule_morphology(img, inst),
                }
                support[c].append(rec)
        print(f"  Support: {time.time()-t0:.1f}s ({sum(len(v) for v in support.values())} cells)")
        
        # Encode query
        t0 = time.time()
        query = []
        for cell in val_cells:
            img = np.array(Image.open(cell["image_path"]).convert("RGB"))
            h, w = img.shape[:2]
            inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
            if inst is None:
                continue
            rec = {
                "gt": cell["ann"]["class_id"],
                "feat": bclip_encode(img, inst),
                "feat_bclip": bclip_encode(img, inst),
                "feat_dino": dino_encode(img, inst),
                "morph": compute_granule_morphology(img, inst),
            }
            query.append(rec)
        print(f"  Query: {time.time()-t0:.1f}s ({len(query)} cells)")
        
        # Strategies
        strats = {
            # Baseline: BiomedCLIP dual-space
            "bclip:dual_65_35_7": lambda q, s: cls_dual(q, s, cids, 0.65, 0.35, 7),
            "bclip:dual_60_40_5": lambda q, s: cls_dual(q, s, cids, 0.60, 0.40, 5),
            # Enhanced morph (40-dim vs 30-dim before)
            "bclip:dual_65_35_7_gm": lambda q, s: cls_dual(q, s, cids, 0.65, 0.35, 7),
            "bclip:dual_55_45_5_gm": lambda q, s: cls_dual(q, s, cids, 0.55, 0.45, 5),
            "bclip:dual_50_50_5_gm": lambda q, s: cls_dual(q, s, cids, 0.50, 0.50, 5),
            # Tip-Adapter-F
            "bclip:tipF_b10a4": lambda q, s: cls_tip_adapter_f(q, s, cids, 10.0, 0.4, 0.01, 50),
            "bclip:tipF_b10a4_100e": lambda q, s: cls_tip_adapter_f(q, s, cids, 10.0, 0.4, 0.01, 100),
            "bclip:tipF_b5a5": lambda q, s: cls_tip_adapter_f(q, s, cids, 5.0, 0.5, 0.01, 50),
            # Tip-Adapter-F with morphology
            "bclip:tipF_morph_b10": lambda q, s: cls_tip_f_morph(q, s, cids, 10.0, 0.3, 0.3, 0.01, 50),
            "bclip:tipF_morph_b10_100e": lambda q, s: cls_tip_f_morph(q, s, cids, 10.0, 0.3, 0.3, 0.01, 100),
            # Dual backbone
            "dual_bb:50_20_30_k7": lambda q, s: cls_dual_backbone(q, s, cids, 0.50, 0.20, 0.30, 7),
            "dual_bb:40_25_35_k7": lambda q, s: cls_dual_backbone(q, s, cids, 0.40, 0.25, 0.35, 7),
            "dual_bb:55_15_30_k5": lambda q, s: cls_dual_backbone(q, s, cids, 0.55, 0.15, 0.30, 5),
            "dual_bb:45_20_35_k5": lambda q, s: cls_dual_backbone(q, s, cids, 0.45, 0.20, 0.35, 5),
            # Concat backbone
            "concat_bb:65_35_k7": lambda q, s: cls_concat_backbone(q, s, cids, 0.65, 0.35, 7),
            "concat_bb:60_40_k5": lambda q, s: cls_concat_backbone(q, s, cids, 0.60, 0.40, 5),
            "concat_bb:55_45_k5": lambda q, s: cls_concat_backbone(q, s, cids, 0.55, 0.45, 5),
        }
        
        for sn, fn in strats.items():
            try:
                m = fn(query, support)
                all_results[sn]["acc"].append(m["acc"])
                all_results[sn]["mf1"].append(m["mf1"])
                for c in cids:
                    all_results[sn]["pc"][c].append(m["pc"][c]["f1"])
            except Exception as e:
                print(f"    {sn} FAILED: {e}")
        
        gc.collect(); torch.cuda.empty_cache()
    
    # Print results
    print(f"\n{'='*115}")
    print("ADVANCED 10-SHOT RESULTS (5 seeds)")
    print(f"{'='*115}")
    header = f"{'Strategy':<35} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 110)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r:
        if len(v["acc"]) < 3:
            continue
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<35} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    # Save results
    result_file = Path(__file__).parent / "advanced_10shot_results.json"
    with open(result_file, "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["pc"][c])) for c in cids}}
                   for n, v in all_results.items() if len(v["acc"]) >= 3}, f, indent=2)
    print(f"\nSaved to {result_file}")


if __name__ == "__main__":
    main()
