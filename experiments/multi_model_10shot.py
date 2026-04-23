#!/usr/bin/env python3
"""
Multi-model TRUE 10-shot classification experiment.
Zero data leakage: only 10 cells/class for support, all else is inference.

Models:
  1. BiomedCLIP ViT-B/16 (biomedical CLIP, 512-dim)
  2. DINOv2 ViT-S/14 (self-supervised, 384-dim)
  3. DINOv2 ViT-B/14 (self-supervised, 768-dim)
  4. DinoBloom ViT-B/14 (blood cell specialized, 768-dim)

Strategies per model:
  - Prototype (cosine)
  - kNN (k=5,7)
  - Tip-Adapter (training-free cache)
  - Dual-space (visual + 30-dim morphology)
  - Tip-Adapter + morphology
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


# ========== Data ==========

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


# ========== Model loaders ==========

def load_biomedclip():
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    b = _load_model_bundle("auto")
    return {"model": b["model"], "preprocess": b["preprocess"], "dim": 512, "name": "biomedclip",
            "encode_fn": "biomedclip", "input_size": 224}


def load_dinov2_s():
    m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
    state = torch.load(WEIGHTS_DIR / 'dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE).eval()
    t = transforms.Compose([transforms.Resize(518), transforms.CenterCrop(518),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return {"model": m, "transform": t, "dim": 384, "name": "dinov2_s",
            "encode_fn": "dino", "input_size": 518}


def load_dinov2_b():
    m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
    state = torch.load(WEIGHTS_DIR / 'dinov2_vitb14_pretrain.pth', map_location='cpu', weights_only=True)
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE).eval()
    t = transforms.Compose([transforms.Resize(518), transforms.CenterCrop(518),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return {"model": m, "transform": t, "dim": 768, "name": "dinov2_b",
            "encode_fn": "dino", "input_size": 518}


def load_dinobloom():
    m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
    state = torch.load(WEIGHTS_DIR / 'dinobloom_vitb14.pth', map_location='cpu', weights_only=False)
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE).eval()
    t = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return {"model": m, "transform": t, "dim": 768, "name": "dinobloom_b",
            "encode_fn": "dino", "input_size": 224}


# ========== Feature extraction ==========

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


def encode_biomedclip(model_info, image, inst):
    model, preprocess = model_info["model"], model_info["preprocess"]
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


def encode_dino(model_info, image, inst):
    model, transform = model_info["model"], model_info["transform"]
    crop = crop_cell(image, inst, margin=0.15, mask_bg=False)
    t = transform(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = model(t)
        f /= f.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu().numpy().astype(np.float32)


def compute_morphology(image, inst):
    """30-dim enhanced morphology."""
    from biomedclip_query_adaptive_classifier import compute_morphology_features
    base = compute_morphology_features(image=image, instance=inst)
    x1, y1, x2, y2 = inst.bbox
    cell_region = image[y1:y2, x1:x2].copy()
    mask_region = inst.mask[y1:y2, x1:x2]
    if cell_region.size == 0 or not mask_region.any():
        return np.concatenate([base, np.zeros(18, dtype=np.float32)])
    pixels = cell_region[mask_region]
    hsv = cv2.cvtColor(cell_region, cv2.COLOR_RGB2HSV)
    hp = hsv[mask_region]
    gray = cv2.cvtColor(cell_region, cv2.COLOR_RGB2GRAY)
    gm = gray[mask_region]
    r, g, b = pixels[:,0].astype(float), pixels[:,1].astype(float), pixels[:,2].astype(float)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lm = lap[mask_region]
    hist = cv2.calcHist([gray], [0], mask_region.astype(np.uint8)*255, [16], [0,256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    m_g, s_g = float(np.mean(gm)), float(np.std(gm)) + 1e-6
    dark_thr = np.percentile(gm, 25)
    dark_mask = (gray < dark_thr) & mask_region
    dark_area = np.sum(dark_mask)
    n_dark = 0
    if dark_area > 10:
        cnts, _ = cv2.findContours(dark_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_dark = len([c for c in cnts if cv2.contourArea(c) > 5])
    edges = cv2.Canny(gray, 50, 150)
    em = edges[mask_region]
    extra = np.array([
        float(np.mean(hp[:,0]))/180, float(np.std(hp[:,0]))/180,
        float(np.mean(hp[:,1]))/255, float(np.std(hp[:,1]))/255,
        float(np.mean(hp[:,2]))/255, float(np.std(hp[:,2]))/255,
        float(np.mean(r/(g+1e-6))),
        float(np.mean((r-g)/(r+g+1e-6))),
        float(np.mean((r-b)/(r+b+1e-6))),
        float(np.var(lm))/1000 if len(lm)>0 else 0,
        float(np.mean(np.abs(lm)))/100 if len(lm)>0 else 0,
        float(-np.sum(hist*np.log(hist+1e-10))),
        float(np.mean(((gm.astype(float)-m_g)/s_g)**3)),
        float(np.sum(gm<dark_thr)/len(gm)),
        float(np.sum(em>0)/len(em)) if len(em)>0 else 0,
        float(n_dark)/5,
        float(dark_area)/(float(np.sum(mask_region))+1e-6),
        0.0
    ], dtype=np.float32)
    return np.concatenate([base, extra])


def encode_cell(model_info, image_path, ann, idx):
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    inst = ann2inst(ann, h, w, idx + 1)
    if inst is None:
        return None
    enc_fn = encode_biomedclip if model_info["encode_fn"] == "biomedclip" else encode_dino
    feat = enc_fn(model_info, img, inst)
    morph = compute_morphology(img, inst)
    return {"gt": ann["class_id"], "feat": feat, "morph": morph}


# ========== Classifiers ==========

def cls_prototype(query, support, cids):
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0); protos[c] = p / np.linalg.norm(p)
    gt, pred = [], []
    for r in query:
        scores = [float(r["feat"] @ protos[c]) for c in cids]
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_knn(query, support, cids, k=5):
    all_s = []; all_y = []
    for c in cids:
        for s in support[c]:
            all_s.append(s["feat"]); all_y.append(c)
    all_s = np.stack(all_s)
    gt, pred = [], []
    for r in query:
        sims = all_s @ r["feat"]
        topk_idx = np.argsort(sims)[::-1][:k]
        votes = defaultdict(float)
        for i in topk_idx:
            votes[all_y[i]] += sims[i]
        gt.append(r["gt"]); pred.append(max(votes, key=votes.get))
    return metrics(gt, pred, cids)


def cls_tip_adapter(query, support, cids, beta=5.0, alpha=0.5):
    n_c = len(cids)
    cid2i = {c: i for i, c in enumerate(cids)}
    keys, labels = [], []
    for c in cids:
        for s in support[c]:
            keys.append(s["feat"])
            l = np.zeros(n_c, np.float32); l[cid2i[c]] = 1.0; labels.append(l)
    keys = np.stack(keys); labels = np.stack(labels)
    protos = {}
    for c in cids:
        p = np.stack([s["feat"] for s in support[c]]).mean(0)
        protos[c] = p / np.linalg.norm(p)
    gt, pred = [], []
    for r in query:
        pl = np.array([float(r["feat"] @ protos[c]) for c in cids])
        d = np.sum((keys - r["feat"])**2, axis=1)
        cl = np.exp(-beta * d) @ labels
        f = alpha * pl + (1-alpha) * cl
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(f))])
    return metrics(gt, pred, cids)


def cls_dual(query, support, cids, vw=0.65, mw=0.35, k=7):
    sm = []
    for c in cids:
        for s in support[c]:
            sm.append(s["morph"])
    sm = np.stack(sm)
    gm, gs = sm.mean(0), sm.std(0) + 1e-8
    sf = {c: np.stack([s["feat"] for s in support[c]]) for c in cids}
    snm = {c: (np.stack([s["morph"] for s in support[c]]) - gm) / gs for c in cids}
    gt, pred = [], []
    for r in query:
        qm = (r["morph"] - gm) / gs
        scores = []
        for c in cids:
            vs = sf[c] @ r["feat"]
            md = np.array([np.linalg.norm(qm - snm[c][i]) for i in range(len(snm[c]))])
            ms = 1.0 / (1.0 + md)
            comb = vw * vs + mw * ms
            top = np.sort(comb)[::-1][:k]
            scores.append(float(top.mean()))
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_tip_dual(query, support, cids, beta=10.0, alpha=0.3, mw=0.3):
    """Tip-Adapter fused with morphology."""
    n_c = len(cids)
    cid2i = {c: i for i, c in enumerate(cids)}
    keys, labels = [], []
    for c in cids:
        for s in support[c]:
            keys.append(s["feat"])
            l = np.zeros(n_c, np.float32); l[cid2i[c]] = 1.0; labels.append(l)
    keys = np.stack(keys); labels = np.stack(labels)
    protos = {}
    for c in cids:
        p = np.stack([s["feat"] for s in support[c]]).mean(0)
        protos[c] = p / np.linalg.norm(p)
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm)
    gm, gs = sm.mean(0), sm.std(0) + 1e-8
    snm = {c: (np.stack([s["morph"] for s in support[c]]) - gm)/gs for c in cids}
    gt, pred = [], []
    for r in query:
        pl = np.array([float(r["feat"] @ protos[c]) for c in cids])
        d = np.sum((keys - r["feat"])**2, axis=1)
        cl = np.exp(-beta * d) @ labels
        qm = (r["morph"] - gm) / gs
        ms = []
        for c in cids:
            dd = np.array([np.linalg.norm(qm - snm[c][i]) for i in range(len(snm[c]))])
            ms.append(float(np.mean(1.0/(1.0+np.sort(dd)[:5]))))
        ms = np.array(ms)
        f = alpha * pl + (1-alpha-mw) * cl + mw * ms
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(f))])
    return metrics(gt, pred, cids)


# ========== Main ==========

def run_model(model_info, train_cells, val_cells, cids):
    name = model_info["name"]
    results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        random.seed(seed)
        pc = defaultdict(list)
        for cell in train_cells:
            pc[cell["ann"]["class_id"]].append(cell)
        support_cells = {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}
        
        t0 = time.time()
        support = {}
        for c in cids:
            recs = []
            for cell in support_cells[c]:
                r = encode_cell(model_info, cell["image_path"], cell["ann"], cell["idx"])
                if r: recs.append(r)
            support[c] = recs
        print(f"    Support: {time.time()-t0:.1f}s")
        
        t0 = time.time()
        query = []
        for cell in val_cells:
            r = encode_cell(model_info, cell["image_path"], cell["ann"], cell["idx"])
            if r: query.append(r)
        print(f"    Query: {time.time()-t0:.1f}s ({len(query)} cells)")
        
        strats = {
            "proto": lambda q, s: cls_prototype(q, s, cids),
            "knn5": lambda q, s: cls_knn(q, s, cids, 5),
            "knn7": lambda q, s: cls_knn(q, s, cids, 7),
            "tip_b5": lambda q, s: cls_tip_adapter(q, s, cids, 5.0, 0.5),
            "tip_b10": lambda q, s: cls_tip_adapter(q, s, cids, 10.0, 0.5),
            "tip_b20": lambda q, s: cls_tip_adapter(q, s, cids, 20.0, 0.5),
            "tip_b10a3": lambda q, s: cls_tip_adapter(q, s, cids, 10.0, 0.3),
            "dual_65_35_7": lambda q, s: cls_dual(q, s, cids, 0.65, 0.35, 7),
            "dual_60_40_5": lambda q, s: cls_dual(q, s, cids, 0.60, 0.40, 5),
            "dual_55_45_5": lambda q, s: cls_dual(q, s, cids, 0.55, 0.45, 5),
            "dual_70_30_7": lambda q, s: cls_dual(q, s, cids, 0.70, 0.30, 7),
            "dual_50_50_5": lambda q, s: cls_dual(q, s, cids, 0.50, 0.50, 5),
            "tip_dual_b10": lambda q, s: cls_tip_dual(q, s, cids, 10.0, 0.3, 0.3),
            "tip_dual_b5": lambda q, s: cls_tip_dual(q, s, cids, 5.0, 0.3, 0.3),
        }
        
        for sn, fn in strats.items():
            key = f"{name}:{sn}"
            m = fn(query, support)
            results[key]["acc"].append(m["acc"])
            results[key]["mf1"].append(m["mf1"])
            for c in cids:
                results[key]["pc"][c].append(m["pc"][c]["f1"])
    
    return results


def main():
    print("=" * 90)
    print("MULTI-MODEL TRUE 10-SHOT CLASSIFICATION")
    print("  Zero data leakage | 10 cells/class | 5 seeds")
    print("=" * 90)
    
    cids = sorted(CLASS_NAMES.keys())
    train_cells = build_cell_index("train")
    val_cells = build_cell_index("val")
    print(f"Train index: {len(train_cells)} cells | Val index: {len(val_cells)} cells")
    per_class = defaultdict(int)
    for c in train_cells:
        per_class[c["ann"]["class_id"]] += 1
    for c in cids:
        print(f"  Train {CLASS_NAMES[c]}: {per_class[c]}")
    
    all_results = {}
    
    # 1. BiomedCLIP
    print(f"\n{'='*60}\nMODEL: BiomedCLIP (ViT-B/16, 512-dim)\n{'='*60}")
    try:
        mi = load_biomedclip()
        r = run_model(mi, train_cells, val_cells, cids)
        all_results.update(r)
        del mi; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 2. DINOv2-S
    print(f"\n{'='*60}\nMODEL: DINOv2-S (ViT-S/14, 384-dim)\n{'='*60}")
    try:
        mi = load_dinov2_s()
        r = run_model(mi, train_cells, val_cells, cids)
        all_results.update(r)
        del mi; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 3. DINOv2-B
    print(f"\n{'='*60}\nMODEL: DINOv2-B (ViT-B/14, 768-dim)\n{'='*60}")
    try:
        mi = load_dinov2_b()
        r = run_model(mi, train_cells, val_cells, cids)
        all_results.update(r)
        del mi; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 4. DinoBloom-B
    print(f"\n{'='*60}\nMODEL: DinoBloom-B (blood cell specialist, 768-dim)\n{'='*60}")
    try:
        mi = load_dinobloom()
        r = run_model(mi, train_cells, val_cells, cids)
        all_results.update(r)
        del mi; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Print final results
    print("\n" + "=" * 120)
    print("FINAL RESULTS: Multi-Model 10-Shot Classification (5 seeds, zero leakage)")
    print("=" * 120)
    header = f"{'Model:Strategy':<40} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 115)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<40} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    # Summary by model
    print("\n" + "=" * 80)
    print("BEST PER MODEL:")
    print("=" * 80)
    models = set(n.split(":")[0] for n in all_results.keys())
    for model in sorted(models):
        model_results = {k: v for k, v in all_results.items() if k.startswith(model + ":")}
        if not model_results:
            continue
        best = max(model_results.items(), key=lambda x: np.mean(x[1]["mf1"]))
        n, v = best
        print(f"  {model}:")
        print(f"    Best strategy: {n.split(':')[1]}")
        print(f"    Acc = {np.mean(v['acc']):.4f} ± {np.std(v['acc']):.4f}")
        print(f"    mF1 = {np.mean(v['mf1']):.4f} ± {np.std(v['mf1']):.4f}")
        for c in cids:
            print(f"    {CLASS_NAMES[c]}: F1 = {np.mean(v['pc'][c]):.4f}")
    
    with open(Path(__file__).parent / "multi_model_results.json", "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "acc_std": float(np.std(v["acc"])), "mf1_std": float(np.std(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["pc"][c])) for c in cids}}
                   for n, v in all_results.items()}, f, indent=2)
    print("\nSaved to experiments/multi_model_results.json")


if __name__ == "__main__":
    main()
