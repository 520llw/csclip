#!/usr/bin/env python3
"""
Run one model at a time to avoid OOM. Pass model name as argument.
Usage:
  python run_each_model.py biomedclip
  python run_each_model.py dinov2_s
  python run_each_model.py dinov2_b
  python run_each_model.py dinobloom
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


def compute_morphology(image, inst):
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


# ========== Encoders ==========

def make_biomedclip_encoder():
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
    return encode


def make_dino_encoder(model_name):
    if model_name == "dinov2_s":
        m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
        state = torch.load(WEIGHTS_DIR / 'dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
        sz = 518
    elif model_name == "dinov2_b":
        m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
        state = torch.load(WEIGHTS_DIR / 'dinov2_vitb14_pretrain.pth', map_location='cpu', weights_only=True)
        sz = 518
    elif model_name == "dinobloom":
        m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=224)
        state = torch.load(WEIGHTS_DIR / 'dinobloom_vitb14.pth', map_location='cpu', weights_only=False)
        sz = 224
    else:
        raise ValueError(model_name)
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE).eval()
    t = transforms.Compose([transforms.Resize(sz), transforms.CenterCrop(sz),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    def encode(image, inst):
        crop = crop_cell(image, inst, margin=0.15, mask_bg=False)
        inp = t(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = m(inp)
            f /= f.norm(dim=-1, keepdim=True)
        return f.squeeze(0).cpu().numpy().astype(np.float32)
    return encode


# ========== Classifiers ==========

def cls_prototype(query, support, cids):
    protos = {}
    for c in cids:
        feats = np.stack([s["feat"] for s in support[c]])
        p = feats.mean(0); protos[c] = p / np.linalg.norm(p)
    gt, pred = [], []
    for r in query:
        gt.append(r["gt"]); pred.append(cids[int(np.argmax([r["feat"]@protos[c] for c in cids]))])
    return metrics(gt, pred, cids)


def cls_knn(query, support, cids, k=5):
    all_s = []; all_y = []
    for c in cids:
        for s in support[c]: all_s.append(s["feat"]); all_y.append(c)
    all_s = np.stack(all_s)
    gt, pred = [], []
    for r in query:
        sims = all_s @ r["feat"]
        topk = np.argsort(sims)[::-1][:k]
        votes = defaultdict(float)
        for i in topk: votes[all_y[i]] += sims[i]
        gt.append(r["gt"]); pred.append(max(votes, key=votes.get))
    return metrics(gt, pred, cids)


def cls_tip(query, support, cids, beta=5.0, alpha=0.5):
    n_c = len(cids); cid2i = {c: i for i, c in enumerate(cids)}
    keys, labels = [], []
    for c in cids:
        for s in support[c]:
            keys.append(s["feat"])
            l = np.zeros(n_c, np.float32); l[cid2i[c]] = 1.0; labels.append(l)
    keys = np.stack(keys); labels = np.stack(labels)
    protos = {}
    for c in cids:
        p = np.stack([s["feat"] for s in support[c]]).mean(0); protos[c] = p/np.linalg.norm(p)
    gt, pred = [], []
    for r in query:
        pl = np.array([float(r["feat"]@protos[c]) for c in cids])
        d = np.sum((keys - r["feat"])**2, axis=1)
        cl = np.exp(-beta * d) @ labels
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(alpha*pl + (1-alpha)*cl))])
    return metrics(gt, pred, cids)


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


def cls_tip_dual(query, support, cids, beta=10.0, alpha=0.3, mw=0.3):
    n_c = len(cids); cid2i = {c: i for i, c in enumerate(cids)}
    keys, labels = [], []
    for c in cids:
        for s in support[c]:
            keys.append(s["feat"]); l = np.zeros(n_c, np.float32); l[cid2i[c]]=1.0; labels.append(l)
    keys = np.stack(keys); labels = np.stack(labels)
    protos = {}
    for c in cids:
        p = np.stack([s["feat"] for s in support[c]]).mean(0); protos[c] = p/np.linalg.norm(p)
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    gt, pred = [], []
    for r in query:
        pl = np.array([float(r["feat"]@protos[c]) for c in cids])
        d = np.sum((keys - r["feat"])**2, axis=1)
        cl = np.exp(-beta * d) @ labels
        qm = (r["morph"]-gm)/gs
        ms = []
        for c in cids:
            dd = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms.append(float(np.mean(1.0/(1.0+np.sort(dd)[:5]))))
        ms = np.array(ms)
        f = alpha*pl + (1-alpha-mw)*cl + mw*ms
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(f))])
    return metrics(gt, pred, cids)


# ========== Main ==========

def encode_cells(encode_fn, cells):
    recs = []
    for cell in cells:
        img = np.array(Image.open(cell["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
        if inst is None:
            continue
        feat = encode_fn(img, inst)
        morph = compute_morphology(img, inst)
        recs.append({"gt": cell["ann"]["class_id"], "feat": feat, "morph": morph})
    return recs


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "biomedclip"
    print(f"Running TRUE 10-shot for model: {model_name}")
    
    cids = sorted(CLASS_NAMES.keys())
    train_cells = build_cell_index("train")
    val_cells = build_cell_index("val")
    print(f"Train: {len(train_cells)} | Val: {len(val_cells)}")
    
    if model_name == "biomedclip":
        encode_fn = make_biomedclip_encoder()
    else:
        encode_fn = make_dino_encoder(model_name)
    print(f"Model loaded on {DEVICE}")
    
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        random.seed(seed)
        pc = defaultdict(list)
        for cell in train_cells:
            pc[cell["ann"]["class_id"]].append(cell)
        support_cells = {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}
        
        # Encode support (only 40 cells)
        t0 = time.time()
        support_flat = []
        for c in cids:
            support_flat.extend(support_cells[c])
        support_recs_flat = encode_cells(encode_fn, support_flat)
        support_recs = defaultdict(list)
        for r in support_recs_flat:
            support_recs[r["gt"]].append(r)
        print(f"  Support: {time.time()-t0:.1f}s ({len(support_recs_flat)} cells)")
        
        # Encode query
        t0 = time.time()
        query_recs = encode_cells(encode_fn, val_cells)
        print(f"  Query: {time.time()-t0:.1f}s ({len(query_recs)} cells)")
        
        # Strategies
        strats = {
            "proto": lambda q, s: cls_prototype(q, s, cids),
            "knn5": lambda q, s: cls_knn(q, s, cids, 5),
            "knn7": lambda q, s: cls_knn(q, s, cids, 7),
            "tip_b5a5": lambda q, s: cls_tip(q, s, cids, 5.0, 0.5),
            "tip_b10a5": lambda q, s: cls_tip(q, s, cids, 10.0, 0.5),
            "tip_b20a5": lambda q, s: cls_tip(q, s, cids, 20.0, 0.5),
            "tip_b10a3": lambda q, s: cls_tip(q, s, cids, 10.0, 0.3),
            "tip_b5a3": lambda q, s: cls_tip(q, s, cids, 5.0, 0.3),
            "dual_65_35_7": lambda q, s: cls_dual(q, s, cids, 0.65, 0.35, 7),
            "dual_60_40_5": lambda q, s: cls_dual(q, s, cids, 0.60, 0.40, 5),
            "dual_55_45_5": lambda q, s: cls_dual(q, s, cids, 0.55, 0.45, 5),
            "dual_70_30_7": lambda q, s: cls_dual(q, s, cids, 0.70, 0.30, 7),
            "dual_50_50_5": lambda q, s: cls_dual(q, s, cids, 0.50, 0.50, 5),
            "dual_80_20_7": lambda q, s: cls_dual(q, s, cids, 0.80, 0.20, 7),
            "tip_dual_b10": lambda q, s: cls_tip_dual(q, s, cids, 10.0, 0.3, 0.3),
            "tip_dual_b5": lambda q, s: cls_tip_dual(q, s, cids, 5.0, 0.3, 0.3),
            "tip_dual_b20": lambda q, s: cls_tip_dual(q, s, cids, 20.0, 0.3, 0.3),
        }
        
        for sn, fn in strats.items():
            key = f"{model_name}:{sn}"
            m = fn(query_recs, support_recs)
            all_results[key]["acc"].append(m["acc"])
            all_results[key]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[key]["pc"][c].append(m["pc"][c]["f1"])
        
        # Force garbage collection between seeds
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print results
    print(f"\n{'='*110}")
    print(f"RESULTS: {model_name} TRUE 10-shot (5 seeds)")
    print(f"{'='*110}")
    header = f"{'Strategy':<35} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 105)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<35} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    result_file = Path(__file__).parent / f"results_{model_name}.json"
    with open(result_file, "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "acc_std": float(np.std(v["acc"])), "mf1_std": float(np.std(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["pc"][c])) for c in cids}}
                   for n, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {result_file}")


if __name__ == "__main__":
    main()
