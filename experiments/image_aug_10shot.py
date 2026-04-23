#!/usr/bin/env python3
"""
Image-level augmentation of support cells.
Instead of feature-space manipulation, augment the actual images before encoding.
This produces genuinely different features from different views of the same cell.

Augmentations per support cell:
- Original (1)
- Horizontal flip (1)
- Different margins (0.05, 0.20) (2)
- Color jitter (brightness/contrast) (2)
= 6 total views -> 60 per class

Then run classification with augmented support + transductive + cascade.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import random
import gc
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image, ImageEnhance
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from biomedclip_zeroshot_cell_classify import InstanceInfo

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
WEIGHTS_DIR = Path("/home/xut/csclip/model_weights")
CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


def load_yolo(lp):
    anns = []
    if not lp.exists(): return anns
    for line in open(lp):
        p = line.strip().split()
        if len(p) < 7: continue
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
    xs = [pts[i]*w for i in range(0, len(pts), 2)]
    ys = [pts[i]*h for i in range(1, len(pts), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(h, w))
    if len(rr) == 0: return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return InstanceInfo(instance_id=iid, class_id=ann["class_id"],
                        bbox=(max(0, int(np.min(cc))), max(0, int(np.min(rr))),
                              min(w, int(np.max(cc))+1), min(h, int(np.max(rr))+1)), mask=mask)


def crop_cell(image, inst, margin=0.15, mask_bg=False, bg_val=128):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2-x1, y2-y1
    mx, my = int(bw*margin), int(bh*margin)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()
    if mask_bg:
        mc = inst.mask[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
        crop = np.where(mc[..., None], crop, np.full_like(crop, bg_val))
    return crop


def augment_crops(image, inst):
    """Generate multiple augmented crops for a single cell."""
    crops = []

    # Original: cell_margin=0.10, mask_bg=True (standard encoding)
    crops.append(("orig", crop_cell(image, inst, margin=0.10, mask_bg=True)))

    # Horizontal flip
    orig = crop_cell(image, inst, margin=0.10, mask_bg=True)
    crops.append(("hflip", np.fliplr(orig).copy()))

    # Tight crop
    crops.append(("tight", crop_cell(image, inst, margin=0.05, mask_bg=True)))

    # Loose crop (more context)
    crops.append(("loose", crop_cell(image, inst, margin=0.20, mask_bg=True)))

    # Without background masking (natural context)
    crops.append(("nomasktight", crop_cell(image, inst, margin=0.08, mask_bg=False)))

    # Brightness enhancement
    bimg = (np.clip(image.astype(float) * 1.15, 0, 255)).astype(np.uint8)
    crops.append(("bright", crop_cell(bimg, inst, margin=0.10, mask_bg=True)))

    # Darker
    dimg = (np.clip(image.astype(float) * 0.85, 0, 255)).astype(np.uint8)
    crops.append(("dark", crop_cell(dimg, inst, margin=0.10, mask_bg=True)))

    # Vertical flip
    crops.append(("vflip", np.flipud(crop_cell(image, inst, margin=0.10, mask_bg=True)).copy()))

    return crops


def select_support_cells(cells, seed, cids):
    random.seed(seed)
    per_class = defaultdict(list)
    for i, c in enumerate(cells):
        cid = c["ann"]["class_id"]
        if cid in cids:
            per_class[cid].append(i)
    support = {}
    for c in cids:
        support[c] = random.sample(per_class[c], min(N_SHOT, len(per_class[c])))
    return support


def metrics(gt, pred, cids):
    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    pc, f1s = {}, []
    for c in cids:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        pp = sum(1 for p in pred if p == c)
        gp = sum(1 for g in gt if g == c)
        pr = tp/pp if pp else 0.0
        rc = tp/gp if gp else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        pc[c] = {"p": pr, "r": rc, "f1": f1, "n": gp}
        f1s.append(f1)
    return {"acc": correct/total if total else 0, "mf1": float(np.mean(f1s)), "pc": pc}


def main():
    cids = sorted(CLASS_NAMES.keys())
    train_cells = build_cell_index("train")

    # Load pre-cached query features
    d_bclip_val = np.load(CACHE_DIR / "biomedclip_val.npz")
    q_bclip, q_morph, q_labels = d_bclip_val["feats"], d_bclip_val["morphs"], d_bclip_val["labels"]
    d_dino_val = np.load(CACHE_DIR / "dinov2_s_val.npz")
    q_dino = d_dino_val["feats"]

    # Load BiomedCLIP
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    bclip_model, bclip_preprocess = bundle["model"], bundle["preprocess"]

    # Load DINOv2
    dino_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=518)
    dino_state = torch.load(WEIGHTS_DIR / 'dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
    dino_model.load_state_dict(dino_state, strict=False)
    dino_model = dino_model.to(DEVICE).eval()
    dino_transform = transforms.Compose([
        transforms.Resize(518), transforms.CenterCrop(518),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def encode_bclip(crop_img):
        inp = bclip_preprocess(Image.fromarray(crop_img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = bclip_model.encode_image(inp)
            f /= f.norm(dim=-1, keepdim=True)
        return f.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_dino(crop_img):
        inp = dino_transform(Image.fromarray(crop_img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = dino_model(inp)
            f /= f.norm(dim=-1, keepdim=True)
        return f.squeeze(0).cpu().numpy().astype(np.float32)

    # Also load train features for morph and Fisher weights
    d_morph_train = np.load(CACHE_DIR / "biomedclip_train.npz")
    morph_train_all = d_morph_train["morphs"]
    labels_train_all = d_morph_train["labels"]

    eos_morph = morph_train_all[labels_train_all==3]
    neu_morph = morph_train_all[labels_train_all==4]
    n_dims = morph_train_all.shape[1]
    fisher_w = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos_morph[:,d])-np.mean(neu_morph[:,d]))**2 / (np.var(eos_morph[:,d])+np.var(neu_morph[:,d])+1e-10)
        fisher_w[d] = 1.0 + f * 2.0

    # Morphology computer
    from experiments.extract_features import compute_granule_morphology

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        np.random.seed(seed)
        support_cell_idx = select_support_cells(train_cells, seed, cids)

        # Encode support cells with augmentation
        aug_bclip = defaultdict(list)
        aug_dino = defaultdict(list)
        aug_morph = defaultdict(list)
        orig_bclip = {}
        orig_dino = {}
        orig_morph = {}

        for c in cids:
            o_b, o_d, o_m = [], [], []
            for cell_i in support_cell_idx[c]:
                cell = train_cells[cell_i]
                img = np.array(Image.open(cell["image_path"]).convert("RGB"))
                h, w = img.shape[:2]
                inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
                if inst is None:
                    continue

                morph_feat = compute_granule_morphology(img, inst)
                augmented_crops = augment_crops(img, inst)

                for aug_name, crop in augmented_crops:
                    bf = encode_bclip(crop)
                    df = encode_dino(crop)
                    aug_bclip[c].append(bf)
                    aug_dino[c].append(df)
                    aug_morph[c].append(morph_feat)

                # Original only
                orig_crop = crop_cell(img, inst, margin=0.10, mask_bg=True)
                ctx_crop = crop_cell(img, inst, margin=0.30, mask_bg=False)
                bf_cell = encode_bclip(orig_crop)
                bf_ctx = encode_bclip(ctx_crop)
                bf_fused = 0.85*bf_cell + 0.15*bf_ctx
                bf_fused /= np.linalg.norm(bf_fused) + 1e-10
                o_b.append(bf_fused)
                o_d.append(encode_dino(orig_crop))
                o_m.append(morph_feat)

            aug_bclip[c] = np.array(aug_bclip[c])
            aug_dino[c] = np.array(aug_dino[c])
            aug_morph[c] = np.array(aug_morph[c])
            orig_bclip[c] = np.array(o_b)
            orig_dino[c] = np.array(o_d)
            orig_morph[c] = np.array(o_m)

        print(f"  Augmented support per class: {[aug_bclip[c].shape[0] for c in cids]}")
        print(f"  Original support per class:  {[orig_bclip[c].shape[0] for c in cids]}")

        def classify_dual(s_b, s_d, s_m, name_tag, bw=0.45, dw=0.20, mw=0.35, k=7):
            sm_all = np.concatenate([s_m[c] for c in cids])
            gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
            snm = {c: (s_m[c]-gm)/gs for c in cids}
            gt, pred = [], []
            for i in range(len(q_labels)):
                qm = (q_morph[i]-gm)/gs
                scores = []
                for c in cids:
                    vs_b = s_b[c] @ q_bclip[i]
                    vs_d = s_d[c] @ q_dino[i]
                    md = np.linalg.norm(qm - snm[c], axis=1)
                    ms = 1.0/(1.0+md)
                    comb = bw*vs_b + dw*vs_d + mw*ms
                    scores.append(float(np.sort(comb)[::-1][:k].mean()))
                gt.append(int(q_labels[i]))
                pred.append(cids[int(np.argmax(scores))])
            return metrics(gt, pred, cids)

        def trans_cascade(s_b_init, s_d_init, s_m_init, n_iter=2, top_k=5,
                          conf_thr=0.025, cascade_thr=0.01,
                          bw=0.45, dw=0.20, mw=0.35, k=7):
            s_b = {c: s_b_init[c].copy() for c in cids}
            s_d = {c: s_d_init[c].copy() for c in cids}
            s_m = {c: s_m_init[c].copy() for c in cids}
            for _t in range(n_iter):
                sm_all = np.concatenate([s_m[c] for c in cids])
                gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
                snm = {c: (s_m[c]-gm)/gs for c in cids}
                preds, margins_a = [], []
                for i in range(len(q_labels)):
                    qm = (q_morph[i]-gm)/gs
                    scores = []
                    for c in cids:
                        vs_b = s_b[c] @ q_bclip[i]
                        vs_d = s_d[c] @ q_dino[i]
                        md = np.linalg.norm(qm - snm[c], axis=1)
                        ms = 1.0/(1.0+md)
                        comb = bw*vs_b + dw*vs_d + mw*ms
                        scores.append(float(np.sort(comb)[::-1][:k].mean()))
                    s_arr = np.array(scores)
                    sorted_s = np.sort(s_arr)[::-1]
                    preds.append(cids[int(np.argmax(s_arr))])
                    margins_a.append(sorted_s[0]-sorted_s[1])
                preds = np.array(preds)
                margins_a = np.array(margins_a)
                for c in cids:
                    c_mask = (preds == c) & (margins_a > conf_thr)
                    c_idx = np.where(c_mask)[0]
                    if len(c_idx) == 0: continue
                    sorted_idx = c_idx[np.argsort(margins_a[c_idx])[::-1][:top_k]]
                    s_b[c] = np.concatenate([s_b_init[c], q_bclip[sorted_idx]*0.5])
                    s_d[c] = np.concatenate([s_d_init[c], q_dino[sorted_idx]*0.5])
                    s_m[c] = np.concatenate([s_m_init[c], q_morph[sorted_idx]])

            sm_all = np.concatenate([s_m[c] for c in cids])
            gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
            snm = {c: (s_m[c]-gm)/gs for c in cids}
            snm_w = {c: (s_m[c]-gm)/gs * fisher_w for c in cids}
            gt, pred = [], []
            for i in range(len(q_labels)):
                qm = (q_morph[i]-gm)/gs
                qm_w = qm * fisher_w
                scores = {}
                for c in cids:
                    vs_b = s_b[c] @ q_bclip[i]
                    vs_d = s_d[c] @ q_dino[i]
                    md = np.linalg.norm(qm - snm[c], axis=1)
                    ms = 1.0/(1.0+md)
                    comb = bw*vs_b + dw*vs_d + mw*ms
                    scores[c] = float(np.sort(comb)[::-1][:k].mean())
                s_arr = np.array([scores[c] for c in cids])
                top1 = cids[int(np.argmax(s_arr))]
                margin = np.sort(s_arr)[::-1][0]-np.sort(s_arr)[::-1][1]
                if top1 in [3, 4] and margin < cascade_thr:
                    for gc in [3, 4]:
                        md_w = np.linalg.norm(qm_w - snm_w[gc], axis=1)
                        mscore = float(np.mean(1.0/(1.0+np.sort(md_w)[:5])))
                        vs_b_s = float(np.sort(s_b[gc] @ q_bclip[i])[::-1][:3].mean())
                        vs_d_s = float(np.sort(s_d[gc] @ q_dino[i])[::-1][:3].mean())
                        scores[gc] = 0.30*vs_b_s + 0.15*vs_d_s + 0.55*mscore
                    top1 = 3 if scores[3] > scores[4] else 4
                gt.append(int(q_labels[i]))
                pred.append(top1)
            return metrics(gt, pred, cids)

        # Test 1: Augmented support (no transductive)
        for kk in [5, 7, 10]:
            name = f"img_aug_k{kk}"
            m = classify_dual(aug_bclip, aug_dino, aug_morph, name, k=kk)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Test 2: Original support (baseline re-encoded)
        name = "orig_reencoded"
        m = classify_dual(orig_bclip, orig_dino, orig_morph, name)
        all_results[name]["acc"].append(m["acc"])
        all_results[name]["mf1"].append(m["mf1"])
        for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Test 3: Aug + transductive + cascade
        for cthr in [0.008, 0.010, 0.012]:
            name = f"img_aug_trans_cas_t{cthr}"
            m = trans_cascade(aug_bclip, aug_dino, aug_morph, cascade_thr=cthr)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Test 4: Original + transductive + cascade (should match previous best)
        name = "orig_trans_cas"
        m = trans_cascade(orig_bclip, orig_dino, orig_morph)
        all_results[name]["acc"].append(m["acc"])
        all_results[name]["mf1"].append(m["mf1"])
        for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # Test 5: Aug support, different weight configs
        for bw in [0.40, 0.45, 0.50]:
            for dw in [0.15, 0.20, 0.25]:
                mwt = 1.0 - bw - dw
                if mwt < 0.1: continue
                name = f"img_aug_tc_b{bw}_d{dw}_m{mwt:.2f}"
                m = trans_cascade(aug_bclip, aug_dino, aug_morph,
                                  bw=bw, dw=dw, mw=mwt)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    # Cleanup GPU
    del bclip_model, dino_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*130}")
    print("IMAGE AUGMENTATION RESULTS (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<50} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:25]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}, Eos={np.mean(best[1]['pc'][3]):.4f}")


if __name__ == "__main__":
    main()
