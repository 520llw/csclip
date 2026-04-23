#!/usr/bin/env python3
"""
Extract features from Phikon-v2 (pathology-specific ViT-L) and test 10-shot.
Phikon-v2 is pre-trained on 450M pathology patches using DINOv2 framework.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from biomedclip_zeroshot_cell_classify import InstanceInfo

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def crop_cell(image, inst, margin=0.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2-x1, y2-y1
    mx, my = int(bw*margin), int(bh*margin)
    return image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()


def ann2inst(ann, h, w, iid):
    pts = ann["points"]
    xs = [pts[i]*w for i in range(0, len(pts), 2)]
    ys = [pts[i]*h for i in range(1, len(pts), 2)]
    rr, cc = sk_polygon(ys, xs, shape=(h, w))
    if len(rr) == 0: return None
    mask = np.zeros((h, w), dtype=bool); mask[rr, cc] = True
    return InstanceInfo(instance_id=iid, class_id=ann["class_id"],
        bbox=(max(0,int(np.min(cc))), max(0,int(np.min(rr))),
              min(w,int(np.max(cc))+1), min(h,int(np.max(rr))+1)), mask=mask)


def main():
    from transformers import AutoModel, AutoImageProcessor

    model_dir = Path("/home/xut/csclip/model_weights/phikon_v2")
    print("Loading Phikon-v2...")
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(DEVICE).eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Output dim: checking...")

    for split in ["train", "val"]:
        cells = build_cell_index(split)
        print(f"\n{split}: {len(cells)} cells")
        feats_list, labels_list = [], []
        t0 = time.time()
        cur_img_path = None
        cur_img = None

        for idx, cell in enumerate(cells):
            if cell["image_path"] != cur_img_path:
                cur_img_path = cell["image_path"]
                cur_img = np.array(Image.open(cur_img_path).convert("RGB"))

            h, w = cur_img.shape[:2]
            inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
            if inst is None: continue

            crop = crop_cell(cur_img, inst, margin=0.15)
            pil_crop = Image.fromarray(crop)
            inputs = processor(images=pil_crop, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)
                feat = outputs.last_hidden_state[:, 0]
                feat = feat / feat.norm(dim=-1, keepdim=True)

            feats_list.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
            labels_list.append(cell["ann"]["class_id"])

            if idx % 200 == 0:
                print(f"  [{idx}/{len(cells)}] {time.time()-t0:.1f}s")

        feats = np.stack(feats_list)
        labels = np.array(labels_list)
        print(f"  Feature shape: {feats.shape}")
        print(f"  Time: {time.time()-t0:.1f}s")

        out_path = CACHE_DIR / f"phikon_v2_{split}.npz"
        existing = np.load(CACHE_DIR / f"biomedclip_{split}.npz")
        np.savez(out_path, feats=feats, morphs=existing["morphs"], labels=labels)
        print(f"  Saved to {out_path}")

    # Quick 10-shot test
    print("\n" + "="*80)
    print("QUICK 10-SHOT TEST WITH PHIKON-V2")
    print("="*80)

    d_train = np.load(CACHE_DIR / "phikon_v2_train.npz")
    d_val = np.load(CACHE_DIR / "phikon_v2_val.npz")
    bclip_train = np.load(CACHE_DIR / "biomedclip_train.npz")["feats"]
    bclip_val = np.load(CACHE_DIR / "biomedclip_val.npz")["feats"]

    cids = sorted(CLASS_NAMES.keys())
    SEEDS = [42, 123, 456, 789, 2026]

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        random.seed(seed)
        pc = defaultdict(list)
        for i, l in enumerate(d_train["labels"]): pc[int(l)].append(i)
        support_idx = {c: random.sample(pc[c], min(10, len(pc[c]))) for c in cids}
        s_ph = {c: d_train["feats"][support_idx[c]] for c in cids}
        s_bc = {c: bclip_train[support_idx[c]] for c in cids}
        s_morph = {c: d_train["morphs"][support_idx[c]] for c in cids}

        sm = np.concatenate([s_morph[c] for c in cids])
        gm, gs = sm.mean(0), sm.std(0)+1e-8

        for mode_name, feats_val, feats_sup in [
            ("phikon_only", d_val["feats"], s_ph),
            ("bclip_only", bclip_val, s_bc),
        ]:
            snm = {c: (s_morph[c]-gm)/gs for c in cids}
            gt, pred = [], []
            for i in range(len(d_val["labels"])):
                qm = (d_val["morphs"][i]-gm)/gs
                scores = []
                for c in cids:
                    vs = feats_sup[c] @ feats_val[i]
                    md = np.linalg.norm(qm-snm[c], axis=1)
                    ms = 1.0/(1.0+md)
                    comb = 0.65*vs + 0.35*ms
                    scores.append(float(np.sort(comb)[::-1][:7].mean()))
                gt.append(int(d_val["labels"][i]))
                pred.append(cids[int(np.argmax(scores))])

            total = len(gt); correct = sum(g==p for g,p in zip(gt,pred))
            f1s = []
            for c in cids:
                tp = sum(1 for g,p in zip(gt,pred) if g==c and p==c)
                pp = sum(1 for p in pred if p==c)
                gp = sum(1 for g in gt if g==c)
                pr = tp/pp if pp else 0; rc = tp/gp if gp else 0
                f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
                f1s.append(f1)
                all_results[mode_name]["pc"][c].append(f1)
            all_results[mode_name]["acc"].append(correct/total)
            all_results[mode_name]["mf1"].append(float(np.mean(f1s)))

        # Triple backbone: BiomedCLIP + Phikon + morph
        for bw, pw, mw in [(0.40, 0.25, 0.35), (0.35, 0.30, 0.35),
                            (0.45, 0.20, 0.35), (0.30, 0.35, 0.35),
                            (0.35, 0.25, 0.40)]:
            name = f"triple_bw{bw}_pw{pw}"
            snm = {c: (s_morph[c]-gm)/gs for c in cids}
            gt, pred = [], []
            for i in range(len(d_val["labels"])):
                qm = (d_val["morphs"][i]-gm)/gs
                scores = []
                for c in cids:
                    vs_b = s_bc[c] @ bclip_val[i]
                    vs_p = s_ph[c] @ d_val["feats"][i]
                    md = np.linalg.norm(qm-snm[c], axis=1)
                    ms = 1.0/(1.0+md)
                    comb = bw*vs_b + pw*vs_p + mw*ms
                    scores.append(float(np.sort(comb)[::-1][:7].mean()))
                gt.append(int(d_val["labels"][i]))
                pred.append(cids[int(np.argmax(scores))])
            total = len(gt); correct = sum(g==p for g,p in zip(gt,pred))
            f1s = []
            for c in cids:
                tp = sum(1 for g,p in zip(gt,pred) if g==c and p==c)
                pp = sum(1 for p in pred if p==c)
                gp = sum(1 for g in gt if g==c)
                pr = tp/pp if pp else 0; rc = tp/gp if gp else 0
                f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
                f1s.append(f1)
                all_results[name]["pc"][c].append(f1)
            all_results[name]["acc"].append(correct/total)
            all_results[name]["mf1"].append(float(np.mean(f1s)))

    print(f"\n{'Strategy':<40} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}")
    print("-"*100)
    for name in sorted(all_results.keys(), key=lambda x: -np.mean(all_results[x]["mf1"])):
        v = all_results[name]
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<40} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}")


if __name__ == "__main__":
    main()
