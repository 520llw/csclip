#!/usr/bin/env python3
"""
Cascade 10-shot: Two-stage classification for Eos/Neu disambiguation.
Stage 1: Separate Lymphocyte/Macrophage from granulocytes (easy)
Stage 2: Within granulocytes, use specialized features to distinguish Eos/Neu

Also: Fine-sweep dual-backbone parameters, per-class weight optimization.
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
from experiments.advanced_10shot import (
    load_yolo, build_cell_index, ann2inst, metrics, crop_cell,
    compute_granule_morphology, load_biomedclip, load_dinov2s,
    DATA_ROOT, WEIGHTS_DIR, CLASS_NAMES, N_SHOT, SEEDS, DEVICE
)


# ========== Classifiers ==========

def cls_dual_bb(query, support, cids, bw, dw, mw, k):
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    sf_b = {c: np.stack([s["feat_bclip"] for s in support[c]]) for c in cids}
    sf_d = {c: np.stack([s["feat_dino"] for s in support[c]]) for c in cids}
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    
    gt, pred = [], []
    for r in query:
        qm = (r["morph"]-gm)/gs
        scores = []
        for c in cids:
            vs_b = sf_b[c] @ r["feat_bclip"]
            vs_d = sf_d[c] @ r["feat_dino"]
            md = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_cascade(query, support, cids):
    """Two-stage cascade:
    Stage 1: 4-class kNN with dual-backbone → get initial prediction + confidence
    Stage 2: If prediction is Eos or Neu AND confidence is low →
             Use specialized Eos/Neu morph classifier
    """
    gran_ids = [3, 4]  # Eos, Neu
    mono_ids = [5, 6]  # Lym, Mac
    
    # Morph normalization
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    
    sf_b = {c: np.stack([s["feat_bclip"] for s in support[c]]) for c in cids}
    sf_d = {c: np.stack([s["feat_dino"] for s in support[c]]) for c in cids}
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    
    # Eos/Neu specific morph features (color ratios, granule features)
    eos_neu_morph_idx = list(range(12, 40))  # enhanced color + granule features
    
    gt, pred = [], []
    refined_count = 0
    for r in query:
        qm = (r["morph"]-gm)/gs
        
        # Stage 1: Full 4-class scoring
        scores = {}
        for c in cids:
            vs_b = sf_b[c] @ r["feat_bclip"]
            vs_d = sf_d[c] @ r["feat_dino"]
            md = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms = 1.0/(1.0+md)
            comb = 0.40*vs_b + 0.25*vs_d + 0.35*ms
            scores[c] = float(np.sort(comb)[::-1][:7].mean())
        
        s_arr = np.array([scores[c] for c in cids])
        top1 = cids[int(np.argmax(s_arr))]
        s_sorted = np.sort(s_arr)[::-1]
        confidence = s_sorted[0] - s_sorted[1]
        
        # Stage 2: Refine Eos/Neu when uncertain
        if top1 in gran_ids and confidence < 0.02:
            # Use morph-heavy re-scoring for Eos vs Neu only
            eos_morph = snm[3][:, eos_neu_morph_idx] if len(eos_neu_morph_idx) <= snm[3].shape[1] else snm[3]
            neu_morph = snm[4][:, eos_neu_morph_idx] if len(eos_neu_morph_idx) <= snm[4].shape[1] else snm[4]
            q_morph = qm[eos_neu_morph_idx] if len(eos_neu_morph_idx) <= len(qm) else qm
            
            eos_d = np.array([np.linalg.norm(q_morph-eos_morph[i]) for i in range(len(eos_morph))])
            neu_d = np.array([np.linalg.norm(q_morph-neu_morph[i]) for i in range(len(neu_morph))])
            eos_s = float(np.mean(1.0/(1.0+np.sort(eos_d)[:5])))
            neu_s = float(np.mean(1.0/(1.0+np.sort(neu_d)[:5])))
            
            # Morph-boosted re-score
            visual_eos = 0.40*float(np.sort(sf_b[3]@r["feat_bclip"])[::-1][:3].mean()) + \
                         0.25*float(np.sort(sf_d[3]@r["feat_dino"])[::-1][:3].mean())
            visual_neu = 0.40*float(np.sort(sf_b[4]@r["feat_bclip"])[::-1][:3].mean()) + \
                         0.25*float(np.sort(sf_d[4]@r["feat_dino"])[::-1][:3].mean())
            
            eos_final = visual_eos + 0.50 * eos_s
            neu_final = visual_neu + 0.50 * neu_s
            top1 = 3 if eos_final > neu_final else 4
            refined_count += 1
        
        gt.append(r["gt"])
        pred.append(top1)
    
    return metrics(gt, pred, cids), refined_count


def cls_class_weight(query, support, cids, weights):
    """Per-class optimized weights: {class_id: (bw, dw, mw, k)}."""
    sm = []
    for c in cids:
        for s in support[c]: sm.append(s["morph"])
    sm = np.stack(sm); gm, gs = sm.mean(0), sm.std(0)+1e-8
    sf_b = {c: np.stack([s["feat_bclip"] for s in support[c]]) for c in cids}
    sf_d = {c: np.stack([s["feat_dino"] for s in support[c]]) for c in cids}
    snm = {c: (np.stack([s["morph"] for s in support[c]])-gm)/gs for c in cids}
    
    gt, pred = [], []
    for r in query:
        qm = (r["morph"]-gm)/gs
        scores = []
        for c in cids:
            bw, dw, mw, k = weights.get(c, (0.40, 0.25, 0.35, 7))
            vs_b = sf_b[c] @ r["feat_bclip"]
            vs_d = sf_d[c] @ r["feat_dino"]
            md = np.array([np.linalg.norm(qm-snm[c][i]) for i in range(len(snm[c]))])
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            scores.append(float(np.sort(comb)[::-1][:k].mean()))
        gt.append(r["gt"]); pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


# ========== Main ==========

def main():
    print("=" * 90)
    print("CASCADE + FINE-SWEEP 10-SHOT EXPERIMENT")
    print("=" * 90)
    
    cids = sorted(CLASS_NAMES.keys())
    train_cells = build_cell_index("train")
    val_cells = build_cell_index("val")
    
    print("Loading models...")
    bclip_encode, _ = load_biomedclip()
    dino_encode, _ = load_dinov2s()
    print("Models loaded")
    
    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})
    
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        random.seed(seed)
        pc = defaultdict(list)
        for cell in train_cells:
            pc[cell["ann"]["class_id"]].append(cell)
        support_cells = {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}
        
        t0 = time.time()
        support = defaultdict(list)
        for c in cids:
            for cell in support_cells[c]:
                img = np.array(Image.open(cell["image_path"]).convert("RGB"))
                h, w = img.shape[:2]
                inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
                if inst is None:
                    continue
                support[c].append({
                    "gt": cell["ann"]["class_id"],
                    "feat_bclip": bclip_encode(img, inst),
                    "feat_dino": dino_encode(img, inst),
                    "morph": compute_granule_morphology(img, inst),
                })
        print(f"  Support: {time.time()-t0:.1f}s")
        
        t0 = time.time()
        query = []
        for cell in val_cells:
            img = np.array(Image.open(cell["image_path"]).convert("RGB"))
            h, w = img.shape[:2]
            inst = ann2inst(cell["ann"], h, w, cell["idx"]+1)
            if inst is None:
                continue
            query.append({
                "gt": cell["ann"]["class_id"],
                "feat_bclip": bclip_encode(img, inst),
                "feat_dino": dino_encode(img, inst),
                "morph": compute_granule_morphology(img, inst),
            })
        print(f"  Query: {time.time()-t0:.1f}s ({len(query)} cells)")
        
        # Fine sweep of dual backbone parameters
        for bw in [0.35, 0.40, 0.45]:
            for dw in [0.15, 0.20, 0.25]:
                mw = 1.0 - bw - dw
                if mw < 0.15 or mw > 0.50:
                    continue
                for k in [5, 7]:
                    name = f"db:{bw:.2f}_{dw:.2f}_{mw:.2f}_k{k}"
                    m = cls_dual_bb(query, support, cids, bw, dw, mw, k)
                    all_results[name]["acc"].append(m["acc"])
                    all_results[name]["mf1"].append(m["mf1"])
                    for c in cids:
                        all_results[name]["pc"][c].append(m["pc"][c]["f1"])
        
        # Cascade
        m, rc = cls_cascade(query, support, cids)
        all_results["cascade"]["acc"].append(m["acc"])
        all_results["cascade"]["mf1"].append(m["mf1"])
        for c in cids:
            all_results["cascade"]["pc"][c].append(m["pc"][c]["f1"])
        print(f"  Cascade refined {rc} cells")
        
        # Class-specific weights: boost morph for Eos, visual for Lym
        weight_configs = {
            "cw_eos_morph_heavy": {
                3: (0.30, 0.15, 0.55, 7),  # Eos: heavy morph
                4: (0.35, 0.25, 0.40, 7),  # Neu: balanced
                5: (0.50, 0.25, 0.25, 5),  # Lym: visual heavy
                6: (0.45, 0.25, 0.30, 5),  # Mac: visual moderate
            },
            "cw_eos_morph_boost": {
                3: (0.25, 0.20, 0.55, 5),
                4: (0.35, 0.25, 0.40, 5),
                5: (0.55, 0.20, 0.25, 7),
                6: (0.45, 0.20, 0.35, 7),
            },
            "cw_eos_dino_boost": {
                3: (0.30, 0.35, 0.35, 7),  # Eos: heavy DINOv2
                4: (0.35, 0.30, 0.35, 7),
                5: (0.50, 0.15, 0.35, 5),
                6: (0.45, 0.15, 0.40, 5),
            },
            "cw_uniform_morph": {
                3: (0.30, 0.20, 0.50, 5),
                4: (0.30, 0.20, 0.50, 5),
                5: (0.30, 0.20, 0.50, 5),
                6: (0.30, 0.20, 0.50, 5),
            },
        }
        
        for wn, wc in weight_configs.items():
            m = cls_class_weight(query, support, cids, wc)
            all_results[wn]["acc"].append(m["acc"])
            all_results[wn]["mf1"].append(m["mf1"])
            for c in cids:
                all_results[wn]["pc"][c].append(m["pc"][c]["f1"])
        
        gc.collect(); torch.cuda.empty_cache()
    
    # Results
    print(f"\n{'='*115}")
    print("CASCADE + FINE-SWEEP RESULTS (5 seeds)")
    print(f"{'='*115}")
    header = f"{'Strategy':<40} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 115)
    
    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:25]:
        if len(v["acc"]) < 5:
            continue
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<40} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    
    result_file = Path(__file__).parent / "cascade_results.json"
    with open(result_file, "w") as f:
        json.dump({n: {"acc": float(np.mean(v["acc"])), "mf1": float(np.mean(v["mf1"])),
                        "per_class": {str(c): float(np.mean(v["pc"][c])) for c in cids}}
                   for n, v in all_results.items() if len(v["acc"]) >= 5}, f, indent=2)
    print(f"\nSaved to {result_file}")


if __name__ == "__main__":
    main()
