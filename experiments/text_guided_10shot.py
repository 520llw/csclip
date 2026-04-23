#!/usr/bin/env python3
"""
Text-guided few-shot classification for BALF cells.
Uses BiomedCLIP's text encoder to create text-aware prototypes that
disambiguate visually similar cell types.

Also tests:
- SVM / Logistic regression on frozen features
- NCM with Mahalanobis distance
- Text-visual prototype fusion with transductive + cascade
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import random
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]

# Cytological descriptions for BiomedCLIP text encoding
CELL_DESCRIPTIONS = {
    3: [
        "eosinophil granulocyte with bright red-orange eosin-stained granules",
        "eosinophil cell with bilobed nucleus and prominent eosinophilic cytoplasmic granules",
        "eosinophil with large red granules and segmented nucleus in bronchoalveolar lavage",
        "a round cell with abundant red-staining granules characteristic of eosinophils",
    ],
    4: [
        "neutrophil granulocyte with multilobed nucleus and fine pale granules",
        "neutrophil cell with segmented nucleus in bronchoalveolar lavage fluid",
        "polymorphonuclear neutrophil with light purple cytoplasm and lobulated nucleus",
        "a cell with multi-segmented dark nucleus and faint cytoplasmic granules typical of neutrophils",
    ],
    5: [
        "lymphocyte with large dark nucleus and thin rim of blue cytoplasm",
        "small round lymphocyte cell in bronchoalveolar lavage fluid",
        "mononuclear lymphocyte with high nucleus-to-cytoplasm ratio",
        "a small round cell with dense chromatin and minimal cytoplasm characteristic of lymphocytes",
    ],
    6: [
        "macrophage with large irregular shape and foamy or vacuolated cytoplasm",
        "alveolar macrophage cell with eccentric nucleus and phagocytic inclusions",
        "large phagocytic macrophage in bronchoalveolar lavage with abundant cytoplasm",
        "a large cell with irregular borders and pale vacuolated cytoplasm typical of macrophages",
    ],
}

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_cache(model, split):
    d = np.load(CACHE_DIR / f"{model}_{split}.npz")
    return d["feats"], d["morphs"], d["labels"]


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


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels): pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def encode_text_prototypes():
    """Encode cell type text descriptions with BiomedCLIP."""
    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    from open_clip import get_tokenizer
    bundle = _load_model_bundle("auto")
    model = bundle["model"]

    bclip_dir = "/home/xut/csclip/labeling_tool/weights/biomedclip"
    tokenizer = get_tokenizer(f"local-dir:{bclip_dir}")

    text_prototypes = {}
    for cid, descs in CELL_DESCRIPTIONS.items():
        feats = []
        for desc in descs:
            tokens = tokenizer([desc]).to(DEVICE)
            with torch.no_grad():
                tf = model.encode_text(tokens)
                tf /= tf.norm(dim=-1, keepdim=True)
            feats.append(tf.squeeze(0).cpu().numpy().astype(np.float32))
        text_prototypes[cid] = np.mean(feats, axis=0)
        text_prototypes[cid] /= np.linalg.norm(text_prototypes[cid]) + 1e-10

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return text_prototypes


def cls_text_visual_fusion(q_bclip, q_dino, q_morph, q_labels,
                            s_bclip, s_dino, s_morph,
                            cids, text_protos,
                            bw=0.40, dw=0.15, mw=0.30, tw=0.15, k=7):
    """Dual-backbone + text prototype fusion."""
    sm_all = np.concatenate([s_morph[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_morph[c]-gm)/gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        scores = []
        for c in cids:
            vs_b = s_bclip[c] @ q_bclip[i]
            vs_d = s_dino[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            text_sim = float(text_protos[c] @ q_bclip[i])
            comb = bw*vs_b + dw*vs_d + mw*ms
            visual_score = float(np.sort(comb)[::-1][:k].mean())
            scores.append(visual_score + tw * text_sim)
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmax(scores))])
    return metrics(gt, pred, cids)


def cls_text_trans_cascade(q_bclip, q_dino, q_morph, q_labels,
                            s_bclip_init, s_dino_init, s_morph_init,
                            cids, text_protos, morph_weights,
                            bw=0.40, dw=0.15, mw=0.30, tw=0.15, k=7,
                            n_iter=2, top_k=5, conf_thr=0.025, cascade_thr=0.01):
    """Text-guided + transductive + cascade."""
    s_b = {c: s_bclip_init[c].copy() for c in cids}
    s_d = {c: s_dino_init[c].copy() for c in cids}
    s_m = {c: s_morph_init[c].copy() for c in cids}

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
                visual_score = float(np.sort(comb)[::-1][:k].mean())
                text_sim = float(text_protos[c] @ q_bclip[i])
                scores.append(visual_score + tw * text_sim)
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
            s_b[c] = np.concatenate([s_bclip_init[c], q_bclip[sorted_idx]*0.5])
            s_d[c] = np.concatenate([s_dino_init[c], q_dino[sorted_idx]*0.5])
            s_m[c] = np.concatenate([s_morph_init[c], q_morph[sorted_idx]])

    sm_all = np.concatenate([s_m[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (s_m[c]-gm)/gs for c in cids}
    snm_w = {c: (s_m[c]-gm)/gs * morph_weights for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm)/gs
        qm_w = qm * morph_weights
        scores = {}
        for c in cids:
            vs_b = s_b[c] @ q_bclip[i]
            vs_d = s_d[c] @ q_dino[i]
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0/(1.0+md)
            comb = bw*vs_b + dw*vs_d + mw*ms
            visual_score = float(np.sort(comb)[::-1][:k].mean())
            text_sim = float(text_protos[c] @ q_bclip[i])
            scores[c] = visual_score + tw * text_sim
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


def cls_svm(q_feats, q_labels, s_feats, s_labels, cids, C=1.0, kernel='rbf'):
    """SVM classifier on fused features."""
    X_train = np.array(s_feats)
    y_train = np.array(s_labels)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(q_feats)
    svm = SVC(C=C, kernel=kernel, decision_function_shape='ovr')
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test).tolist()
    gt = [int(l) for l in q_labels]
    return metrics(gt, pred, cids)


def cls_logreg(q_feats, q_labels, s_feats, s_labels, cids, C=1.0):
    """Logistic regression on fused features."""
    X_train = np.array(s_feats)
    y_train = np.array(s_labels)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(q_feats)
    lr = LogisticRegression(C=C, max_iter=1000, multi_class='multinomial',
                             class_weight='balanced', solver='lbfgs')
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test).tolist()
    gt = [int(l) for l in q_labels]
    return metrics(gt, pred, cids)


def cls_ncm_mahalanobis(q_feats, q_labels, s_feats_dict, cids, alpha=0.5):
    """NCM with regularized Mahalanobis distance."""
    prototypes = {c: np.mean(s_feats_dict[c], axis=0) for c in cids}
    all_s = np.concatenate([s_feats_dict[c] for c in cids])
    global_cov = np.cov(all_s.T) + alpha * np.eye(all_s.shape[1])
    try:
        cov_inv = np.linalg.inv(global_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(all_s.shape[1])

    gt, pred = [], []
    for i in range(len(q_labels)):
        dists = []
        for c in cids:
            diff = q_feats[i] - prototypes[c]
            d = float(diff @ cov_inv @ diff)
            dists.append(d)
        gt.append(int(q_labels[i]))
        pred.append(cids[int(np.argmin(dists))])
    return metrics(gt, pred, cids)


def main():
    bclip_train, morph_train, labels_train = load_cache("biomedclip", "train")
    bclip_val, morph_val, labels_val = load_cache("biomedclip", "val")
    dino_train, _, _ = load_cache("dinov2_s", "train")
    dino_val, _, _ = load_cache("dinov2_s", "val")

    cids = sorted(CLASS_NAMES.keys())

    # Fisher weights for cascade
    eos, neu = morph_train[labels_train==3], morph_train[labels_train==4]
    n_dims = morph_train.shape[1]
    fisher_w = np.ones(n_dims, np.float32)
    for d in range(n_dims):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        fisher_w[d] = 1.0 + f * 2.0

    # Encode text prototypes
    print("Encoding text prototypes...")
    text_protos = encode_text_prototypes()
    print("Text prototype similarities:")
    for c1 in cids:
        for c2 in cids:
            sim = float(text_protos[c1] @ text_protos[c2])
            print(f"  {CLASS_NAMES[c1][:3]}-{CLASS_NAMES[c2][:3]}: {sim:.4f}", end="")
        print()

    all_results = defaultdict(lambda: {"acc": [], "mf1": [], "pc": defaultdict(list)})

    for seed in SEEDS:
        print(f"\nSeed {seed}...")
        np.random.seed(seed)
        support_idx = select_support(labels_train, seed, cids)
        s_bclip = {c: bclip_train[support_idx[c]] for c in cids}
        s_dino = {c: dino_train[support_idx[c]] for c in cids}
        s_morph = {c: morph_train[support_idx[c]] for c in cids}

        # 1) Text-visual fusion (no transductive/cascade)
        for tw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for bw in [0.40, 0.45]:
                dw = 0.20
                mw_v = 1.0 - bw - dw - tw
                if mw_v < 0.05: continue
                name = f"textvis_b{bw}_d{dw}_m{mw_v:.2f}_t{tw}"
                m = cls_text_visual_fusion(
                    bclip_val, dino_val, morph_val, labels_val,
                    s_bclip, s_dino, s_morph, cids, text_protos,
                    bw, dw, mw_v, tw)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 2) Text + transductive + cascade
        for tw in [0.05, 0.10, 0.15]:
            bw = 0.40
            dw = 0.15
            mw_v = 1.0 - bw - dw - tw
            for cthr in [0.008, 0.010, 0.012]:
                name = f"text_tc_tw{tw}_ct{cthr}"
                m = cls_text_trans_cascade(
                    bclip_val, dino_val, morph_val, labels_val,
                    s_bclip, s_dino, s_morph, cids, text_protos, fisher_w,
                    bw, dw, mw_v, tw, cascade_thr=cthr)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 3) SVM on concatenated features
        s_feats_list, s_labs_list = [], []
        for c in cids:
            sm_all_tmp = np.concatenate([s_morph[cc] for cc in cids])
            gm_t = sm_all_tmp.mean(0)
            gs_t = sm_all_tmp.std(0)+1e-8
            for idx in range(len(support_idx[c])):
                f = np.concatenate([
                    s_bclip[c][idx],
                    s_dino[c][idx],
                    (s_morph[c][idx]-gm_t)/gs_t
                ])
                s_feats_list.append(f)
                s_labs_list.append(c)

        q_feats_fused = []
        sm_all_tmp = np.concatenate([s_morph[cc] for cc in cids])
        gm_t = sm_all_tmp.mean(0)
        gs_t = sm_all_tmp.std(0)+1e-8
        for i in range(len(labels_val)):
            f = np.concatenate([bclip_val[i], dino_val[i], (morph_val[i]-gm_t)/gs_t])
            q_feats_fused.append(f)
        q_feats_fused = np.array(q_feats_fused)

        for C in [0.01, 0.1, 1.0, 10.0]:
            for kernel in ['rbf', 'linear']:
                name = f"svm_{kernel}_C{C}"
                m = cls_svm(q_feats_fused, labels_val, s_feats_list, s_labs_list, cids, C, kernel)
                all_results[name]["acc"].append(m["acc"])
                all_results[name]["mf1"].append(m["mf1"])
                for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 4) Logistic Regression
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            name = f"logreg_C{C}"
            m = cls_logreg(q_feats_fused, labels_val, s_feats_list, s_labs_list, cids, C)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

        # 5) NCM with Mahalanobis (BiomedCLIP features only)
        for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
            name = f"ncm_mahal_a{alpha}"
            s_dict = {c: bclip_train[support_idx[c]] for c in cids}
            m = cls_ncm_mahalanobis(bclip_val, labels_val, s_dict, cids, alpha)
            all_results[name]["acc"].append(m["acc"])
            all_results[name]["mf1"].append(m["mf1"])
            for c in cids: all_results[name]["pc"][c].append(m["pc"][c]["f1"])

    print(f"\n{'='*130}")
    print("TEXT-GUIDED & ML CLASSIFIER RESULTS (5 seeds)")
    print(f"{'='*130}")
    header = f"{'Strategy':<50} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'Astd':>5} {'Fstd':>5}"
    print(header)
    print("-" * 130)

    sorted_r = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for name, v in sorted_r[:30]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    print(f"\n--- Best by Eos F1 ---")
    sorted_eos = sorted(all_results.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for name, v in sorted_eos[:15]:
        pc_str = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{name:<50} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} "
              f"{pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")

    best = sorted_r[0]
    print(f"\nBEST: {best[0]} → mF1={np.mean(best[1]['mf1']):.4f}")


if __name__ == "__main__":
    main()
