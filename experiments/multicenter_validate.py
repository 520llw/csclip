#!/usr/bin/env python3
"""
MultiCenter dataset validation: feature extraction + SADC classification + PAMSR segmentation.
Validates generalization of our methods on a 7-class multi-center dataset.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys, gc, random, itertools, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MC_ROOT = Path("/home/xut/csclip/cell_datasets/MultiCenter_organized")
D2_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
WEIGHTS = Path("/home/xut/csclip/model_weights")
CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MC_CLASSES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
MC_ALL_CLASSES = {0: "CCEC", 1: "RBC", 2: "SEC", 3: "Eosinophil",
                  4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}

from biomedclip_zeroshot_cell_classify import InstanceInfo


def find_images(data_root, split):
    """Find images with both .png and .jpg extensions."""
    idir = data_root / "images" / split
    if not idir.exists():
        print(f"  WARNING: {idir} does not exist!")
        return []
    imgs = sorted(list(idir.glob("*.png")) + list(idir.glob("*.jpg")) + list(idir.glob("*.jpeg")))
    return imgs


def load_yolo(lp, target_classes):
    anns = []
    if not lp.exists():
        return anns
    for line in open(lp):
        p = line.strip().split()
        if len(p) < 7:
            continue
        c = int(p[0])
        if c in target_classes:
            anns.append({"class_id": c, "points": [float(x) for x in p[1:]]})
    return anns


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
                              min(w, int(np.max(cc)) + 1), min(h, int(np.max(rr)) + 1)), mask=mask)


def crop_cell(image, inst, margin=0.15, mask_bg=False, bg_val=128):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    crop = image[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)].copy()
    if mask_bg:
        mc = inst.mask[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        crop = np.where(mc[..., None], crop, np.full_like(crop, bg_val))
    return crop


def compute_morph(image, inst):
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
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for freq in [0.1, 0.2]:
            kern = cv2.getGaborKernel((9, 9), 2.0, theta, 1.0 / freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
            gabor_responses.append(float(np.mean(np.abs(filtered[mask]))))

    pad_gray = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    lbp_val = np.zeros_like(gray, dtype=float)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        lbp_val += (pad_gray[1 + dy:gray.shape[0] + 1 + dy, 1 + dx:gray.shape[1] + 1 + dx] > gray).astype(float)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = thresh & (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 2]
    n_granules = len(areas) / max(1, np.sum(mask) / 100)
    mean_gs = float(np.mean(areas)) / 100 if areas else 0
    std_gs = float(np.std(areas)) / 100 if len(areas) > 1 else 0

    hist = cv2.calcHist([gray], [0], mask.astype(np.uint8) * 255, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    m_g, s_g = float(np.mean(gm)), float(np.std(gm)) + 1e-6
    dark_thr = np.percentile(gm, 25)
    dark_mask = (gray < dark_thr) & mask
    dark_area = np.sum(dark_mask)
    n_lobes = 0
    if dark_area > 10:
        cnts_d, _ = cv2.findContours(dark_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        n_lobes = len([c for c in cnts_d if cv2.contourArea(c) > 5])
    edges = cv2.Canny(gray, 50, 150)
    em = edges[mask]

    extra = np.array([
        float(np.mean(hp[:, 0])) / 180, float(np.std(hp[:, 0])) / 180,
        float(np.mean(hp[:, 1])) / 255, float(np.std(hp[:, 1])) / 255,
        float(np.mean(hp[:, 2])) / 255, float(np.std(hp[:, 2])) / 255,
        float(np.mean(r / (g + 1e-6))), float(np.mean((r - g) / (r + g + 1e-6))),
        float(np.mean((r - b) / (r + b + 1e-6))),
        float(np.var(lm)) / 1000 if len(lm) > 0 else 0,
        float(np.mean(np.abs(lm))) / 100 if len(lm) > 0 else 0,
        float(np.mean(gabor_responses)) / 100, float(np.std(gabor_responses)) / 100,
        float(np.mean(lbp_val[mask])) / 8, float(np.std(lbp_val[mask])) / 8,
        n_granules, mean_gs, std_gs,
        float(-np.sum(hist * np.log(hist + 1e-10))),
        float(np.mean(((gm.astype(float) - m_g) / s_g) ** 3)),
        float(dark_area) / (float(np.sum(mask)) + 1e-6), float(n_lobes) / 5,
        float(np.sum(em > 0) / len(em)) if len(em) > 0 else 0,
        float(np.sum(gm < dark_thr) / len(gm)),
        float(np.percentile(r, 75) - np.percentile(r, 25)) / 255,
        float(np.mean(r > g)),
        float(np.mean(r)) / 255 - float(np.mean(g)) / 255,
        float(np.std(r) - np.std(g)) / 255,
    ], dtype=np.float32)
    return np.concatenate([base, extra])


# =================== Feature Extraction ===================

def build_cell_index(data_root, split, target_classes):
    images = find_images(data_root, split)
    ldir = data_root / "labels_polygon" / split
    cells = []
    for ip in images:
        anns = load_yolo(ldir / (ip.stem + ".txt"), target_classes)
        for i, ann in enumerate(anns):
            cells.append({"image_path": str(ip), "ann": ann, "idx": i})
    return cells


def extract_all_features(data_root, dataset_name, target_classes):
    """Extract BiomedCLIP, Phikon-v2, DINOv2 features + morphology."""
    cache_bc = CACHE_DIR / f"{dataset_name}_biomedclip"
    cache_ph = CACHE_DIR / f"{dataset_name}_phikon_v2"
    cache_dn = CACHE_DIR / f"{dataset_name}_dinov2_s"

    for split in ["train", "val"]:
        bc_path = CACHE_DIR / f"{dataset_name}_biomedclip_{split}.npz"
        ph_path = CACHE_DIR / f"{dataset_name}_phikon_v2_{split}.npz"
        dn_path = CACHE_DIR / f"{dataset_name}_dinov2_s_{split}.npz"

        if bc_path.exists() and ph_path.exists() and dn_path.exists():
            print(f"  Cache exists for {dataset_name}/{split}, skipping extraction")
            continue

        cells = build_cell_index(data_root, split, target_classes)
        if not cells:
            print(f"  No cells found for {dataset_name}/{split}")
            continue
        print(f"  {dataset_name}/{split}: {len(cells)} cells")

        bc_feats, ph_feats, dn_feats, morphs, labels = [], [], [], [], []
        bc_model, bc_prep = None, None
        ph_model, ph_proc = None, None
        dn_model, dn_trans = None, None

        if not bc_path.exists():
            from labeling_tool.fewshot_biomedclip import _load_model_bundle
            b = _load_model_bundle("auto")
            bc_model, bc_prep = b["model"], b["preprocess"]
            print("    BiomedCLIP loaded")

        if not ph_path.exists():
            from transformers import AutoModel, AutoImageProcessor
            ph_proc = AutoImageProcessor.from_pretrained(str(WEIGHTS / "phikon_v2"))
            ph_model = AutoModel.from_pretrained(str(WEIGHTS / "phikon_v2")).to(DEVICE).eval()
            print("    Phikon-v2 loaded")

        if not dn_path.exists():
            dn_model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                                          num_classes=0, img_size=518)
            state = torch.load(WEIGHTS / 'dinov2_vits14_pretrain.pth', map_location='cpu',
                               weights_only=True)
            dn_model.load_state_dict(state, strict=False)
            dn_model = dn_model.to(DEVICE).eval()
            dn_trans = transforms.Compose([
                transforms.Resize(518), transforms.CenterCrop(518),
                transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
            print("    DINOv2-S loaded")

        t0 = time.time()
        cur_path, cur_img = None, None

        for i, cell in enumerate(cells):
            if cell["image_path"] != cur_path:
                cur_path = cell["image_path"]
                cur_img = np.array(Image.open(cur_path).convert("RGB"))

            h, w = cur_img.shape[:2]
            inst = ann2inst(cell["ann"], h, w, cell["idx"] + 1)
            if inst is None:
                continue

            morph = compute_morph(cur_img, inst)
            morphs.append(morph)
            labels.append(cell["ann"]["class_id"])

            if bc_model is not None:
                cc = crop_cell(cur_img, inst, margin=0.10, mask_bg=True)
                ctx = crop_cell(cur_img, inst, margin=0.30, mask_bg=False)
                ct = bc_prep(Image.fromarray(cc)).unsqueeze(0).to(DEVICE)
                cxt = bc_prep(Image.fromarray(ctx)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    cf = bc_model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
                    xf = bc_model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
                fused = 0.85 * cf + 0.15 * xf
                fused /= fused.norm(dim=-1, keepdim=True)
                bc_feats.append(fused.squeeze(0).cpu().numpy().astype(np.float32))

            if ph_model is not None:
                crop = crop_cell(cur_img, inst, margin=0.15)
                inputs = ph_proc(images=Image.fromarray(crop), return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = ph_model(**inputs)
                    feat = out.last_hidden_state[:, 0]
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                ph_feats.append(feat.squeeze(0).cpu().numpy().astype(np.float32))

            if dn_model is not None:
                crop = crop_cell(cur_img, inst, margin=0.15)
                inp = dn_trans(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    f = dn_model(inp); f /= f.norm(dim=-1, keepdim=True)
                dn_feats.append(f.squeeze(0).cpu().numpy().astype(np.float32))

            if (i + 1) % 200 == 0:
                print(f"    [{i + 1}/{len(cells)}] {time.time() - t0:.1f}s", flush=True)

        morphs_arr = np.stack(morphs)
        labels_arr = np.array(labels)
        print(f"    Done: {len(labels)} cells in {time.time() - t0:.1f}s")

        if bc_feats:
            np.savez_compressed(bc_path, feats=np.stack(bc_feats), morphs=morphs_arr, labels=labels_arr)
            print(f"    Saved BiomedCLIP → {bc_path}")
        if ph_feats:
            np.savez_compressed(ph_path, feats=np.stack(ph_feats), morphs=morphs_arr, labels=labels_arr)
            print(f"    Saved Phikon-v2 → {ph_path}")
        if dn_feats:
            np.savez_compressed(dn_path, feats=np.stack(dn_feats), morphs=morphs_arr, labels=labels_arr)
            print(f"    Saved DINOv2-S → {dn_path}")

        del bc_model, ph_model, dn_model
        gc.collect()
        torch.cuda.empty_cache()


# =================== SADC v3 Classification ===================

def augment_support_all(s_bc, s_ph, s_dn, s_morph, cids, n_aug=20, seed=0):
    rng_state = random.getstate()
    np_state = np.random.get_state()
    a_bc, a_ph, a_dn, a_morph = {}, {}, {}, {}
    for c in cids:
        random.seed(seed + c)
        np.random.seed(seed + c)
        n = len(s_bc[c])
        if n < 2:
            a_bc[c], a_ph[c], a_dn[c], a_morph[c] = s_bc[c], s_ph[c], s_dn[c], s_morph[c]
            continue
        pairs = [(random.sample(range(n), 2), np.random.beta(0.6, 0.6)) for _ in range(n_aug)]
        bc_aug, ph_aug, dn_aug, m_aug = [], [], [], []
        for (i, j), lam in pairs:
            bc_m = lam * s_bc[c][i] + (1 - lam) * s_bc[c][j]
            bc_m /= np.linalg.norm(bc_m) + 1e-8; bc_aug.append(bc_m)
            ph_m = lam * s_ph[c][i] + (1 - lam) * s_ph[c][j]
            ph_m /= np.linalg.norm(ph_m) + 1e-8; ph_aug.append(ph_m)
            dn_m = lam * s_dn[c][i] + (1 - lam) * s_dn[c][j]
            dn_m /= np.linalg.norm(dn_m) + 1e-8; dn_aug.append(dn_m)
            m_aug.append(lam * s_morph[c][i] + (1 - lam) * s_morph[c][j])
        a_bc[c] = np.concatenate([s_bc[c], np.stack(bc_aug)])
        a_ph[c] = np.concatenate([s_ph[c], np.stack(ph_aug)])
        a_dn[c] = np.concatenate([s_dn[c], np.stack(dn_aug)])
        a_morph[c] = np.concatenate([s_morph[c], np.stack(m_aug)])
    random.setstate(rng_state)
    np.random.set_state(np_state)
    return a_bc, a_ph, a_dn, a_morph


def sadc_classify(q_bc, q_ph, q_dn, q_morph, q_labels, s_bc, s_ph, s_dn, s_morph, cids,
                  bw=0.42, pw=0.18, dw=0.07, mw=0.33, k=7, n_iter=2, n_aug=20):
    """SADC v3 with SFA + ATD (best config from data2)."""
    sb, sp, sd, sm = augment_support_all(s_bc, s_ph, s_dn, s_morph, cids, n_aug)
    sb_orig = {c: s_bc[c].copy() for c in cids}

    for it in range(n_iter):
        sm_all = np.concatenate([sm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
        snm = {c: (sm[c] - gm) / gs for c in cids}

        preds, margins = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i] - gm) / gs
            scores = []
            for c in cids:
                vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
                md = np.linalg.norm(qm - snm[c], axis=1)
                ms = 1.0 / (1.0 + md)
                scores.append(float(np.sort(vs + mw * ms)[::-1][:k].mean()))
            sa = np.array(scores)
            ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))])
            margins.append(ss[0] - ss[1])

        preds = np.array(preds)
        margins = np.array(margins)

        for c in cids:
            cm = (preds == c) & (margins > 0.025)
            ci = np.where(cm)[0]
            if len(ci) == 0:
                continue
            proto_c = sb_orig[c].mean(0)
            dists = np.array([np.linalg.norm(q_bc[idx] - proto_c) for idx in ci])
            div_scores = margins[ci] * (1.0 + 0.3 * dists / (dists.mean() + 1e-8))
            ti = ci[np.argsort(div_scores)[::-1][:5]]
            sb[c] = np.concatenate([sb[c][:len(s_bc[c]) + n_aug], q_bc[ti] * 0.5])
            sp[c] = np.concatenate([sp[c][:len(s_ph[c]) + n_aug], q_ph[ti] * 0.5])
            sd[c] = np.concatenate([sd[c][:len(s_dn[c]) + n_aug], q_dn[ti] * 0.5])
            sm[c] = np.concatenate([sm[c][:len(s_morph[c]) + n_aug], q_morph[ti]])

    sm_all = np.concatenate([sm[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0) + 1e-8
    snm = {c: (sm[c] - gm) / gs for c in cids}

    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i] - gm) / gs
        scores = {}
        for c in cids:
            vs = bw * (sb[c] @ q_bc[i]) + pw * (sp[c] @ q_ph[i]) + dw * (sd[c] @ q_dn[i])
            md = np.linalg.norm(qm - snm[c], axis=1)
            ms = 1.0 / (1.0 + md)
            scores[c] = float(np.sort(vs + mw * ms)[::-1][:k].mean())
        gt.append(int(q_labels[i]))
        pred.append(max(scores, key=scores.get))

    total = len(gt)
    correct = sum(int(g == p) for g, p in zip(gt, pred))
    f1s = []
    pc = {}
    for c in cids:
        tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        pp = sum(1 for p in pred if p == c)
        gp = sum(1 for g in gt if g == c)
        pr = tp / pp if pp else 0
        rc = tp / gp if gp else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        pc[c] = f1
        f1s.append(f1)
    return {"acc": correct / total, "mf1": np.mean(f1s), "pc": pc}


# =================== PAMSR Segmentation ===================

def match_masks(gt_masks, pred_masks, iou_thr=0.5):
    tp, matched = 0, set()
    n_pred = pred_masks.max() if pred_masks.max() > 0 else 0
    for pid in range(1, n_pred + 1):
        pm = pred_masks == pid
        if pm.sum() < 30:
            continue
        best, bi = 0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched:
                continue
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou = inter / union if union > 0 else 0
            if iou > best:
                best, bi = iou, gi
        if best >= iou_thr and bi >= 0:
            tp += 1
            matched.add(bi)
    return tp, n_pred - tp, len(gt_masks) - tp


def load_gt_masks(lbl_path, h, w, target_classes):
    anns = []
    if not lbl_path.exists():
        return anns
    for line in open(lbl_path):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = int(parts[0])
        if cid not in target_classes:
            continue
        pts = [float(x) for x in parts[1:]]
        xs = [pts[i] * w for i in range(0, len(pts), 2)]
        ys = [pts[i] * h for i in range(1, len(pts), 2)]
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        if len(rr) == 0:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        anns.append(mask)
    return anns


def pamsr(model, img, primary_d=50, secondary_ds=[40, 65],
          cellprob=-3.0, rescue_prob_thr=1.0, min_area=80, overlap_thr=0.2):
    h, w = img.shape[:2]
    masks_p, flows_p, _ = model.eval(img, diameter=primary_d,
                                     cellprob_threshold=cellprob, channels=[0, 0])
    final = masks_p.copy()
    nid = final.max() + 1
    secondary_cells = []
    for sd in secondary_ds:
        masks_s, flows_s, _ = model.eval(img, diameter=sd,
                                         cellprob_threshold=cellprob, channels=[0, 0])
        prob_map_s = flows_s[2]
        for cid in range(1, masks_s.max() + 1):
            cm = masks_s == cid
            area = cm.sum()
            if area < min_area:
                continue
            overlap = np.sum(final[cm] > 0)
            if overlap > overlap_thr * area:
                continue
            mp = float(prob_map_s[cm].mean())
            secondary_cells.append({"mask": cm, "mp": mp, "d": sd, "area": area})
        del masks_s, flows_s
        gc.collect()

    for i, c1 in enumerate(secondary_cells):
        c1["has_consensus"] = False
        for j, c2 in enumerate(secondary_cells):
            if i == j or c1["d"] == c2["d"]:
                continue
            inter = np.logical_and(c1["mask"], c2["mask"]).sum()
            union = np.logical_or(c1["mask"], c2["mask"]).sum()
            if union > 0 and inter / union > 0.3:
                c1["has_consensus"] = True
                break

    rescued = 0
    for c in sorted(secondary_cells, key=lambda x: -x["mp"]):
        if not (c["mp"] > rescue_prob_thr and c["has_consensus"]):
            continue
        if np.sum(final[c["mask"]] > 0) > overlap_thr * c["area"]:
            continue
        final[c["mask"] & (final == 0)] = nid
        nid += 1
        rescued += 1
    return final, rescued


# =================== Main ===================

def run_classification(dataset_name, data_root, target_classes):
    """Run SADC classification on a dataset."""
    cids = sorted(target_classes.keys())
    label = dataset_name if dataset_name else "data2"
    print(f"\n{'=' * 80}")
    print(f"CLASSIFICATION: {label} ({len(cids)} classes)")
    print(f"{'=' * 80}")

    prefix = f"{dataset_name}_" if dataset_name else ""
    bc_t = np.load(CACHE_DIR / f"{prefix}biomedclip_train.npz")
    bc_v = np.load(CACHE_DIR / f"{prefix}biomedclip_val.npz")
    ph_t = np.load(CACHE_DIR / f"{prefix}phikon_v2_train.npz")
    ph_v = np.load(CACHE_DIR / f"{prefix}phikon_v2_val.npz")
    dn_t = np.load(CACHE_DIR / f"{prefix}dinov2_s_train.npz")
    dn_v = np.load(CACHE_DIR / f"{prefix}dinov2_s_val.npz")

    seeds = [42, 123, 456, 789, 2026]
    all_res = {"acc": [], "mf1": [], "pc": defaultdict(list)}

    for seed in seeds:
        random.seed(seed)
        pc = defaultdict(list)
        for i, l in enumerate(bc_t["labels"]):
            pc[int(l)].append(i)
        si = {c: random.sample(pc[c], min(10, len(pc[c]))) for c in cids}

        m = sadc_classify(
            bc_v["feats"], ph_v["feats"], dn_v["feats"], bc_v["morphs"], bc_v["labels"],
            {c: bc_t["feats"][si[c]] for c in cids},
            {c: ph_t["feats"][si[c]] for c in cids},
            {c: dn_t["feats"][si[c]] for c in cids},
            {c: bc_t["morphs"][si[c]] for c in cids},
            cids)

        all_res["acc"].append(m["acc"])
        all_res["mf1"].append(m["mf1"])
        for c in cids:
            all_res["pc"][c].append(m["pc"][c])
        print(f"  Seed {seed}: Acc={m['acc']:.4f} mF1={m['mf1']:.4f}", flush=True)

    print(f"\n  AVERAGE over {len(seeds)} seeds:")
    print(f"  Acc  = {np.mean(all_res['acc']):.4f} ± {np.std(all_res['acc']):.4f}")
    print(f"  mF1  = {np.mean(all_res['mf1']):.4f} ± {np.std(all_res['mf1']):.4f}")
    for c in cids:
        name = target_classes[c]
        print(f"  {name:<12} F1 = {np.mean(all_res['pc'][c]):.4f} ± {np.std(all_res['pc'][c]):.4f}")
    return all_res


def run_segmentation(dataset_name, data_root, target_classes, max_images=50):
    """Run PAMSR segmentation on a dataset."""
    print(f"\n{'=' * 80}")
    print(f"SEGMENTATION: {dataset_name}")
    print(f"{'=' * 80}")

    from cellpose import models
    model = models.CellposeModel(gpu=True)

    images = find_images(data_root, "val")
    if len(images) > max_images:
        images = images[:max_images]
    lbl_dir = data_root / "labels_polygon" / "val"

    agg = defaultdict(lambda: [0, 0, 0])
    for idx, ip in enumerate(images):
        lp = lbl_dir / (ip.stem + ".txt")
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        gt = load_gt_masks(lp, h, w, target_classes)
        if not gt:
            continue

        masks_def, _, _ = model.eval(img, channels=[0, 0])
        tp, fp, fn = match_masks(gt, masks_def)
        agg["default"][0] += tp; agg["default"][1] += fp; agg["default"][2] += fn
        del masks_def

        masks_opt, _, _ = model.eval(img, diameter=50, cellprob_threshold=-3.0, channels=[0, 0])
        tp, fp, fn = match_masks(gt, masks_opt)
        agg["optimized_d50"][0] += tp; agg["optimized_d50"][1] += fp; agg["optimized_d50"][2] += fn
        del masks_opt

        masks_pamsr, rescued = pamsr(model, img)
        tp, fp, fn = match_masks(gt, masks_pamsr)
        agg["PAMSR"][0] += tp; agg["PAMSR"][1] += fp; agg["PAMSR"][2] += fn
        del masks_pamsr
        gc.collect()

        print(f"  [{idx + 1}/{len(images)}] {ip.name}", flush=True)

    print(f"\n  {'Method':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-' * 65}")
    for name, (tp, fp, fn) in sorted(agg.items(), key=lambda x: -(2 * x[1][0] / (2 * x[1][0] + x[1][1] + x[1][2] + 1e-10))):
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"  {name:<20} {tp:>5} {fp:>5} {fn:>5} {p:>7.4f} {r:>7.4f} {f1:>7.4f}")
    return agg


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-DATASET VALIDATION")
    print("=" * 80)

    mc_imgs_train = find_images(MC_ROOT, "train")
    mc_imgs_val = find_images(MC_ROOT, "val")
    d2_imgs_train = find_images(D2_ROOT, "train")
    d2_imgs_val = find_images(D2_ROOT, "val")

    print(f"\nMultiCenter: train={len(mc_imgs_train)} val={len(mc_imgs_val)} images")
    print(f"data2:       train={len(d2_imgs_train)} val={len(d2_imgs_val)} images")

    if not mc_imgs_val:
        print("\n*** MultiCenter images not found! Skipping MC validation. ***")
        print("*** Only running data2 validation. ***\n")

    datasets = []

    if d2_imgs_val:
        datasets.append(("data2", D2_ROOT, MC_CLASSES))

    if mc_imgs_val:
        datasets.append(("multicenter", MC_ROOT, MC_CLASSES))

    for ds_name, ds_root, ds_classes in datasets:
        print(f"\n{'#' * 80}")
        print(f"DATASET: {ds_name}")
        print(f"{'#' * 80}")

        if ds_name == "data2":
            cache_name = ""
            bc_check = CACHE_DIR / "biomedclip_train.npz"
        else:
            cache_name = ds_name
            bc_check = CACHE_DIR / f"{ds_name}_biomedclip_train.npz"

        if not bc_check.exists():
            print(f"\nExtracting features for {ds_name}...")
            extract_all_features(ds_root, cache_name if cache_name else "data2_tmp", ds_classes)

        try:
            run_classification(cache_name, ds_root, ds_classes)
        except Exception as e:
            print(f"  Classification error: {e}")

        run_segmentation(ds_name, ds_root, ds_classes, max_images=40)

    print(f"\n{'=' * 80}")
    print("ALL VALIDATIONS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
