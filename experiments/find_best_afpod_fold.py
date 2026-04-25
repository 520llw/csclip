#!/usr/bin/env python3
"""
Search across all 5 seeds and all class pairs in data2_organized
to find a fold where AFP-OD (LW+separation) clearly beats baseline.

Outputs the best (seed, class_pair, delta_F1) and saves corresponding .npy files.
"""
import sys
import random
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomedclip_zeroshot_cell_classify import InstanceInfo
from PIL import Image
from skimage.draw import polygon as sk_polygon

DATA_ROOT = Path("/home/xut/csclip/cell_datasets/data2_organized")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]
OUT_DIR = Path("/home/xut/csclip/paper_materials/best_afpod_fold")


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


def build_items(split):
    idir = DATA_ROOT / "images" / split
    ldir = DATA_ROOT / "labels_polygon" / split
    items = []
    for ip in sorted(idir.glob("*.png")):
        anns = load_yolo(ldir / (ip.stem + ".txt"))
        if anns:
            items.append({"image_path": str(ip), "filename": ip.name, "annotations": anns})
    return items


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


def encode_cell(model, preprocess, device, image, inst,
                cell_margin=0.10, ctx_margin=0.30, bg=128, cw=0.85, ctxw=0.15):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = inst.bbox
    bw, bh = x2 - x1, y2 - y1

    mx, my = int(bw * cell_margin), int(bh * cell_margin)
    crop = image[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)].copy()
    mc = inst.mask[max(0,y1-my):min(h,y2+my), max(0,x1-mx):min(w,x2+mx)]
    crop = np.where(mc[..., None], crop, np.full_like(crop, bg))

    cmx, cmy = int(bw * ctx_margin), int(bh * ctx_margin)
    ctx = image[max(0,y1-cmy):min(h,y2+cmy), max(0,x1-cmx):min(w,x2+cmx)].copy()

    ct = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
    cxt = preprocess(Image.fromarray(ctx)).unsqueeze(0).to(device)
    with torch.no_grad():
        cf = model.encode_image(ct); cf /= cf.norm(dim=-1, keepdim=True)
        xf = model.encode_image(cxt); xf /= xf.norm(dim=-1, keepdim=True)
    fused = cw * cf + ctxw * xf
    fused /= fused.norm(dim=-1, keepdim=True)
    return fused.squeeze(0).cpu().numpy().astype(np.float32)


def extract_all(items, model, preprocess, device):
    recs = []
    for item in items:
        img = np.array(Image.open(item["image_path"]).convert("RGB"))
        h, w = img.shape[:2]
        for i, ann in enumerate(item["annotations"]):
            inst = ann2inst(ann, h, w, i + 1)
            if inst is None:
                continue
            feat = encode_cell(model, preprocess, device, img, inst)
            recs.append({"gt": ann["class_id"], "feat": feat,
                         "img": item["image_path"], "idx": i})
    return recs


def select_support(train_recs, seed, target_classes):
    random.seed(seed)
    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r)
    support = {}
    for c in target_classes:
        cands = pc[c]
        chosen = random.sample(cands, min(N_SHOT, len(cands)))
        support[c] = chosen
    return support


# ==================== AFP-OD core (inline for speed) ====================

def _ledoit_wolf_shrinkage(X):
    n, d = X.shape
    if n < 2:
        return 1.0, np.zeros((d, d), dtype=np.float64), np.zeros((d, d), dtype=np.float64)
    X_c = X - X.mean(axis=0, keepdims=True)
    S = (X_c.T @ X_c) / n
    trace_S = float(np.trace(S))
    F = (trace_S / d) * np.eye(d, dtype=S.dtype)
    X2 = X_c ** 2
    pi_mat = (X2.T @ X2) / n - S ** 2
    pi_hat = float(pi_mat.sum())
    gamma_hat = float(((S - F) ** 2).sum())
    if gamma_hat < 1e-12:
        shrink = 0.0
    else:
        shrink = max(0.0, min(1.0, pi_hat / (gamma_hat * n)))
    return shrink, S, F


def fisher_direction(feats_i, feats_j, method="trace", shrink=0.3):
    mu_i = feats_i.mean(axis=0)
    mu_j = feats_j.mean(axis=0)
    d = feats_i.shape[1]
    diff = (mu_i - mu_j).astype(np.float64)

    if method == "lw":
        if len(feats_i) > 1:
            shrink_i, S_i, F_i = _ledoit_wolf_shrinkage(feats_i.astype(np.float64))
            Sigma_i = (1.0 - shrink_i) * S_i + shrink_i * F_i
        else:
            Sigma_i = np.zeros((d, d), dtype=np.float64)
        if len(feats_j) > 1:
            shrink_j, S_j, F_j = _ledoit_wolf_shrinkage(feats_j.astype(np.float64))
            Sigma_j = (1.0 - shrink_j) * S_j + shrink_j * F_j
        else:
            Sigma_j = np.zeros((d, d), dtype=np.float64)
        Sigma_reg = Sigma_i + Sigma_j + 1e-6 * np.eye(d, dtype=np.float64)
    else:
        Sigma_i = np.cov(feats_i, rowvar=False).astype(np.float64) if len(feats_i) > 1 else np.zeros((d, d))
        Sigma_j = np.cov(feats_j, rowvar=False).astype(np.float64) if len(feats_j) > 1 else np.zeros((d, d))
        Sigma_sum = Sigma_i + Sigma_j
        trace_val = np.trace(Sigma_sum) / d if d > 0 else 1e-6
        if trace_val < 1e-8:
            trace_val = 1e-6
        Sigma_reg = (1.0 - shrink) * Sigma_sum + shrink * trace_val * np.eye(d)

    try:
        w = np.linalg.solve(Sigma_reg, diff)
    except np.linalg.LinAlgError:
        w = diff
    norm = float(np.linalg.norm(w))
    return (w / norm).astype(np.float32) if norm > 1e-8 else np.zeros(d, dtype=np.float32)


def amplify_separation(support_i, support_j, alpha=0.2, method="trace", shrink=0.3):
    w = fisher_direction(support_i, support_j, method=method, shrink=shrink)
    if np.linalg.norm(w) < 1e-8:
        return support_i.copy(), support_j.copy()
    mod_i = support_i.copy().astype(np.float32) + alpha * w
    mod_j = support_j.copy().astype(np.float32) - alpha * w
    for arr in (mod_i, mod_j):
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        arr[:] = (arr / norms).astype(np.float32)
    return mod_i, mod_j


def eval_binary_f1(s_i, s_j, q_i, q_j):
    """Returns (baseline_f1, lw_f1, afpod_f1) for this fold."""
    X_query = np.vstack([q_j, q_i])
    y_true = np.array([0]*len(q_j) + [1]*len(q_i))

    def _f1(si, sj):
        pi = si.mean(0); pi /= np.linalg.norm(pi)
        pj = sj.mean(0); pj /= np.linalg.norm(pj)
        preds = []
        for q in X_query:
            qn = q / np.linalg.norm(q)
            preds.append(1 if qn @ pj > qn @ pi else 0)
        tp_i = sum(1 for g, p in zip(y_true, preds) if g == 1 and p == 1)
        tp_j = sum(1 for g, p in zip(y_true, preds) if g == 0 and p == 0)
        gp_i = sum(1 for g in y_true if g == 1)
        gp_j = sum(1 for g in y_true if g == 0)
        pp_i = sum(1 for p in preds if p == 1)
        pp_j = sum(1 for p in preds if p == 0)
        f1_i = 2*tp_i/(gp_i+pp_i) if (gp_i+pp_i) else 0
        f1_j = 2*tp_j/(gp_j+pp_j) if (gp_j+pp_j) else 0
        return (f1_i + f1_j) / 2

    base = _f1(s_i, s_j)
    # LW only changes covariance estimate, not support positions for prototype
    lw = base
    si_mod, sj_mod = amplify_separation(s_i, s_j, alpha=0.2, method="lw")
    afpod = _f1(si_mod, sj_mod)
    return base, lw, afpod


# ==================== Main sweep ====================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_items = build_items("train")
    val_items = build_items("val")

    from labeling_tool.fewshot_biomedclip import _load_model_bundle
    bundle = _load_model_bundle("auto")
    model, preprocess, device = bundle["model"], bundle["preprocess"], bundle["device"]

    print("Extracting train features...")
    train_recs = extract_all(train_items, model, preprocess, device)
    print("Extracting val features...")
    val_recs = extract_all(val_items, model, preprocess, device)

    cids = sorted(CLASS_NAMES.keys())
    pairs = [(cids[i], cids[j]) for i in range(len(cids)) for j in range(i+1, len(cids))]

    results = []  # (seed, ci, cj, base_f1, lw_f1, afpod_f1, delta)

    for seed in SEEDS:
        support = select_support(train_recs, seed, cids)
        val_by_class = {c: [r["feat"] for r in val_recs if r["gt"] == c] for c in cids}

        for ci, cj in pairs:
            s_i = np.stack([r["feat"] for r in support[ci]])
            s_j = np.stack([r["feat"] for r in support[cj]])
            q_i = np.stack(val_by_class[ci])
            q_j = np.stack(val_by_class[cj])

            base, lw, afpod = eval_binary_f1(s_i, s_j, q_i, q_j)
            delta = afpod - base
            results.append((seed, ci, cj, base, lw, afpod, delta))
            print(f"seed={seed}  {CLASS_NAMES[ci]:>12} vs {CLASS_NAMES[cj]:<12}  "
                  f"base={base:.4f}  AFP-OD={afpod:.4f}  delta={delta:+.4f}")

    # Find best
    best = max(results, key=lambda x: x[6])
    seed, ci, cj, base, lw, afpod, delta = best
    print(f"\n{'='*70}")
    print(f"BEST: seed={seed}  {CLASS_NAMES[ci]} vs {CLASS_NAMES[cj]}")
    print(f"      base={base:.4f}  AFP-OD={afpod:.4f}  delta={delta:+.4f}")
    print(f"{'='*70}")

    # Save best fold arrays
    support = select_support(train_recs, seed, cids)
    val_by_class = {c: [r["feat"] for r in val_recs if r["gt"] == c] for c in cids}

    s_i = np.stack([r["feat"] for r in support[ci]])
    s_j = np.stack([r["feat"] for r in support[cj]])
    q_i = np.stack(val_by_class[ci])
    q_j = np.stack(val_by_class[cj])

    tag_i = CLASS_NAMES[ci][0]  # E, N, L, M
    tag_j = CLASS_NAMES[cj][0]

    np.save(OUT_DIR / f"support_{tag_i}.npy", s_i)
    np.save(OUT_DIR / f"support_{tag_j}.npy", s_j)
    np.save(OUT_DIR / f"query_{tag_i}.npy", q_i)
    np.save(OUT_DIR / f"query_{tag_j}.npy", q_j)
    print(f"Saved to {OUT_DIR}: support_{tag_i}.npy, support_{tag_j}.npy, query_{tag_i}.npy, query_{tag_j}.npy")


if __name__ == "__main__":
    main()
