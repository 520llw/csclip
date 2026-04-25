#!/usr/bin/env python3
"""
Extract one fold of Macrophage (M) vs Neutrophil (N) support/query features
from data2_organized for covariance-ellipse visualization.

Outputs (to /home/xut/csclip/paper_materials/m_vs_n_fold/):
  support_M.npy  (10, D)
  support_N.npy  (10, D)
  query_M.npy    (n, D)
  query_N.npy    (m, D)
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
TARGET_CLASSES = {4: "Neutrophil", 6: "Macrophage"}   # N vs M
N_SHOT = 10
SEED = 42
OUT_DIR = Path("/home/xut/csclip/paper_materials/m_vs_n_fold")


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


def select_support(train_recs, seed):
    random.seed(seed)
    pc = defaultdict(list)
    for r in train_recs:
        pc[r["gt"]].append(r)
    support = {}
    for c in TARGET_CLASSES:
        cands = pc[c]
        chosen = random.sample(cands, min(N_SHOT, len(cands)))
        support[c] = chosen
    return support


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

    # Select 10-shot support for M and N
    support = select_support(train_recs, SEED)

    # Build arrays
    support_M = np.stack([r["feat"] for r in support[6]])   # Macrophage
    support_N = np.stack([r["feat"] for r in support[4]])   # Neutrophil

    query_M = np.stack([r["feat"] for r in val_recs if r["gt"] == 6])
    query_N = np.stack([r["feat"] for r in val_recs if r["gt"] == 4])

    np.save(OUT_DIR / "support_M.npy", support_M)
    np.save(OUT_DIR / "support_N.npy", support_N)
    np.save(OUT_DIR / "query_M.npy", query_M)
    np.save(OUT_DIR / "query_N.npy", query_N)

    print(f"Saved to {OUT_DIR}")
    print(f"  support_M: {support_M.shape}")
    print(f"  support_N: {support_N.shape}")
    print(f"  query_M:   {query_M.shape}")
    print(f"  query_N:   {query_N.shape}")


if __name__ == "__main__":
    main()
