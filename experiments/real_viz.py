#!/usr/bin/env python3
"""
Generate real image visualizations showing:
1. Segmentation: Default vs PAMSR on actual images
2. Classification: GT vs predicted labels on cell crops
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys, gc, random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon as sk_polygon

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

D2 = Path("/home/xut/csclip/cell_datasets/data2_organized")
CACHE = Path("/home/xut/csclip/experiments/feature_cache")
OUT = Path("/home/xut/csclip/experiments/figures")
CLASSES = {3: "Eos", 4: "Neu", 5: "Lym", 6: "Mac"}
COLORS = {3: (255, 80, 80), 4: (80, 180, 255), 5: (80, 255, 80), 6: (255, 200, 80)}
cids = sorted(CLASSES.keys())


def load_gt_anns(lp, h, w):
    anns = []
    if not lp.exists(): return anns
    for line in open(lp):
        p = line.strip().split()
        if len(p) < 7: continue
        c = int(p[0])
        if c not in CLASSES: continue
        pts = [float(x) for x in p[1:]]
        xs = np.array([pts[i]*w for i in range(0,len(pts),2)])
        ys = np.array([pts[i]*h for i in range(1,len(pts),2)])
        rr, cc = sk_polygon(ys, xs, shape=(h,w))
        if len(rr)==0: continue
        mask = np.zeros((h,w), dtype=bool)
        mask[rr,cc] = True
        cx, cy = int(xs.mean()), int(ys.mean())
        anns.append({"cid": c, "mask": mask, "cx": cx, "cy": cy,
                      "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    return anns


def draw_masks_on_image(img, masks_arr, alpha=0.3):
    overlay = img.copy()
    np.random.seed(42)
    for mid in range(1, masks_arr.max()+1):
        m = masks_arr == mid
        if m.sum() < 30: continue
        color = np.random.randint(50, 255, 3).tolist()
        overlay[m] = color
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)


def draw_gt_on_image(img, anns, alpha=0.3):
    overlay = img.copy()
    for ann in anns:
        c = COLORS[ann["cid"]]
        overlay[ann["mask"]] = c
        contours, _ = cv2.findContours(ann["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, c, 2)
        cv2.putText(overlay, CLASSES[ann["cid"]], (ann["cx"]-15, ann["cy"]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)


def pamsr(model, img, primary_d=50, secondary_ds=[40, 65], cellprob=-3.0):
    h, w = img.shape[:2]
    masks_p, flows_p, _ = model.eval(img, diameter=primary_d, cellprob_threshold=cellprob, channels=[0,0])
    final = masks_p.copy()
    nid = final.max()+1
    secondary_cells = []
    for sd in secondary_ds:
        masks_s, flows_s, _ = model.eval(img, diameter=sd, cellprob_threshold=cellprob, channels=[0,0])
        prob_s = flows_s[2]
        for cid in range(1, masks_s.max()+1):
            cm = masks_s==cid
            if cm.sum()<80: continue
            if np.sum(final[cm]>0) > 0.2*cm.sum(): continue
            mp = float(prob_s[cm].mean())
            secondary_cells.append({"mask":cm, "mp":mp, "d":sd})
        del masks_s, flows_s; gc.collect()
    for i, c1 in enumerate(secondary_cells):
        c1["cons"] = False
        for j, c2 in enumerate(secondary_cells):
            if i==j or c1["d"]==c2["d"]: continue
            inter = np.logical_and(c1["mask"], c2["mask"]).sum()
            union = np.logical_or(c1["mask"], c2["mask"]).sum()
            if union>0 and inter/union>0.3: c1["cons"]=True; break
    for c in sorted(secondary_cells, key=lambda x:-x["mp"]):
        if c["mp"]>1.0 and c["cons"]:
            if np.sum(final[c["mask"]]>0) <= 0.2*c["mask"].sum():
                final[c["mask"]&(final==0)] = nid; nid+=1
    return final


def viz_segmentation():
    """3 example images: GT | Default | PAMSR"""
    from cellpose import models
    model = models.CellposeModel(gpu=True)

    imgs = sorted((D2/"images"/"val").glob("*.png"))
    sel = [imgs[0], imgs[5], imgs[20]]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    titles = ["Ground Truth", "Default CellposeSAM (F1=0.51)", "PAMSR Optimized (F1=0.73)"]

    for row, ip in enumerate(sel):
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        lp = D2/"labels_polygon"/"val"/(ip.stem+".txt")
        anns = load_gt_anns(lp, h, w)

        gt_vis = draw_gt_on_image(img.copy(), anns)

        masks_def, _, _ = model.eval(img, channels=[0,0])
        def_vis = draw_masks_on_image(img.copy(), masks_def)
        n_def = masks_def.max()

        masks_pamsr = pamsr(model, img)
        pamsr_vis = draw_masks_on_image(img.copy(), masks_pamsr)
        n_pamsr = masks_pamsr.max()

        crop_h, crop_w = h//2, w//2
        sy, sx = h//4, w//4

        for col, (vis, title) in enumerate([(gt_vis, titles[0]),
                                             (def_vis, titles[1]),
                                             (pamsr_vis, titles[2])]):
            axes[row, col].imshow(vis[sy:sy+crop_h, sx:sx+crop_w])
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(title, fontsize=14, fontweight='bold')
            if col == 1:
                axes[row, col].text(5, 30, f"Detected: {n_def}", fontsize=11,
                                   color='white', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
            elif col == 2:
                axes[row, col].text(5, 30, f"Detected: {n_pamsr}", fontsize=11,
                                   color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
            elif col == 0:
                axes[row, col].text(5, 30, f"GT cells: {len(anns)}", fontsize=11,
                                   color='white', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))

        del masks_def, masks_pamsr; gc.collect()

    plt.suptitle("Cell Segmentation: GT vs Default vs PAMSR (data2_organized)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT/"segmentation_real.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def viz_classification():
    """Show cell crops with GT vs predicted labels."""
    bc_t = np.load(CACHE/"biomedclip_train.npz")
    bc_v = np.load(CACHE/"biomedclip_val.npz")
    ph_t = np.load(CACHE/"phikon_v2_train.npz")
    ph_v = np.load(CACHE/"phikon_v2_val.npz")
    dn_t = np.load(CACHE/"dinov2_s_train.npz")
    dn_v = np.load(CACHE/"dinov2_s_val.npz")

    random.seed(42)
    pc = defaultdict(list)
    for i, l in enumerate(bc_t["labels"]): pc[int(l)].append(i)
    si = {c: random.sample(pc[c], min(10, len(pc[c]))) for c in cids}

    sb = {c: bc_t["feats"][si[c]] for c in cids}
    sp = {c: ph_t["feats"][si[c]] for c in cids}
    sd = {c: dn_t["feats"][si[c]] for c in cids}
    sm = {c: bc_t["morphs"][si[c]] for c in cids}
    sb_orig = {c: sb[c].copy() for c in cids}

    bw, pw, dw, mw, k = 0.42, 0.18, 0.07, 0.33, 7
    for it in range(2):
        sm_all = np.concatenate([sm[c] for c in cids])
        gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
        snm = {c: (sm[c]-gm)/gs for c in cids}
        preds, margins = [], []
        for i in range(len(bc_v["labels"])):
            qm = (bc_v["morphs"][i]-gm)/gs
            scores = []
            for c in cids:
                vs = bw*(sb[c]@bc_v["feats"][i])+pw*(sp[c]@ph_v["feats"][i])+dw*(sd[c]@dn_v["feats"][i])
                md = np.linalg.norm(qm-snm[c], axis=1); ms = 1.0/(1.0+md)
                scores.append(float(np.sort(vs+mw*ms)[::-1][:k].mean()))
            sa = np.array(scores); ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))]); margins.append(ss[0]-ss[1])
        preds = np.array(preds); margins = np.array(margins)
        for c in cids:
            cm = (preds==c)&(margins>0.025); ci = np.where(cm)[0]
            if len(ci)==0: continue
            proto_c = sb_orig[c].mean(0)
            dists = np.array([np.linalg.norm(bc_v["feats"][idx]-proto_c) for idx in ci])
            div_scores = margins[ci]*(1.0+0.3*dists/(dists.mean()+1e-8))
            ti = ci[np.argsort(div_scores)[::-1][:5]]
            sb[c] = np.concatenate([sb_orig[c], bc_v["feats"][ti]*0.5])
            sp[c] = np.concatenate([{c:ph_t["feats"][si[c]] for c in cids}[c], ph_v["feats"][ti]*0.5])
            sd[c] = np.concatenate([{c:dn_t["feats"][si[c]] for c in cids}[c], dn_v["feats"][ti]*0.5])
            sm[c] = np.concatenate([{c:bc_t["morphs"][si[c]] for c in cids}[c], bc_v["morphs"][ti]])

    sm2 = np.concatenate([sm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0)+1e-8
    snm = {c: (sm[c]-gm2)/gs2 for c in cids}
    final_gt, final_pred = [], []
    for i in range(len(bc_v["labels"])):
        qm = (bc_v["morphs"][i]-gm2)/gs2
        scores = {}
        for c in cids:
            vs = bw*(sb[c]@bc_v["feats"][i])+pw*(sp[c]@ph_v["feats"][i])+dw*(sd[c]@dn_v["feats"][i])
            md = np.linalg.norm(qm-snm[c], axis=1); ms = 1.0/(1.0+md)
            scores[c] = float(np.sort(vs+mw*ms)[::-1][:k].mean())
        final_gt.append(int(bc_v["labels"][i]))
        final_pred.append(max(scores, key=scores.get))

    from biomedclip_zeroshot_cell_classify import InstanceInfo
    def load_yolo(lp):
        anns = []
        if not lp.exists(): return anns
        for line in open(lp):
            p = line.strip().split()
            if len(p)<7: continue
            c = int(p[0])
            if c in CLASSES: anns.append({"class_id":c, "points":[float(x) for x in p[1:]]})
        return anns

    imgs = sorted((D2/"images"/"val").glob("*.png"))
    all_crops = []
    cell_idx = 0
    for ip in imgs:
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]
        anns = load_yolo(D2/"labels_polygon"/"val"/(ip.stem+".txt"))
        for ann in anns:
            if cell_idx >= len(final_gt): break
            pts = ann["points"]
            xs = [pts[i]*w for i in range(0,len(pts),2)]
            ys = [pts[i]*h for i in range(1,len(pts),2)]
            x1, y1 = max(0, int(min(xs))-10), max(0, int(min(ys))-10)
            x2, y2 = min(w, int(max(xs))+10), min(h, int(max(ys))+10)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                cell_idx += 1
                continue
            crop = cv2.resize(crop, (80, 80))
            gt_c = final_gt[cell_idx]
            pred_c = final_pred[cell_idx]
            all_crops.append({"crop": crop, "gt": gt_c, "pred": pred_c, "correct": gt_c==pred_c})
            cell_idx += 1

    per_class_correct = {c: [] for c in cids}
    per_class_wrong = {c: [] for c in cids}
    for item in all_crops:
        if item["correct"]:
            per_class_correct[item["gt"]].append(item)
        else:
            per_class_wrong[item["gt"]].append(item)

    fig, axes = plt.subplots(4, 12, figsize=(24, 9))
    for row_i, c in enumerate(cids):
        correct = per_class_correct[c][:8]
        wrong = per_class_wrong[c][:4]
        items = correct + wrong
        for col_i in range(12):
            ax = axes[row_i, col_i]
            if col_i < len(items):
                item = items[col_i]
                ax.imshow(item["crop"])
                if item["correct"]:
                    ax.set_title(f"{CLASSES[item['gt']]}", fontsize=8, color='green', fontweight='bold')
                    for spine in ax.spines.values(): spine.set_color('green'); spine.set_linewidth(2)
                else:
                    ax.set_title(f"GT:{CLASSES[item['gt']]} P:{CLASSES[item['pred']]}", fontsize=7, color='red', fontweight='bold')
                    for spine in ax.spines.values(): spine.set_color('red'); spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row_i, 0].set_ylabel(f"{CLASSES[c]}\n({len(per_class_correct[c])}ok/{len(per_class_wrong[c])}err)",
                                   fontsize=10, fontweight='bold', rotation=0, labelpad=60)

    plt.suptitle("10-Shot Classification: Green=Correct, Red=Misclassified (seed=42)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    path = OUT/"classification_real.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def viz_full_image_classification():
    """Show a full image with classification overlay."""
    bc_t = np.load(CACHE/"biomedclip_train.npz")
    bc_v = np.load(CACHE/"biomedclip_val.npz")
    ph_t = np.load(CACHE/"phikon_v2_train.npz")
    ph_v = np.load(CACHE/"phikon_v2_val.npz")
    dn_t = np.load(CACHE/"dinov2_s_train.npz")
    dn_v = np.load(CACHE/"dinov2_s_val.npz")

    random.seed(42)
    pc = defaultdict(list)
    for i, l in enumerate(bc_t["labels"]): pc[int(l)].append(i)
    si = {c: random.sample(pc[c], min(10, len(pc[c]))) for c in cids}

    bw, pw, dw, mw, k = 0.42, 0.18, 0.07, 0.33, 7
    sb = {c: bc_t["feats"][si[c]] for c in cids}
    sp = {c: ph_t["feats"][si[c]] for c in cids}
    sd = {c: dn_t["feats"][si[c]] for c in cids}
    sm = {c: bc_t["morphs"][si[c]] for c in cids}

    sm_all = np.concatenate([sm[c] for c in cids])
    gm, gs = sm_all.mean(0), sm_all.std(0)+1e-8
    snm = {c: (sm[c]-gm)/gs for c in cids}

    cell_idx = 0
    imgs = sorted((D2/"images"/"val").glob("*.png"))
    ip = imgs[5]  # pick one with diverse cells

    img = np.array(Image.open(ip).convert("RGB"))
    h, w = img.shape[:2]
    anns = load_gt_anns(D2/"labels_polygon"/"val"/(ip.stem+".txt"), h, w)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    gt_vis = img.copy()
    for ann in anns:
        c = COLORS[ann["cid"]]
        contours, _ = cv2.findContours(ann["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_vis, contours, -1, c, 3)
        cv2.putText(gt_vis, CLASSES[ann["cid"]], (ann["cx"]-12, ann["cy"]-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    # Count cells per image up to this image
    cell_start = 0
    for prev_ip in imgs:
        if prev_ip == ip: break
        prev_anns_count = len(load_gt_anns(D2/"labels_polygon"/"val"/(prev_ip.stem+".txt"), h, w))
        cell_start += prev_anns_count

    pred_vis = img.copy()
    for i, ann in enumerate(anns):
        vi = cell_start + i
        if vi >= len(bc_v["labels"]): break
        qm = (bc_v["morphs"][vi]-gm)/gs
        scores = {}
        for c in cids:
            vs = bw*(sb[c]@bc_v["feats"][vi])+pw*(sp[c]@ph_v["feats"][vi])+dw*(sd[c]@dn_v["feats"][vi])
            md = np.linalg.norm(qm-snm[c], axis=1); ms = 1.0/(1.0+md)
            scores[c] = float(np.sort(vs+mw*ms)[::-1][:k].mean())
        pred_c = max(scores, key=scores.get)
        c = COLORS[pred_c]
        correct = pred_c == ann["cid"]
        contours, _ = cv2.findContours(ann["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 3 if correct else 4
        cv2.drawContours(pred_vis, contours, -1, c if correct else (255,0,0), thickness)
        label = CLASSES[pred_c]
        if not correct: label = f"X-{label}"
        cv2.putText(pred_vis, label, (ann["cx"]-12, ann["cy"]-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, c if correct else (255,0,0), 2)

    crop_h, crop_w = h//2, w//2
    sy, sx = h//4, w//4
    ax1.imshow(gt_vis[sy:sy+crop_h, sx:sx+crop_w])
    ax1.set_title("Ground Truth Labels", fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax2.imshow(pred_vis[sy:sy+crop_h, sx:sx+crop_w])
    ax2.set_title("10-Shot ATD Predictions (Red X = Error)", fontsize=14, fontweight='bold')
    ax2.axis('off')

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=np.array(COLORS[c])/255, label=CLASSES[c]) for c in cids]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.suptitle(f"Full Image Classification: {ip.stem}", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT/"classification_full_image.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Segmentation visualization ===")
    viz_segmentation()
    print("\n=== Classification crops ===")
    viz_classification()
    print("\n=== Full image classification ===")
    viz_full_image_classification()
    print("\nDone!")


if __name__ == "__main__":
    main()
