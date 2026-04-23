"""
PBC Dataset Few-shot Classification Evaluation
===============================================
验证 BALF-Analyzer 分类模块（S3–S5）在外周血涂片数据集的泛化性。

数据集：PBC_dataset_normal_DIB（Nature Scientific Data, Acevedo 2020）
图像：360×363 RGB，单细胞居中，背景含 RBC

两组实验：
  Group A (4-class BALF-mapping):
      eosinophil→Eos, neutrophil→Neu, lymphocyte→Lym, monocyte→Mac
  Group B (8-class full PBC):
      basophil, eosinophil, erythroblast, ig, lymphocyte,
      monocyte, neutrophil, platelet

评估协议：10-shot × 5 seeds，nested 留出评估，报告 mF1 ± std

掩码策略：Cellpose 分割取距图像中心最近的实例；若无检出则用全图 bbox

运行：
    /data/software/mamba/envs/cel/bin/python experiments/pbc_fewshot_eval.py
"""

import sys, os, random, time, warnings
import numpy as np
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/xut/csclip')
sys.path.insert(0, '/home/xut/csclip/sam3')

# ── 配置 ──────────────────────────────────────────────────────────────────────
PBC_ROOT = Path('/home/xut/csclip/cell_datasets/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB')
CACHE_DIR = Path('/tmp/pbc_feat_cache')
CACHE_DIR.mkdir(exist_ok=True)

N_SHOT = 10
SEEDS  = [42, 123, 456, 789, 1024]

# Group A: 4-class BALF-mapping
GROUP_A = {
    'eosinophil': 'Eosinophil',
    'neutrophil':  'Neutrophil',
    'lymphocyte':  'Lymphocyte',
    'monocyte':    'Macrophage',
}

# Group B: 8-class full PBC
GROUP_B = {
    'basophil':    'Basophil',
    'eosinophil':  'Eosinophil',
    'erythroblast':'Erythroblast',
    'ig':          'IG',
    'lymphocyte':  'Lymphocyte',
    'monocyte':    'Monocyte',
    'neutrophil':  'Neutrophil',
    'platelet':    'Platelet',
}

# ── 全局模型（懒加载）─────────────────────────────────────────────────────────
_models = {}


def get_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_models():
    if _models:
        return _models
    device = get_device()
    print(f"[Device: {device}]")

    # Cellpose
    print("[Loading Cellpose cpsam ...]", flush=True)
    from cellpose import models as cp_models
    import torch
    _models['cp'] = cp_models.CellposeModel(
        gpu=torch.cuda.is_available(), pretrained_model='cpsam')

    # BiomedCLIP
    print("[Loading BiomedCLIP ...]", flush=True)
    import open_clip
    bc_model, _, bc_prep = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    bc_model = bc_model.to(device).eval()
    _models['bc'] = bc_model
    _models['bc_prep'] = bc_prep

    # Phikon-v2
    print("[Loading Phikon-v2 ...]", flush=True)
    from transformers import AutoImageProcessor, AutoModel
    ph_proc = AutoImageProcessor.from_pretrained(
        '/home/xut/csclip/model_weights/phikon_v2')
    ph_model = AutoModel.from_pretrained(
        '/home/xut/csclip/model_weights/phikon_v2').to(device).eval()
    _models['ph'] = ph_model
    _models['ph_proc'] = ph_proc

    # DINOv2-Small
    print("[Loading DINOv2-S ...]", flush=True)
    import torch
    dn_model = torch.hub.load(
        '/home/xut/csclip/model_weights', 'dinov2_vits14',
        source='local', pretrained=False)
    ckpt = torch.load('/home/xut/csclip/model_weights/dinov2_vits14_pretrain.pth',
                      map_location='cpu')
    dn_model.load_state_dict(ckpt)
    dn_model = dn_model.to(device).eval()
    _models['dn'] = dn_model

    _models['device'] = device
    print("[All models loaded]\n", flush=True)
    return _models


# ── 分割：取距图像中心最近的实例 ─────────────────────────────────────────────
def segment_center_cell(img_rgb, cp_model):
    """对单细胞居中图像，返回距中心最近的 Cellpose 实例掩码及其 bbox。"""
    h, w = img_rgb.shape[:2]
    cy, cx = h / 2, w / 2

    masks, _, _ = cp_model.eval(img_rgb, diameter=None,
                                 flow_threshold=0.4, cellprob_threshold=-2.0,
                                 channels=[0, 0])
    inst_ids = np.unique(masks)
    inst_ids = inst_ids[inst_ids > 0]

    if len(inst_ids) == 0:
        # 无检出：退回全图 bbox
        return np.ones((h, w), dtype=bool), (0, 0, h, w)

    best_id, best_dist = None, float('inf')
    for iid in inst_ids:
        m = masks == iid
        ys, xs = np.where(m)
        dist = ((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = iid

    mask = masks == best_id
    ys, xs = np.where(mask)
    bbox = (ys.min(), xs.min(), ys.max(), xs.max())
    return mask, bbox


# ── 双尺度裁剪 ────────────────────────────────────────────────────────────────
def dual_scale_crop(img_rgb, bbox, cell_margin=0.10, ctx_margin=0.50):
    h, w = img_rgb.shape[:2]
    y0, x0, y1, x1 = bbox
    bh, bw = y1 - y0, x1 - x0

    def clamp_crop(margin):
        dy, dx = int(bh * margin), int(bw * margin)
        r0 = max(0, y0 - dy); r1 = min(h, y1 + dy)
        c0 = max(0, x0 - dx); c1 = min(w, x1 + dx)
        crop = img_rgb[r0:r1, c0:c1]
        return Image.fromarray(crop).resize((224, 224), Image.BILINEAR)

    return clamp_crop(cell_margin), clamp_crop(ctx_margin)


# ── 特征提取 ──────────────────────────────────────────────────────────────────
@np.vectorize
def _relu(x): return max(0.0, x)


def extract_features(img_rgb, mask, bbox, models):
    import torch
    device = models['device']

    cell_crop, ctx_crop = dual_scale_crop(img_rgb, bbox)

    def encode_bc(pil_img):
        t = models['bc_prep'](pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = models['bc'].encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze().cpu().numpy()

    def encode_ph(pil_img):
        inp = models['ph_proc'](images=pil_img, return_tensors='pt')
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = models['ph'](**inp)
        f = out.last_hidden_state[:, 0]
        f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze().cpu().numpy()

    def encode_dn(pil_img):
        import torchvision.transforms as T
        tf = T.Compose([T.ToTensor(),
                        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        t = tf(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = models['dn'](t)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze().cpu().numpy()

    f_bc_cell = encode_bc(cell_crop)
    f_bc_ctx  = encode_bc(ctx_crop)
    f_bc = 0.9 * f_bc_cell + 0.1 * f_bc_ctx
    f_bc = f_bc / (np.linalg.norm(f_bc) + 1e-8)

    f_ph_cell = encode_ph(cell_crop)
    f_ph_ctx  = encode_ph(ctx_crop)
    f_ph = 0.9 * f_ph_cell + 0.1 * f_ph_ctx
    f_ph = f_ph / (np.linalg.norm(f_ph) + 1e-8)

    f_dn_cell = encode_dn(cell_crop)
    f_dn_ctx  = encode_dn(ctx_crop)
    f_dn = 0.9 * f_dn_cell + 0.1 * f_dn_ctx
    f_dn = f_dn / (np.linalg.norm(f_dn) + 1e-8)

    morph = compute_morphology(img_rgb, mask)
    return f_bc, f_ph, f_dn, morph


def compute_morphology(img_rgb, mask):
    """简化版形态学特征（12维：几何+颜色）"""
    from skimage import measure
    import cv2

    area = mask.sum()
    if area < 10:
        return np.zeros(12, dtype=np.float32)

    # 几何
    props = measure.regionprops(mask.astype(np.uint8))[0]
    log_area   = np.log1p(area)
    circularity= 4 * np.pi * area / (props.perimeter ** 2 + 1e-8)
    eccentricity = props.eccentricity
    extent = props.extent

    # 颜色（在掩码内）
    roi = img_rgb[mask]
    r_mean, g_mean, b_mean = roi[:, 0].mean()/255, roi[:, 1].mean()/255, roi[:, 2].mean()/255
    r_std,  g_std,  b_std  = roi[:, 0].std()/255,  roi[:, 1].std()/255,  roi[:, 2].std()/255
    red_gt_green = r_mean / (g_mean + 1e-8)
    red_gt_blue  = r_mean / (b_mean + 1e-8)
    # HSV 均值
    hsv = cv2.cvtColor((img_rgb * mask[:,:,None]).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h_mean = hsv[:,:,0][mask].mean() / 179.0
    s_mean = hsv[:,:,1][mask].mean() / 255.0

    feats = np.array([
        log_area, circularity, eccentricity, extent,
        r_mean, g_mean, b_mean, r_std, g_std, b_std,
        red_gt_green, red_gt_blue
    ], dtype=np.float32)
    return feats


# ── AFP-OD 分类 ───────────────────────────────────────────────────────────────
def mbknn_score(query_feat, support_feats, k=7):
    """MB-kNN 评分（余弦相似度）"""
    sims = np.dot(support_feats, query_feat)
    topk = np.sort(sims)[-k:]
    return topk.mean()


def morph_score(query_m, support_ms):
    """形态学 kNN 评分（归一化欧氏距离倒数）"""
    dists = np.linalg.norm(support_ms - query_m[None], axis=1)
    return 1.0 / (1.0 + dists.mean())


def afpod_classify(query, supports, class_labels, k=7,
                   w_bc=0.42, w_ph=0.18, w_dn=0.07, w_m=0.33,
                   tau=0.15, alpha=0.10):
    """AFP-OD 分类（简化版：双视图混淆检测 + LW Fisher 解耦 + MB-kNN）"""
    from sklearn.covariance import LedoitWolf

    q_bc, q_ph, q_dn, q_m = query
    classes = list(supports.keys())
    n_cls = len(classes)

    # support 特征收集
    sup_bc = {c: np.stack([s[0] for s in supports[c]]) for c in classes}
    sup_ph = {c: np.stack([s[1] for s in supports[c]]) for c in classes}
    sup_dn = {c: np.stack([s[2] for s in supports[c]]) for c in classes}
    sup_m  = {c: np.stack([s[3] for s in supports[c]]) for c in classes}

    # z-score 归一化形态学（基于 support 统计）
    all_m = np.concatenate(list(sup_m.values()), axis=0)
    m_mean, m_std = all_m.mean(0), all_m.std(0) + 1e-8
    sup_mz = {c: (sup_m[c] - m_mean) / m_std for c in classes}
    q_mz = (q_m - m_mean) / m_std

    # ── 双视图混淆对检测（LOO-kNN）──
    def loo_confusion(feats_dict, metric='cosine'):
        R = np.zeros((n_cls, n_cls))
        for i, ci in enumerate(classes):
            fi = feats_dict[ci]
            for j, cj in enumerate(classes):
                if i == j: continue
                fj = feats_dict[cj]
                confused = 0
                for idx in range(len(fi)):
                    loo = np.delete(fi, idx, axis=0)
                    cands = np.concatenate([loo, fj], axis=0)
                    if metric == 'cosine':
                        sims = cands @ fi[idx]
                    else:
                        sims = -np.linalg.norm(cands - fi[idx][None], axis=1)
                    best_pool = np.argmax(sims)
                    if best_pool >= len(loo):
                        confused += 1
                R[i, j] = confused / len(fi)
        return R

    R_feat  = loo_confusion(sup_bc, 'cosine')
    R_morph = loo_confusion(sup_mz, 'euclidean')

    confused_pairs = set()
    for i in range(n_cls):
        for j in range(i+1, n_cls):
            sym = R_feat[i,j] + R_feat[j,i] + R_morph[i,j] + R_morph[j,i]
            if sym / 2 >= tau:
                confused_pairs.add((i, j))

    # ── LW Fisher 解耦 ──
    def lw_fisher_dir(feat_i, feat_j):
        try:
            cov_i = LedoitWolf().fit(feat_i).covariance_
            cov_j = LedoitWolf().fit(feat_j).covariance_
        except Exception:
            d = feat_i.shape[1]
            cov_i = np.eye(d) * feat_i.var()
            cov_j = np.eye(d) * feat_j.var()
        W = cov_i + cov_j + 1e-4 * np.eye(cov_i.shape[0])
        diff = feat_i.mean(0) - feat_j.mean(0)
        w = np.linalg.solve(W, diff)
        norm = np.linalg.norm(w) + 1e-8
        return w / norm

    # 克隆 support 特征用于扰动
    sup_bc_p = {c: sup_bc[c].copy() for c in classes}
    sup_ph_p = {c: sup_ph[c].copy() for c in classes}
    sup_dn_p = {c: sup_dn[c].copy() for c in classes}

    for (i, j) in confused_pairs:
        ci, cj = classes[i], classes[j]
        for sup_dict in [sup_bc_p, sup_ph_p, sup_dn_p]:
            w = lw_fisher_dir(sup_dict[ci], sup_dict[cj])
            sup_dict[ci] = sup_dict[ci] + alpha * w
            sup_dict[cj] = sup_dict[cj] - alpha * w

    # 归一化
    for c in classes:
        for sup_dict in [sup_bc_p, sup_ph_p, sup_dn_p]:
            norms = np.linalg.norm(sup_dict[c], axis=1, keepdims=True) + 1e-8
            sup_dict[c] = sup_dict[c] / norms

    # ── MB-kNN 打分 ──
    scores = {}
    for c in classes:
        s_bc = mbknn_score(q_bc, sup_bc_p[c], k)
        s_ph = mbknn_score(q_ph, sup_ph_p[c], k)
        s_dn = mbknn_score(q_dn, sup_dn_p[c], k)
        s_m  = morph_score(q_mz, sup_mz[c])
        scores[c] = w_bc * s_bc + w_ph * s_ph + w_dn * s_dn + w_m * s_m

    pred = max(scores, key=scores.get)
    return pred


# ── 特征缓存 ──────────────────────────────────────────────────────────────────
def get_cache_path(cls_name, img_name):
    return CACHE_DIR / f"{cls_name}_{Path(img_name).stem}.npz"


def load_or_extract(img_path, cls_name, models):
    cache = get_cache_path(cls_name, img_path.name)
    if cache.exists():
        d = np.load(cache)
        return d['bc'], d['ph'], d['dn'], d['m']

    img_rgb = np.array(Image.open(img_path).convert('RGB'))
    mask, bbox = segment_center_cell(img_rgb, models['cp'])
    f_bc, f_ph, f_dn, morph = extract_features(img_rgb, mask, bbox, models)
    np.savez_compressed(cache, bc=f_bc, ph=f_ph, dn=f_dn, m=morph)
    return f_bc, f_ph, f_dn, morph


# ── 评估函数 ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, classes):
    from collections import Counter
    results = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        results[c] = {'P': prec, 'R': rec, 'F1': f1}
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    mf1 = np.mean([v['F1'] for v in results.values()])
    return acc, mf1, results


def run_eval(group_dict, models, tag=''):
    """对给定类别组运行 5-seed 评估"""
    classes = list(group_dict.keys())
    n_cls = len(classes)
    print(f"\n{'='*70}")
    print(f"  {tag}  ({n_cls} classes: {classes})")
    print(f"{'='*70}")

    # 预加载所有图片路径
    all_paths = {}
    for c in classes:
        cls_dir = PBC_ROOT / c
        paths = sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.png'))
        all_paths[c] = paths
        print(f"  {c}: {len(paths)} images")

    # 预提取特征（带缓存）
    print(f"\n[Extracting features (cached) ...]", flush=True)
    all_feats = {}
    for c in classes:
        feats = []
        for i, p in enumerate(all_paths[c]):
            if i % 100 == 0:
                print(f"  {c}: {i}/{len(all_paths[c])}", flush=True)
            feats.append(load_or_extract(p, c, models))
        all_feats[c] = feats
        print(f"  {c}: done ({len(feats)} features)")

    seed_mf1, seed_acc = [], []
    seed_per_class = {c: [] for c in classes}

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed)
        supports = {}
        queries_feat, queries_label = [], []

        for c in classes:
            idxs = list(range(len(all_feats[c])))
            random.shuffle(idxs)
            sup_idxs = idxs[:N_SHOT]
            qry_idxs = idxs[N_SHOT:]

            supports[c] = [all_feats[c][i] for i in sup_idxs]
            for i in qry_idxs:
                queries_feat.append(all_feats[c][i])
                queries_label.append(c)

        # 推理
        y_true, y_pred = [], []
        t0 = time.time()
        for feat, label in zip(queries_feat, queries_label):
            pred = afpod_classify(feat, supports, classes)
            y_true.append(label)
            y_pred.append(pred)

        acc, mf1, per_cls = compute_metrics(y_true, y_pred, classes)
        seed_mf1.append(mf1)
        seed_acc.append(acc)
        for c in classes:
            seed_per_class[c].append(per_cls[c]['F1'])

        print(f"  seed={seed}: Acc={acc:.4f}  mF1={mf1:.4f}  "
              f"[{time.time()-t0:.1f}s]", flush=True)
        for c in classes:
            print(f"    {c:15s}: F1={per_cls[c]['F1']:.4f}  "
                  f"P={per_cls[c]['P']:.4f}  R={per_cls[c]['R']:.4f}")

    # 汇总
    print(f"\n{'─'*70}")
    print(f"  SUMMARY [{tag}]  10-shot × {len(SEEDS)} seeds")
    print(f"  Acc  = {np.mean(seed_acc):.4f} ± {np.std(seed_acc):.4f}")
    print(f"  mF1  = {np.mean(seed_mf1):.4f} ± {np.std(seed_mf1):.4f}")
    print(f"  Per-class F1:")
    for c in classes:
        vals = seed_per_class[c]
        print(f"    {c:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'─'*70}\n")

    return {
        'tag': tag,
        'acc_mean': np.mean(seed_acc), 'acc_std': np.std(seed_acc),
        'mf1_mean': np.mean(seed_mf1), 'mf1_std': np.std(seed_mf1),
        'per_class': {c: {'mean': np.mean(seed_per_class[c]),
                          'std':  np.std(seed_per_class[c])} for c in classes},
    }


# ── 主入口 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("PBC Few-shot Evaluation")
    print(f"  N_shot={N_SHOT}, Seeds={SEEDS}")
    print(f"  Cache: {CACHE_DIR}\n")

    models = load_models()

    # Group A: 4-class BALF-mapping
    res_a = run_eval(GROUP_A, models, tag='Group A: 4-class BALF-mapping')

    # Group B: 8-class full PBC
    res_b = run_eval(GROUP_B, models, tag='Group B: 8-class full PBC')

    # 最终汇总对比
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Experiment':<30} {'Acc':>8} {'mF1':>8}")
    print("-"*70)
    for res in [res_a, res_b]:
        print(f"{res['tag']:<30} "
              f"{res['acc_mean']:.4f}±{res['acc_std']:.4f}  "
              f"{res['mf1_mean']:.4f}±{res['mf1_std']:.4f}")
    print("="*70)
