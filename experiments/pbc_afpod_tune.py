#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.afpod_classify import afpod_classify, calc_metrics

CACHE_DIR = Path('/home/xut/csclip/experiments/feature_cache')
SEEDS = [42, 123, 456, 789, 2026]
N_SHOT = 10
CLASS_NAMES = {1: 'Eosinophil', 4: 'Lymphocyte', 5: 'Monocyte', 6: 'Neutrophil'}
CIDS = sorted(CLASS_NAMES.keys())


def load_cache(model: str):
    d = np.load(CACHE_DIR / f'pbc_balf4_{model}_all.npz')
    feats = np.ascontiguousarray(d['feats'].astype(np.float32))
    morphs = np.ascontiguousarray(d['morphs'].astype(np.float32))
    labels = np.ascontiguousarray(d['labels'].astype(np.int64))
    return feats, morphs, labels


def select_support_query(labels, seed):
    rng = np.random.RandomState(seed)
    support = {}
    query_idx = []
    for c in CIDS:
        idx = np.where(labels == c)[0].copy()
        rng.shuffle(idx)
        support[c] = idx[:N_SHOT]
        query_idx.extend(idx[N_SHOT:].tolist())
    return support, np.array(query_idx, dtype=np.int64)


def append_result(store, name, metrics):
    store[name]['acc'].append(metrics['acc'])
    store[name]['mf1'].append(metrics['mf1'])
    for c in CIDS:
        store[name]['pc'][c].append(metrics['pc'][c]['f1'])


def print_row(name, v):
    pc_str = ' '.join(f"{np.mean(v['pc'][c]):>7.4f}" for c in CIDS)
    print(f"{name:<42} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    bc, morph, labels = load_cache('biomedclip')
    ph, _, _ = load_cache('phikon_v2')
    dn, _, _ = load_cache('dinov2_s')

    mask = np.isin(labels, CIDS)
    bc = np.ascontiguousarray(bc[mask].astype(np.float32))
    ph = np.ascontiguousarray(ph[mask].astype(np.float32))
    dn = np.ascontiguousarray(dn[mask].astype(np.float32))
    morph = np.ascontiguousarray(morph[mask].astype(np.float32))
    labels = np.ascontiguousarray(labels[mask].astype(np.int64))

    print('=' * 140)
    print('PBC AFP-OD tuning scan | group=balf4')
    print('=' * 140, flush=True)

    weight_sets = [
        ('w_balforig', 0.42, 0.18, 0.07, 0.33),
        ('w_visplus', 0.40, 0.25, 0.10, 0.25),
        ('w_balanced', 0.35, 0.25, 0.10, 0.30),
        ('w_bcstrong', 0.45, 0.20, 0.10, 0.25),
    ]
    configs = []
    for alpha in [0.02, 0.05, 0.10, 0.15, 0.20]:
        for det_mode in ['feature_only', 'dualview_intersection', 'dualview_union']:
            for conf_thresh in [0.10, 0.15, 0.20, 0.25]:
                for wname, bw, pw, dw, mw in weight_sets:
                    configs.append({
                        'name': f'a{alpha:.2f}_{det_mode}_t{conf_thresh:.2f}_{wname}',
                        'alpha': alpha,
                        'det_mode': det_mode,
                        'conf_thresh': conf_thresh,
                        'bw': bw,
                        'pw': pw,
                        'dw': dw,
                        'mw': mw,
                    })

    results = defaultdict(lambda: {'acc': [], 'mf1': [], 'pc': defaultdict(list)})

    for seed in SEEDS:
        support_idx, query_idx = select_support_query(labels, seed)
        q_bc = np.ascontiguousarray(bc[query_idx].astype(np.float32))
        q_ph = np.ascontiguousarray(ph[query_idx].astype(np.float32))
        q_dn = np.ascontiguousarray(dn[query_idx].astype(np.float32))
        q_m = np.ascontiguousarray(morph[query_idx].astype(np.float32))
        q_lab = np.ascontiguousarray(labels[query_idx].astype(np.int64))
        s_bc = {c: np.ascontiguousarray(bc[support_idx[c]].astype(np.float32)) for c in CIDS}
        s_ph = {c: np.ascontiguousarray(ph[support_idx[c]].astype(np.float32)) for c in CIDS}
        s_dn = {c: np.ascontiguousarray(dn[support_idx[c]].astype(np.float32)) for c in CIDS}
        s_m = {c: np.ascontiguousarray(morph[support_idx[c]].astype(np.float32)) for c in CIDS}

        for i, cfg in enumerate(configs, start=1):
            scores, _ = afpod_classify(
                q_bc, q_ph, q_dn, q_m,
                s_bc, s_ph, s_dn, s_m, CIDS,
                bw=cfg['bw'], pw=cfg['pw'], dw=cfg['dw'], mw=cfg['mw'],
                alpha=cfg['alpha'], conf_thresh=cfg['conf_thresh'],
                method='lw', shrink=0.3, alpha_blend=0.0,
                detection_mode=cfg['det_mode'],
            )
            pred = [CIDS[int(np.argmax(scores[j]))] for j in range(len(q_lab))]
            append_result(results, cfg['name'], calc_metrics(q_lab.tolist(), pred, CIDS))
            if i == 1 or i % 20 == 0:
                cur = results[cfg['name']]
                print(f"[seed {seed}] {i}/{len(configs)} {cfg['name']}  mF1={np.mean(cur['mf1']):.4f}", flush=True)

    ranked = sorted(results.items(), key=lambda kv: (-np.mean(kv[1]['mf1']), -np.mean(kv[1]['acc'])))
    print('\n' + '=' * 140)
    print(f"{'Method':<42} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Lym':>7} {'Mon':>7} {'Neu':>7}  {'As':>5} {'Fs':>5}")
    print('-' * 140)
    for name, v in ranked[:20]:
        print_row(name, v)

    print('\nTop 5 summary:')
    for rank, (name, v) in enumerate(ranked[:5], start=1):
        print(f"  #{rank}: {name} | mF1={np.mean(v['mf1']):.4f} ± {np.std(v['mf1']):.4f} | Acc={np.mean(v['acc']):.4f}")


if __name__ == '__main__':
    main()
