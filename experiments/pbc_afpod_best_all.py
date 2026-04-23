#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.afpod_classify import afpod_classify, mb_knn_classify, calc_metrics

CACHE_DIR = Path('/home/xut/csclip/experiments/feature_cache')
SEEDS = [42, 123, 456, 789, 2026]
N_SHOT = 10
CLASS_NAMES = {
    0: 'Basophil',
    1: 'Eosinophil',
    2: 'Erythroblast',
    3: 'IG',
    4: 'Lymphocyte',
    5: 'Monocyte',
    6: 'Neutrophil',
    7: 'Platelet',
}
CIDS = sorted(CLASS_NAMES.keys())


def load_cache(model: str):
    d = np.load(CACHE_DIR / f'pbc_all_{model}_all.npz')
    return (
        np.ascontiguousarray(d['feats'].astype(np.float32)),
        np.ascontiguousarray(d['morphs'].astype(np.float32)),
        np.ascontiguousarray(d['labels'].astype(np.int64)),
    )


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
    pc_str = ' '.join(f"{np.mean(v['pc'][c]):>6.4f}" for c in CIDS)
    print(f"{name:<22} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    bc, morph, labels = load_cache('biomedclip')
    ph, _, _ = load_cache('phikon_v2')
    dn, _, _ = load_cache('dinov2_s')

    print('=' * 180)
    print('PBC AFP-OD best-config evaluation | group=all-8')
    print('=' * 180, flush=True)

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

        bl_scores = mb_knn_classify(q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, CIDS,
                                    bw=0.40, pw=0.25, dw=0.10, mw=0.25)
        pred = [CIDS[int(np.argmax(bl_scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'MB_kNN_tuned', calc_metrics(q_lab.tolist(), pred, CIDS))

        scores, _ = afpod_classify(
            q_bc, q_ph, q_dn, q_m,
            s_bc, s_ph, s_dn, s_m, CIDS,
            bw=0.40, pw=0.25, dw=0.10, mw=0.25,
            alpha=0.05, conf_thresh=0.10,
            method='lw', shrink=0.3, alpha_blend=0.0,
            detection_mode='dualview_intersection',
        )
        pred = [CIDS[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'AFPOD_best', calc_metrics(q_lab.tolist(), pred, CIDS))

        print(f"[seed {seed}] MB={np.mean(results['MB_kNN_tuned']['mf1']):.4f} AFPOD_best={np.mean(results['AFPOD_best']['mf1']):.4f}", flush=True)

    print('\n' + '=' * 180)
    print(f"{'Method':<22} {'Acc':>7} {'mF1':>7} " + ' '.join(f'{CLASS_NAMES[c][:3]:>6}' for c in CIDS) + '  As Fs')
    print('-' * 180)
    print_row('MB_kNN_tuned', results['MB_kNN_tuned'])
    print_row('AFPOD_best', results['AFPOD_best'])
    dm = np.mean(results['AFPOD_best']['mf1']) - np.mean(results['MB_kNN_tuned']['mf1'])
    print(f"\nImprovement: ΔmF1={dm:+.4f}")


if __name__ == '__main__':
    main()
