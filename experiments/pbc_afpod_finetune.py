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
    pc_str = ' '.join(f"{np.mean(v['pc'][c]):>7.4f}" for c in CIDS)
    print(f"{name:<42} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    bc, morph, labels = load_cache('biomedclip')
    ph, _, _ = load_cache('phikon_v2')
    dn, _, _ = load_cache('dinov2_s')

    mask = np.isin(labels, CIDS)
    bc = np.ascontiguousarray(bc[mask])
    ph = np.ascontiguousarray(ph[mask])
    dn = np.ascontiguousarray(dn[mask])
    morph = np.ascontiguousarray(morph[mask])
    labels = np.ascontiguousarray(labels[mask])

    print('=' * 140)
    print('PBC AFP-OD fine-tune | group=balf4')
    print('=' * 140, flush=True)

    configs = [
        ('best_guess', 0.02, 'dualview_intersection', 0.10, 0.45, 0.20, 0.10, 0.25),
        ('best_guess_t015', 0.02, 'dualview_intersection', 0.15, 0.45, 0.20, 0.10, 0.25),
        ('best_guess_a005', 0.05, 'dualview_intersection', 0.10, 0.45, 0.20, 0.10, 0.25),
        ('best_guess_a005_t015', 0.05, 'dualview_intersection', 0.15, 0.45, 0.20, 0.10, 0.25),
        ('visplus_inter_t010_a002', 0.02, 'dualview_intersection', 0.10, 0.40, 0.25, 0.10, 0.25),
        ('visplus_inter_t015_a002', 0.02, 'dualview_intersection', 0.15, 0.40, 0.25, 0.10, 0.25),
        ('visplus_inter_t010_a005', 0.05, 'dualview_intersection', 0.10, 0.40, 0.25, 0.10, 0.25),
        ('balanced_inter_t010_a002', 0.02, 'dualview_intersection', 0.10, 0.35, 0.25, 0.10, 0.30),
        ('balanced_inter_t015_a002', 0.02, 'dualview_intersection', 0.15, 0.35, 0.25, 0.10, 0.30),
        ('feature_bcstrong_t010_a002', 0.02, 'feature_only', 0.10, 0.45, 0.20, 0.10, 0.25),
        ('feature_bcstrong_t015_a002', 0.02, 'feature_only', 0.15, 0.45, 0.20, 0.10, 0.25),
        ('feature_visplus_t010_a002', 0.02, 'feature_only', 0.10, 0.40, 0.25, 0.10, 0.25),
    ]

    results = defaultdict(lambda: {'acc': [], 'mf1': [], 'pc': defaultdict(list)})

    for seed in SEEDS:
        support_idx, query_idx = select_support_query(labels, seed)
        q_bc = np.ascontiguousarray(bc[query_idx])
        q_ph = np.ascontiguousarray(ph[query_idx])
        q_dn = np.ascontiguousarray(dn[query_idx])
        q_m = np.ascontiguousarray(morph[query_idx])
        q_lab = np.ascontiguousarray(labels[query_idx])
        s_bc = {c: np.ascontiguousarray(bc[support_idx[c]]) for c in CIDS}
        s_ph = {c: np.ascontiguousarray(ph[support_idx[c]]) for c in CIDS}
        s_dn = {c: np.ascontiguousarray(dn[support_idx[c]]) for c in CIDS}
        s_m = {c: np.ascontiguousarray(morph[support_idx[c]]) for c in CIDS}

        for i, (name, alpha, det_mode, conf_thresh, bw, pw, dw, mw) in enumerate(configs, start=1):
            scores, _ = afpod_classify(
                q_bc, q_ph, q_dn, q_m,
                s_bc, s_ph, s_dn, s_m, CIDS,
                bw=bw, pw=pw, dw=dw, mw=mw,
                alpha=alpha, conf_thresh=conf_thresh,
                method='lw', shrink=0.3, alpha_blend=0.0,
                detection_mode=det_mode,
            )
            pred = [CIDS[int(np.argmax(scores[j]))] for j in range(len(q_lab))]
            append_result(results, name, calc_metrics(q_lab.tolist(), pred, CIDS))
            print(f"[seed {seed}] {i}/{len(configs)} {name} mF1={np.mean(results[name]['mf1']):.4f}", flush=True)

    ranked = sorted(results.items(), key=lambda kv: (-np.mean(kv[1]['mf1']), -np.mean(kv[1]['acc'])))
    print('\n' + '=' * 140)
    print(f"{'Method':<42} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Lym':>7} {'Mon':>7} {'Neu':>7}  {'As':>5} {'Fs':>5}")
    print('-' * 140)
    for name, v in ranked:
        print_row(name, v)

    print('\nTop 5 summary:')
    for rank, (name, v) in enumerate(ranked[:5], start=1):
        print(f"  #{rank}: {name} | mF1={np.mean(v['mf1']):.4f} ± {np.std(v['mf1']):.4f} | Acc={np.mean(v['acc']):.4f}")


if __name__ == '__main__':
    main()
