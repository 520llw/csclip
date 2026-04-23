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
GROUP_CLASS_NAMES = {
    'balf4': {
        1: 'Eosinophil',
        6: 'Neutrophil',
        4: 'Lymphocyte',
        5: 'Monocyte',
    },
    'all': {
        0: 'Basophil',
        1: 'Eosinophil',
        2: 'Erythroblast',
        3: 'IG',
        4: 'Lymphocyte',
        5: 'Monocyte',
        6: 'Neutrophil',
        7: 'Platelet',
    },
}


def load_cache(group: str, model: str):
    d = np.load(CACHE_DIR / f'pbc_{group}_{model}_all.npz')
    return d['feats'], d['morphs'], d['labels']


def select_support_query(labels, seed, cids, n_shot=N_SHOT):
    rng = np.random.RandomState(seed)
    support = {}
    query_idx = []
    for c in cids:
        idx = np.where(labels == c)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        take = min(n_shot, len(idx))
        support[c] = idx[:take]
        query_idx.extend(idx[take:].tolist())
    return support, np.array(query_idx, dtype=np.int64)


def score_only_backbones(q_bc, q_ph, q_dn, s_bc, s_ph, s_dn, cids, bw, pw, dw, k=7):
    scores = np.zeros((len(q_bc), len(cids)), dtype=np.float32)
    for i in range(len(q_bc)):
        for ki, c in enumerate(cids):
            ncls = len(s_bc[c])
            if ncls == 0:
                scores[i, ki] = -np.inf
                continue
            vs = bw * (s_bc[c] @ q_bc[i]) + pw * (s_ph[c] @ q_ph[i]) + dw * (s_dn[c] @ q_dn[i])
            kk = min(k, ncls)
            scores[i, ki] = float(np.sort(vs)[::-1][:kk].mean())
    return scores


def append_result(store, name, metrics, cids):
    store[name]['acc'].append(metrics['acc'])
    store[name]['mf1'].append(metrics['mf1'])
    for c in cids:
        store[name]['pc'][c].append(metrics['pc'][c]['f1'])


def print_row(name, v, cids):
    pc_str = ' '.join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
    print(f"{name:<30} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc_str}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else 'balf4'
    if group not in GROUP_CLASS_NAMES:
        raise ValueError(group)

    bc, morph, labels = load_cache(group, 'biomedclip')
    ph, _, _ = load_cache(group, 'phikon_v2')
    dn, _, _ = load_cache(group, 'dinov2_s')

    cids = sorted(GROUP_CLASS_NAMES[group].keys())
    labels = labels.astype(np.int64)
    mask = np.isin(labels, cids)
    bc, ph, dn, morph, labels = bc[mask], ph[mask], dn[mask], morph[mask], labels[mask]

    print('=' * 120)
    print(f'PBC AFP-OD Ablation | group={group} | n={len(labels)} | classes={cids}')
    print('=' * 120, flush=True)

    results = defaultdict(lambda: {'acc': [], 'mf1': [], 'pc': defaultdict(list)})

    for seed in SEEDS:
        support_idx, query_idx = select_support_query(labels, seed, cids, N_SHOT)
        q_bc, q_ph, q_dn, q_m, q_lab = bc[query_idx], ph[query_idx], dn[query_idx], morph[query_idx], labels[query_idx]
        s_bc = {c: bc[support_idx[c]] for c in cids}
        s_ph = {c: ph[support_idx[c]] for c in cids}
        s_dn = {c: dn[support_idx[c]] for c in cids}
        s_m = {c: morph[support_idx[c]] for c in cids}

        scores = mb_knn_classify(q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, cids)
        pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'MB_kNN', calc_metrics(q_lab.tolist(), pred, cids), cids)

        scores = afpod_classify(q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, cids,
                                alpha=0.10, conf_thresh=0.15, method='lw', shrink=0.3,
                                alpha_blend=0.0, detection_mode='dualview_union')[0]
        pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'AFPOD_full', calc_metrics(q_lab.tolist(), pred, cids), cids)

        scores = afpod_classify(q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, cids,
                                alpha=0.10, conf_thresh=0.15, method='lw', shrink=0.3,
                                alpha_blend=0.0, detection_mode='feature_only')[0]
        pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'AFPOD_no_dualview', calc_metrics(q_lab.tolist(), pred, cids), cids)

        scores = afpod_classify(q_bc, q_ph, q_dn, q_m, s_bc, s_ph, s_dn, s_m, cids,
                                alpha=0.0, conf_thresh=0.15, method='lw', shrink=0.3,
                                alpha_blend=0.0, detection_mode='dualview_union')[0]
        pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'AFPOD_no_disentangle', calc_metrics(q_lab.tolist(), pred, cids), cids)

        scores = score_only_backbones(q_bc, q_ph, q_dn, s_bc, s_ph, s_dn, cids, bw=0.42, pw=0.18, dw=0.07)
        pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'NoMorph_kNN', calc_metrics(q_lab.tolist(), pred, cids), cids)

        scores = score_only_backbones(q_bc, q_ph, q_dn, s_bc, s_ph, s_dn, cids, bw=1.0, pw=0.0, dw=0.0)
        pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'BiomedCLIP_only', calc_metrics(q_lab.tolist(), pred, cids), cids)

        scores = score_only_backbones(q_bc, q_ph, q_dn, s_bc, s_ph, s_dn, cids, bw=0.42, pw=0.18, dw=0.0)
        pred = [cids[int(np.argmax(scores[i]))] for i in range(len(q_lab))]
        append_result(results, 'BC+PH_only', calc_metrics(q_lab.tolist(), pred, cids), cids)

        print(f'[seed {seed}] MB={np.mean(results["MB_kNN"]["mf1"]):.4f}  AFPOD_full={np.mean(results["AFPOD_full"]["mf1"]):.4f}', flush=True)

    print('\n' + '=' * 120)
    print(f"{'Method':<30} {'Acc':>7} {'mF1':>7} " + ' '.join(f'{GROUP_CLASS_NAMES[group][c][:3]:>7}' for c in cids) + '  As Fs')
    print('-' * 120)
    order = ['BiomedCLIP_only', 'BC+PH_only', 'NoMorph_kNN', 'MB_kNN', 'AFPOD_no_disentangle', 'AFPOD_no_dualview', 'AFPOD_full']
    for name in order:
        print_row(name, results[name], cids)

    best = 'AFPOD_full'
    base = 'MB_kNN'
    print('\nImprovements vs MB_kNN:')
    for name in order:
        dm = np.mean(results[name]['mf1']) - np.mean(results[base]['mf1'])
        print(f'  {name:<24} ΔmF1={dm:+.4f}')


if __name__ == '__main__':
    main()
