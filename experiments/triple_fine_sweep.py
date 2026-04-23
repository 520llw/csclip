#!/usr/bin/env python3
"""Fine sweep around the new best: bpd_38_20_07_35."""
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

CACHE_DIR = Path("/home/xut/csclip/experiments/feature_cache")
CLASS_NAMES = {3: "Eosinophil", 4: "Neutrophil", 5: "Lymphocyte", 6: "Macrophage"}
N_SHOT = 10
SEEDS = [42, 123, 456, 789, 2026]


def load_cache(m, s):
    d = np.load(CACHE_DIR / f"{m}_{s}.npz")
    return d["feats"], d["morphs"], d["labels"]


def metrics(gt, pred, cids):
    total = len(gt); correct = sum(int(g==p) for g,p in zip(gt,pred))
    pc, f1s = {}, []
    for c in cids:
        tp = sum(1 for g,p in zip(gt,pred) if g==c and p==c)
        pp = sum(1 for p in pred if p==c)
        gp = sum(1 for g in gt if g==c)
        pr = tp/pp if pp else 0; rc = tp/gp if gp else 0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
        pc[c] = {"p": pr, "r": rc, "f1": f1}; f1s.append(f1)
    return {"acc": correct/total, "mf1": np.mean(f1s), "pc": pc}


def select_support(labels, seed, cids):
    random.seed(seed)
    pc = defaultdict(list)
    for i, l in enumerate(labels): pc[int(l)].append(i)
    return {c: random.sample(pc[c], min(N_SHOT, len(pc[c]))) for c in cids}


def run_pipeline(q_bc, q_ph, q_dn, q_morph, q_labels,
                  s_bc0, s_ph0, s_dn0, s_morph0,
                  cids, fw, bw, pw, dw, mw, k, ni, tk, cf, ct, cmw):
    sm = np.concatenate([s_morph0[c] for c in cids])
    gm, gs = sm.mean(0), sm.std(0)+1e-8
    sb, sp, sd, smm = ({c: v[c].copy() for c in cids} for v in [s_bc0, s_ph0, s_dn0, s_morph0])
    for _ in range(ni):
        snm = {c: (smm[c]-gm)/gs for c in cids}
        preds, margins = [], []
        for i in range(len(q_labels)):
            qm = (q_morph[i]-gm)/gs
            scores = []
            for c in cids:
                vs = bw*(sb[c]@q_bc[i]) + pw*(sp[c]@q_ph[i]) + dw*(sd[c]@q_dn[i])
                md = np.linalg.norm(qm-snm[c],axis=1); ms = 1.0/(1.0+md)
                scores.append(float(np.sort(vs + mw*ms)[::-1][:k].mean()))
            sa = np.array(scores); ss = np.sort(sa)[::-1]
            preds.append(cids[int(np.argmax(sa))]); margins.append(ss[0]-ss[1])
        preds, margins = np.array(preds), np.array(margins)
        for c in cids:
            cm = (preds==c)&(margins>cf); ci = np.where(cm)[0]
            if len(ci)==0: continue
            ti = ci[np.argsort(margins[ci])[::-1][:tk]]
            sb[c] = np.concatenate([s_bc0[c], q_bc[ti]*0.5])
            sp[c] = np.concatenate([s_ph0[c], q_ph[ti]*0.5])
            sd[c] = np.concatenate([s_dn0[c], q_dn[ti]*0.5])
            smm[c] = np.concatenate([s_morph0[c], q_morph[ti]])
    sm2 = np.concatenate([smm[c] for c in cids])
    gm2, gs2 = sm2.mean(0), sm2.std(0)+1e-8
    snm = {c: (smm[c]-gm2)/gs2 for c in cids}
    snmw = {c: snm[c]*fw for c in cids}
    gt, pred = [], []
    for i in range(len(q_labels)):
        qm = (q_morph[i]-gm2)/gs2; qmw = qm*fw
        scores = {}
        for c in cids:
            vs = bw*(sb[c]@q_bc[i]) + pw*(sp[c]@q_ph[i]) + dw*(sd[c]@q_dn[i])
            md = np.linalg.norm(qm-snm[c],axis=1); ms = 1.0/(1.0+md)
            scores[c] = float(np.sort(vs + mw*ms)[::-1][:k].mean())
        sa = np.array([scores[c] for c in cids])
        t1 = cids[int(np.argmax(sa))]; mg = np.sort(sa)[::-1][0]-np.sort(sa)[::-1][1]
        if t1 in [3,4] and mg < ct:
            for gc in [3,4]:
                mdw = np.linalg.norm(qmw-snmw[gc],axis=1)
                msc = float(np.mean(1.0/(1.0+np.sort(mdw)[:5])))
                vbs = float(np.sort(sb[gc]@q_bc[i])[::-1][:3].mean())
                vps = float(np.sort(sp[gc]@q_ph[i])[::-1][:3].mean())
                scores[gc] = 0.25*vbs + 0.20*vps + cmw*msc
            t1 = 3 if scores[3]>scores[4] else 4
        gt.append(int(q_labels[i])); pred.append(t1)
    return metrics(gt, pred, cids)


def main():
    bc_t, mt, lt = load_cache("biomedclip", "train")
    bc_v, mv, lv = load_cache("biomedclip", "val")
    ph_t, _, _ = load_cache("phikon_v2", "train")
    ph_v, _, _ = load_cache("phikon_v2", "val")
    dn_t, _, _ = load_cache("dinov2_s", "train")
    dn_v, _, _ = load_cache("dinov2_s", "val")
    cids = sorted(CLASS_NAMES.keys())
    ndim = mt.shape[1]
    eos, neu = mt[lt==3], mt[lt==4]
    fw = np.ones(ndim, np.float32)
    for d in range(ndim):
        f = (np.mean(eos[:,d])-np.mean(neu[:,d]))**2 / (np.var(eos[:,d])+np.var(neu[:,d])+1e-10)
        fw[d] = 1.0 + f*2.0

    ar = defaultdict(lambda: {"acc":[],"mf1":[],"pc":defaultdict(list)})
    for seed in SEEDS:
        print(f"Seed {seed}...")
        si = select_support(lt, seed, cids)
        sbc = {c: bc_t[si[c]] for c in cids}
        sph = {c: ph_t[si[c]] for c in cids}
        sdn = {c: dn_t[si[c]] for c in cids}
        sm = {c: mt[si[c]] for c in cids}

        for bw in [0.36, 0.38, 0.40, 0.42]:
            for pw in [0.18, 0.20, 0.22, 0.24]:
                for dw in [0.05, 0.07, 0.09]:
                    mw = 1.0-bw-pw-dw
                    if mw < 0.25 or mw > 0.45: continue
                    for ct in [0.008, 0.010, 0.012]:
                        for cmw in [0.45, 0.50, 0.55]:
                            n = f"b{bw}_p{pw}_d{dw}_m{mw:.2f}_ct{ct}_cm{cmw}"
                            try:
                                m = run_pipeline(bc_v,ph_v,dn_v,mv,lv,sbc,sph,sdn,sm,cids,fw,
                                                  bw,pw,dw,mw,7,2,5,0.025,ct,cmw)
                                ar[n]["acc"].append(m["acc"]); ar[n]["mf1"].append(m["mf1"])
                                for c in cids: ar[n]["pc"][c].append(m["pc"][c]["f1"])
                            except: pass

    print(f"\n{'='*140}")
    print("TRIPLE FINE SWEEP (5 seeds)")
    print(f"{'='*140}")
    h = f"{'Strategy':<60} {'Acc':>7} {'mF1':>7} {'Eos':>7} {'Neu':>7} {'Lym':>7} {'Mac':>7}  {'As':>5} {'Fs':>5}"
    print(h); print("-"*140)
    sr = sorted(ar.items(), key=lambda x: -np.mean(x[1]["mf1"]))
    for n,v in sr[:25]:
        if len(v["acc"])<3: continue
        pc = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{n:<60} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc}  {np.std(v['acc']):>5.3f} {np.std(v['mf1']):>5.3f}")
    b = sr[0]
    print(f"\n*** BEST: {b[0]} -> mF1={np.mean(b[1]['mf1']):.4f}, Eos={np.mean(b[1]['pc'][3]):.4f} ***")
    print(f"\n--- Best by Eos F1 ---")
    se = sorted(ar.items(), key=lambda x: -np.mean(x[1]["pc"][3]))
    for n,v in se[:10]:
        if len(v["acc"])<3: continue
        pc = " ".join(f"{np.mean(v['pc'][c]):>7.4f}" for c in cids)
        print(f"{n:<60} {np.mean(v['acc']):>7.4f} {np.mean(v['mf1']):>7.4f} {pc}")


if __name__ == "__main__":
    main()
