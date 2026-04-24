#!/usr/bin/env python3
"""
Generate all data-driven paper figures for BALF-Analyzer.
Uses final data from EXPERIMENT_RESULTS_SUMMARY.md.
Output: paper_materials/figures/ directory.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path("/home/xut/csclip/paper_materials/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Journal-quality style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'eos': '#E74C3C',
    'neu': '#3498DB',
    'lym': '#2ECC71',
    'mac': '#F39C12',
    'primary': '#2980B9',
    'secondary': '#27AE60',
    'accent': '#E67E22',
    'gray': '#7F8C8D',
    'light_gray': '#BDC3C7',
}


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close(fig)


# ============================================================
# Fig.4: Main Classification Results vs SOTA (data2, 10-shot)
# ============================================================
def fig4_main_classification():
    methods = [
        'NCM\n(BC)',
        'kNN\nk=7',
        'Label\nProp.',
        'Tip-\nAdapter',
        'Tip-Adapter\n-F',
        'EM-Dirichlet',
        'Linear\nProbe',
        'Ensemble',
        'MB-kNN\n(Baseline)',
        'AFP-OD\nP3c (Ours)',
    ]
    acc = [0.7757, 0.7964, 0.7788, 0.8658, 0.8658, 0.5586, 0.7140, 0.7484, 0.848, 0.863]
    mf1 = [0.6557, 0.6592, 0.6150, 0.7415, 0.7413, 0.5586, 0.6842, 0.7484, 0.7252, 0.7563]
    eos = [0.2933, 0.2999, 0.2559, 0.3900, 0.3887, 0.2039, 0.4119, 0.4404, 0.4465, 0.5018]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(methods))
    w = 0.25

    bars1 = ax.bar(x - w, acc, w, label='Accuracy', color='#3498DB', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, mf1, w, label='Macro-F1', color='#2ECC71', edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + w, eos, w, label='Eosinophil F1', color='#E74C3C', edgecolor='white', linewidth=0.5)

    # Highlight ours
    for bars in [bars1, bars2, bars3]:
        bars[-1].set_edgecolor('#2C3E50')
        bars[-1].set_linewidth(2)
    for bars in [bars1, bars2, bars3]:
        bars[-2].set_edgecolor('#7F8C8D')
        bars[-2].set_linewidth(1.5)
        bars[-2].set_linestyle('--')

    ax.set_ylabel('Score')
    ax.set_title('Fig. 4: 10-Shot Classification Results on BALF data2 (4 classes, 25-run average)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8.5)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7563, color='#E74C3C', linestyle=':', linewidth=1, alpha=0.5)

    # Add value labels on top bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=6.5, rotation=90)

    plt.tight_layout()
    save(fig, 'fig4_main_classification.png')


# ============================================================
# Fig.5: Ablation Study (4 subplots)
# ============================================================
def fig5_ablation():
    fig = plt.figure(figsize=(16, 12))

    # ---- (a) Backbone Combination ----
    ax1 = fig.add_subplot(2, 2, 1)
    backbones = ['BC\nonly', 'PH\nonly', 'DN\nonly', 'BC+PH', 'BC+DN', 'BC+PH+DN\n(ours)', 'w/o\nMorph']
    acc_a = [0.8476, 0.8087, 0.7701, 0.8591, 0.8567, 0.8653, 0.8479]
    mf1_a = [0.7039, 0.6721, 0.6454, 0.7359, 0.7262, 0.7408, 0.7202]
    eos_a = [0.3083, 0.3343, 0.2390, 0.4178, 0.3384, 0.4092, 0.3859]

    x = np.arange(len(backbones))
    w = 0.25
    ax1.bar(x - w, acc_a, w, label='Accuracy', color='#3498DB')
    ax1.bar(x, mf1_a, w, label='Macro-F1', color='#2ECC71')
    ax1.bar(x + w, eos_a, w, label='Eos F1', color='#E74C3C')
    ax1.set_ylabel('Score')
    ax1.set_title('(a) Backbone Combination Ablation', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones, fontsize=9)
    ax1.legend(loc='upper left', frameon=True, edgecolor='gray')
    ax1.set_ylim(0, 1.0)
    for i, (a, m, e) in enumerate(zip(acc_a, mf1_a, eos_a)):
        ax1.text(i, max(a, m, e) + 0.02, f'mF1={m:.3f}', ha='center', fontsize=8, color='#27AE60', fontweight='bold')

    # ---- (b) Separation Strength α ----
    ax2 = fig.add_subplot(2, 2, 2)
    alphas = [0.05, 0.10, 0.20]
    mf1_b = [0.745, 0.756, 0.748]
    eos_b = [0.484, 0.502, 0.483]

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(alphas, mf1_b, 'o-', color='#2980B9', linewidth=2, markersize=10, label='Macro-F1')[0]
    l2 = ax2_twin.plot(alphas, eos_b, 's--', color='#E74C3C', linewidth=2, markersize=10, label='Eos F1')[0]
    ax2.axvline(x=0.10, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(0.10, 0.758, 'α=0.10\n(ours)', ha='center', fontsize=9, color='#7F8C8D')

    ax2.set_xlabel('Separation Strength α')
    ax2.set_ylabel('Macro-F1', color='#2980B9')
    ax2_twin.set_ylabel('Eosinophil F1', color='#E74C3C')
    ax2.set_title('(b) AFP-OD Separation Strength α', fontweight='bold')
    ax2.set_ylim(0.73, 0.76)
    ax2_twin.set_ylim(0.47, 0.51)
    ax2.legend([l1, l2], ['Macro-F1', 'Eos F1'], loc='upper left', frameon=True, edgecolor='gray')
    for a, m, e in zip(alphas, mf1_b, eos_b):
        ax2.text(a, m + 0.001, f'{m:.3f}', ha='center', fontsize=9, color='#2980B9', fontweight='bold')
        ax2_twin.text(a, e + 0.0015, f'{e:.3f}', ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    # ---- (c) AFP-OD Stepwise Ablation ----
    ax3 = fig.add_subplot(2, 2, 3)
    configs = ['MB-kNN\n(baseline)', 'P1\nTrace', 'P2\nLW', 'P3a\nMorph-PLS', 'P3b\nIntersect', 'P3c\nUnion (ours)']
    mf1_c = [0.7252, 0.7477, 0.7485, 0.7425, 0.7491, 0.7563]
    eos_c = [0.4465, 0.4920, 0.4933, 0.4789, 0.4903, 0.5018]

    x = np.arange(len(configs))
    w = 0.35
    bars_m = ax3.bar(x - w/2, mf1_c, w, label='Macro-F1', color='#2ECC71', edgecolor='white')
    bars_e = ax3.bar(x + w/2, eos_c, w, label='Eos F1', color='#E74C3C', edgecolor='white')

    # Highlight ours
    bars_m[-1].set_edgecolor('#2C3E50')
    bars_m[-1].set_linewidth(2)
    bars_e[-1].set_edgecolor('#2C3E50')
    bars_e[-1].set_linewidth(2)

    ax3.set_ylabel('Score')
    ax3.set_title('(c) AFP-OD Stepwise Ablation', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, fontsize=9)
    ax3.legend(loc='upper left', frameon=True, edgecolor='gray')
    ax3.set_ylim(0.42, 0.78)
    for i, (m, e) in enumerate(zip(mf1_c, eos_c)):
        ax3.text(i - w/2, m + 0.005, f'{m:.3f}', ha='center', fontsize=8, color='#27AE60')
        ax3.text(i + w/2, e + 0.005, f'{e:.3f}', ha='center', fontsize=8, color='#C0392B')

    # ---- (d) N-shot Curve ----
    ax4 = fig.add_subplot(2, 2, 4)
    n_shots = [1, 3, 5, 10, 20]
    acc_d = [0.5047, 0.4883, 0.5916, 0.8582, 0.8550]
    mf1_d = [0.4245, 0.4027, 0.4955, 0.7330, 0.7535]
    eos_d = [0.1032, 0.2210, 0.1986, 0.3815, 0.4887]
    acc_std = [0.176, 0.326, 0.160, 0.014, 0.022]
    mf1_std = [0.111, 0.253, 0.136, 0.024, 0.024]

    ax4.errorbar(n_shots, acc_d, yerr=acc_std, fmt='o-', color='#3498DB', linewidth=2, markersize=8, capsize=4, label='Accuracy')
    ax4.errorbar(n_shots, mf1_d, yerr=mf1_std, fmt='s-', color='#2ECC71', linewidth=2, markersize=8, capsize=4, label='Macro-F1')
    ax4.plot(n_shots, eos_d, '^--', color='#E74C3C', linewidth=2, markersize=8, label='Eos F1')

    ax4.axvline(x=10, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.7)
    ax4.text(10, 0.55, '10-shot\n(ours)', ha='center', fontsize=9, color='#7F8C8D')
    ax4.annotate('Inflection\npoint', xy=(10, 0.7330), xytext=(14, 0.60),
                 arrowprops=dict(arrowstyle='->', color='#7F8C8D'), fontsize=9, color='#7F8C8D')

    ax4.set_xlabel('N-shot (labeled cells per class)')
    ax4.set_ylabel('Score')
    ax4.set_title('(d) N-shot Performance Curve', fontweight='bold')
    ax4.legend(loc='lower right', frameon=True, edgecolor='gray')
    ax4.set_ylim(0, 1.05)
    ax4.set_xticks(n_shots)

    fig.suptitle('Fig. 5: Ablation Study Results (data2, 10-shot unless noted)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, 'fig5_ablation_study.png')


# ============================================================
# Fig.7: Segmentation Results (PAMSR 3-dataset + WBC gold)
# ============================================================
def fig7_segmentation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: PAMSR incremental on 3 datasets
    ax = axes[0]
    datasets = ['data2\n(BALF)', 'data1\n(Clinical)', 'WBC-Seg\n(External)']
    single_f1 = [0.7261, 0.6532, 0.8120]
    pamsr_f1 = [0.7276, 0.6556, 0.8186]
    delta_r = [0.0121, 0.0205, 0.0184]

    x = np.arange(len(datasets))
    w = 0.3
    bars1 = ax.bar(x - w/2, single_f1, w, label='Single-scale', color='#BDC3C7', edgecolor='white')
    bars2 = ax.bar(x + w/2, pamsr_f1, w, label='PAMSR (Ours)', color='#2980B9', edgecolor='white')

    # Add delta labels
    for i, (s, p, d) in enumerate(zip(single_f1, pamsr_f1, delta_r)):
        ax.annotate('', xy=(i + w/2, p), xytext=(i - w/2, s),
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
        ax.text(i, max(s, p) + 0.015, f'+{d:.4f}\nRecall', ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    ax.set_ylabel('F1 Score @ IoU≥0.5')
    ax.set_title('(a) PAMSR Incremental Gains', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left', frameon=True, edgecolor='gray')
    ax.set_ylim(0.6, 0.88)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.003, f'{h:.4f}',
                    ha='center', va='bottom', fontsize=9)

    # Right: WBC-Seg detailed metrics
    ax = axes[1]
    metrics = ['Precision', 'Recall', 'F1']
    wbc_vals = [0.9220, 0.8552, 0.8874]
    colors_bar = ['#3498DB', '#2ECC71', '#9B59B6']
    bars = ax.bar(metrics, wbc_vals, color=colors_bar, edgecolor='white', width=0.5)
    ax.set_ylim(0.8, 0.95)
    ax.set_ylabel('Score')
    ax.set_title('(b) WBC-Seg Gold Standard (22,683 GT polygons)', fontweight='bold')
    for bar, val in zip(bars, wbc_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    fig.suptitle('Fig. 7: Segmentation Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, 'fig7_segmentation.png')


# ============================================================
# Fig.8: Cross-dataset Generalization
# ============================================================
def fig8_cross_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Classification across datasets
    ax = axes[0]
    methods = ['NCM\n(BC)', 'kNN\nk=5', 'MB-kNN', 'AFP-OD\nP3c (ours)']
    data2_mf1 = [0.6557, 0.6582, 0.7252, 0.7563]
    mc_mf1 = [0.3798, 0.3300, 0.3190, 0.3190]
    pbc_mf1 = [None, None, 0.8495, 0.8577]  # Only MB-kNN and ours

    x = np.arange(len(methods))
    w = 0.25
    bars1 = ax.bar(x - w, data2_mf1, w, label='data2 (BALF)', color='#3498DB')
    bars2 = ax.bar(x, mc_mf1, w, label='MultiCenter', color='#E67E22')
    bars3 = ax.bar(x + w, [v if v else 0 for v in pbc_mf1], w, label='PBC (external)', color='#9B59B6')

    # Gray out missing bars
    bars3[0].set_visible(False)
    bars3[1].set_visible(False)

    ax.set_ylabel('Macro-F1')
    ax.set_title('(a) Cross-Dataset Classification Generalization', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.legend(loc='upper right', frameon=True, edgecolor='gray')
    ax.set_ylim(0, 1.0)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(pbc_mf1):
        if v:
            ax.text(i + w, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # Right: MultiCenter class distribution + limitation note
    ax = axes[1]
    classes = ['Neu', 'Mac', 'Lym', 'Eos']
    mc_counts = [86, 9.6, 3.8, 0.4]
    colors_pie = ['#3498DB', '#F39C12', '#2ECC71', '#E74C3C']
    explode = (0, 0, 0, 0.15)
    wedges, texts, autotexts = ax.pie(mc_counts, explode=explode, labels=classes, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90,
                                       textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('(b) MultiCenter Validation Class Distribution\n(Extreme Imbalance: Eos only 0.4%)', fontweight='bold')

    fig.suptitle('Fig. 8: Cross-Dataset Robustness & Limitations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, 'fig8_cross_dataset.png')


# ============================================================
# Fig.6: Confusion Matrix (improved version)
# ============================================================
def fig6_confusion():
    # Data from confusion matrix seed42
    # Using the per-sample CSV to compute confusion
    import pandas as pd
    csv_path = Path("/home/xut/csclip/paper_materials/confusion/per_sample_predictions_seed42.csv")
    if not csv_path.exists():
        print("SKIP fig6: confusion CSV not found")
        return

    df = pd.read_csv(csv_path)
    label_map = {3: 0, 4: 1, 5: 2, 6: 3}
    class_names = ['Eos', 'Neu', 'Lym', 'Mac']
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for _, row in df.iterrows():
        t = label_map.get(int(row['true_label']), int(row['true_label']))
        p = label_map.get(int(row['pred_label']), int(row['pred_label']))
        cm[t, p] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Count version
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('(a) Confusion Matrix (Counts)', fontweight='bold')

    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                           color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14, fontweight='bold')

    # Normalized version
    ax = axes[1]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('(b) Row-Normalized (Recall)', fontweight='bold')

    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                           color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('Normalized Count' if True else 'Count')

    fig.suptitle('Fig. 6: Confusion Matrix — Adaptive Cascade (Seed 42, n=1316)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, 'fig6_confusion_matrix.png')


# ============================================================
# Fig.S1: Per-class F1 Radar Chart
# ============================================================
def fig_s1_radar():
    from math import pi

    categories = ['Eos', 'Neu', 'Lym', 'Mac']
    N = len(categories)

    # Data: MB-kNN vs AFP-OD P3c
    mbknn = [0.4465, 0.6784, 0.9310, 0.8448]
    afpod = [0.5018, 0.734, 0.931, 0.859]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    mbknn += mbknn[:1]
    afpod += afpod[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, mbknn, 'o-', linewidth=2, label='MB-kNN (baseline)', color='#BDC3C7')
    ax.fill(angles, mbknn, alpha=0.15, color='#BDC3C7')
    ax.plot(angles, afpod, 'o-', linewidth=2, label='AFP-OD P3c (ours)', color='#2980B9')
    ax.fill(angles, afpod, alpha=0.15, color='#2980B9')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    ax.set_title('Per-Class F1 Comparison\n(data2, 10-shot)', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, edgecolor='gray')

    plt.tight_layout()
    save(fig, 'fig_s1_radar_perclass.png')


if __name__ == '__main__':
    print("Generating all paper figures...")
    fig4_main_classification()
    fig5_ablation()
    fig7_segmentation()
    fig8_cross_dataset()
    fig6_confusion()
    fig_s1_radar()
    print(f"\nAll figures saved to: {OUT_DIR}")
