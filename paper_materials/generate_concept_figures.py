#!/usr/bin/env python3
"""Generate conceptual diagram figures for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT_DIR = Path("/home/xut/csclip/paper_materials/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close(fig)


# ============================================================
# Fig.1: BALF-Analyzer Overall Architecture
# ============================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, color='#ECF0F1', edge='#2C3E50', fontsize=10, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
                fontweight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color='#2C3E50'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Title
    ax.text(7, 9.6, 'Fig. 1: BALF-Analyzer Overall Architecture', ha='center', va='top',
            fontsize=16, fontweight='bold')

    # Stage 1: Input
    box(4.5, 8.5, 5, 0.7, 'BALF Microscopy Field Image', '#D5DBDB', bold=True)
    arrow(7, 8.5, 7, 8.0)

    # Stage 2: Cascade Segmentation
    box(3.5, 7.2, 7, 0.7, 'Cascade Segmentation Frontend', '#AED6F1', bold=True)
    box(1.5, 6.2, 4.5, 0.7, 'Cellpose 4.1.1 (cpsam)\nPrimary-Anchor Multi-Scale Rescue', '#D6EAF8')
    box(8.0, 6.2, 4.5, 0.7, 'SAM3 Interactive Refinement\n(Human-in-the-Loop)', '#D6EAF8')
    arrow(7, 7.2, 3.75, 6.9)
    arrow(7, 7.2, 10.25, 6.9)
    arrow(3.75, 6.2, 5.5, 5.7)
    arrow(10.25, 6.2, 8.5, 5.7)

    # Stage 3: Feature Extraction
    box(3.5, 4.9, 7, 0.7, 'Multi-Backbone Feature Extraction', '#A9DFBF', bold=True)
    box(0.3, 3.9, 3.0, 0.7, 'BiomedCLIP\n(512-d)', '#D5F5E3')
    box(3.8, 3.9, 3.0, 0.7, 'Phikon-v2\n(1024-d)', '#D5F5E3')
    box(7.3, 3.9, 3.0, 0.7, 'DINOv2-S\n(384-d)', '#D5F5E3')
    box(10.8, 3.9, 2.5, 0.7, '40-d Hand-crafted\nMorphology', '#D5F5E3')
    arrow(5.5, 4.9, 1.8, 4.6)
    arrow(5.5, 4.9, 5.3, 4.6)
    arrow(8.5, 4.9, 8.8, 4.6)
    arrow(8.5, 4.9, 12.0, 4.6)
    arrow(1.8, 3.9, 5.5, 3.4)
    arrow(5.3, 3.9, 6.0, 3.4)
    arrow(8.8, 3.9, 7.5, 3.4)
    arrow(12.0, 3.9, 8.5, 3.4)

    # Stage 4: Support Set + AFP-OD
    box(3.5, 2.6, 7, 0.7, 'Adaptive Fisher Prototype Oriented Decoupling (AFP-OD)', '#F9E79F', bold=True)
    box(0.5, 1.6, 4.0, 0.7, 'Dual-View Confusion Detection\n(LOO + Centroid views)', '#FCF3CF')
    box(5.5, 1.6, 4.0, 0.7, 'Ledoit-Wolf Shrinkage\n→ Fisher Direction', '#FCF3CF')
    box(10.0, 1.6, 3.5, 0.7, 'Confidence-Gated\nPrototype Shift (α=0.10)', '#FCF3CF')
    arrow(5.5, 2.6, 2.5, 2.3)
    arrow(6.5, 2.6, 7.5, 2.3)
    arrow(8.5, 2.6, 11.75, 2.3)
    arrow(2.5, 1.6, 5.5, 1.3)
    arrow(7.5, 1.6, 6.5, 1.3)
    arrow(11.75, 1.6, 8.5, 1.3)

    # Stage 5: Classification + HITL
    box(3.5, 0.3, 7, 0.7, 'Multi-Backbone Weighted kNN (k=7) + Human-in-the-Loop Review', '#F5B7B1', bold=True)
    arrow(7, 1.6, 7, 1.0)

    # Two-level correction principle label
    ax.text(0.1, 5.5, 'Two-Level\nCorrection', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#8E44AD', rotation=90)
    ax.annotate('', xy=(0.1, 7.0), xytext=(0.1, 4.0),
                arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=2))
    ax.text(0.1, 7.3, 'Proposal\nLevel', ha='center', va='bottom', fontsize=8, color='#8E44AD')
    ax.text(0.1, 3.7, 'Prototype\nLevel', ha='center', va='top', fontsize=8, color='#8E44AD')

    # PAMSR label on left
    ax.text(13.5, 6.5, 'PAMSR', ha='center', va='center', fontsize=10, fontweight='bold', color='#2980B9',
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#2980B9'))

    # AFP-OD label on right
    ax.text(13.5, 2.0, 'AFP-OD', ha='center', va='center', fontsize=10, fontweight='bold', color='#D35400',
            bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor='#D35400'))

    plt.tight_layout()
    save(fig, 'fig1_architecture.png')


# ============================================================
# Fig.3: AFP-OD Conceptual Diagram
# ============================================================
def fig3_afpod_concept():
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(8, 9.6, 'Fig. 3: AFP-OD Conceptual Diagram', ha='center', va='top',
            fontsize=16, fontweight='bold')

    def circle(x, y, r, color, label, fontsize=9):
        c = plt.Circle((x, y), r, facecolor=color, edgecolor='#2C3E50', linewidth=1.5, alpha=0.7)
        ax.add_patch(c)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    def arrow(x1, y1, x2, y2, color='#2C3E50', style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5))

    # ---- (a) Dual-View Confusion Detection ----
    ax.text(3.5, 8.8, '(a) Dual-View Confusion Detection', ha='center', fontsize=12, fontweight='bold')

    # Support set scatter
    np.random.seed(42)
    # Class A (Eos)
    xa = np.random.randn(8) * 0.3 + 2.5
    ya = np.random.randn(8) * 0.3 + 6.5
    ax.scatter(xa, ya, c='#E74C3C', s=80, alpha=0.7, edgecolors='white', linewidth=0.5, label='Eosinophil')
    # Class B (Neu)
    xb = np.random.randn(8) * 0.3 + 3.5
    yb = np.random.randn(8) * 0.3 + 6.5
    ax.scatter(xb, yb, c='#3498DB', s=80, alpha=0.7, edgecolors='white', linewidth=0.5, label='Neutrophil')
    # Prototypes
    ax.scatter([2.5], [6.5], c='#E74C3C', s=300, marker='*', edgecolors='#2C3E50', linewidth=1.5, zorder=5)
    ax.scatter([3.5], [6.5], c='#3498DB', s=300, marker='*', edgecolors='#2C3E50', linewidth=1.5, zorder=5)
    ax.text(2.5, 5.9, 'μ_Eos', ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
    ax.text(3.5, 5.9, 'μ_Neu', ha='center', fontsize=9, color='#3498DB', fontweight='bold')

    # Dashed ellipse showing overlap region
    ellipse = mpatches.Ellipse((3.0, 6.5), 1.5, 1.0, angle=0, fill=False,
                                edgecolor='#8E44AD', linestyle='--', linewidth=2)
    ax.add_patch(ellipse)
    ax.text(3.0, 7.3, 'Confusion\nRegion', ha='center', fontsize=8, color='#8E44AD')

    # View labels
    ax.text(1.5, 7.8, 'View 1:\nLeave-One-Out', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#2980B9'))
    ax.text(4.5, 7.8, 'View 2:\nCentroid', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#2980B9'))

    # ---- (b) LW-Fisher Direction ----
    ax.text(8.5, 8.8, '(b) Ledoit-Wolf → Fisher Direction', ha='center', fontsize=12, fontweight='bold')

    # Covariance ellipses
    ellipse1 = mpatches.Ellipse((7.5, 6.5), 1.2, 0.8, angle=30, fill=True,
                                 facecolor='#E74C3C', alpha=0.2, edgecolor='#E74C3C', linewidth=2)
    ellipse2 = mpatches.Ellipse((8.5, 6.5), 1.2, 0.8, angle=-30, fill=True,
                                 facecolor='#3498DB', alpha=0.2, edgecolor='#3498DB', linewidth=2)
    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)

    # Fisher direction arrow
    ax.annotate('', xy=(9.5, 7.2), xytext=(6.5, 5.8),
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=3))
    ax.text(8.0, 5.3, r'$\mathbf{v}_{ij} = (\hat\Sigma_i + \hat\Sigma_j)^{-1}(\mu_i - \mu_j)$',
            ha='center', fontsize=10, color='#27AE60', fontweight='bold')

    # Shrinkage annotation
    ax.text(6.0, 7.8, r'LW shrink: $\hat\Sigma = (1-\lambda^*)\Sigma + \lambda^*\frac{\mathrm{tr}(\Sigma)}{d}I$',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#E8F8F5', edgecolor='#27AE60'))

    # ---- (c) Confidence-Gated Prototype Shift ----
    ax.text(13.0, 8.8, '(c) Confidence-Gated Shift', ha='center', fontsize=12, fontweight='bold')

    # Before shift
    ax.scatter([12.5], [6.5], c='#E74C3C', s=400, marker='*', edgecolors='#2C3E50', linewidth=2, zorder=5)
    ax.scatter([13.5], [6.5], c='#3498DB', s=400, marker='*', edgecolors='#2C3E50', linewidth=2, zorder=5)
    ax.text(12.5, 5.9, 'μ\'_Eos', ha='center', fontsize=9, color='#E74C3C')
    ax.text(13.5, 5.9, 'μ\'_Neu', ha='center', fontsize=9, color='#3498DB')

    # Shift arrows
    ax.annotate('', xy=(12.0, 7.0), xytext=(12.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
    ax.annotate('', xy=(14.0, 7.0), xytext=(13.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2))
    ax.text(12.0, 7.3, '+αv', ha='center', fontsize=9, color='#E74C3C')
    ax.text(14.0, 7.3, '-αv', ha='center', fontsize=9, color='#3498DB')

    # Confidence gate
    ax.text(13.0, 5.2, 'Enable only if\nΔ_conf < 0.025', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor='#F39C12'))

    # Query sample
    ax.scatter([13.0], [6.8], c='#2C3E50', s=100, marker='x', linewidth=3, zorder=6)
    ax.text(13.0, 7.1, 'Query q', ha='center', fontsize=8, color='#2C3E50')

    # Bottom summary
    ax.text(8, 0.8, 'Union of confused pairs from both views → LW-regularized Fisher direction → Confidence-gated shift',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', edgecolor='#2C3E50', linewidth=2))

    ax.set_aspect('equal')
    plt.tight_layout()
    save(fig, 'fig3_afpod_concept.png')


# ============================================================
# Fig.2: PAMSR Mechanism Diagram
# ============================================================
def fig2_pamsr_mechanism():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(7, 5.7, 'Fig. 2: PAMSR — Primary-Anchor Multi-Scale Rescue', ha='center', fontsize=15, fontweight='bold')

    def box(x, y, w, h, text, color, fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='#2C3E50', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

    # Input image
    box(0.5, 3.5, 2.0, 1.2, 'Input\nBALF Image', '#D5DBDB', fontsize=10)
    arrow(2.5, 4.1, 3.3, 4.1)

    # Primary scale
    box(3.3, 3.5, 2.2, 1.2, 'Primary Scale\nd*=50', '#AED6F1', fontsize=10)
    arrow(5.5, 4.1, 6.3, 4.1)

    # Anchor masks
    box(6.3, 3.5, 2.2, 1.2, 'Anchor Masks\nMp (baseline)', '#D6EAF8', fontsize=10)

    # Auxiliary scales (below)
    box(3.3, 1.5, 2.2, 1.2, 'Auxiliary\nd={40, 65}', '#AED6F1', fontsize=10)
    arrow(3.3, 3.5, 3.3, 2.7)
    arrow(4.4, 2.7, 4.4, 3.5)

    # Candidate generation
    box(6.3, 1.5, 2.2, 1.2, 'Candidate\nInstances Ci', '#D6EAF8', fontsize=10)
    arrow(5.5, 2.1, 6.3, 2.1)

    # Filter gates
    box(9.0, 3.5, 2.2, 1.2, 'Gate 1: IoU < 0.2\n(non-overlap)', '#F9E79F', fontsize=9)
    box(9.0, 1.5, 2.2, 1.2, 'Gate 2: cellprob > τ\n(confidence)', '#F9E79F', fontsize=9)
    box(9.0, 0.2, 2.2, 1.0, 'Gate 3: Cross-scale\nconsensus (optional)', '#F9E79F', fontsize=9)

    arrow(8.5, 4.1, 9.0, 4.1)
    arrow(8.5, 2.1, 9.0, 2.1)
    arrow(10.1, 1.5, 10.1, 1.2)

    # Final output
    box(12.0, 2.5, 1.8, 1.5, 'Final\nMasks', '#82E0AA', fontsize=11)
    arrow(11.2, 4.1, 12.0, 3.2)
    arrow(11.2, 2.1, 12.0, 3.3)
    arrow(11.2, 0.7, 12.0, 2.8)

    # Key principle annotation
    ax.text(7, 0.3, 'Principle: Anchor masks are protected; auxiliary scale only rescues verifiable omissions',
            ha='center', fontsize=10, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor='#F8F9F9', edgecolor='#2C3E50'))

    plt.tight_layout()
    save(fig, 'fig2_pamsr_mechanism.png')


if __name__ == '__main__':
    print("Generating conceptual figures...")
    fig1_architecture()
    fig3_afpod_concept()
    fig2_pamsr_mechanism()
    print(f"\nConceptual figures saved to: {OUT_DIR}")
