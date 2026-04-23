"""
Generate patent-style line drawings for patent_draft_zh.md.
风格：黑白线描，无填色，Arial / 思源黑体中文，白底，细边框，适合发明专利附图。

输出目录：/home/xut/csclip/patent_figures/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np
import matplotlib as mpl

# ---------- 全局样式 ---------- #
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUT_DIR = Path("/home/xut/csclip/patent_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LW = 1.2       # 统一线宽
FS = 10        # 基础字号
FS_SM = 9
FS_LG = 11


def save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"✓ {path}")


def box(ax, x, y, w, h, text, *, fs=FS, lw=LW, style="round", zorder=3):
    """在 (x,y) 左下角绘制矩形并在中心写文字。返回中心坐标。"""
    if style == "round":
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           fc="white", ec="black", lw=lw, zorder=zorder)
    elif style == "square":
        p = Rectangle((x, y), w, h, fc="white", ec="black", lw=lw,
                      zorder=zorder)
    elif style == "dashed":
        p = Rectangle((x, y), w, h, fc="white", ec="black", lw=lw,
                      linestyle="--", zorder=zorder)
    else:
        raise ValueError(style)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=zorder + 1)
    return (x + w / 2, y + h / 2)


def diamond(ax, cx, cy, w, h, text, *, fs=FS_SM, lw=LW):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
           (cx - w / 2, cy)]
    poly = mpatches.Polygon(pts, closed=True, fc="white", ec="black",
                            lw=lw, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)
    return (cx, cy)


def arrow(ax, xy_from, xy_to, *, text=None, fs=FS_SM, style="->", lw=LW,
          rad=0.0, ls="-"):
    ap = FancyArrowPatch(xy_from, xy_to,
                         arrowstyle=style, mutation_scale=14,
                         color="black", lw=lw, linestyle=ls,
                         connectionstyle=f"arc3,rad={rad}", zorder=2)
    ax.add_patch(ap)
    if text is not None:
        mx = (xy_from[0] + xy_to[0]) / 2
        my = (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my, text, fontsize=fs, ha="center", va="center",
                backgroundcolor="white", zorder=5)


def clean(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# =====================================================================
# 图 1 整体流程图
# =====================================================================
def fig_1_pipeline():
    fig, ax = plt.subplots(figsize=(8.0, 10.5))
    clean(ax, (0, 10), (0, 13.5))

    # Stages (y 从上到下)
    stages = [
        ("1. 输入细胞显微视野图像", 11.6),
        ("2. 级联分割：Cellpose 初分割 → SAM3 精修\n(IoU > 0.5 采纳精修掩码)", 10.05),
        ("3. 双尺度裁剪\n细胞裁剪 (扩 10%) + 上下文裁剪 (扩 50%)", 8.5),
        ("4. 多骨干特征提取\nBiomedCLIP (512) + Phikon-v2 (1024) + DINOv2-S (384)\n+ 40 维形态学特征", 6.95),
        ("5. Support 集构建 (每类 N=10)", 5.4),
        ("6. 双视图留一法 kNN 混淆检测\n特征空间 ∪ 形态学空间，阈值 $\\tau=0.15$", 3.85),
        ("7. Ledoit-Wolf 自适应协方差收缩\n→ Fisher 判别方向 $w_{ij}$", 2.3),
        ("8. Support 原型定向扰动\n$f \\leftarrow f \\pm \\alpha \\cdot w_{ij}$，$\\alpha=0.10$", 0.75),
    ]
    w = 6.8
    centers = []
    for txt, y in stages:
        c = box(ax, (10 - w) / 2, y, w, 1.0, txt, fs=FS_SM)
        centers.append(c)

    # 箭头连接
    for i in range(len(centers) - 1):
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        arrow(ax, (x1, y1 - 0.5), (x2, y2 + 0.5))

    # 最终输出框
    out_c = box(ax, 1.6, -0.8, 6.8, 1.0,
                "9. 多骨干加权 kNN 分类\n($w_{BC}=0.42, w_{PH}=0.18, w_{DN}=0.07, w_{m}=0.33; k=7$)",
                fs=FS_SM)
    arrow(ax, (centers[-1][0], centers[-1][1] - 0.5),
          (out_c[0], out_c[1] + 0.5))

    ax.set_ylim(-1.2, 13.5)
    ax.text(5, 13.2, "图 1  本发明方法整体流程图",
            ha="center", va="center", fontsize=FS_LG, weight="bold")
    save(fig, "图1_整体流程图.png")


# =====================================================================
# 图 2 Cellpose + SAM3 级联分割
# =====================================================================
def fig_2_cascade_seg():
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    clean(ax, (0, 16), (0, 7.5))

    # Input image (symbolic microscope field)
    img_box = box(ax, 0.3, 2.5, 2.5, 2.5, "", lw=LW)
    # 在 input 方框里画几个代表细胞的圆
    rng = np.random.default_rng(3)
    for _ in range(7):
        cx = 0.6 + rng.uniform(0, 1.9)
        cy = 2.8 + rng.uniform(0, 1.9)
        r = rng.uniform(0.12, 0.25)
        ax.add_patch(Circle((cx, cy), r, fc="white", ec="black", lw=0.8,
                            zorder=5))
    ax.text(1.55, 5.3, "输入视野图像 $I$", ha="center", fontsize=FS_SM)

    # Cellpose branch
    cp = box(ax, 4.0, 4.3, 2.6, 1.0, "Cellpose 4.1.1\n(cyto3, 自动分割)",
             fs=FS_SM)
    arrow(ax, (2.8, 4.5), (4.0, cp[1]))

    cp_out = box(ax, 7.2, 4.3, 2.7, 1.0,
                 "初始掩码 $M_{\\mathrm{init}}$\n包围盒 $B_i$", fs=FS_SM)
    arrow(ax, (6.6, cp[1]), (7.2, cp_out[1]))

    # SAM3 branch
    sam = box(ax, 4.0, 1.7, 2.6, 1.0, "SAM3\n(bbox 作为稀疏提示)", fs=FS_SM)
    arrow(ax, (2.8, 3.3), (4.0, sam[1]))

    sam_out = box(ax, 7.2, 1.7, 2.7, 1.0, "精修掩码 $M_{\\mathrm{sam3}}$",
                  fs=FS_SM)
    arrow(ax, (6.6, sam[1]), (7.2, sam_out[1]))

    # bbox 从 Cellpose 传给 SAM3
    arrow(ax, (8.55, 4.3), (5.3, 2.7), style="->", rad=0.25, ls="--")
    ax.text(7.3, 3.55, "$B_i$ 提示", fontsize=FS_SM - 1, style="italic")

    # Decision — enlarged diamond, symmetric arrows hit vertices
    dia_cx, dia_cy = 11.95, 3.5
    dia_w, dia_h = 2.2, 2.2
    dia = diamond(ax, dia_cx, dia_cy, dia_w, dia_h,
                  "$\\mathrm{IoU}(M_{\\mathrm{sam3}}, M_{\\mathrm{init}}) > 0.5?$",
                  fs=FS_SM - 2)
    left_v = (dia_cx - dia_w / 2, dia_cy)
    right_v = (dia_cx + dia_w / 2, dia_cy)
    arrow(ax, (9.9, 4.8), left_v)
    arrow(ax, (9.9, 2.2), left_v)

    # Outputs — symmetric arrows: same dx=0.95, dy=±1.3
    out_yes = box(ax, 14.0, 4.35, 1.5, 0.9,
                  "是\n采用 $M_{\\mathrm{sam3}}$", fs=FS_SM - 1)
    out_no = box(ax, 14.0, 1.75, 1.5, 0.9,
                 "否\n采用 $M_{\\mathrm{init}}$", fs=FS_SM - 1)
    arrow(ax, right_v, (14.0, 4.8))
    arrow(ax, right_v, (14.0, 2.2))

    ax.set_xlim(-0.2, 15.8)
    ax.text(7.8, 7.0, "图 2  Cellpose + SAM3 级联分割流程示意图",
            ha="center", va="center", fontsize=FS_LG, weight="bold")
    save(fig, "图2_级联分割.png")


# =====================================================================
# 图 3 双尺度裁剪
# =====================================================================
def fig_3_dual_crop():
    fig, ax = plt.subplots(figsize=(10, 5.3))
    clean(ax, (0, 14), (0, 7))

    # Left: 原图 + bbox 扩展示意
    ax.add_patch(Rectangle((0.3, 0.5), 4.5, 5.2, fc="white", ec="black",
                           lw=LW))
    # 细胞主体
    cell = Circle((2.55, 3.1), 0.55, fc="white", ec="black", lw=LW)
    ax.add_patch(cell)
    # 细胞 bbox (原始)
    ax.add_patch(Rectangle((2.0, 2.55), 1.1, 1.1, fc="none", ec="black",
                           lw=LW, linestyle=":"))
    ax.text(2.55, 2.25, "原始包围盒 $B$", fontsize=FS_SM - 1,
            ha="center", style="italic")
    # 10% 扩展
    ax.add_patch(Rectangle((1.89, 2.44), 1.32, 1.32, fc="none",
                           ec="black", lw=LW))
    ax.text(0.4, 3.9, "扩 10%\n(细胞裁剪)", fontsize=FS_SM)
    # 50% 扩展
    ax.add_patch(Rectangle((1.45, 2.0), 2.20, 2.20, fc="none", ec="black",
                           lw=LW, linestyle="--"))
    ax.text(0.4, 1.1, "扩 50%\n(上下文裁剪)", fontsize=FS_SM)
    ax.text(2.55, 5.95, "输入: 细胞实例 + 包围盒",
            ha="center", fontsize=FS_SM)

    # 中间箭头（从输入框右边缘到 c1/c2 左边缘，水平对称）
    arrow(ax, (4.8, 4.35), (6.0, 4.35))
    arrow(ax, (4.8, 2.15), (6.0, 2.15))

    # 细胞裁剪 → 224×224
    c1 = box(ax, 6.0, 3.7, 2.1, 1.3,
             "细胞裁剪 $I_{\\mathrm{cell}}$\nresize $224 \\times 224$", fs=FS_SM)
    # 上下文裁剪 → 224×224
    c2 = box(ax, 6.0, 1.5, 2.1, 1.3,
             "上下文裁剪 $I_{\\mathrm{ctx}}$\nresize $224 \\times 224$", fs=FS_SM)

    # 编码器 VFM
    enc1 = box(ax, 9.2, 3.7, 1.8, 1.3, "VFM 编码器\n(共享权重)",
               fs=FS_SM)
    enc2 = box(ax, 9.2, 1.5, 1.8, 1.3, "VFM 编码器\n(共享权重)",
               fs=FS_SM)
    arrow(ax, (8.1, c1[1]), (9.2, enc1[1]))
    arrow(ax, (8.1, c2[1]), (9.2, enc2[1]))

    # 特征 → 加权融合（汇聚到 fuse 左边缘中心）
    fuse = box(ax, 12.0, 2.6, 1.8, 1.3,
               "$f = 0.9 \\cdot f_{\\mathrm{cell}}$\n$+ \\; 0.1 \\cdot f_{\\mathrm{ctx}}$", fs=FS_SM)
    arrow(ax, (11.0, enc1[1]), (12.0, fuse[1]))
    arrow(ax, (11.0, enc2[1]), (12.0, fuse[1]))

    ax.text(7, 6.6, "图 3  双尺度裁剪与加权特征融合示意图",
            ha="center", va="center", fontsize=FS_LG, weight="bold")
    save(fig, "图3_双尺度裁剪.png")


# =====================================================================
# 图 4 多骨干特征提取
# =====================================================================
def fig_4_multi_backbone():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    clean(ax, (0, 14), (0, 8))

    # 输入
    inp = box(ax, 0.3, 3.5, 2.2, 1.2,
              "细胞裁剪 $I_{\\mathrm{cell}}$\n上下文裁剪 $I_{\\mathrm{ctx}}$\n($224 \\times 224$)", fs=FS_SM)

    # 三个骨干 + 形态学
    backbones = [
        ("BiomedCLIP\nViT-B/16 (冻结)", "$f_{\\mathrm{BC}} \\in R^{512}$", 6.0),
        ("Phikon-v2\nViT-L/14 (冻结)", "$f_{\\mathrm{PH}} \\in R^{1024}$", 4.2),
        ("DINOv2-Small\nViT-S/14 (冻结)", "$f_{\\mathrm{DN}} \\in R^{384}$", 2.4),
    ]
    # 输入框 → 各骨干框（分散起点，轻微弯曲）
    src_y = [4.7, 4.1, 3.5]
    for (name, out, y), sy in zip(backbones, src_y):
        b = box(ax, 3.6, y, 2.6, 1.2, name, fs=FS_SM)
        f = box(ax, 7.2, y, 2.3, 1.2, out, fs=FS_SM)
        arrow(ax, (2.5, sy), (3.6, b[1]), rad=0.08)
        arrow(ax, (6.2, b[1]), (7.2, f[1]))

    # 形态学支路
    mask_box = box(ax, 3.6, 0.4, 2.6, 1.2,
                   "分割掩码 $M$\n(Cellpose+SAM3)", fs=FS_SM)
    morph = box(ax, 7.2, 0.4, 2.3, 1.2,
                "形态学 $m \\in R^{40}$\n(面积, 周长, 圆度,\n颜色/颗粒度...)",
                fs=FS_SM - 1)
    arrow(ax, (2.5, 3.5), (3.6, mask_box[1]), rad=-0.08)
    arrow(ax, (6.2, mask_box[1]), (7.2, morph[1]))

    # 加权评分
    score = box(ax, 10.5, 2.8, 3.2, 2.2,
                "多骨干加权评分\n"
                "$s(q,c) = \\sum w_b \\cdot \\cos(f_j^b, q^b)$\n"
                "$+ \\; \\frac{w_m}{1 + \\|m_j - q_m\\|}$\n\n"
                "$w_{\\mathrm{BC}}=0.42 \\quad w_{\\mathrm{PH}}=0.18$\n"
                "$w_{\\mathrm{DN}}=0.07 \\quad w_{\\mathrm{m}}=0.33$", fs=FS_SM - 1)

    # 四个特征框 → 评分框（分散终点到 score 左边缘，直线）
    score_targets = [4.6, 4.0, 3.4, 2.8]
    for y, ty in zip([6.6, 4.8, 3.0, 1.0], score_targets):
        arrow(ax, (9.5, y), (score[0] - 1.6, ty), rad=0.0)

    ax.text(7, 7.6, "图 4  多骨干视觉基础模型与形态学特征并行提取结构图",
            ha="center", va="center", fontsize=FS_LG, weight="bold")
    save(fig, "图4_多骨干结构.png")


# =====================================================================
# 图 5 双视图混淆检测
# =====================================================================
def fig_5_dual_view():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    # 合成 4 类（2D 投影）演示混淆
    rng = np.random.default_rng(42)
    labels = ["Eos", "Mac", "Neu", "Lym"]
    markers = ["o", "s", "^", "D"]

    # --- 特征空间 --- 让 Eos/Mac 显著重叠
    centers_f = [(0.4, 0.6), (0.9, 0.5), (2.2, 2.3), (3.5, 0.8)]
    # --- 形态学空间 --- 让 Neu/Mac 重叠（示例）
    centers_m = [(0.5, 2.8), (2.3, 1.2), (1.8, 1.0), (3.5, 3.0)]

    for ax, centers, title in zip(
            axes,
            [centers_f, centers_m],
            ["(a) 特征空间（BiomedCLIP，余弦 kNN）",
             "(b) 形态学空间（z-score，欧氏 kNN）"]):
        for (cx, cy), lab, mk in zip(centers, labels, markers):
            pts = rng.normal(loc=(cx, cy), scale=0.28, size=(10, 2))
            ax.scatter(pts[:, 0], pts[:, 1], marker=mk, s=48,
                       facecolors="none", edgecolors="black", lw=LW,
                       label=lab)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("维度 1", fontsize=FS_SM)
        ax.set_ylabel("维度 2", fontsize=FS_SM)
        ax.set_title(title, fontsize=FS_SM)
        for s in ax.spines.values():
            s.set_linewidth(LW)
        ax.legend(loc="upper right", fontsize=FS_SM - 1, frameon=True,
                  edgecolor="black")

    # 在特征空间画出混淆类对椭圆
    from matplotlib.patches import Ellipse
    axes[0].add_patch(Ellipse((0.65, 0.55), 2.1, 1.2, angle=0,
                              fc="none", ec="black", lw=1.5, ls="--"))
    axes[0].text(0.65, -0.25, "$R_{\\mathrm{feat}}(\\mathrm{Eos}, \\mathrm{Mac}) \\geq \\tau$",
                 ha="center", fontsize=FS_SM - 1)

    axes[1].add_patch(Ellipse((2.05, 1.1), 2.0, 1.0, angle=0,
                              fc="none", ec="black", lw=1.5, ls="--"))
    axes[1].text(2.05, -0.25, "$R_{\\mathrm{morph}}(\\mathrm{Neu}, \\mathrm{Mac}) \\geq \\tau$",
                 ha="center", fontsize=FS_SM - 1)

    fig.suptitle(
        "图 5  双视图混淆检测：$P = \\{(i,j) \\mid R_{\\mathrm{feat}}\\geq \\tau \\vee R_{\\mathrm{morph}}\\geq \\tau\\}$",
        fontsize=FS_LG, weight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "图5_双视图混淆检测.png")


# =====================================================================
# 图 6 LW-Fisher 流程图
# =====================================================================
def fig_6_lw_fisher():
    fig, ax = plt.subplots(figsize=(9.5, 7))
    clean(ax, (0, 10), (0, 10))

    steps = [
        (5, 9.0, "输入：混淆对 $(c_i, c_j)$ 及两类 support 特征"),
        (5, 7.6, "计算均值 $\\mu_c$ 与样本协方差 $S_c$\n($c \\in \\{i, j\\}$)"),
        (5, 5.8, "计算收缩强度\n"
                  "$\\pi_c = \\sum_{p,q} \\mathrm{Var}(S_{c,pq})$\n"
                  "$\\gamma_c = \\| S_c - (\\mathrm{tr} S_c / D) \\cdot I \\|_F^2$"),
        (5, 4.05, "$\\rho_c = \\min(1, \\max(0, \\pi_c / (\\gamma_c \\cdot N)))$"),
        (5, 2.65, "$\\Sigma_c^{\\mathrm{LW}} = (1 - \\rho_c) S_c + \\rho_c \\cdot (\\mathrm{tr} S_c / D) \\cdot I$"),
        (5, 1.25, "$w_{ij} = (\\Sigma_i^{\\mathrm{LW}} + \\Sigma_j^{\\mathrm{LW}} + \\varepsilon \\cdot I)^{-1} \\cdot (\\mu_i - \\mu_j)$"),
        (5, -0.1, "单位化：$w_{ij} \\leftarrow w_{ij} / \\| w_{ij} \\|_2$"),
    ]
    widths = [6.6, 4.4, 7.0, 5.4, 5.8, 6.4, 4.4]
    heights = [0.7, 0.9, 1.5, 0.8, 0.8, 0.8, 0.7]
    centers = []
    for (cx, cy, txt), w, h in zip(steps, widths, heights):
        box(ax, cx - w / 2, cy - h / 2, w, h, txt, fs=FS_SM)
        centers.append((cx, cy))

    for i in range(len(centers) - 1):
        arrow(ax, (centers[i][0], centers[i][1] - heights[i] / 2),
              (centers[i + 1][0], centers[i + 1][1] + heights[i + 1] / 2))

    ax.set_ylim(-0.5, 10.5)
    ax.text(5, 10.2, "图 6  Ledoit-Wolf 自适应 Fisher 判别方向计算流程图",
            ha="center", fontsize=FS_LG, weight="bold")
    save(fig, "图6_LW_Fisher流程.png")


# =====================================================================
# 图 7 support 原型定向解耦几何示意
# =====================================================================
def fig_7_prototype_decouple():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    rng = np.random.default_rng(0)
    # 两类 support（初始重叠明显）
    mu_i = np.array([1.6, 1.9])
    mu_j = np.array([2.4, 2.1])
    S_i = rng.normal(loc=mu_i, scale=0.38, size=(10, 2))
    S_j = rng.normal(loc=mu_j, scale=0.38, size=(10, 2))

    # 假设 Fisher 方向 w_ij ≈ μ_i − μ_j 方向（与专利公式一致）
    w = (mu_i - mu_j)
    w = w / np.linalg.norm(w)
    alpha = 0.9  # 放大便于观察

    panels = [
        (axes[0], (S_i, S_j),
         "(a) 扰动前：两类 support 在特征空间中重叠", False),
        (axes[1], (S_i + alpha * w, S_j - alpha * w),
         "(b) 扰动后：沿 Fisher 方向 $\\pm\\alpha \\cdot w_{ij}$ 反向平移", True),
    ]
    for ax, pts, title, show_pert in panels:
        Pi, Pj = pts
        ax.scatter(Pi[:, 0], Pi[:, 1], marker="o", s=70,
                   facecolors="none", edgecolors="black", lw=LW,
                   label="类 i (Eos)")
        ax.scatter(Pj[:, 0], Pj[:, 1], marker="s", s=70,
                   facecolors="none", edgecolors="black", lw=LW,
                   label="类 j (Mac)")
        mi = Pi.mean(0); mj = Pj.mean(0)
        ax.scatter(*mi, marker="o", s=160, facecolors="black", zorder=5)
        ax.scatter(*mj, marker="s", s=160, facecolors="black", zorder=5)
        # 根据左右子图调整 μ 标注偏移，避免和散点重叠
        off_i = np.array([0.18, 0.28]) if show_pert else np.array([0.10, 0.20])
        off_j = np.array([0.18, 0.28]) if show_pert else np.array([0.10, 0.20])
        ax.annotate("$\\mu_i$", mi + off_i, fontsize=FS_SM, weight="bold")
        ax.annotate("$\\mu_j$", mj + off_j, fontsize=FS_SM, weight="bold")

        if show_pert:
            # 扰动箭头：从旧均值指向新均值
            mi0, mj0 = S_i.mean(0), S_j.mean(0)
            ax.annotate("", xy=mi, xytext=mi0,
                        arrowprops=dict(arrowstyle="->", lw=1.4,
                                        color="black", linestyle="--"))
            ax.annotate("", xy=mj, xytext=mj0,
                        arrowprops=dict(arrowstyle="->", lw=1.4,
                                        color="black", linestyle="--"))
            # 标记扰动符号（远离散点）
            ax.text(mi0[0] + (mi[0] - mi0[0]) / 2 - 0.08,
                    mi0[1] + (mi[1] - mi0[1]) / 2 + 0.38,
                    "$+\\alpha \\cdot w_{ij}$", fontsize=FS_SM, style="italic")
            ax.text(mj0[0] + (mj[0] - mj0[0]) / 2 + 0.08,
                    mj0[1] + (mj[1] - mj0[1]) / 2 - 0.42,
                    "$-\\alpha \\cdot w_{ij}$", fontsize=FS_SM, style="italic")
        else:
            # (a) 在下方标注 Fisher 方向向量
            mid = (mi + mj) / 2
            ax.annotate("", xy=mid + 0.9 * w, xytext=mid - 0.9 * w,
                        arrowprops=dict(arrowstyle="->", lw=1.6,
                                        color="black"))
            ax.text(mid[0] + 0.9 * w[0] + 0.10,
                    mid[1] + 0.9 * w[1] - 0.55,
                    "Fisher 方向 $w_{ij}$", fontsize=FS_SM, style="italic")

        ax.set_xlim(-0.5, 5.0); ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=FS_SM)
        for s in ax.spines.values():
            s.set_linewidth(LW)
        ax.legend(loc="lower right", fontsize=FS_SM - 1, frameon=True,
                  edgecolor="black")

    fig.suptitle("图 7  Support 原型沿 Fisher 方向定向解耦的几何示意图",
                 fontsize=FS_LG, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "图7_原型解耦几何.png")


# =====================================================================
# 图 8 BALF-Analyzer UI 示意图
# =====================================================================
def fig_8_ui():
    fig, ax = plt.subplots(figsize=(11, 6.8))
    clean(ax, (0, 16), (0, 10))

    # 顶栏
    box(ax, 0.2, 8.7, 15.6, 0.9,
        "BALF-Analyzer  |  数据集：data2_organized  |  用户：医师 A   [导出 JSON / CSV]",
        fs=FS_SM)

    # 左侧数据集/类别列表
    box(ax, 0.2, 0.2, 3.0, 8.2,
        "数据集列表\n\n• data2\n• MultiCenter\n• data1\n\n"
        "类别 (N=10)\n━━━━━━━━\n◉ Eos      10/10\n◉ Neu      10/10\n"
        "◉ Lym      10/10\n◉ Mac      10/10\n\n"
        "[开始批量预标注]\n[重置 Support]",
        fs=FS_SM)

    # 中央：细胞缩略图网格
    grid_x0, grid_y0 = 3.5, 0.2
    grid_w, grid_h = 8.5, 8.2
    ax.add_patch(Rectangle((grid_x0, grid_y0), grid_w, grid_h,
                           fc="white", ec="black", lw=LW))
    ax.text(grid_x0 + grid_w / 2, grid_y0 + grid_h - 0.35,
            "细胞预标注结果（鼠标悬停查看置信度，双击可修正）",
            ha="center", fontsize=FS_SM)
    # 6×5 缩略图方格
    cols, rows = 6, 5
    cw = (grid_w - 0.6) / cols
    ch = (grid_h - 1.0) / rows
    rng = np.random.default_rng(1)
    labels = ["Eos", "Neu", "Lym", "Mac"]
    conf_levels = [0.95, 0.88, 0.72, 0.51, 0.99, 0.83]
    for r in range(rows):
        for c in range(cols):
            x0 = grid_x0 + 0.3 + c * cw
            y0 = grid_y0 + 0.3 + r * ch
            ax.add_patch(Rectangle((x0, y0), cw - 0.1, ch - 0.3,
                                   fc="white", ec="black", lw=0.8))
            # 圆形代表细胞
            ax.add_patch(Circle((x0 + (cw - 0.1) / 2,
                                  y0 + (ch - 0.3) / 2),
                                 0.22, fc="white", ec="black", lw=0.8))
            lab = labels[rng.integers(0, 4)]
            conf = conf_levels[(r * cols + c) % len(conf_levels)]
            # 低置信度加虚线框以示需审核
            if conf < 0.6:
                ax.add_patch(Rectangle((x0 - 0.02, y0 - 0.02),
                                        cw - 0.06, ch - 0.26,
                                        fc="none", ec="black", lw=1.3,
                                        linestyle="--"))
            ax.text(x0 + (cw - 0.1) / 2, y0 + 0.1,
                    f"{lab} ({conf:.2f})", ha="center", fontsize=FS_SM - 2)

    # 右侧：统计 / 操作面板
    box(ax, 12.3, 0.2, 3.5, 8.2,
        "统计面板\n━━━━━━━━\n\n已预标注：263 个\n低置信度（< 0.6）：41\n\n"
        "类别分布：\n  Eos   10 (3.8%)\n  Neu   26 (9.9%)\n  Lym  145 (55.1%)\n"
        "  Mac   82 (31.2%)\n\n[审核低置信度]\n[确认全部]\n[撤销操作]",
        fs=FS_SM)

    ax.text(8, 9.7, "图 8  BALF-Analyzer Web 标注系统主界面示意图",
            ha="center", fontsize=FS_LG, weight="bold")
    # 底部图例
    ax.text(8, -0.15, "（虚线框表示置信度 < 0.6 需人工审核）",
            ha="center", fontsize=FS_SM - 1, style="italic")
    save(fig, "图8_UI界面.png")


# =====================================================================
# 图 9 实验结果对比（使用真实数据）
# =====================================================================
def fig_9_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = [
        "MB-kNN\n(基线)",
        "+Fisher\n(trace)",
        "+LW\n收缩",
        "+双视图\n交集",
        "+双视图并集\n(本发明)",
    ]
    mf1 = [0.7252, 0.7477, 0.7485, 0.7491, 0.7563]
    eos = [0.4465, 0.4920, 0.4933, 0.4903, 0.5018]

    x = np.arange(len(methods))
    w = 0.6

    hatches = ["", "///", "...", "xxx", ""]
    # 让最终方法为实心深色，其余为空心填充hatch
    for i, (v, h) in enumerate(zip(mf1, hatches)):
        ax1.bar(x[i], v, w, fc="white" if i < 4 else "#2b2b2b",
                ec="black", lw=LW, hatch=h)
    ax1.set_ylim(0.70, 0.78)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=FS_SM - 1)
    ax1.set_ylabel("宏平均 F1 (mF1)", fontsize=FS_SM)
    ax1.set_title("(a) 整体宏 F1", fontsize=FS_SM)
    for i, v in enumerate(mf1):
        ax1.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=FS_SM - 1)
    for s in ax1.spines.values():
        s.set_linewidth(LW)
    ax1.tick_params(width=LW)
    ax1.grid(axis="y", linestyle=":", lw=0.6, alpha=0.6)

    for i, (v, h) in enumerate(zip(eos, hatches)):
        ax2.bar(x[i], v, w, fc="white" if i < 4 else "#2b2b2b",
                ec="black", lw=LW, hatch=h)
    ax2.set_ylim(0.40, 0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=FS_SM - 1)
    ax2.set_ylabel("嗜酸性粒细胞 F1 (Eos F1)", fontsize=FS_SM)
    ax2.set_title("(b) 稀有类 Eos F1", fontsize=FS_SM)
    for i, v in enumerate(eos):
        ax2.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=FS_SM - 1)
    for s in ax2.spines.values():
        s.set_linewidth(LW)
    ax2.tick_params(width=LW)
    ax2.grid(axis="y", linestyle=":", lw=0.6, alpha=0.6)

    # 注释提升幅度
    ax1.annotate("", xy=(4, 0.7563), xytext=(0, 0.7252),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax1.text(2.0, 0.775, "+3.12%", fontsize=FS_SM, ha="center",
             weight="bold")
    ax2.annotate("", xy=(4, 0.5018), xytext=(0, 0.4465),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax2.text(2.0, 0.545, "+5.53%", fontsize=FS_SM, ha="center",
             weight="bold")

    fig.suptitle("图 9  BALF data2 数据集上各方法性能对比\n"
                 "(10-shot, 嵌套 5 折 CV × 5 种子, 共 25 次评估)",
                 fontsize=FS_LG, weight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "图9_实验结果对比.png")


# =====================================================================
if __name__ == "__main__":
    fig_1_pipeline()
    fig_2_cascade_seg()
    fig_3_dual_crop()
    fig_4_multi_backbone()
    fig_5_dual_view()
    fig_6_lw_fisher()
    fig_7_prototype_decouple()
    fig_8_ui()
    fig_9_results()
    print(f"\n全部附图已生成至：{OUT_DIR}")
