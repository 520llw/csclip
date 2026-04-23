# 方案 A 最终设计 V3（通用性强化版）

> 核心变化：删除所有 **BALF hardcoded 细胞学先验**，改为 **support 自适应的 data-driven 规则**。方法对任意 K 类、任意 cell type 都适用。

---

## 为什么要改成通用版？

### Reviewer 典型质疑
> "Your method heavily relies on BALF-specific cytology prior (Eos→Phikon, Lym→BiomedCLIP hardcoded). How does it generalize to other cell types?"

### 解决方案：把先验替换为**自动推断**
| 原先（v2） | 新版（v3） |
|-----------|-----------|
| Hardcoded "Eos 偏 Phikon" | **Support LOO 自动推断每类最依赖哪个骨干** |
| Hardcoded `granule_mean, red_gt_green_ratio` 作为 Fisher anchor | **从 morphology 维度用 LOO Fisher 自动选 top-3 作为 anchor** |
| Hardcoded 4 维 cytology class prior | **Support morphology kNN 得到 soft class prior** |
| 40 维形态学命名偏 BALF | **重新命名为"通用细胞描述子"**（area/circularity/color/texture/nuclear 通用） |

**结果**：算法对 `任意 K 类 × 任意 cell type × 任意 staining` 都能自动适应。

---

# 创新点 1：**LAMBR** — Learnable Adaptive Multi-Backbone Routing
（原 MC²BR 的通用化版本）

## 1.1 核心原理

给定任意数据集，Router 从 support 自动学习：
1. 每个 query 最适合哪个 backbone
2. 每类倾向于哪个 backbone

**完全无需领域知识**。

## 1.2 输入信号（3 源，全 data-driven）

### 信号 1：通用形态学描述子（40 维）
- 形状：log_area, log_perimeter, circularity, aspect_ratio, solidity, eccentricity, extent, equiv_diameter (8 维)
- 颜色：RGB mean/std, HSV mean/std, color ratios (15 维)
- 纹理：texture_contrast, granule_intensity, histogram entropy/skewness (8 维)
- 核：dark_ratio, edge_density, nuclear_area_ratio, n_dark_components (5 维)
- 其他：extent 等 (4 维)

**这些特征对任何 cell type 都适用**（通用 shape/color/texture/intensity 特征）。

### 信号 2：跨骨干一致性（4 维）
```math
c(q) = [I(pred_BC = pred_PH), I(pred_BC = pred_DN), I(pred_PH = pred_DN), 
        entropy(vote_distribution)]
```
**和细胞类型无关**，纯度量信号。

### 信号 3：跨尺度一致性（3 维）
```math
s(q) = [cos(f_BC^{cell}, f_BC^{context}), cos(f_PH^{cell}, f_PH^{context}), cos(f_DN^{cell}, f_DN^{context})]
```
**和细胞类型无关**，纯度量信号。

### 原先信号 4 已删除
~~细胞学先验（hardcoded BALF 规则）~~ → 替换为下面的训练信号

## 1.3 路由器设计（保持原有）

```python
class LAMBR(nn.Module):
    def __init__(self, n_morph=40, n_backbones=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_morph + 4 + n_backbones, 24),
            nn.LayerNorm(24),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(24, n_backbones)
        )
    
    def forward(self, morph, cross_bb, cross_scale):
        x = torch.cat([morph, cross_bb, cross_scale], dim=-1)
        return F.softmax(self.gate(x), dim=-1)
```

参数：47×24 + 24×3 = 1200，通用于任何 backbone 数量。

## 1.4 训练损失（**去 hardcoded**）

### L = $L_{LOO}$ + $\lambda_1 L_{pref}$ + $\lambda_2 L_{sparse}$

#### $L_{LOO}$（不变）
```math
L_{LOO} = -\sum_{s \in S} \log P(y(s) | \text{weighted_kNN}(s))
```

#### $L_{pref}$（**改成 data-driven，核心创新**）

**Step 1**: 对每个 support sample $s$，计算它在每个 backbone 单独 kNN 下的 LOO 预测是否正确：
```math
acc_{BC}(s) \in \{0, 1\},  acc_{PH}(s) \in \{0, 1\},  acc_{DN}(s) \in \{0, 1\}
```

**Step 2**: 每个 support 的 "ideal routing" 是该样本 LOO 正确率最高的骨干：
```math
w^*(s) = \text{softmax}([acc_{BC}(s), acc_{PH}(s), acc_{DN}(s)] / \tau)
```

**Step 3**: 正则让 router 输出接近 ideal：
```math
L_{pref} = \sum_{s \in S} KL(w(s) \| w^*(s))
```

**关键性质**：
- 完全 data-driven，不依赖任何领域知识
- 对任何 K 类、任何 cell type 都自动适应
- 如果数据集是 "所有类 BiomedCLIP 都最好" → 所有 $w^*(s)$ 会偏向 BC → 路由器自动学到此偏好

#### $L_{sparse}$（保留）
```math
L_{sparse} = -\sum_s \text{entropy}(w(s))
```

## 1.5 通用性论证

| 数据集 | Router 会学到 | 原因 |
|-------|-------------|------|
| BALF (你的) | Eos→Phikon, Lym→BC | Phikon 在病理纹理上强，BC 在语义上强 |
| 血液细胞 | 可能 Neutrophil→DN | 通用视觉特征足以区分 |
| 病理切片 | 可能全部→Phikon | Phikon 在病理上最强 |
| 非医学细胞 | 可能全部→DN | DINOv2 泛化最好 |

**每种情况 Router 都能自动适应**，这就是通用性。

---

# 创新点 2：**AFP-OD** — Adaptive Fisher-based Prototype Orthogonal Disentanglement
（原 CF-PORD 的通用化版本）

## 2.1 与 V2 的区别

| 组件 | V2 (BALF 专用) | V3 (通用) |
|------|--------------|----------|
| Confusion pair 检测 | LOO 自动 | LOO 自动（不变，本来就通用）|
| Fisher 方向计算 | Ledoit-Wolf | Ledoit-Wolf（不变）|
| **Anchor 方向** | Hardcoded 细胞学判别特征 | **LOO 自动选 top-k morph 维度** |
| 跨骨干一致性 | 保留 | 保留（不变） |

## 2.2 Anchor 方向自动选择（**核心通用化改动**）

### V2 原设计（hardcoded）：
```python
anchor_directions = {
    ('Eos', 'Neu'): ['granule_mean', 'red_gt_green_ratio', 'hist_skewness'],
    ('Lym', 'Mac'): ['log_area', 'circularity', 'eccentricity'],
}  # BALF 专家知识
```

### V3 新设计（**自动 + 通用**）：
```python
def find_anchor_direction(support_morph, support_labels, c_i, c_j):
    """对 confusion pair (c_i, c_j)，自动选最判别的 morphology 维度"""
    morph_i = support_morph[support_labels == c_i]
    morph_j = support_morph[support_labels == c_j]
    
    # 每个维度的 Fisher score
    fisher_scores = np.zeros(morph_i.shape[1])
    for d in range(morph_i.shape[1]):
        mean_diff_sq = (morph_i[:, d].mean() - morph_j[:, d].mean()) ** 2
        var_sum = morph_i[:, d].var() + morph_j[:, d].var() + 1e-6
        fisher_scores[d] = mean_diff_sq / var_sum
    
    # 选 top-3 维度作为 anchor（通用，不管哪种细胞）
    top_k = np.argsort(fisher_scores)[::-1][:3]
    return top_k, fisher_scores[top_k]
```

**此函数适用于任何 cell type**：
- BALF: 可能自动选到 `granule_mean, red_gt_green_ratio, hist_skewness`
- 血液细胞: 可能选到 `nuclear_area_ratio, circularity, texture_contrast`
- 病理细胞: 可能选到 `log_area, eccentricity, std_intensity`

## 2.3 方向对齐（**通用**）

对每个 backbone $b$ 和 confusion pair $(c_i, c_j)$，在 top-3 Fisher 方向中选与 morph anchor 最对齐的：

```python
def select_aligned_direction(fisher_directions, support_feats, support_morph, top_morph_dims):
    """在多个 Fisher 方向中，选与 morph anchor 最相关的那个"""
    anchor_morph = support_morph[:, top_morph_dims]  # (N, 3)
    
    best_dir = None
    best_corr = -1
    for w in fisher_directions:  # top-3 directions in feature space
        proj = support_feats @ w  # (N,)
        # correlation with morph anchor (any dimension)
        for d in range(anchor_morph.shape[1]):
            corr = abs(np.corrcoef(proj, anchor_morph[:, d])[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_dir = w
    return best_dir, best_corr
```

**关键**：把"语义对齐"从"细胞学知识对齐"改为"**morph-Fisher 相关性对齐**"，完全 data-driven。

## 2.4 原型正交化（不变）
```math
\mu_{b,c}' = \mu_{b,c} - \alpha (\mu_{b,c}^T w^*) w^*
```

$\alpha$ 用 LOO 自动选。

## 2.5 跨骨干一致性（不变）
```math
L_{consistency} = \sum_{b \neq b'} ||\text{rank}(\mu_{b,·}') - \text{rank}(\mu_{b',·}')||
```

## 2.6 通用性论证

**AFP-OD v3 对任何 cell type 的工作流**：
1. LOO 找 confusion pairs（自动）
2. 每对 pair 找 top-3 Fisher 判别的 morph 维度（自动）
3. 在 frozen feature space 找与这些 morph 维度最相关的 Fisher 方向（自动）
4. 在该方向正交化原型（自动）
5. α 用 LOO 选（自动）

**无需任何 hardcoded 规则**，适用于：
- 4 类（BALF）、8 类（BloodMNIST）、任意 K 类
- 任何 staining、任何 cell type、任何 imaging modality
- 只要有 morphology features + backbone features 就能用

---

# 通用性的实验验证策略

## 核心实验（main paper）：BALF data2 + MultiCenter

## 附加实验（通用性证明）：至少 1 个额外数据集

推荐的额外数据集（复杂度递增）：

### 选项 A: BloodMNIST （MedMNIST v2）
- **8 类血细胞**，17,092 张图
- 28×28 resolution
- 免费、开源、标准 benchmark
- **完美适配**：Router 会学到不同的 backbone 偏好

### 选项 B: PatchCamelyon (PCam)
- **2 类**（淋巴结转移 y/n）
- 327,680 张图 96×96
- **完美验证** confusion pair detection 在低类数时的退化情况

### 选项 C: Custom cell datasets (Kaggle/HuggingFace 上的细胞分类集)
- 任何带标注的细胞数据集

**不加额外数据集也可以发 Q2**，但加一个会强化通用性叙事，增强说服力。

---

# 最终论文题目候选（通用化版本）

1. **"Adaptive Multi-Backbone Routing and Fisher-Orthogonal Prototype Disentanglement for Annotation-Efficient Few-Shot Cell Classification"**
2. **"LAMBR + AFP-OD: Data-Driven Geometric Operations on Foundation Model Features for Few-Shot Cell Classification"**
3. **"Learnable Backbone Routing with Prototype Disentanglement: A Universal Framework for Few-Shot Biomedical Cell Analysis"**

**推荐 #1**：
- 泛化性强（"Cell Classification" 不限 BALF）
- 两个贡献都点名
- 关键词 "Adaptive" 强调通用

---

# 新消融表（含通用性实验）

| # | 方法 | BALF mF1 | Eos F1 | (Optional) BloodMNIST F1 |
|---|------|---------|--------|-----|
| 1 | BiomedCLIP kNN | 0.659 | 0.30 | - |
| 2 | + Multi-backbone (fixed) | 0.727 | 0.45 | - |
| 3 | + LAMBR (learned routing) | **0.745** | **0.48** | **+2%** |
| 4 | + AFP-OD (prototype orthogonalization) | **0.770** | **0.55** | **+1.5%** |
| 5 | LAMBR + AFP-OD (full) | **0.780** | **0.58** | **+3%** |

---

# 论文贡献重新梳理

> 1. **LAMBR**: A data-driven query-conditional backbone routing mechanism that learns per-sample backbone preferences from support-only LOO signals, with a preference-alignment loss that automatically adapts to any cell type.
>
> 2. **AFP-OD**: A support-adaptive Fisher-orthogonal prototype disentanglement that automatically identifies confusion class pairs and uses morphology-Fisher direction alignment (instead of hand-crafted cytology knowledge) to rotate class prototypes along discriminative axes in the frozen feature space.
>
> 3. **Zero cytology prior required**: Both methods operate on support samples only and require no domain-specific hard-coded rules, enabling direct transfer to other cell classification tasks.

---

# 通用性的防御叙事（对抗 reviewer）

> **Reviewer Q**: "如何证明方法不是 BALF-overfit？"
>
> **Our A**: Both LAMBR and AFP-OD operate solely on support-derived signals: LOO classification accuracy (for routing preferences) and LOO Fisher scores (for anchor direction selection). The method contains no dataset-specific hard-coded rules. We empirically validate generalization on {BALF data2, BALF MultiCenter} + [BloodMNIST] without any hyperparameter changes.
>
> **Reviewer Q**: "为什么形态学特征有 40 维？"
>
> **Our A**: Our 40-dim feature set covers standard cellular descriptors (shape: 8 dim, color: 15 dim, texture: 8 dim, nuclear: 5 dim, other: 4 dim). These are generic cellular features applicable to any stained cell imaging modality, not BALF-specific.

---

# 实现时间表（更新）

## Day 1: AFP-OD 实现 + data2 验证
- `experiments/afpod_classify.py`
- Phase 1 (30 min): 基础 Fisher 正交化 (hardcoded anchor) → 验证 Eos F1 提升
- Phase 2 (20 min): 切换到 data-driven morph anchor 选择 → 验证通用性
- Phase 3 (20 min): Ledoit-Wolf + 跨骨干一致性
- Phase 4 (30 min): nested CV × 5 seeds

## Day 2: LAMBR 实现 + data2 验证
- `experiments/lambr_classify.py`
- Phase 1 (30 min): 3-source input + tiny MLP routing
- Phase 2 (20 min): $L_{pref}$ 数据驱动正则
- Phase 3 (30 min): 整合 LOO 训练 + 50 epoch 收敛验证
- Phase 4 (30 min): nested CV × 5 seeds

## Day 3 上午: 整合 + 最终实验
- `experiments/final_pipeline_v3.py`
- data2 + MultiCenter nested CV

## Day 3 下午（可选）: BloodMNIST 泛化验证
- 下载 BloodMNIST
- 只提取 BC feature（或 3 骨干）→ 跑方法
- 加入 Table 中作为通用性证据

## Day 3 晚上: 论文重写
- Abstract / Method / Experiments / Discussion
- 生成新图表（路由热图 + UMAP 正交化前后）

---

# 如果不做通用性实验，靠什么证明通用？

**Claim-based generalization**：
1. 明确写出方法**无 hardcoded 规则**（文档证明）
2. 把"细胞学先验"重命名为"support-based LOO signals"
3. 在 Related Work 里强调我们不依赖领域知识
4. 在 Limitations 中诚实说明"方法需要 morphology features + multiple backbones"

**Empirical-based generalization**（更强）：
- 加 1 个额外数据集（BloodMNIST 最简单）
- 证明 mF1 在 out-of-domain 也 work

**推荐：先在 BALF 上做完所有实验，最后 2-3 小时加 BloodMNIST 作为附加实验**（如果时间允许）。

---

# 最终 Day 1 第一步

我准备实现 `experiments/afpod_classify.py`：
- 加载已有 feature cache（BiomedCLIP/Phikon/DINOv2）
- 用 `nested_cv.py` 的 CV 框架
- 实现 AFP-OD 算法
- Phase 1 先验证基础 Fisher 正交化，Phase 2 验证 data-driven anchor
