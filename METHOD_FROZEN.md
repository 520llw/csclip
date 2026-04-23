# BALF 细胞分析系统 —— 方法定档文档 (METHOD_FROZEN v1.0)

**定档日期**：2026-04-18
**状态**：🔒 方法冻结，后续只做实验验证、系统工程化、写作
**目标期刊**：CMIG / CMPB（二区，应用型 + 系统型）

---

## 0. 系统定位（一句话）

> 一套**训练自由（training-free）**的 BALF 细胞全链路分析系统：融合多视觉基础模型（BiomedCLIP + Phikon-v2 + DINOv2）+ 通用分割器（Cellpose + SAM3）+ 少样本分类（AFP-OD），通过 10-shot 人工标注即可完成全片细胞分类，显著降低细胞学诊断的标注成本。

## 1. 整体流水线（End-to-End Pipeline）

```
原始 BALF 图像（2048×2048 等）
        │
        ▼
┌──────────────────────────┐
│  S1. 细胞候选提取         │
│  Cellpose 4.1.1 (cyto3)   │    → 实例 mask, bbox
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  S2. Mask 精修            │
│  SAM3 (可选 refinement)   │    → 精细边界 mask
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  S3. 双尺度裁剪           │
│  cell (90%) + context 10% │    → 每个实例 2 个 crop
└──────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  S4. 多骨干 VFM 特征提取（并行）          │
│  ┌─BiomedCLIP (512d)  ─┐                 │
│  ┼─Phikon-v2   (1024d) ┼─→ 拼接         │
│  └─DINOv2-S   (384d)  ─┘                 │
│  + 形态学特征（40d, 数据驱动）            │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  S5. 少样本分类：AFP-OD   │
│  (仅需 10-shot support)   │    → 类别概率
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  S6. 标注工具人工审核     │
│  labeling_tool/ (FastAPI)  │    → 最终标签入库
└──────────────────────────┘
```

**关键特性**：
- **零训练**：S1/S2/S3/S4 全部冻结模型参数，S5 只在 support 集上做轻量化推理侧处理
- **即插即用**：每个模块可替换（Cellpose → StarDist，BiomedCLIP → PLIP，AFP-OD → Tip-Adapter）
- **形态学可解释**：40 维形态学特征贯穿始终，提供可解释性

---

## 2. 模块详细定档

### 2.1 S1/S2 分割：Cellpose + SAM3 级联 🔒

| 项 | 定档值 |
|---|-------|
| 主分割器 | Cellpose 4.1.1（`cyto3` 预训练权重） |
| Mask 精修 | SAM3（meta, 2025Q4，`sam3_h` checkpoint ~3.4GB） |
| 输入尺寸 | 原图 resize 最长边到 2048 |
| Cellpose 参数 | `diameter=None`（自动估计）, `flow_threshold=0.4`, `cellprob_threshold=0.0` |
| SAM3 prompt | 由 Cellpose mask 的 bbox 自动转化（无人工 prompt） |
| 级联策略 | Cellpose 输出 mask → bbox → SAM3 refinement → IoU > 0.5 则采用 SAM3 结果 |

**不改的理由**：Cellpose 已是细胞分割事实标准；SAM3 作为 refinement 而非主分割器，风险可控。

**论文要补的消融**（非方法改动，仅验证）：
- 纯 Cellpose vs 纯 SAM3 vs Cellpose+SAM3 级联 的 **mean IoU / mAP@50**
- 在有 GT mask 的子集上（若 data2 无 GT mask，需手工标 30 张做验证集）

### 2.2 S3 双尺度裁剪 🔒

| 项 | 定档值 |
|---|-------|
| 细胞裁剪 | bbox 向外扩 10% → resize 到 224×224 |
| 上下文裁剪 | bbox 向外扩 50% → resize 到 224×224 |
| 融合方式 | 两尺度特征各 encode 后按 `0.9 · cell + 0.1 · context` 加权 |

**依据**：`biomedclip_fewshot_support_experiment.py` 中已验证该比例在 BALF 上最优。

### 2.3 S4 多骨干 VFM 特征提取 🔒

| 骨干 | 维度 | 权重 | 领域 |
|------|------|------|------|
| BiomedCLIP (open_clip 3.3.0) | 512 | **0.42** | 生物医学图文对齐 |
| Phikon-v2 (HuggingFace) | 1024 | **0.18** | 病理专用 ViT |
| DINOv2-Small | 384 | **0.07** | 通用自监督 |
| 形态学（数据驱动 40d）| 40 | **0.33** | 面积/周长/圆度/颜色/偏心率等 |

**融合公式**：
```
score(q, s) = 0.42·⟨f_BC(q), f_BC(s)⟩
            + 0.18·⟨f_PH(q), f_PH(s)⟩
            + 0.07·⟨f_DN(q), f_DN(s)⟩
            + 0.33·(1 / (1 + ‖m(q) - m(s)‖_z))
```
其中 `m(·)` 为 z-score 归一化后的形态学向量。

**权重确定**：通过 data2 上 validation grid search 得到，**冻结用于所有数据集**，不再 tune。

**论文要补的消融**：
- 单骨干 / 双骨干 / 三骨干 的 mF1 对比，证明多骨干必要性
- 去掉形态学（mw=0）对 Eosinophil F1 的影响

### 2.4 S5 少样本分类：AFP-OD P3c 🔒

**全称**：Adaptive Fisher-based Prototype Orthogonal Disentanglement with Dual-View Confusion Detection

**核心思想**：在 kNN 分类前，对 support 样本沿 Fisher 判别方向做轻微扰动，把易混淆的类对拉开。

#### 算法定档伪代码

```
输入：support 特征 S = {S_c}，support 形态 M = {M_c}，query 特征 Q
      类别集合 C，阈值 τ=0.15，扰动强度 α=0.10，kNN k=7

# Step 1: 双视图混淆对检测（feature + morph）
R_feat = LOO_kNN_cosine(S, C, k=5)        # (c_i, c_j) → 混淆率
R_morph = LOO_kNN_euclidean(Z(M), C, k=5)  # Z = z-score
Pairs = { (c_i, c_j) | R_feat ≥ τ OR R_morph ≥ τ }   # union 模式

# Step 2: 逐混淆对计算 Fisher 方向并扰动 support
for (c_i, c_j) in Pairs:
    Σ_i = LedoitWolf(S_{c_i});  Σ_j = LedoitWolf(S_{c_j})
    w = (Σ_i + Σ_j + εI)^{-1} · (μ_i - μ_j);  w ← w / ||w||
    S_{c_i} += α · w;  S_{c_j} -= α · w

归一化所有 S_c 到单位球面

# Step 3: Multi-Backbone kNN 分类
for each q ∈ Q:
    for each class c:
        vs = 0.42·⟨S_{BC,c}, q_BC⟩ + 0.18·⟨S_{PH,c}, q_PH⟩ + 0.07·⟨S_{DN,c}, q_DN⟩
        ms = 1 / (1 + ||Z(M_c) - Z(q_m)||)
        score_c = mean(top_k(vs + 0.33·ms))
    pred(q) = argmax_c score_c
```

#### 关键超参数（冻结）

| 参数 | 值 | 来源 |
|------|---|------|
| 混淆对阈值 τ | 0.15 | data2 扫参最优 |
| 扰动强度 α | 0.10 | data2 扫参最优（0.05/0.10/0.20 中最好）|
| kNN 的 k | 7 | 与 MB_kNN baseline 一致 |
| 协方差估计 | Ledoit-Wolf（解析收缩）| Phase 2 确认 > 固定 trace shrinkage |
| 检测模式 | dual-view **union** | Phase 3c 确认 > intersection / feature-only |
| 检测用骨干 | BiomedCLIP | 默认最强骨干 |

#### 对比消融链条（论文中保留的对比实验）

| 方法 | mF1 | Eos F1 | ΔmF1 |
|------|-----|--------|------|
| MB_kNN (baseline) | 0.7252 | 0.4465 | — |
| + Fisher (trace shrink) | 0.7477 | 0.4920 | +2.25% |
| + Ledoit-Wolf shrinkage | 0.7485 | 0.4933 | +2.34% |
| + Morph-anchored blend（**负面结果**）| 0.7425 | 0.4789 | +1.74% |
| + Dual-view intersection | 0.7491 | 0.4903 | +2.39% |
| **+ Dual-view union（P3c，最终方法）** | **0.7563** | **0.5018** | **+3.12%** |

**不再扩展的方向**（明确放弃，避免精力分散）：
- ❌ Phase 4 cross-backbone rank consistency
- ❌ Morph-anchored PLS direction（已验证负面）
- ❌ 自适应 α_ij（工程复杂度 vs 收益不划算）
- ❌ 核 Fisher / SVM margin 方向替代（已足够好）

### 2.5 S6 标注工具 🔒

| 组件 | 定档 |
|------|------|
| 后端 | FastAPI + SQLite |
| 前端 | 原生 HTML/JS/CSS（无框架依赖）|
| 核心功能 | 数据集管理 / 自动预标注 / 人工审核 / 导出 |
| 支持数据 | 单张图 / 批量文件夹 / 已标注重新审核 |
| 入口 | `python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000` |

**要补的工程项**（不是方法改动）：
- ✅ 已有：数据集管理、标注、分类
- ⏳ 待加：**"一键全片自动预标注"** 按钮（批量走完整 S1→S5 pipeline）
- ⏳ 待加：标注节省统计面板（已审核 / 自动通过 / 待审核）
- ⏳ 待加：演示视频录制脚本

---

## 3. 论文中的贡献声明（Contribution Statement）

```
We present BALF-Analyzer, the first training-free end-to-end system for
few-shot BALF cell classification that combines multiple vision foundation
models with classical statistical disentanglement. Our contributions are:

(1) A cascaded segmentation pipeline (Cellpose + SAM3) that extracts
    cell instances without any cell-specific training.

(2) A multi-backbone VFM ensemble (BiomedCLIP + Phikon-v2 + DINOv2)
    augmented with 40-dimensional data-driven morphological features,
    providing complementary biomedical, pathological, and self-supervised
    cues for few-shot classification.

(3) AFP-OD (Adaptive Fisher-based Prototype Orthogonal Disentanglement),
    a novel post-hoc support-set regularization technique that detects
    confusable class pairs via dual-view (feature + morphology) leave-one-out
    k-NN and pushes them apart along Ledoit-Wolf-shrunk Fisher directions,
    yielding +3.12% mF1 / +5.53% Eosinophil F1 over the multi-backbone kNN
    baseline in 10-shot nested cross-validation.

(4) An open-source FastAPI-based annotation system with one-click
    batch auto-labeling, demonstrated to reduce manual annotation workload
    by XX% on BALF cytology workflows (quantified in §X).
```

---

## 4. 复现协议（Reproducibility）

### 4.1 环境
- **硬件**：1× GPU（≥8GB，实验用 12GB）
- **系统**：Linux（`cel` conda 环境，Python 3.9.21，PyTorch 2.5.1+CUDA12）
- **关键依赖版本**：
  - `cellpose == 4.1.1`
  - `open_clip_torch == 3.3.0`
  - `sam3`（本地源码安装自 `/home/xut/csclip/sam3`）
  - `numpy, scipy, scikit-learn, scikit-image, opencv-python, pillow, monai`

### 4.2 评估协议（冻结）
- **主数据集**：`cell_datasets/data2_organized/`（180 张，4 类有效：Eosinophil/Neutrophil/Lymphocyte/Macrophage）
- **Support 采样**：`N_SHOT = 10`，per-class 随机抽取
- **评估**：**Nested 5-fold CV × 5 seeds = 25 次评估**（seeds = `[42, 123, 456, 789, 2026]`）
- **指标**：mF1（主指标）、Eosinophil F1（关键痛点类）、per-class F1、accuracy
- **代码入口**：`experiments/afpod_classify.py`

### 4.3 跨数据集协议（待执行，不改方法）
- `data1_organized/` (2698 张)
- `MultiCenter_organized/` (2372 张)
- 使用**完全相同的超参数和方法**（α=0.10, τ=0.15, k=7, union mode），不做 per-dataset tuning

---

## 5. 与 SOTA 的定位（论文 Related Work 用）

| 方法 | 类型 | 我们的关系 |
|------|------|-----------|
| Cellpose（Nat. Methods 2021）| 通用细胞分割 | **使用**，作为 S1 |
| SAM / SAM2 / SAM3（Meta）| 通用分割大模型 | **使用**，作为 S2 |
| BiomedCLIP (Microsoft 2023) | 医学图文模型 | **使用**，作为 F1 骨干 |
| Phikon-v2 (Owkin 2024) | 病理 ViT | **使用**，作为 F2 骨干 |
| DINOv2 (Meta 2023) | 通用自监督 | **使用**，作为 F3 骨干 |
| Tip-Adapter (ECCV 2022) | CLIP few-shot | **对比 baseline**（待实现）|
| ProtoNet (NeurIPS 2017) | 经典 few-shot | **对比 baseline**（待实现）|
| FSOD-VFM (arXiv 2026) | VFM few-shot detection | **理念相似**，但我们面向分类而非检测 |
| SADC+ATD（本组前作，若有）| BALF 专用 | **对比 baseline**（如存在）|

---

## 6. 方法冻结清单（Checklist）

- [x] 分割架构（Cellpose 4.1.1 + SAM3 级联）
- [x] 双尺度裁剪比例（10% + 50%）
- [x] 多骨干权重（0.42 / 0.18 / 0.07 / 0.33）
- [x] 40 维形态学特征集合
- [x] AFP-OD 超参数（α=0.10, τ=0.15, k=7, LW, union）
- [x] 评估协议（nested 5-fold × 5 seeds）
- [x] 主指标定义（mF1, Eos F1）

🔒 **本文档签署后，以上所有项目不再修改**。若跨数据集验证失败（P3c 在 data1/MultiCenter 上 ΔmF1 < +1%），才允许回退到本文档修订 AFP-OD 参数，其他模块保持不动。

---

## 7. 后续工作清单（全部为非方法工作）

### Phase A：实验补全（2 周）
- [ ] 跨数据集验证：data1 + MultiCenter
- [ ] 标注节省曲线：1/3/5/10/20/50-shot
- [ ] 分割消融：Cellpose vs SAM3 vs 级联
- [ ] 多骨干消融：7 种组合（3 单 + 3 对 + 1 全）
- [ ] Baseline 补强：Tip-Adapter、ProtoNet、Cellpose+LR

### Phase B：系统工程（1 周）
- [ ] Web 工具"一键批量预标注"
- [ ] 端到端 demo pipeline 脚本
- [ ] 推理速度 / 显存报告
- [ ] GitHub 仓库整理 + README + 安装文档
- [ ] 演示视频（3-5 分钟）

### Phase C：写作（1 周）
- [ ] Introduction + Related Work
- [ ] System Overview（核心章节，占 30% 篇幅）
- [ ] Method (AFP-OD)（占 20% 篇幅）
- [ ] Experiments + Case Study（占 30% 篇幅）
- [ ] Discussion + Conclusion + 图表

### 目标期刊（按优先级）
1. **Computerized Medical Imaging and Graphics (CMIG)** — 二区，IF 5.4
2. **Computer Methods and Programs in Biomedicine (CMPB)** — 二区，IF 4.9
3. **Artificial Intelligence in Medicine (AIIM)** — 二区，IF 6.1

---

**文档维护者**：Cascade + 用户
**下次更新时间**：Phase A 跨数据集结果出来后（如验证失败才更新方法部分）
