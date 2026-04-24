# 论文/专利素材图汇总（最终版）

生成时间：2026-04-24  
所有 PAMSR 对比图均为 **真实 CellposeSAM + PAMSR 运行结果**（非模拟）。

---

## 一、BALF 显微视野原图 + 分割掩码 + 中心坐标

**选用图像**：`data1_organized/images/train/1483.jpg`（原始分辨率 853×640 px）

**细胞构成**：
- Cell 0: **Macrophage**（巨噬细胞，临床意义主细胞）
- Cell 1: **Neutrophil**（中性粒细胞，炎性浸润指标）

**特点**：2 个独立无粘连细胞，位于画面中心区域，四周边距充足（最小边距 > 23%），满足论文配图要求。

### 输出文件（`balf_field/`）

| 文件名 | 说明 |
|--------|------|
| `1483_original.jpg` | 原始显微视野图 |
| `1483_mask_only.png` | 纯分割掩码图（按细胞类别着色） |
| `1483_mask_overlay.png` | 半透明掩码叠加原图 |
| `1483_annotated.png` | **推荐主图**：原图 + 掩码 + 检测框 + 十字中心 + 类别标签 |
| `1483_cell_centers.txt` | 核心细胞像素中心坐标与 BBox 文本文件 |

**细胞中心坐标**：
- Macrophage: (325.4, 290.7)
- Neutrophil: (436.4, 330.9)

---

## 二、FastAPI 标注工具界面截图

**用途**：嵌入论文图 3（或方法章节），展示复核队列 + SAM3 提示交互界面。

### 输出文件（`ui_mockup/`）

| 文件名 | 说明 |
|--------|------|
| `fastapi_review_queue_sam3_mockup.png` | 1600×1000 px 高分辨率模拟截图 |
| `fastapi_review_queue_sam3_mockup_1200.png` | 1200×750 px 论文嵌入用缩小版 |

**界面元素说明**：
- 左侧：Pending / Confirmed / Rejected 复核队列列表（含真实 BALF 缩略图）
- 中上：主视野画布，含 3 个 SAM3 Prompt 标记（P1/P2 正例点、P3 负例点）
- 中下：工具栏（Pan / Zoom / Positive Point / Negative Point / Box Prompt / Auto-mask / Clear）
- 右侧：SAM3 Prompt 列表 + 分类置信度条 + Confirm / Reject / Re-prompt 操作按钮

---

## 三、PAMSR 多尺度分割定性结果配图（真实运行结果）

**重要声明**：以下对比图均基于 `CellposeSAM` 真实运行生成，非模拟。  
**运行环境**：`cel` conda 环境，Cellpose 4.1.1 + CellposeSAM (`cpsam`)  
**通用方法参数**：
- Single-scale：diameter=50, cellprob_threshold=-3.0
- PAMSR (Ours)：primary_d=50, secondary_ds=[40, 65], consensus rescue, prob_thr=0.0

### 输出文件（`pamsr_real/`）

#### data2（主实验集，2048×1536）

| 文件名 | 说明 |
|--------|------|
| `pamsr_real_2022-06-10-14-09-32-87353.png` | **案例 1**：GT=77，PAMSR 救援 3 个边缘碎裂细胞 |
| `pamsr_real_2022-06-10-14-05-26-85638.png` | **案例 2**：GT=66，PAMSR 救援 1 个小淋巴细胞 |
| `pamsr_real_2022-06-10-14-03-51-27123.png` | **案例 3**：GT=67，PAMSR 救援 1 个被遮挡中性粒细胞 |
| `pamsr_real_2022-06-10-14-34-55-71733.png` | **案例 4**：GT=61，PAMSR 救援 1 个弱边界信号嗜酸性粒细胞 |
| `pamsr_real_summary_all_4cases.png` | **四组纵向拼接总图**（18 MB，适合整页排版） |

#### data1（临床 BALF，853×640）

| 文件名 | 说明 |
|--------|------|
| `pamsr_real_data1_1131.png` | **案例 5**：GT=4，PAMSR 救援 1 个被遮挡细胞，FN 1→0，零 FP 增长 |

#### MultiCenter（多中心数据，853×640）

| 文件名 | 说明 |
|--------|------|
| `pamsr_real_multicenter_1222.png` | **案例 6**：GT=15，PAMSR 救援 2 个 GT 细胞，TP 10→12，FN 5→3 |
| `pamsr_real_multicenter_1238.png` | 备用：GT=3，PAMSR 救援 1 个但 FP 同步增长（不推荐主图） |

#### WBC-Seg（外部血液涂片，4160×3120）

> ⚠️ **注意**：CellposeSAM 在 WBC-Seg 上表现极差（单图 TP<10/146），PAMSR 无法从极低的基线中救援出真正的 GT 细胞（rescued 均为额外 FP）。因此**不生成 WBC 单图 PAMSR 演示**，仅引用数据集级量化统计（见下方）。

每张对比图均包含 4 列：
- (a) Original：原始显微实拍图
- (b) Single-scale（青色轮廓）：单尺度分割结果
- (c) PAMSR (Ours)（绿色轮廓）：多尺度修复结果，**黄色圈标注被救援细胞**
- (d) Ground Truth：Polygon 标注真值（按类别着色）

### 六组案例的量化对比

| # | Dataset | Image | GT | Single TP/FP/FN | PAMSR TP/FP/FN | Rescued | Repair Note |
|---|---------|-------|----|-----------------|----------------|---------|-------------|
| 1 | data2 | 87353 | 77 | 56 / 13 / 21 | 59 / 13 / 18 | **3** | 多尺度共识救援 3 个图像边缘碎裂细胞 |
| 2 | data2 | 85638 | 66 | 63 / 5 / 3 | 64 / 5 / 2 | **1** | 次级尺度 d=40 找回主尺度遗漏的小淋巴细胞 |
| 3 | data2 | 27123 | 67 | 61 / 14 / 6 | 62 / 14 / 5 | **1** | PAMSR 救援巨噬细胞簇中被遮挡的中性粒细胞 |
| 4 | data2 | 71733 | 61 | 54 / 23 / 7 | 55 / 23 / 6 | **1** | 共识救援恢复弱边界信号的嗜酸性粒细胞 |
| 5 | data1 | 1131 | 4 | 3 / 0 / 1 | 4 / 0 / 0 | **1** | 找回被部分遮挡的巨噬细胞，零 FP 代价 |
| 6 | MultiCenter | 1222 | 15 | 10 / 1 / 5 | 12 / 2 / 3 | **2** | 多尺度找回 2 个分散小细胞，FN 5→3 |

### WBC-Seg 数据集级统计（无可行单图演示）

| 指标 | Single-scale | PAMSR | Δ |
|------|-------------|-------|---|
| Precision | 0.2537 | 0.2782 | **+0.0245** |
| Recall | 0.4733 | 0.5015 | **+0.0282** |
| F1 | 0.3301 | 0.3515 | **+0.0214** |

> 虽然单图 TP 极低导致难以展示直观的单细胞救援，但数据集级统计表明 PAMSR 仍带来稳定的 Precision/Recall 双提升。

### 辅助文件

| 文件名 | 说明 |
|--------|------|
| `pamsr_real_repair_log_table.png` | 修复日志表格图（含 GT / Single / PAMSR 指标对比） |
| `pamsr_real_summary.txt` | data2 逐图量化指标文本摘要 |
| `pamsr_real_final_datasets.txt` | data1 / MultiCenter / WBC 扫描结果摘要 |
| `pamsr_cross_dataset_best_cases.png` | **跨数据集最佳案例汇总图**（data1 + data2 + MultiCenter 三行并排） |

---

## 四、混淆矩阵热力图 + 逐样本预测文件

**实验配置**：Seed = 42，10-shot support，Adaptive kNN + Morphology  
**样本量**：n = 1316（data2 验证集全部细胞）  
**指标**：Accuracy = 0.852，Macro-F1 = 0.746

### 输出文件（`confusion/`）

| 文件名 | 说明 |
|--------|------|
| `per_sample_predictions_seed42.csv` | **逐样本预测文件**：含 sample_idx, true_label, pred_label, true_name, pred_name, confidence_margin, refined 标志, 各类别 score |
| `confusion_matrix_seed42.png` | **混淆矩阵热力图（计数版）**，300 dpi，适合论文直接插入 |
| `confusion_matrix_normalized_seed42.png` | **混淆矩阵热力图（行归一化百分比版）**，便于观察各类别 recall |
| `metrics_seed42.json` | 该种子下各类别 precision / recall / f1 及 macro 指标 JSON |

---

## 快速引用（LaTeX）

素材根目录统一为 `paper_materials/`：

```latex
% 混淆矩阵
\includegraphics[width=0.48\textwidth]{paper_materials/confusion/confusion_matrix_seed42.png}

% BALF 原图+掩码
\includegraphics[width=0.48\textwidth]{paper_materials/balf_field/1483_annotated.png}

% PAMSR 四组真实对比总图（单栏大图）
\includegraphics[width=\textwidth]{paper_materials/pamsr_real/pamsr_real_summary_all_4cases.png}

% PAMSR 跨数据集单图演示（按需选用）
\includegraphics[width=\textwidth]{paper_materials/pamsr_real/pamsr_real_data1_1131.png}
\includegraphics[width=\textwidth]{paper_materials/pamsr_real/pamsr_real_multicenter_1222.png}

% PAMSR 跨数据集汇总图（推荐补充材料图）
\includegraphics[width=\textwidth]{paper_materials/pamsr_real/pamsr_cross_dataset_best_cases.png}

% PAMSR 修复日志表格
\includegraphics[width=\textwidth]{paper_materials/pamsr_real/pamsr_real_repair_log_table.png}

% FastAPI 界面
\includegraphics[width=\textwidth]{paper_materials/ui_mockup/fastapi_review_queue_sam3_mockup_1200.png}
```

---

## 备注

- 所有 PNG 均按 300 dpi 保存，可直接用于期刊印刷。
- **PAMSR 对比图为真实 CellposeSAM 运行结果**，在 `cel` 环境（Python 3.9, Cellpose 4.1.1, PyTorch 2.5.1 + CUDA）下生成，黄色圈为代码自动标注的被救援细胞位置。
- data2 四组案例由 `generate_real_pamsr_v3.py` 生成；data1 / MultiCenter / WBC 由 `generate_real_pamsr_final_datasets.py` 生成。
- 如需在其他数据集上生成 PAMSR 对比图，可修改上述脚本中的数据集配置并重新运行。
