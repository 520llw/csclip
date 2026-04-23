# 论文待补数据清单

**文档用途**：用户后续补充的数据入口，按优先级排序。已完成部分见 `EXPERIMENT_RESULTS_SUMMARY.md`。

---

## P0 - 必须补（否则影响论文可接受性）

### 1. BALF data2 的独立分割 GT（30 张手工标注）
- **理由**：当前 BALF 分割仅用 WBC-Seg 间接验证，审稿人可能质疑。
- **需求**：挑选 data2 验证集 30 张图，手工标注 WBC 多边形（约 1–2 小时）。
- **产出指标**：BALF 原生数据上的分割 P/R/F1，填入论文 §4.6。
- **占位位置**：`paper_full_zh.md` §5.4 局限性第 4 点。

### 2. 跨数据集（MultiCenter）的 AFP-OD 调优结果
- **理由**：当前 AFP-OD 在 MultiCenter 上与 MB-kNN 持平（0.3190），论文中诚实呈现，但审稿人可能要求改进。
- **需求**：
  - 对 MultiCenter 调优 $\tau$（0.10 / 0.12 / 0.15 / 0.18 / 0.20）
  - 测试类敏感 $\tau_c$
  - 测试自适应 $\alpha_{ij}$（按 support 可分性）
- **产出**：至少一个变体在 MultiCenter 上的 mF1 ≥ 0.34（超越 NCM）。
- **占位位置**：`paper_full_zh.md` 表 6 新增"AFP-OD (adaptive τ)"行。

### 3. Support 标注时间的真实测量
- **理由**：§4.8 中"5 分钟选 support"为估计值，需由真实用户计时验证。
- **需求**：2–3 名医师在标注系统中分别完成一次 10-shot support 选择流程，记录时间。
- **产出**：真实平均时间 ± 标准差。
- **占位位置**：§4.8 标注效率段。

---

## P1 - 强烈建议补（提升论文分量）

### 4. 临床前瞻性小规模验证
- **理由**：CMIG 偏好带临床样本的工作。
- **需求**：1 家合作医院提供 ≥ 20 份新 BALF 样本，系统完成分类后由医师 review，报告错误率与标注时间节省。
- **产出**：新增 §4.9 临床试点章节。

### 5. 与本组之前发表方法的比较（若存在）
- **理由**：论文中应体现对既有工作的超越。
- **需求**：如存在同组前期 BALF 分类工作（SADC/ATD 之外），加入表 2 对比。
- **占位位置**：表 2 新增 "SADC+ATD（本组 v1）" 行。

### 6. 推理速度的完整 breakdown
- **理由**：§4.8 当前只给总时间，细分更有说服力。
- **需求**：在同一测试机分别计时 Cellpose / SAM3 / 三骨干 / 形态学 / AFP-OD，平均 10 张图。
- **产出**：§4.8 新增"运行时性能 breakdown"表。

---

## P2 - 锦上添花（可作为 Supplementary）

### 7. 形态学特征重要性排序（SHAP / Fisher 可视化）
- **理由**：提升可解释性叙事。
- **需求**：用 Fisher 得分或 mutual information 对 40 维形态学排序，保留 top-10 热力图。
- **产出**：Supplementary 图。

### 8. BC/PH/DN 特征 t-SNE 可视化
- **理由**：直观展示多骨干互补。
- **需求**：data2 验证集 1316 细胞在三骨干下的 t-SNE 各 1 张。
- **产出**：Supplementary 图 S1。

### 9. AFP-OD 扰动前后的 support 嵌入可视化
- **理由**：让读者"看见"解耦效果。
- **需求**：UMAP 投影 support 集，对比 $\alpha=0$ vs $\alpha=0.10$。
- **产出**：图 7 的补充示意（可合并）。

### 10. 其他 VFM 的对比（可选）
- **理由**：说明选择 BC+PH+DN 的合理性。
- **需求**：测试 CLIP / MedCLIP / UNI / PathCLIP 等作为替代骨干，比较 mF1。
- **产出**：表 4 扩展。

---

## P3 - 投稿前再处理（不阻塞初稿）

### 11. 参考文献完整版
- 目前标记为 `[CITE]`，投稿前需按 CMIG 格式整理约 30–40 条。
- 优先顺序：Cellpose → SAM → BiomedCLIP/Phikon/DINOv2 → ProtoNet/Tip-Adapter → Fisher/LW → BALF 相关工作。

### 12. 英文翻译
- 当前为中文全稿，投稿需翻译为英文。
- 建议：先投二区英文期刊（CMIG 必须英文），保留中文版作为专利/教学素材。

### 13. 图 9 合成（骨干消融 + N-shot 曲线）
- 当前论文表格形式存在；如需图形，脚本：`experiments/figures/ablation_classification.png` 已有骨干柱状图，N-shot 曲线需新绘。
- 建议由现有数据直接 matplotlib 出图，不需再实验。

### 14. 图 8 （UI 界面）补真实截图
- 当前 `patent_figures/图8_UI界面.png` 为示意图，投稿前补系统真实截屏。

### 15. Cover letter / Highlights
- CMIG 要求 3–5 条 Highlights，每条 ≤ 85 字符。建议：
  - First training-free BALF cell few-shot classification framework combining multi-VFM features.
  - AFP-OD uses dual-view confusion detection and LW-shrunk Fisher directions to disentangle support prototypes.
  - mF1 +3.12% and Eos F1 +5.53% over the multi-backbone kNN baseline in nested 5-fold CV.
  - Independent WBC-Seg validation confirms industrial-grade segmentation (F1 = 0.887).
  - Open-source FastAPI annotation system reduces cell labeling by 99.25%.

---

## 状态看板（随时更新）

- [ ] P0-1 BALF 原生分割 GT
- [ ] P0-2 MultiCenter AFP-OD 调优
- [ ] P0-3 真实用户标注时长
- [ ] P1-4 临床前瞻性验证
- [ ] P1-5 本组前作对比
- [ ] P1-6 推理速度 breakdown
- [ ] P2-7 形态学特征重要性
- [ ] P2-8 多骨干 t-SNE
- [ ] P2-9 AFP-OD 嵌入可视化
- [ ] P2-10 其他 VFM 对比
- [ ] P3-11 参考文献完整化
- [ ] P3-12 英文翻译
- [ ] P3-13 图 9 合成
- [ ] P3-14 UI 真实截图
- [ ] P3-15 Cover letter / Highlights

---

**维护说明**：每完成一项时在 `paper_full_zh.md` 中对应位置删除 `[数据占位]` 并填入真值，同时在本文档打勾。
