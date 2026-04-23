# BALF-Analyzer 论文框架

> 目标期刊：CMIG（中科院二区）
> 定位：BALF 场景下的 training-free few-shot 细胞分析框架
> 参考范式：FSOD-VFM（arXiv:2602.03137）—— 仅作为范式参考，本文做独立的 BALF 场景创新
> 维护说明：本文件是论文结构的**唯一骨架来源**。数字引用以
> `EXPERIMENT_RESULTS_SUMMARY.md`、`MULTI_DATASET_RESULTS_SUMMARY.md`、
> `MULTISCALE_EXPERIMENT_REPORT.md` 为准。
> 更新时间：2026-04-21

---

## 0. 一句话立意

我们面对的不是"分类点创新"或"分割点创新"的问题，
而是"**如何用极少标注（10-shot）+ 纯基础模型组合 + 完全 training-free 的方式，
在真实临床 BALF 细胞学场景中实现可落地的自动化分析**"。

这个立意决定了全文所有章节围绕一件事展开：
**架构 × 少样本 × 跨数据集 × 临床可落地**。

---

## 1. 工作核心定位

### 1.1 这不是什么
- 不是"训练了一个新分割网络"的论文
- 不是"提出了一个新 few-shot 分类器"的论文
- 不是"刷榜"的论文

### 1.2 这是什么
- 一套 **training-free** 的 **BALF 细胞少样本分析框架**
- 在 **5 个数据集** 上系统验证
- 对应 **FSOD-VFM 范式在医学 BALF 场景的映射与改造**

### 1.3 与 FSOD-VFM 的差异化（不能照抄）

| 层级 | FSOD-VFM | 本文 |
|------|----------|------|
| Proposal 层 | UPN + graph diffusion 重加权 | **PAMSR**：主尺度锚定 + 多尺度一致性救援 |
| Prototype 层 | DINOv2 + nearest-class | **AFP-OD**：双视图混淆检测 + LW Fisher 定向解耦 |
| 特征层 | 单骨干 DINOv2 | **BC + PH + DN + 40 维形态学** 四路互补 |
| 交互层 | 无 | **SAM3 + 十三点 / 框 / 人工修正** 的人机协同精修 |
| 验证层 | Pascal / COCO / CD-FSOD | **data2 / data1 / WBC-Seg / PBC / MultiCenter** 五数据集三层证据 |
| 场景 | 通用目标检测 | **真实 BALF 临床场景 + 多中心挑战** |

**核心差异**：我们是"医学临床 + 多层修复 + 多骨干 + 跨中心真实挑战"，
不是通用 few-shot 检测的医学复刻。

---

## 2. 14 条创新点全量盘点

### 系统架构层
1. **面向 BALF 的 training-free 10-shot 全链路分析框架**（⭐⭐⭐⭐⭐）
2. **两层修复架构**：proposal 层 PAMSR + prototype 层 AFP-OD（⭐⭐⭐⭐⭐）

### 分割方法层
3. **PAMSR**（Primary-Anchor Multi-Scale Rescue）（⭐⭐⭐⭐）
4. **领域自适应的 CellposeSAM 参数标定流程**（⭐⭐⭐）

### 分类方法层
5. **AFP-OD**（Adaptive Fisher Prototype with Oriented Decoupling）（⭐⭐⭐⭐）
6. **四路异构特征互补**（BC + PH + DN + 40 维形态学）（⭐⭐⭐⭐）
7. **双尺度细胞裁剪**（cell 90% + context 10%）（⭐⭐⭐）

### 约束 / 先验层
8. **12 维形态学硬约束规则**（⭐⭐⭐）

### 人机协同层
9. **SAM3 驱动的多模式精修接口**（text / box / 13-point / 自由点击）（⭐⭐⭐）
10. **置信度驱动的人机审核流**（⭐⭐⭐）
11. **开源 FastAPI 标注 / 审核 Web 系统**（⭐⭐⭐）

### 验证层
12. **三层证据金字塔验证范式**（主 / 临床金标准 / 外部泛化）（⭐⭐⭐⭐）
13. **Nested 5-fold × 5 seeds 严格少样本协议**（⭐⭐）

### 负面发现
14. **揭示 BALF 域下文本 zero-shot 失败**（⭐⭐⭐）

---

## 3. 进贡献列表的 4 条

| 排序 | 贡献 | 对应创新点 |
|------|------|------------|
| **C1** | Training-free 10-shot BALF 全链路框架 + 两层修复架构 | 1 + 2 |
| **C2** | PAMSR（proposal 层多尺度救援） | 3 |
| **C3** | AFP-OD（prototype 层定向解耦） | 5 |
| **C4** | 人机协同系统 + 三层证据验证范式 | 9 + 10 + 11 + 12 |

创新点 4 / 6 / 7 / 8 / 13 / 14 **进正文 Method 与 Experiments**，
但**不进贡献列表**（避免贡献稀释）。

---

## 4. 论文章节骨架

### Title 候选
- **BALF-Analyzer: A Training-Free Few-Shot Framework for BALF Cell Segmentation and Classification**

### Abstract（Few-shot 社区风格，不放百分比 headline）
- 背景：BALF 诊断价值与标注瓶颈
- 挑战：域内标注稀缺 + 类不平衡 + 域偏移 + 文本 zero-shot 失败
- 方法：training-free，10-shot，基础模型组合
  - Cascaded segmentation (Cellpose + SAM3 refinement)
  - Multi-backbone features (BC + PH + DN + morphology)
  - PAMSR (proposal-level multi-scale rescue)
  - AFP-OD (prototype-level oriented decoupling)
  - Human-in-the-loop review
- 结果（骨架数字）：
  - `data2`: mF1 0.7563，Eos F1 0.5018
  - `WBC-Seg`: 独立金标准分割 F1 0.8874
  - `data1`（真实临床金标准）：分割 F1 0.6532，分类 mF1 0.511
  - `PBC`（外部泛化）：mF1 0.8577
  - PAMSR 三数据集一致 +0.2% / +0.4% / +0.8%
- 结论：10-shot 即可端到端落地

### 1. Introduction
- BALF 临床意义 + 人工判读瓶颈
- 现状：全监督需千级标注；文本 zero-shot 在 BALF 对齐失败；tip-adapter 等 40 样本过拟合
- 研究空白：医学 few-shot 缺乏基础模型组合式范式
- 贡献列表（4 条，即 §3）

### 2. Related Work
- 细胞分割基础模型：Cellpose、SAM 系列
- 医学视觉 / 语义基础模型：BiomedCLIP、Phikon-v2、DINOv2、DINO-Bloom
- Few-shot 分类：Tip-Adapter / EM-Dirichlet / Label Propagation / Linear Probe
- Foundation model composition for few-shot：FSOD-VFM 及相关
- BALF / WBC 分析

### 3. Method
#### 3.1 Overall Architecture
- 流程图（图 1）：Image → Cellpose → (SAM3 refinement 可选)
  → 多骨干特征 → AFP-OD 分类 → 人机审核
- 设计原则：training-free、两层修复（proposal 层 + prototype 层）

#### 3.2 Cascaded Segmentation Frontend
- Cellpose 4.1.1 (cpsam)
- 领域参数适配（data-specific diameter / cellprob）
- SAM3 作为人机协同阶段的精修工具（定位为交互优化而非主自动增益）

#### 3.3 PAMSR: Primary-Anchor Multi-Scale Rescue
- 动机
- 算法（主尺度 + 辅尺度 + 共识门控 + 置信筛选）
- 伪代码
- 复杂度

#### 3.4 Multi-Backbone Feature Extraction
- BiomedCLIP（语义）
- Phikon-v2（病理纹理）
- DINOv2-S（几何结构）
- 40 维形态学
- 双尺度裁剪（cell 90% + context 10%）
- 融合方式

#### 3.5 AFP-OD: Adaptive Fisher Prototype with Oriented Decoupling
- 动机：support 空间类间混淆
- 双视图混淆检测
- Ledoit-Wolf 收缩 Fisher 方向
- 原型定向扰动（α, conf_threshold）
- 伪代码

#### 3.6 Human-in-the-Loop Refinement
- 低置信度审核阈值
- SAM3 交互式精修（框 / 十三点 / 人工框）
- FastAPI 系统

### 4. Experiments

#### 4.1 Datasets（5 个）
- `data2_organized`：BALF 主数据集
- `data1_organized`：真实临床 + 分割金标准
- `WBC-Seg`：独立分割金标准
- `PBC`：外部分类泛化
- `MultiCenter`：跨中心挑战

#### 4.2 Implementation
- 硬件 / 环境 / 版本
- 超参表

#### 4.3 Protocol
- 10-shot per class
- Nested 5-fold × 5 seeds
- IoU ≥ 0.5 分割匹配

#### 4.4 Main Classification Results（`data2`）
- 表：vs 8 种 SOTA
- AFP-OD `+3.12%` mF1，Eos F1 `+5.53%`

#### 4.5 Segmentation Results
- `WBC-Seg` 独立金标准（F1 0.8874）
- `data1` 金标准（F1 0.6532）
- PAMSR 三数据集一致性（+0.2% / +0.4% / +0.8%）

#### 4.6 Clinical Gold-Standard Validation（`data1`）
- 分类 AFP-OD +3.7% mF1
- 分割 F1 0.6532

#### 4.7 External Generalization（`PBC`）
- 8 类 AFP-OD mF1 0.8577

#### 4.8 Cross-Center Challenge（`MultiCenter`）
- 诚实写成局限性

#### 4.9 Ablations
- 骨干组合（BC / PH / DN 九配置） ✅ 已完成
- α 分离强度 (0.05 / 0.10 / 0.20) ✅ 已完成
- AFP-OD 阶梯消融（P1 / P2 / P3a/b/c） ✅ 已完成
- N-shot 曲线（1 / 3 / 5 / 10 / 20） ✅ 已完成
- 形态学（有 / 无）整合在骨干消融里 ✅ 已完成
- PAMSR 共识 vs 非共识 ✅ 已完成

#### 4.10 Runtime
- 端到端 15-25 秒

### 5. Discussion
- 与 FSOD-VFM 范式的差异与本文独立性
- 两层修复架构（PAMSR + AFP-OD）的系统合理性
- 多骨干互补的必要性
- 局限性：
  - 跨中心极端不平衡
  - 稀缺类（Eos）绝对数值仍低
  - 文本 zero-shot 在 BALF 失败
- 未来工作：更大预训练医学基础模型 / 层级分类 / 主动学习

### 6. Conclusion

### References

### Appendix
- 可视化
- 超参细节
- 人机协同系统截图

---

## 5. 证据层级地图（写作时的"唯一数字来源"）

| 维度 | 最强证据数据集 | 关键数字 |
|------|---------------|---------|
| 主分类性能 | `data2` | AFP-OD mF1 0.7563，+3.12% |
| 主分割验证 | `WBC-Seg` | F1 0.8874（独立金标准） |
| 真实临床场景 | `data1` | 分割 F1 0.6532；分类 mF1 0.511 |
| 外部分类泛化 | `PBC` | AFP-OD mF1 0.8577 |
| 跨中心鲁棒性 | `MultiCenter` | 分割参数可迁移；分类受极端不平衡影响 |
| 多尺度一致性 | `data2 / data1 / WBC-Seg` | +0.2% / +0.4% / +0.8% |

所有数字写作时**只从 `MULTI_DATASET_RESULTS_SUMMARY.md` 与
`MULTISCALE_EXPERIMENT_REPORT.md` 引用**，不再从零散日志里找。

---

## 6. 消融实验现状与策略

### 已完成的消融（直接进 §4.9）
| 维度 | 数据来源 | 状态 |
|------|---------|------|
| AFP-OD 阶梯（MB-kNN → P1/P2/P3a/b/c）on `data2` | `EXPERIMENT_RESULTS_SUMMARY §1.3` | ✅ |
| α 分离强度（0.05/0.10/0.20）on `data2` | `§1.4` | ✅ |
| 骨干组合九配置（BC/PH/DN 及其组合） | `nshot_ablation_results.txt` | ✅ |
| 形态学特征 on/off | 骨干消融中覆盖 | ✅ |
| N-shot 曲线（1/3/5/10/20） | 同上 | ✅ |
| PAMSR 主尺度 / 辅尺度 / 共识开关 | `MULTISCALE_EXPERIMENT_REPORT.md` | ✅ |
| AFP-OD 跨数据集一致性（`data1` / `PBC`） | 各数据集报告 | ✅ |

### 暂不补做（返修阶段再决定）
> 用户决策（2026-04-21）：**先写论文说明模块有效性，不阻塞补消融。**
> 返修阶段根据审稿意见按需补充。

- **两层修复架构四象限**：因为分割后还有人机手动修正环节，
  纯自动 2×2 消融不能完全反映真实流程，推迟到返修再评估
- **双尺度裁剪（最终 setup）**：当前骨干消融已侧面支撑
- **形态学特征 vs 硬约束解耦**：不拆开，作为整体"形态学先验"叙事
- **AFP-OD 置信度阈值扫描**：固定本文取值，不做扫描

### 写作策略
- **§4.9 Ablations 只写已完成项**
- 对暂缺的消融，**Method 章节用文字说明模块有效性的直觉**，
  并在 Discussion 中自述"更细粒度消融留待后续工作"
- 所有"本文取值 α=0.10、conf_thr=0.025、ctx_w=0.1"一次性列进超参表

---

## 7. 风险清单（写作前必须对齐）

| 风险 | 处理策略 |
|------|---------|
| 99.25% 标注缩减被审稿人挑 | 摘要和贡献里**不再**作为 headline，只用 "10-shot + training-free" |
| SAM3 被质疑独立贡献 | 定位为**人机协同阶段精修工具**，不进主实验对比 |
| PAMSR 单尺度 vs 多尺度提升幅度小 | 用**三数据集一致性**表述，而非单点幅度 |
| AFP-OD 相对 MB-kNN 只 +3.12% | 用**跨数据集一致提升**（data1 +3.7%, PBC +0.81%）+ Eos +5.53% |
| MultiCenter 分类下降 | 明确作为 Cross-Center Challenge 写入 Discussion / Limitations |
| 分割没有外部 SOTA 对比 | WBC-Seg 绝对分数 + PAMSR 一致性支撑；必要时返修阶段补 StarDist |
| 统计显著性 | 锦上添花项，暂不阻塞写作；后续可补 paired test |
| 缺四象限消融 | Discussion 交代为 future work；审稿若追问再补 |

---

## 8. 写作推进顺序

1. **先定贡献列表**（§3）—— 所有后续章节都要对齐这 4 条
2. **再写 Method**（§3 of paper）—— 只有 Method 稳了，Experiments 的数字才有落点
3. **再写 Experiments**（§4 of paper）—— 按本文档 §4 骨架直接填
4. **再写 Introduction**（§1 of paper）—— 放到最后再写开头，效率最高
5. **最后写 Abstract + Conclusion**

---

## 9. 维护原则

- 本文档是论文结构的**唯一骨架来源**，不再到处维护多份大纲
- 数字只引用 `MULTI_DATASET_RESULTS_SUMMARY.md` / `EXPERIMENT_RESULTS_SUMMARY.md` /
  `MULTISCALE_EXPERIMENT_REPORT.md`
- 任何新实验产生结果，**先更新结果汇总文档，再回来动本框架**
- 启发论文 FSOD-VFM 只在 §2 Related Work 和 §5 Discussion 各出现一次，
  避免过度依附
- **写作纪律**：所有章节写完后**必须进行自检**，对照 §10 期刊规范 与 §11 写作规范
- **知识边界纪律**：遇到不确定的期刊规范 / 格式 / 术语，**必须查官方资料并写入本文件**

---

## 10. Journal Submission Rules（目标期刊：CMIG）

**期刊全称**：Computerized Medical Imaging and Graphics
**出版社**：Elsevier
**ISSN**：0895-6111
**分区**：中科院医学影像二区
**官方 Guide for Authors**：
`https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/publish/guide-for-authors`

### 10.1 Abstract 硬约束
- **上限 250 词**（≤ 250 words，Elsevier 官方明确要求）
- **非结构化**（本文选此；连续段落，不分 Background / Methods 小节）
- 简洁陈述目的 / 主要结果 / 主要结论
- **禁止在摘要里放参考文献**
- 首次出现的缩写必须原地展开（BALF / PAMSR / AFP-OD 均需）

### 10.2 Highlights 硬约束
- **3–5 条** bullet points
- 每条 **≤ 85 字符（含空格）**
- 出现在 title page，与 manuscript 分开提交

### 10.3 Keywords 硬约束
- **1–7 个关键词**
- 英文
- **避免** 由 "and" / "of" 连接的多词短语
- 推荐选单词关键词

### 10.4 Reference 格式
- **Elsevier Harvard 风格**（作者-年份）
- In-text 示例：
  - 两作者：`(Stevenson and Bryant, 2000)`
  - 三作者及以上：`(Pancost et al., 2007)`
- CSL 文件：
  `https://raw.githubusercontent.com/citation-style-language/styles/master/dependent/computerized-medical-imaging-and-graphics.csl`

### 10.5 其他硬约束
- **SI 单位制**必须使用
- 图 / 表允许彩色（网络免费，印刷可能收费）
- 推荐使用 Elsevier `elsarticle` LaTeX 模板（CMIG 支持 Word 与 LaTeX）

### 10.6 未查清的条目（遇到写作需要时再补查）
- [ ] Manuscript 总长度上限（如有）
- [ ] Graphical Abstract 是否必需
- [ ] Figure / Table 最大数量上限
- [ ] CRediT 作者贡献声明格式
- [ ] Data Availability Statement 要求

---

## 11. 写作规范（Tier-1 style checklist）

> 每写完一个章节，**必须**逐项打钩。

### 11.1 禁用词 / 禁用句式
- [ ] 不用 `novel / state-of-the-art / significantly`
- [ ] 不用 `to the best of our knowledge`
- [ ] 不用 `comprehensive / extensive / thorough`
- [ ] 不用 `as we show / we believe`（hedge）
- [ ] 不用 "首次 / 唯一 / 最优" 类 over-claim

### 11.2 句式要求
- [ ] 每句一个核心思想，**方法句 ≤ 30 词**
- [ ] 连续被动语态 ≤ 2 句
- [ ] 缩写首次出现原地展开

### 11.3 术语一致性
- [ ] 贡献锚点必须原样复用：
      `two-level correction principle` / `PAMSR` / `AFP-OD` /
      `human-in-the-loop` / `three-tier evidence design`
- [ ] 不临时换同义词

### 11.4 数字规范
- [ ] 所有数字在 `MULTI_DATASET_RESULTS_SUMMARY.md` / `EXPERIMENT_RESULTS_SUMMARY.md` /
      `MULTISCALE_EXPERIMENT_REPORT.md` 中可溯源
- [ ] mF1 差值写作 `+3.1 mF1`，**禁止** 光写 `+3.1`
- [ ] 绝对 F1 保留 3 位小数（如 `0.756`）

### 11.5 风险规避
- [ ] 不使用 "99.25% 标注缩减" headline
- [ ] SAM3 不作为主对比项，定位为交互工具
- [ ] MultiCenter 的弱结果只在 Discussion / Limitations 出现
- [ ] AFP-OD / PAMSR 幅度用事实数字而非形容词修饰

---

## 12. Abstract（v4 定稿）

> **词数 250（CMIG 上限）** · 非结构化 · 自检已通过 §11

Automated analysis of bronchoalveolar lavage fluid (BALF) cytology is hindered by the high cost of pixel- and instance-level annotation required by fully supervised pipelines. Vision–language zero-shot alternatives, in turn, fail on BALF because textual class names are not visually discriminative in this domain. We present **BALF-Analyzer**, a training-free few-shot framework that requires only 10 labeled cells per class and no gradient updates. The framework follows a **two-level correction principle**. At the proposal level, *Primary-Anchor Multi-Scale Rescue* (PAMSR) uses one optimal scale as an anchor and admits candidates from auxiliary scales only under cross-scale consensus and probability gating. At the prototype level, *Adaptive Fisher Prototype with Oriented Decoupling* (AFP-OD) locates confused class pairs on the support set and shifts the affected prototypes along a Ledoit–Wolf–shrunk Fisher direction under a per-query confidence gate. An open human-in-the-loop system with SAM3-driven interactive refinement integrates the framework into a deployable review pipeline. Across five BALF and WBC datasets, BALF-Analyzer achieves a macro F1 of **0.756** on the primary BALF benchmark (+3.1 mF1 over a multi-backbone kNN baseline), a segmentation F1 of **0.887** on an external WBC gold standard, and a macro F1 of **0.858** on an external PBC benchmark; PAMSR additionally delivers recall-driven F1 gains on all three evaluated segmentation datasets. These results suggest that a training-free, 10-shot pipeline, when equipped with two-level correction, provides a practical path toward deployable BALF cell analysis.

### Highlights（≤ 85 字符 / 条，待自检）
> 待在 Abstract 定稿后单独一轮起草 + 自检。
