# 论文写作大纲（CMIG 格式，中文版）

**目标期刊**：Computerized Medical Imaging and Graphics (CMIG, 二区, IF 5.4)
**类型**：Research Article
**篇幅目标**：正文 8000–10000 中文字（对应英文 7000–8000 词）
**图表预算**：6 张图 + 5 张表（CMIG 通常接受 ≤ 10 figures）

---

## 论文标题（中英双语）

- **中文**：基于多视觉基础模型与自适应 Fisher 原型解耦的支气管肺泡灌洗液细胞少样本分类系统
- **英文**：BALF-Analyzer: A Training-Free Few-Shot BALF Cell Classification System via Multi-Foundation-Model Features and Adaptive Fisher Prototype Disentanglement

---

## 篇章结构（对应 CMIG 投稿要求）

### Abstract（350 词以内，结构化）
- Background（50 词）
- Methods（100 词）：三大创新点概括
- Results（100 词）：data2 主结果 + WBC-Seg 分割 + 标注节省
- Conclusions（50 词）

### Keywords（6 个）
支气管肺泡灌洗液 / 少样本分类 / 视觉基础模型 / Fisher 判别 / 原型解耦 / 细胞学辅助诊断

---

## 1. 引言（Introduction，约 800–1000 字）
**目标**：激发问题、陈述 gap、列出贡献

- 1.1 **临床背景**：BALF 细胞分类在间质肺病/嗜酸性肺炎/肺部肿瘤的诊断价值；Eos ≥ 25% 是关键诊断阈值
- 1.2 **问题**：人工分类耗时 30–60 min/样本，观察者间一致性差
- 1.3 **现有 AI 局限**：
  - 监督学习需要大量标注
  - 单一基础模型语义信息不足
  - 分割与分类割裂
  - 对稀有混淆类（Eos/Neu/Mac）处理不佳
- 1.4 **我们的思路**：training-free + multi-VFM + 针对性解耦
- 1.5 **三大贡献**（对应专利权利要求）：
  1. 级联分割 + 双尺度裁剪的通用前端
  2. 三骨干 + 形态学互补特征评分
  3. AFP-OD 少样本原型解耦算法
- 1.6 **文章组织说明**

**[图 1]**：BALF-Analyzer 整体流水线图（已有 `patent_figures/图1_整体流程图.png`）

---

## 2. 相关工作（Related Work，约 600–800 字）
- 2.1 **细胞分割**：Cellpose 系列 + SAM 系列
- 2.2 **医学图像基础模型**：BiomedCLIP / Phikon-v2 / DINOv2 / 比较优劣
- 2.3 **少样本分类**：ProtoNet / Tip-Adapter / Label Propagation / 共性——对"混淆类对"无专门处理
- 2.4 **经典判别方法**：Fisher LDA / Ledoit-Wolf 收缩 / 重新审视价值
- 2.5 **与本文定位的区别**：首个把"双视图混淆检测 + LW-Fisher 原型解耦"用于多骨干 VFM 少样本细胞分类

---

## 3. 方法（Methods，约 2500–3000 字，含公式）
**核心章节，占篇幅 30%**

### 3.1 问题形式化（notation 统一）
- query image → 分割 → 每个实例 → 特征 → 分类
- 定义：$\mathcal{S}_c$, $\mathbf{f}^{BC/PH/DN}$, $\mathbf{m}$, $N$-shot, $C$ 类

### 3.2 级联分割（Cascaded Segmentation）
- Cellpose 4.1.1 (`cpsam`) 自动检测
- SAM3 bbox 提示精修
- IoU > 0.5 采用精修
- **[图 2]** 级联分割框图（已有）

### 3.3 双尺度裁剪与多骨干特征（Dual-Scale Crop + Multi-Backbone）
- **[图 3]** 双尺度裁剪示意
- 细胞裁剪（扩 10%）+ 上下文裁剪（扩 50%）
- 融合公式 $f = 0.9 f_\text{cell} + 0.1 f_\text{ctx}$
- **[图 4]** 多骨干并行编码
- BC/PH/DN 冻结 + 40 维形态学
- 评分函数公式（MB-kNN 基线）

### 3.4 AFP-OD 自适应 Fisher 原型解耦（核心创新）

#### 3.4.1 双视图混淆检测
- 特征视图 LOO-kNN（余弦）
- 形态学视图 LOO-kNN（欧氏 + z-score）
- **并集准则**（理论论证）
- **[图 5]** 双视图混淆检测示意

#### 3.4.2 Ledoit-Wolf Fisher 方向
- LW 收缩估计
- 最优收缩率解析解
- Fisher 方向计算 + 单位化
- **[图 6]** LW-Fisher 流程

#### 3.4.3 原型定向解耦
- 扰动规则 $\tilde{f}^{(i)} = f^{(i)} + \alpha w_{ij}$
- 多混淆对累加
- 单位球面重归一化
- **[图 7]** 解耦几何示意

#### 3.4.4 算法伪代码（AlgorithmBox 1）

### 3.5 人机协同标注系统（System）
- FastAPI + SQLite + 原生 JS
- 一键批量预标注 + 低置信度审核流
- **[图 8]** UI 界面示意（可选，可放 Supplementary）

---

## 4. 实验（Experiments，约 2500–3000 字）
**占篇幅 30%**

### 4.1 数据集
- **data2_organized**（主）：6,631 实例，4 类，分布说明
- **WBC-Seg**（分割金标准）：152 张 val，22,683 GT 多边形
- **MultiCenter**（跨中心泛化）：极端 Neu 主导

### 4.2 评估协议
- Nested 5-fold CV × 5 seeds = 25 次评估
- 指标：mF1 / Eos F1 / Acc / 各类 F1
- 所有超参在 data2 训练集确定，不进行测试集调优

### 4.3 分类主结果
- **[表 1]** data2 主结果（方法堆叠递进表）
- **[表 2]** 对比 8 种 SOTA（Tip-Adapter / LP / NCM / kNN / LR ...）
- **核心数字**：mF1 0.7252 → 0.7563（+3.12%），Eos F1 0.4465 → 0.5018（+5.53%）

### 4.4 消融实验
- **[表 3]** AFP-OD 组件消融（trace vs LW / feature-only vs dual-view / intersection vs union）
- **[表 4]** α 扫描（0.05/0.10/0.20）
- **[图 9 (a)]** 骨干组合消融 7 种（柱状图）
- **[图 9 (b)]** N-shot 曲线（1/3/5/10/20-shot）

### 4.5 分割验证（WBC-Seg）
- **[表 5]** WBC-Seg 验证集上 CellposeSAM 实例分割结果
- P=0.9220 / R=0.8552 / **F1=0.8874** / TP=19,398 / Pred=21,038 / GT=22,683
- 证明分割前端可独立作为工业级工具
- **[图 10]** 分割结果可视化（已有 `wbc_seg_compare_v2.png`）

### 4.6 跨数据集泛化
- **[表 6]** MultiCenter 上 AFP-OD vs 各 baseline
- 诚实报告：极端不平衡下 NCM 反而最优，AFP-OD 不再占优
- 讨论：建议未来工作

### 4.7 标注效率与临床价值
- 99.25% 标注缩减率
- Eos F1 绝对值提升的临床意义

### 4.8 运行时性能
- 单张图像端到端 15–25 秒
- 支持一键批量处理

---

## 5. 讨论（Discussion，约 1000–1200 字）
- 5.1 **双视图并集准则优于交集的原因**：真混淆类高收益 + 非混淆类近零代价的非对称性
- 5.2 **Morph-anchored PLS 失效的反思**：形态学仅适合做检测信号，不适合做方向锚点
- 5.3 **临床意义**：Eos F1 +5.53% 对嗜酸性肺炎诊断的影响
- 5.4 **局限性**：
  - 10-shot 最优，极低 shot（1/3）表现不稳
  - MultiCenter 极端不平衡下泛化受限
  - 未在非 BALF 细胞上验证
- 5.5 **未来工作**：
  - 自适应 α_{ij}
  - 非 BALF 场景扩展
  - 前瞻性临床研究
  - LLM-Pathology 协作

---

## 6. 结论（Conclusions，约 200–300 字）
- 重申三大贡献
- 关键数字
- 系统开源价值

---

## 图表总表

| 编号 | 内容 | 来源 | 状态 |
|------|------|------|------|
| 图 1 | 整体流水线 | `patent_figures/图1_整体流程图.png` | ✅ 已有 |
| 图 2 | 级联分割流程 | `patent_figures/图2_级联分割.png` | ✅ 已有 |
| 图 3 | 双尺度裁剪 | `patent_figures/图3_双尺度裁剪.png` | ✅ 已有 |
| 图 4 | 多骨干结构 | `patent_figures/图4_多骨干结构.png` | ✅ 已有 |
| 图 5 | 双视图混淆检测 | `patent_figures/图5_双视图混淆检测.png` | ✅ 已有 |
| 图 6 | LW-Fisher 流程 | `patent_figures/图6_LW_Fisher流程.png` | ✅ 已有 |
| 图 7 | 原型解耦几何 | `patent_figures/图7_原型解耦几何.png` | ✅ 已有 |
| 图 8 | UI 界面（可选） | `patent_figures/图8_UI界面.png` | ✅ 已有（可放 Supp）|
| 图 9 | 消融结果（骨干+N-shot 子图）| 需合并现有 `ablation_*.png` | ⏳ 待补 |
| 图 10 | 分割可视化 | `experiments/figures/wbc_seg_compare_v2.png` | ✅ 已有 |
| 表 1 | 分类主结果 | EXPERIMENT_RESULTS_SUMMARY §1.3 | ✅ 已有 |
| 表 2 | SOTA 对比 | EXPERIMENT_RESULTS_SUMMARY §1.7 | ✅ 已有 |
| 表 3 | AFP-OD 消融 | EXPERIMENT_RESULTS_SUMMARY §1.3 | ✅ 已有 |
| 表 4 | α 扫描 | EXPERIMENT_RESULTS_SUMMARY §1.4 | ✅ 已有 |
| 表 5 | WBC-Seg 分割 | EXPERIMENT_RESULTS_SUMMARY §2.3 | ✅ 已有 |
| 表 6 | MultiCenter | EXPERIMENT_RESULTS_SUMMARY §3.2 | ✅ 已有 |

---

## 写作风格要点
- 全程用**客观被动语态**（"本文""本方法""实验结果表明"）
- 数字全部带**小数点三位以下**（如 0.7563）避免误差
- 关键创新点用**加粗**（"双视图并集""Ledoit-Wolf Fisher"）
- 公式编号到节（如 Eq. (3.4.1-1)）
- 图表引用一律 "**如图 X 所示**"、"**详见表 X**"
- 避免"我们认为""明显地"等主观表述
