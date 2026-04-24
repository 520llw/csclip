# BALF-Analyzer 论文图集（完整版）

生成时间：2026-04-24  
所有图表基于 `EXPERIMENT_RESULTS_SUMMARY.md` 中的最终数据生成，英文标注，300 dpi。

---

## 主图（按论文出现顺序）

| 图号 | 文件名 | 说明 | 论文位置 |
|------|--------|------|----------|
| Fig.1 | `fig1_architecture.png` | BALF-Analyzer 整体架构流程图（含两层修正原则、PAMSR、AFP-OD） | §3.1 |
| Fig.2 | `fig2_pamsr_qualitative.png` | PAMSR 跨数据集定性对比（data1 + data2 + MultiCenter） | §3.3 / §4.5 |
| Fig.3 | `fig3_afpod_concept.png` | AFP-OD 概念示意图（双视图混淆检测 → LW-Fisher → 置信度门控偏移） | §3.5 |
| Fig.4 | `fig4_main_classification.png` | data2 主分类结果：8种SOTA vs MB-kNN vs AFP-OD P3c | §4.4 |
| Fig.5 | `fig5_ablation_study.png` | 消融实验汇总（4子图：骨干组合 / α曲线 / 阶梯消融 / N-shot曲线） | §4.9 |
| Fig.6 | `fig6_confusion_matrix.png` | 混淆矩阵（计数版 + 行归一化版，Seed 42, n=1316） | §4.4 |
| Fig.7 | `fig7_segmentation.png` | 分割结果（PAMSR三数据集增量 + WBC-Seg金标准） | §4.5 |
| Fig.8 | `fig8_cross_dataset.png` | 跨数据集泛化与局限性（MultiCenter + PBC） | §4.7 / §4.8 |
| Fig.9 | `fig9_tsne_features.png` | t-SNE 特征可视化（BiomedCLIP vs 融合特征） | §3.4 |
| Fig.10 | `fig10_hitl_interface.png` | 人机协同审核界面（FastAPI + SAM3） | §3.6 |

## 辅助图 / 补充材料

| 图号 | 文件名 | 说明 |
|------|--------|------|
| Fig.S1 | `fig_s1_radar_perclass.png` | 各类别F1雷达图（MB-kNN vs AFP-OD） |
| — | `fig2_pamsr_mechanism.png` | PAMSR 机制流程图（单独使用） |
| — | `fig2b_dual_scale_crop.png` | 双尺度裁剪示意图（Cell 110% + Context 150%） |
| — | `fig_balf_field_demo.png` | BALF 显微视野示例（Macrophage + Neutrophil） |

## 数据溯源

所有数字均来自以下文档：
- `paper_package/EXPERIMENT_RESULTS_SUMMARY.md` — 主实验数据
- `paper_package/MULTI_DATASET_RESULTS_SUMMARY.md` — 跨数据集数据
- `paper_package/MULTISCALE_EXPERIMENT_REPORT.md` — PAMSR分割数据

## LaTeX 引用示例

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_main_classification.png}
\caption{10-Shot classification results on BALF data2.}
\label{fig:main_results}
\end{figure}
```

## 制作脚本

- `generate_all_paper_figures.py` — 数据驱动图（Fig.4–8, Fig.6, Fig.S1）
- `generate_concept_figures.py` — 概念示意图（Fig.1, Fig.2 mechanism, Fig.3）
- `generate_real_pamsr_v3.py` — PAMSR真实运行对比（data2）
- `generate_real_pamsr_final_datasets.py` — PAMSR跨数据集扫描（data1/MultiCenter/WBC）
- `generate_cross_dataset_summary.py` — 跨数据集汇总图组合
