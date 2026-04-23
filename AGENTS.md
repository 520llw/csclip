# AGENTS.md — BALF Cell Classification & Segmentation Project

## 项目概述

本项目是基于 **BiomedCLIP** 的 BALF（支气管肺泡灌洗液）细胞"初筛-核查-分割-分类"全链路科研优化系统。
核心目标：通过技术手段极小化人工标注工作量，支撑发明专利和中科院二区论文。

主要技术栈：
- **视觉-语言模型**：BiomedCLIP（open_clip_torch 3.3.0，本地权重）
- **分割模型**：Cellpose 4.1.1、SAM3（Meta，checkpoint ~3.4GB）
- **特征提取器**：DINOv2、DINO-Bloom、Phikon-v2、PLIP
- **Web 标注工具**：FastAPI + 原生 JS（`labeling_tool/`）
- **实验脚本**：Python CLI 脚本（`experiments/`）

---

## 目录结构

```
/home/xut/csclip/               # 项目根目录
├── *.py                        # 4 个重建的核心模块（被 labeling_tool 与实验脚本共同导入）
├── experiments/                # 实验与评估脚本（基线、10-shot、消融、诊断等）
│   ├── feature_cache/          # 预提取的 .npz 特征缓存（按模型/数据集/划分存放）
│   ├── figures/                # 结果图表输出
│   ├── visualizations/         # 可视化结果
│   └── *.py / *.sh             # 实验脚本与运行入口
├── labeling_tool/              # FastAPI 标注系统
│   ├── main.py                 # FastAPI 应用入口
│   ├── database.py             # 数据集/标注/元数据管理
│   ├── model.py                # 后端模型逻辑（BiomedCLIP + Cellpose + SAM3）
│   ├── feature_extractors/     # 多模型特征提取器工厂（BiomedCLIP/DINOv2/Phikon等）
│   ├── prompt_tuner.py         # CoOp prompt tuning 实现
│   ├── morphology_constraints.py # 12 维形态学硬约束
│   ├── fewshot_biomedclip.py   # Few-shot 分类器准备与预测
│   ├── hybrid_classifier.py    # 混合自适应分类器
│   ├── static/                 # 前端静态资源（HTML/JS/CSS）
│   └── requirements.txt        # Web 工具依赖（FastAPI/Uvicorn等）
├── cell_datasets/              # 数据集目录
│   ├── data2_organized/        # 主实验集（180 张，7 类，4 类有效）
│   ├── data1_organized/        # 2698 张，有标注
│   ├── MultiCenter_organized/  # 2372 张，多中心数据
│   └── Tao_Divide/             # 20,580 张，无标注，用于验证
├── model_weights/              # 本地模型权重
│   ├── dinobloom_vitb14.pth
│   ├── dinov2_vitb14_pretrain.pth
│   ├── dinov2_vits14_pretrain.pth
│   ├── phikon_v2/
│   └── plip / plip_direct/
└── sam3/                       # SAM3 源码（带 setup.py / pyproject.toml）
    └── sam3/
        └── build_sam3_image_model 等
```

---

## 环境要求

### Conda 环境
- **实验环境**：`cel`（Python 3.9.21）
  - PyTorch 2.5.1 + CUDA 12.x
  - 路径：`/data/software/mamba/envs/cel/bin/python`
- **Web 标注环境**：`research_assistant`（Windows / Python 3.11.15）
  - PyTorch 2.10.0+cu128
  - 仅用于 `labeling_tool/` 的本地开发部署

### 核心依赖
```text
torch >= 2.5.1
torchvision
cellpose == 4.1.1
open_clip_torch == 3.3.0
sam3 (本地源码安装，路径 /home/xut/csclip/sam3)
fastapi >= 0.135.2
uvicorn >= 0.42.0
numpy, scipy, scikit-learn, scikit-image, opencv-python, pillow, matplotlib, monai
```

### 环境变量
- `MEDSAM_ROOT`：若未设置，代码默认指向 `/home/xut/csclip`（通过 `labeling_tool/paths.py` 解析）。
- BiomedCLIP 本地权重目录需存在 `open_clip_config.json` 或同类配置文件。

---

## 关键外部模块（根目录 .py 文件）

项目根目录下有 **4 个重建的核心模块**，原为 MedSAM 项目的 stub，现已被完整实现。
**修改这些文件时需特别谨慎**，因为 `labeling_tool/` 与 `experiments/` 均直接依赖它们。

| 文件 | 职责 |
|------|------|
| `biomedclip_zeroshot_cell_classify.py` | `InstanceInfo` 数据类、设备解析 `resolve_device`、权重目录验证 |
| `biomedclip_fewshot_support_experiment.py` | 双尺度（cell 90% + context 10%）裁剪与特征编码 |
| `biomedclip_query_adaptive_classifier.py` | 12 维形态学特征（面积/周长/圆度/颜色/偏心率等）构建与归一化 |
| `biomedclip_hybrid_adaptive_classifier.py` | 混合打分：全局原型 + 文本原型 + support 亲和度 + 自适应 scaling |

**导入约定**：
- 实验脚本中常通过 `sys.path.insert(0, '/home/xut/csclip')` 或 `sys.path.insert(0, '/home/xut/csclip/sam3')` 来导入根目录模块与 SAM3。
- `test_imports.py` 提供了所有核心模块的冒烟测试。

---

## 运行入口

### 实验脚本（Linux / `cel` 环境）
所有实验均通过根目录下的 `.sh` 脚本触发，统一写入 `/tmp/*_output.txt`：

```bash
bash run_baseline.sh      # 基线诊断实验 → baseline_diagnosis.py
bash run_10shot.sh        # 10-shot 分类 → ten_shot_classify.py
bash run_10shot_v2.sh     # v2 版本
bash run_ablation.sh      # 消融实验 → ablation_study.py
bash run_advanced.sh      # 高级实验
bash run_diagnosis.sh     # 深度诊断
bash run_knn.sh           # KNN 实验
bash run_optimize.sh      # 优化实验
bash run_cellpose_test.sh # Cellpose 测试
```

### Web 标注服务（`labeling_tool/`）
```bash
conda activate research_assistant   # 或等效环境
cd /home/xut/csclip
python -m uvicorn labeling_tool.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 代码风格与约定

- **类型注解**：大量使用 `from __future__ import annotations`、`dataclass`、类型提示。
- **NumPy / PyTorch 混合**：分类器内部常以 NumPy 进行逻辑运算，边缘用 `torch` 做张量编码。
- **路径处理**：优先使用 `pathlib.Path`，避免硬编码 Windows 路径。
- **设备解析**：统一使用 `resolve_device("auto")`，默认 `"cuda"` 若可用否则 `"cpu"`。

---

## 特征缓存机制

`experiments/feature_cache/` 存放预提取的图像特征，命名规范：
```
{model}_{dataset}_{split}.npz
# 例如：biomedclip_val.npz、multicenter_dinov2_s_train.npz
```
缓存格式通常为 NumPy 数组字典（`features`、`labels`、`image_ids` 等）。

---

## 重要研究发现与约束

1. **文本原型完全无效**：BiomedCLIP 的文本编码器对细胞类名的余弦相似度高达 0.92–0.96，图像-文本对齐仅 0.31–0.35。因此**纯零样本文本路径已被放弃**，分类策略以图像原型 + 形态学约束为核心。
2. **嗜酸性粒细胞（Eosinophil）是最大痛点**：基线 precision 仅 ~18.9%，需依赖形态学硬约束（颗粒度、双叶核特征）提升。
3. **淋巴细胞表现最好**：F1 ≈ 0.89，得益于其独特的小圆形态。
4. **Prompt Tuning（CoOp）已集成**：在 `labeling_tool/prompt_tuner.py` 中，通过学习连续 prompt 向量残差校正文本原型。

---

## 新代理快速上手指南

1. **验证环境**：运行 `python test_imports.py`，确保 4 个核心模块与 SAM3 可正常导入。
2. **跑通基线**：执行 `bash run_baseline.sh`，查看 `/tmp/baseline_output.txt`。
3. **修改分类逻辑**：先读 `biomedclip_hybrid_adaptive_classifier.py`，再读 `labeling_tool/hybrid_classifier.py` 的封装层。
4. **添加实验**：在 `experiments/` 新建脚本，并在根目录新建 `run_xxx.sh` 作为统一入口。
5. **修改 Web 后端**：从 `labeling_tool/main.py` 的 FastAPI 路由入手，模型逻辑在 `labeling_tool/model.py`。

---

## 文件修改优先级

以下文件被多处依赖，修改前请先运行 `python test_imports.py` 确认无破坏：
- `biomedclip_zeroshot_cell_classify.py`
- `biomedclip_fewshot_support_experiment.py`
- `biomedclip_query_adaptive_classifier.py`
- `biomedclip_hybrid_adaptive_classifier.py`
- `labeling_tool/paths.py`
- `labeling_tool/fewshot_biomedclip.py`
- `labeling_tool/hybrid_classifier.py`
