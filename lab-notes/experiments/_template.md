---
date: YYYY-MM-DD
type: experiment
exp_id: <slug-与文件名一致>
status: running          # running | done | failed | paused
tags: []
commit: <git hash 或留空>
seed: 42
---

# <具体到现象 / 变化点的标题>

## 背景与动机

<为什么做这次实验；相较上次 / baseline 的变化点。>

## 方法变化点

- <变化 1>
- <变化 2>

## 配置

- 数据集 / 划分：
- 模型 / 规模：
- 优化器 / LR / 调度：
- Batch size（effective / per-device）：
- 硬件：

## 结果

| 指标 | baseline | 本次 | Δ |
|---|---|---|---|
| Top-1 | TODO | TODO | TODO |

<必要时贴 loss 曲线截图 / log 路径。>

## 观察与分析

<看到了什么、为什么可能是这样；若未达预期，提出 2-3 个假设。>

## 结论

<当前结论是否支持 / 反驳假设？若反驳，下一步打算怎样改？>

## 遗留问题

- [ ] TODO
