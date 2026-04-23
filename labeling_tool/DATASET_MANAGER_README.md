# 增强型数据管理系统文档

## 概述

基于对CVAT、Label Studio等主流数据标注工具的调研，我们为BALF细胞标注系统实现了一个增强型数据管理系统。新系统提供更完善的数据集组织、元数据管理、进度跟踪和批量操作功能。

## 新增功能

### 1. 数据集元数据管理

每个数据集现在支持以下元数据字段：
- **描述 (description)**: 数据集的详细说明
- **标签 (tags)**: 用于分类和过滤的标签列表
- **状态 (status)**: active(活跃) / archived(归档) / pending(待处理)
- **优先级 (priority)**: high(高) / normal(正常) / low(低)
- **负责人 (assigned_to)**: 数据集负责人
- **备注 (notes)**: 额外备注信息
- **创建/更新时间**: 自动记录

元数据存储在 `dataset_metadata.json` 文件中，与 `datasets.json` 分离。

### 2. 增强的API端点

#### 获取增强数据集列表
```
GET /api/datasets/enhanced?status=active&tag=medical&sort_by=progress&sort_order=desc
```

参数:
- `status`: 按状态过滤 (active/archived/pending)
- `tag`: 按标签过滤
- `sort_by`: 排序字段 (name/progress/created/updated/train_count)
- `sort_order`: 排序方向 (asc/desc)

返回包含进度统计的完整数据集信息。

#### 数据集元数据管理
```
GET /api/datasets/{group_id}/metadata    # 获取元数据
PUT /api/datasets/{group_id}/metadata    # 更新元数据
```

#### 数据集概览统计
```
GET /api/datasets/summary
```

返回所有数据集的汇总统计信息：
- 总数据集数量
- 总图片数量
- 平均标注进度
- 状态/优先级分布
- 所有标签列表

#### 批量操作
```
POST /api/datasets/bulk_action
```

支持的操作：
- `exclude`: 批量隐藏数据集
- `restore`: 批量恢复数据集
- `update_tags`: 批量更新标签

#### 分页图片列表
```
GET /api/images/paginated?group_id=xxx&page=1&page_size=100&filter_type=unlabeled
```

参数:
- `page`: 页码
- `page_size`: 每页数量
- `filter_type`: 过滤类型 (all/labeled/unlabeled)
- `search`: 文件名搜索

### 3. 前端数据管理面板

新增 `dataset-manager.js` 模块，提供以下功能：

#### 数据集卡片视图
- 显示数据集名称、图片数量、类别数
- 可视化标注进度条
- 显示描述、标签等元数据
- 快速编辑和打开按钮

#### 过滤和搜索
- 按状态过滤
- 按标签过滤
- 实时搜索（名称和描述）
- 多维度排序

#### 批量操作
- 多选数据集
- 批量隐藏/恢复
- 批量更新标签

#### 元数据编辑
- 模态框编辑元数据
- 实时保存

### 4. 标注进度跟踪

每个数据集自动计算并显示：
- 标注进度百分比
- 已标注图片数 / 总图片数
- 平均每个图片的标注数量
- 类别分布统计

## 文件变更

### 后端修改
- `main.py`: 添加新的API端点和辅助函数
- 新增元数据存储: `dataset_metadata.json`

### 前端修改
- `static/dataset-manager.js`: 新增数据管理模块
- `static/index.html`: 添加数据集管理面板UI和样式

## 使用方法

### 查看数据集管理面板
1. 点击工具栏中的"数据集管理"按钮
2. 在面板中可以看到所有数据集的卡片视图
3. 使用顶部工具栏进行过滤和排序

### 编辑数据集元数据
1. 在数据集卡片上点击"编辑"按钮
2. 在弹出的模态框中编辑描述、标签、状态等信息
3. 点击保存

### 批量操作
1. 使用复选框选择多个数据集
2. 从"批量操作"下拉菜单选择操作类型
3. 确认执行

### 查看统计
调用 `/api/datasets/summary` 获取整体统计信息，可用于仪表盘展示。

## 性能优化

- 分页加载: 大数据集使用分页而非全量加载
- 本地过滤: 搜索功能在本地进行，减少API调用
- 增量更新: 元数据更新只传输变更字段

## 未来扩展

可进一步添加的功能：
- 数据集版本控制
- 团队协作和权限管理
- 数据质量报告
- 与其他系统的数据同步
