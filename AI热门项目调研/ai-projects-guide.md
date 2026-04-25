# 🔥 近一个月GitHub最火的5个AI开源项目 - 配置与使用指南

> 数据来源：GitHub API 2026-04-24 | 作者：晚晚整理

---

## 🥇 1. autoresearch — 76.2K ⭐

**作者：** @karpathy  
**仓库：** https://github.com/karpathy/autoresearch  
**核心功能：** 让AI智能体在单GPU上自主进行LLM训练研究， overnight自动实验

### 环境要求
- NVIDIA GPU（推荐H100，其他GPU也可）
- Python 3.10+
- `uv` 包管理器

### 安装配置

```bash
# 1. 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch

# 3. 安装依赖
uv sync

# 4. 下载训练数据和训练tokenizer（一次性，约2分钟）
uv run prepare.py

# 5. 手动运行单次训练实验（约5分钟）
uv run train.py
```

### 核心文件说明

| 文件 | 作用 | 谁来修改 |
|------|------|---------|
| `prepare.py` | 常量、数据准备、运行时工具 | **不修改** |
| `train.py` | GPT模型、优化器、训练循环 | **AI智能体修改** |
| `program.md` | 智能体指令基线 | **人类修改优化** |

### 运行AI智能体

```
在仓库目录启动 Claude/Codex，关闭所有权限，然后提示：
"Hi have a look at program.md and let's kick off a new experiment! let's do the setup first."
```

### 设计特点
- **固定时间预算：** 每次训练严格5分钟（wall clock），每小时约12次实验
- **评估指标：** `val_bpb`（validation bits per byte），越低越好，与词表大小无关
- **自包含：** 仅依赖PyTorch和几个小包，无分布式训练

### 小平台调参建议（MacBook/小GPU）

```python
# prepare.py 中调整：
MAX_SEQ_LEN = 256        # 降低序列长度
EVAL_TOKENS = 较小值     # 减少验证数据

# train.py 中调整：
DEPTH = 4                # 降低模型深度（默认8）
WINDOW_PATTERN = "L"     # 使用简单注意力（替代SSSL）
TOTAL_BATCH_SIZE = 2**14  # 降低总批次
```

---

## 🥈 2. mempalace — 49.4K ⭐

**作者：** @MemPalace  
**仓库：** https://github.com/MemPalace/mempalace  
**核心功能：** 本地优先的AI记忆系统，96.6% R@5检索准确率，零API调用

### 环境要求
- Python 3.9+
- 向量存储后端（默认ChromaDB）
- 约300MB磁盘空间（默认embedding模型）

### 安装配置

```bash
pip install mempalace

# 初始化项目
mempalace init ~/projects/myapp
```

### 快速使用

```bash
# 挖掘内容到记忆宫殿
mempalace mine ~/projects/myapp                    # 项目文件
mempalace mine ~/.claude/projects/ --mode convos   # Claude Code会话

# 搜索记忆
mempalace search "why did we switch to GraphQL"

# 为新会话加载上下文
mempalace wake-up
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **Wings（翼）** | 人员/项目级别的组织单元 |
| **Rooms（房间）** | 主题分类 |
| **Drawers（抽屉）** | 原始内容存储 |

### 基准测试性能

| 模式 | R@5 | 需要LLM |
|------|-----|---------|
| Raw（纯语义搜索） | **96.6%** | 不需要 |
| Hybrid v4 | **98.4%** | 不需要 |
| Hybrid + LLM重排 | ≥99% | 需要 |

### MCP服务器

29个MCP工具覆盖：
- 宫殿读写
- 知识图谱操作
- 跨翼导航
- 抽屉管理
- 智能体日记

### Claude Code Hooks（自动保存）

```bash
# 周期性保存和上下文压缩前自动保存
# 详见：mempalaceofficial.com/guide/hooks
```

---

## 🥉 3. caveman — 45.3K ⭐

**作者：** @JuliusBrussee  
**仓库：** https://github.com/JuliusBrussee/caveman  
**核心功能：** Claude Code技能，用"洞穴人"说话方式减少75%输出token

### 一句话哲学
> **"why use many token when few do trick"**

### 各平台安装

| 智能体 | 安装命令 |
|--------|---------|
| **Claude Code** | `claude plugin marketplace add JuliusBrussee/caveman && claude plugin install caveman@caveman` |
| **Codex** | Clone repo → `/plugins` → 搜索 "Caveman" → 安装 |
| **Gemini CLI** | `gemini extensions install https://github.com/JuliusBrussee/caveman` |
| **Cursor** | `npx skills add JuliusBrussee/caveman -a cursor` |
| **Windsurf** | `npx skills add JuliusBrussee/caveman -a windsurf` |
| **Copilot** | `npx skills add JuliusBrussee/caveman -a github-copilot` |
| **其他40+智能体** | `npx skills add JuliusBrussee/caveman` |

### 使用方式

```
# 启动洞穴人模式
/caveman
/caveman lite      # 轻度：去掉填充词，保留语法
/caveman full      # 完整：默认模式，去掉冠词，碎片化
/caveman ultra     # 极致：最大压缩，电报风格
/caveman wenyan    # 文言文：古典中文压缩

# 停止
"stop caveman" 或 "normal mode"
```

### 附加技能

| 技能 | 命令 | 功能 |
|------|------|------|
| caveman-commit | `/caveman-commit` | 简洁提交信息（≤50字符） |
| caveman-review | `/caveman-review` | 单行PR评论 |
| caveman-help | `/caveman-help` | 快速参考卡片 |
| caveman-compress | `/caveman:compress <filepath>` | 压缩记忆文件（节省46%输入token） |

### 效果对比

| 任务 | 正常模式 | Caveman模式 | 节省 |
|------|---------|------------|------|
| 解释React重渲染bug | 1180 tokens | 159 tokens | 87% |
| 修复auth中间件 | 704 tokens | 121 tokens | 83% |
| 设置PostgreSQL连接池 | 2347 tokens | 380 tokens | 84% |
| **平均** | **1214 tokens** | **294 tokens** | **75%** |

---

## 4. career-ops — 39.1K ⭐

**作者：** @santifer  
**仓库：** https://github.com/santifer/career-ops  
**核心功能：** 基于Claude Code的AI求职系统，评估offer、生成定制PDF、批量处理

### 环境要求
- Node.js
- Claude Code 或 Gemini CLI
- Playwright（PDF生成）

### 安装配置

```bash
# 1. 克隆并安装
git clone https://github.com/santifer/career-ops.git
cd career-ops && npm install
npx playwright install chromium

# 2. 检查环境
npm run doctor

# 3. 配置个人资料
cp config/profile.example.yml config/profile.yml
# 编辑 profile.yml 填入你的信息

cp templates/portals.example.yml portals.yml
# 自定义目标公司

# 4. 添加CV
# 在项目根目录创建 cv.md，写入你的简历（Markdown格式）

# 5. 用Claude个性化配置
claude   # 在目录中启动Claude Code
# 然后让Claude调整系统：
# "Change the archetypes to backend engineering roles"
# "Add these 5 companies to portals.yml"
```

### 主要功能

| 功能 | 说明 |
|------|------|
| **自动流水线** | 粘贴URL → 完整评估 + PDF + 追踪 |
| **6模块评估** | 角色摘要、CV匹配、级别策略、薪资调研、个性化、面试准备 |
| **面试故事库** | 积累STAR+Reflection故事，回答任何行为问题 |
| **薪资谈判脚本** | 谈判框架、地理折扣反驳、竞争offer杠杆 |
| **ATS PDF生成** | 关键词注入的CV，Space Grotesk + DM Sans设计 |
| **门户扫描器** | 45+公司预配置（Anthropic、OpenAI等） |
| **批量处理** | 并行评估多个offer |
| **仪表板TUI** | Go语言终端UI浏览、过滤、排序 |

### 使用命令

```bash
/career-ops                # 显示所有命令
/career-ops {粘贴JD}       # 完整自动流水线
/career-ops scan           # 扫描招聘门户
/career-ops pdf            # 生成ATS优化CV
/career-ops batch          # 批量评估
/career-ops tracker        # 查看申请状态
/career-ops deep           # 深度公司研究
```

### 项目结构

```
career-ops/
├── cv.md                    # 你的CV
├── config/profile.yml        # 个人资料
├── modes/                   # 14种评估模式
│   ├── oferta.md            # 单职位评估
│   ├── pdf.md               # PDF生成
│   ├── scan.md              # 门户扫描
│   └── batch.md             # 批量处理
├── templates/               # CV模板和门户配置
├── dashboard/               # Go TUI仪表板
├── data/                    # 追踪数据
├── reports/                 # 评估报告
└── output/                  # 生成PDF
```

---

## 5. graphify — 34.0K ⭐

**作者：** @safishamsi  
**仓库：** https://github.com/safishamsi/graphify  
**核心功能：** 将任何代码/文档/论文/图片/视频文件夹转为可查询的知识图谱

### 支持的工具
- Claude Code
- Codex
- OpenCode
- Cursor
- Gemini CLI
- GitHub Copilot CLI
- OpenClaw
- Factory Droid
- Trae
- Google Antigravity

### 安装

```bash
npx skills add safishamsi/graphify
```

### 使用方法

```bash
# 基本使用 - 在目标文件夹运行
graphify

# 知识图谱构建完成后，即可查询：
# "这个函数在哪里被调用？"
# "这个项目使用什么设计模式？"
# "如何处理用户认证？"
```

### 核心能力

| 能力 | 说明 |
|------|------|
| **代码图谱** | 函数调用关系、依赖分析 |
| **文档图谱** | 论文/文档知识结构提取 |
| **多模态** | 图片、视频内容结构化 |
| **跨文件查询** | 自然语言查询整个项目 |
| **增量更新** | 文件变更后自动更新图谱 |

---

## 📊 综合对比

| 项目 | Stars | 类型 | 核心用途 | 硬件要求 |
|------|-------|------|---------|---------|
| autoresearch | 76.2K | 训练/研究 | 让AI自主做LLM实验 | NVIDIA GPU |
| mempalace | 49.4K | 记忆系统 | 本地AI对话记忆 | 低，CPU即可 |
| caveman | 45.3K | 效率工具 | 减少75% token消耗 | 无 |
| career-ops | 39.1K | 求职工具 | AI辅助找工作 | 无 |
| graphify | 34.0K | 知识图谱 | 代码/文档结构化查询 | 低 |

---

## 💡 使用建议

1. **日常使用caveman**：所有Claude Code用户都应该装，立即节省token费用
2. **记忆用mempalace**：如果你经常用AI编码助手，本地记忆系统让上下文不再丢失
3. **研究用autoresearch**：有GPU且对LLM训练感兴趣时尝试
4. **求职用career-ops**：正在找工作时，让AI帮你筛选和定制简历
5. **代码理解用graphify**：接手新项目时，快速构建知识图谱理解架构

---

> 📅 整理时间：2026-04-24  
> 🔗 来源：GitHub API + 官方README
