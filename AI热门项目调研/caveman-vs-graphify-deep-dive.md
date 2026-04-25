# 🔍 深度分析：Caveman vs Graphify

> 基于官方README、源码结构和实际使用场景的全面技术剖析
> 分析时间：2026-04-24

---

## 一、Caveman（45.3K ⭐）— 极简主义的暴力美学

### 1.1 核心哲学：为什么"少即是多"

Caveman不是简单的提示词工程，而是基于一个被验证的认知科学发现：

> **2026年3月论文《Brevity Constraints Reverse Performance Hierarchies in Language Models》发现：**
> 强制大模型用简短回答，在某些基准测试上**准确率提升了26个百分点**，甚至完全逆转了模型性能层级。

**_verbose ≠ better_**。Verbose（冗长）很多时候只是模型在"思考 aloud"，而不是真正在思考。Caveman通过强制约束输出长度，反而让模型更聚焦核心逻辑。

---

### 1.2 技术实现：这不是简单的"少说废话"

#### 四层压缩策略

| 级别 | 策略 | 技术细节 | 适用场景 |
|------|------|---------|---------|
| **Lite** | 去填充词，保留语法 | 去掉"just/really/basically"等填充词，保持完整句子结构 | 团队协作，需要可读性 |
| **Full** | 去冠词，碎片化 | 去掉a/an/the，允许片段化表达，用"="代替"equals" | 日常编码，平衡效率 |
| **Ultra** | 最大压缩，电报风格 | 缩写一切，用箭头"→"，代码变量名风格 | 快速调试，极限省钱 |
| **Wenyan** | 文言文古典压缩 | 利用中文古典语法的信息密度优势 | 最极限压缩，文化趣味 |

#### 关键机制：始终激活（Always-On）

Caveman通过 **PreToolUse hooks** 实现真正的"始终激活"：

```json
// .codex/hooks.json 示例
{
  "hooks": {
    "SessionStart": [
      {
        "tool": "Bash",
        "command": "echo 'Terse like caveman. Technical substance exact. Only fluff die.'"
      }
    ]
  }
}
```

这意味着：**不是用户在提示词里加一句"请简洁回答"，而是在系统层面强制改写模型的输出风格。**

---

### 1.3 生态系统：三个工具矩阵

| 工具 | 解决什么问题 | 一句话 |
|------|-------------|--------|
| **caveman** | 输出压缩 | "why use many token when few do trick" |
| **cavemem** | 记忆持久化 | "why agent forget when agent can remember" |
| **cavekit** | 构建优化 | "why agent guess when agent can know" |

**组合使用逻辑：**
- cavekit编排构建流程
- caveman压缩agent的"说话"
- cavemem压缩agent的"记忆"

---

### 1.4 实测效果数据（来自官方evals）

| 任务类型 | 正常token | Caveman token | 节省率 | 技术准确性 |
|---------|----------|--------------|--------|----------|
| React重渲染bug解释 | 1180 | 159 | **87%** | 100% |
| Auth中间件修复 | 704 | 121 | **83%** | 100% |
| PostgreSQL连接池设置 | 2347 | 380 | **84%** | 100% |
| Docker多阶段构建 | 1042 | 290 | **72%** | 100% |
| PR安全审查 | 678 | 398 | **41%** | 100% |
| 架构：微服务 vs 单体 | 446 | 310 | **30%** | 100% |
| **加权平均** | **1214** | **294** | **~75%** | **100%** |

**注意：** Caveman不影响thinking/reasoning token（思考过程），只压缩**输出**token。省钱是副产品，**速度和可读性**才是主要收益。

---

### 1.5 高级技能详解

#### caveman-compress：反向压缩（输入端）

```bash
/caveman:compress CLAUDE.md
```

这会：
1. 读取你的CLAUDE.md（人类可读）
2. 生成 `CLAUDE.md`（压缩版，Claude每次session读取，省token）
3. 备份为 `CLAUDE.original.md`（你读和编辑的）

**实测输入端压缩效果：**

| 文件 | 原始tokens | 压缩后 | 节省 |
|------|-----------|--------|------|
| claude-md-preferences.md | 706 | 285 | **59.6%** |
| project-notes.md | 1145 | 535 | **53.3%** |
| claude-md-project.md | 1122 | 636 | **43.3%** |
| todo-list.md | 627 | 388 | **38.1%** |
| **平均** | **898** | **481** | **46%** |

**安全保证：** 代码块、URL、文件路径、命令、标题、日期、版本号——所有技术性内容**原样保留**，只压缩散文。

#### caveman-commit：极简提交信息

```bash
/caveman-commit
# 输出：
# fix(auth): token expiry check uses < not <=
```

遵循 Conventional Commits，≤50字符主题行，强调Why而非What。

#### caveman-review：单行PR评论

```
L42: 🔴 bug: user null. Add guard.
```

没有"我觉得..."、"也许你可以考虑..."等寒暄，直接指出问题。

---

### 1.6 安装细节：不同平台的差异

| 平台 | 安装方式 | 自动激活 | Hook类型 |
|------|---------|---------|---------|
| Claude Code | Plugin marketplace + hooks | ✅ | SessionStart + PreToolUse |
| Codex | Local marketplace + `.codex/hooks.json` | ✅（仅本repo） | SessionStart |
| Gemini CLI | `gemini extensions install` | ✅ | GEMINI.md context |
| Cursor | `npx skills add` | ❌（需手动加rules） | `.cursor/rules/` |
| Windsurf | `npx skills add` | ❌ | `.windsurf/rules/` |
| Cline | `npx skills add` | ❌ | `.clinerules/` |
| Copilot | `npx skills add` | ❌ | `.github/copilot-instructions.md` |

**重要：** Claude Code和Codex支持**PreToolUse hooks**——这是真正的始终激活机制。其他平台需要手动将caveman prompt加入system prompt或rules文件。

---

### 1.7 局限性与注意事项

1. **只压缩输出，不压缩思考：** 如果模型用了extended thinking，那些token不受影响
2. **代码/提交/PR内容保持正常：** Caveman不修改代码本身，只修改自然语言解释
3. **文化差异：** Ultra模式对非英语母语者可能理解困难
4. **不适合所有场景：** 写文档、写博客、解释给非技术人员时，可能需要关闭

---

### 1.8 适合谁用？

- ✅ 每天大量使用Claude Code/Codex的开发者
- ✅ 关心API账单的个人/小团队
- ✅ 已经熟悉技术术语，不需要"手把手解释"的高级用户
- ✅ 想要更快响应速度的场景（CI/CD、快速原型）
- ❌ 需要写用户文档、技术博客的场景
- ❌ 给非技术人员解释技术方案的场景
- ❌ 需要详细推理过程的教学场景

---

## 二、Graphify（34.0K ⭐）— 知识图谱的工程化实践

### 2.1 核心问题：为什么需要知识图谱？

Andrej Karpathy有一个习惯：保持 `/raw` 文件夹，里面丢论文、截图、笔记、推文。问题是：**当文件数量超过50个，grep已经不够用了。**

Graphify解决的核心痛点：

> **"71.5倍更少的token per query"** —— 不是 marketing 数字，是实测：查询图谱 vs 读取原始文件。

---

### 2.2 三阶段处理架构（核心技术）

Graphify不是简单地把文件内容塞进向量数据库。它使用**三阶段分层提取：**

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: AST提取（代码文件）                              │
│  ├── 确定性提取，无需LLM                                   │
│  ├── 提取：类、函数、导入、调用图、文档字符串               │
│  └── 支持25种语言（tree-sitter）                          │
├─────────────────────────────────────────────────────────┤
│  Stage 2: 本地转录（视频/音频）                            │
│  ├── faster-whisper本地运行，音频不出机器                   │
│  ├── 使用corpus god nodes生成domain-aware提示词            │
│  └── 转录缓存：graphify-out/transcripts/                  │
├─────────────────────────────────────────────────────────┤
│  Stage 3: LLM语义提取（文档/论文/图片/转录文本）            │
│  ├── Claude子agent并行提取概念、关系、设计原理             │
│  └── 结果合并到NetworkX图谱                               │
└─────────────────────────────────────────────────────────┘
```

**关键洞察：** 代码用AST（免费、准确、快速），只有文档/图片/视频才用LLM（昂贵但必要）。这种分层设计使得**代码项目的增量更新可以几乎零成本完成。**

---

### 2.3 图结构：三种边类型（诚实性设计）

Graphify最核心的设计哲学是**诚实：**

| 边类型 | 标签 | 含义 | 置信度 |
|--------|------|------|--------|
| **EXTRACTED** | 提取 | 直接从源码中找到的关系（如函数调用） | 1.0 |
| **INFERRED** | 推断 | 合理推断的关系（如语义相似性） | 0.0-1.0 |
| **AMBIGUOUS** | 模糊 | 标记为需要人工审查的关系 | 需确认 |

**这比Vector RAG诚实得多：** 传统RAG把"猜测"和"事实"混在一起，Graphify明确标注每条边的来源。

---

### 2.4 社区检测：不用Embeddings的"语义"聚类

```
传统RAG: 文件 → Embedding → 向量数据库 → 相似度搜索
Graphify: 文件 → 概念图 → Leiden社区检测 → 拓扑结构即语义
```

**Leiden算法**（来自graspologic）基于**边密度**发现社区，不需要单独的embedding步骤：
- `semantically_similar_to` 边已经在图中
- 这些边直接影响社区检测
- 结果是：结构即语义

**输出格式：**

```
graphify-out/
├── graph.html          # 交互式图谱（浏览器打开即可）
├── GRAPH_REPORT.md     # 神节点、惊喜连接、建议问题
├── graph.json          # 持久化图谱（几周后仍可查询）
└── cache/              # SHA256缓存（只处理变更文件）
```

---

### 2.5 Always-On机制：让Agent自动使用图谱

Graphify的killer feature不是生成图谱，而是**让AI助手自动读取图谱**：

#### Claude Code 版本

```bash
graphify claude install    # 安装always-on hook
```

这会做两件事：
1. 在 `CLAUDE.md` 添加："有图谱时先读 GRAPH_REPORT.md"
2. 安装 `PreToolUse` hook：每次Glob/Grep前检查图谱存在

**实际效果：**
```
用户："这个auth逻辑在哪里？"

无Graphify：Claude grep所有文件 → 读大量无关代码 → 浪费token
有Graphify：Claude读GRAPH_REPORT.md → 直接定位到auth社区 → 精准查询
```

#### 各平台Always-On对比

| 平台 | 机制 | 触发时机 |
|------|------|---------|
| Claude Code | PreToolUse hook | Glob/Grep前 |
| Codex | PreToolUse hook | Bash调用前 |
| OpenCode | tool.execute.before插件 | Bash前 |
| Gemini CLI | BeforeTool hook | file-read前 |
| Cursor | `.cursor/rules/graphify.mdc` | 每会话自动 |
| Kiro | `.kiro/steering/graphify.md` | 每会话自动 |
| Aider/OpenClaw/Trae | AGENTS.md | 每会话自动 |

---

### 2.6 查询接口：三种深度

| 命令 | 用途 | 示例 |
|------|------|------|
| `/graphify .` | 构建/更新图谱 | 初始构建或--update增量 |
| `/graphify query "..."` | 子图查询 | "show the auth flow" |
| `/graphify path "A" "B"` | 最短路径 | 两个概念如何连接 |
| `/graphify explain "X"` | 节点解释 | "SwinTransformer"是什么 |

**查询输出包括：** 节点标签、边类型、置信度标签、源文件、源位置——**可以直接作为prompt给LLM使用。**

---

### 2.7 增量更新与自动同步

```bash
# 手动增量更新（只处理变更文件）
graphify ./src --update

# 自动监控（代码变更即时，文档变更通知）
graphify ./src --watch

# Git hooks（提交后自动重建）
graphify hook install
```

**增量更新策略：**
- **代码文件变更：** AST提取（即时、免费、无LLM调用）
- **文档/图片变更：** 通知你运行 `--update`（需要LLM，不能自动）

---

### 2.8 多模态支持矩阵

| 文件类型 | 扩展名 | 提取方式 | 需要额外安装 |
|---------|--------|---------|-------------|
| 代码（25种语言） | .py/.ts/.go/.rs... | tree-sitter AST + 调用图 | 无需 |
| 文档 | .md/.html/.txt | Claude概念提取 | 无需 |
| Office文档 | .docx/.xlsx | 转markdown后提取 | `pip install graphifyy[office]` |
| 论文 | .pdf | 引用挖掘 + 概念提取 | 无需 |
| 图片 | .png/.jpg/.webp | Claude vision | 无需 |
| 视频/音频 | .mp4/.mp3/.mov... | faster-whisper本地转录 | `pip install graphifyy[video]` |
| YouTube/URL | 任意视频URL | yt-dlp下载 + Whisper | `pip install graphifyy[video]` |

**关键：** 视频/音频转录**完全本地**，音频不出机器。使用faster-whisper + yt-dlp。

---

### 2.9 导出生态

| 导出格式 | 命令 | 用途 |
|---------|------|------|
| 交互式HTML | 默认 | 浏览器中点击、搜索、过滤 |
| JSON | 默认 | 程序化查询、持久化 |
| Markdown报告 | 默认 | 人类可读的神节点和连接 |
| Obsidian Vault | `--obsidian` | 个人知识管理 |
| SVG | `--svg` | 矢量图嵌入文档 |
| GraphML | `--graphml` | Gephi/yEd可视化 |
| Neo4j | `--neo4j` | 图数据库导入 |
| MCP Server | `--mcp` | 给其他agent工具调用 |
| Wiki | `--wiki` | agent可爬取的markdown wiki |

---

### 2.10 团队协作工作流

```bash
# 1. 第一个人构建初始图谱并提交
git add graphify-out/
git commit -m "Add initial knowledge graph"

# 2. 队友pull后直接可用（助手自动读GRAPH_REPORT.md）

# 3. 安装post-commit hook，代码变更后自动重建
graphify hook install
```

**推荐的.gitignore：**
```gitignore
# 保留图谱输出，跳过本地文件
graphify-out/cache/        # 可选：提交加速，跳过保持仓库小
graphify-out/manifest.json # mtime文件，clone后失效
graphify-out/cost.json     # 本地token追踪
```

---

### 2.11 局限性与注意事项

1. **初始构建有成本：** 第一次运行需要LLM调用提取语义，后续查询免费
2. **图片/视频提取质量取决于LLM vision能力：** 复杂架构图可能提取不全
3. **需要提交graphify-out/：** 队友需要图谱文件才能受益
4. **小规模项目收益有限：** 6个文件的项目，图谱和直接读文件差别不大
5. **增量更新有延迟：** 文档变更后需要手动 `--update`，不能全自动

---

### 2.12 适合谁用？

- ✅ 接手大型代码库（>50文件）的开发者
- ✅ 需要管理大量论文/文档/笔记的研究者
- ✅ 团队需要共享项目理解的场景
- ✅ 长期维护的项目（上下文跨越数月）
- ✅ 多模态输入（代码+论文+视频+截图）的项目
- ❌ 小型脚本/单文件项目
- ❌ 一次性使用的临时项目
- ❌ 不想提交额外文件到repo的团队

---

## 三、Caveman vs Graphify：对比矩阵

| 维度 | Caveman | Graphify |
|------|---------|----------|
| **解决的问题** | Token成本、响应速度 | 项目理解、上下文检索 |
| **干预层级** | 输出层（改变说话方式） | 输入层（改变信息组织） |
| **技术栈** | Prompt工程 + Hooks | AST + 图算法 + LLM提取 |
| **是否需要LLM** | 不需要额外LLM | 需要（初始提取） |
| **增量成本** | 零（每次session自动） | 低（代码免费，文档需LLM） |
| **团队协作** | 个人工具 | 团队共享资产 |
| **最佳项目规模** | 任何规模 | 中大型（>50文件） |
| **硬件要求** | 无 | CPU即可，可选GPU加速embedding |
| **隐私** | 无额外数据传输 | 文档/图片内容发送给LLM API |
| **节省时间** | 每次查询省75% token | 71.5x信息压缩率 |

---

## 四、组合使用：1+1>2

**最佳实践：** 在同一个项目中同时使用Caveman和Graphify

```bash
# 1. 用Graphify构建项目知识图谱
graphify .
graphify claude install    # 设置always-on

# 2. 用Caveman压缩日常交互
claude plugin install caveman@caveman

# 3. 效果叠加
# - Claude先读GRAPH_REPORT.md（精准定位）
# - 然后用Caveman风格回答（省token）
# - 结果：又快又准又省钱
```

**实际场景模拟：**

```
用户："为什么我们的认证中间件会间歇性失败？"

【无工具】
Claude: grep "auth" → 读20个文件 → 长篇分析 → 2000 tokens

【只用Caveman】
Claude: grep "auth" → 读20个文件 → 简短回答 → 500 tokens（省75%）

【只用Graphify】
Claude: 读GRAPH_REPORT.md → 定位auth社区 → 精准读2个文件 → 详细分析 → 800 tokens

【Caveman + Graphify】
Claude: 读GRAPH_REPORT.md → 定位auth社区 → 精准读2个文件 → 简短回答 → 200 tokens
                        ↑ 精准定位              ↑ 压缩输出
                        = 比原始方案省90% token
```

---

## 五、技术细节补充

### Caveman的Eval验证

Caveman不是拍脑袋说"省75%"，它有严谨的eval框架：

```bash
# 运行评估（需要claude CLI）
uv run python evals/llm_run.py

# 测量token（无需API key，离线运行）
uv run --with tiktoken python evals/measure.py
```

**评估设计：** 三臂对照（verbose control vs terse control vs caveman skill），避免"只是让模型变简洁"的混淆变量。

### Graphify的MCP Server

```bash
# 启动MCP server
python -m graphify.serve graphify-out/graph.json
```

提供工具：
- `query_graph` — 自然语言查询
- `get_node` — 获取节点详情
- `get_neighbors` — 获取邻居节点
- `shortest_path` — 两节点最短路径

这让其他agent（如OpenClaw、Factory Droid）可以直接查询图谱，而不需要读取整个graph.json。

---

## 六、安装速查表

### Caveman

```bash
# Claude Code（推荐）
claude plugin marketplace add JuliusBrussee/caveman
claude plugin install caveman@caveman

# 其他平台
npx skills add JuliusBrussee/caveman -a <agent>

# 使用
/caveman           # 启动
/caveman ultra     # 极致模式
"stop caveman"     # 停止
```

### Graphify

```bash
# 安装
pip install graphifyy   # 注意：两个y
graphify install        # 安装到当前agent

# 构建图谱
graphify .              # 当前目录
graphify ./src --update # 增量更新
graphify . --watch      # 自动监控

# 安装always-on
graphify claude install   # Claude Code
graphify codex install    # Codex
graphify cursor install   # Cursor
# ... 等等

# Git hooks
graphify hook install     # 提交后自动重建
```

---

## 七、总结

| | Caveman | Graphify |
|--|---------|----------|
| **本质** | 语言压缩算法 | 知识结构化系统 |
| **成本模型** | 降低单次交互成本 | 降低项目理解门槛 |
| **收益曲线** | 线性（每次省固定比例） | 指数（项目越大收益越高） |
| **最佳搭档** | 高频使用Claude Code的开发者 | 维护/接手大型项目的技术负责人 |
| **一句话** | 让AI少说废话 | 让AI读懂项目 |

**两者结合 = 精准理解 + 高效表达 = 90% token节省**

---

> 📅 深度分析完成：2026-04-24  
> 🔗 完整报告已上传至 [csclip/AI热门项目调研/caveman-vs-graphify-deep-dive.md](https://github.com/520llw/csclip)
