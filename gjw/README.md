# gjw — 同轮多反馈 MCP 聊天桥

> 让 Cursor/VS Code AI 在一次对话中持续工作，通过浏览器或侧栏发送后续消息，无需反复开启新对话。

gjw 是一个基于 [Model Context Protocol (MCP)](https://modelcontextprotocol.io) 的本地聊天桥接服务。它在 AI 与浏览器之间建立双向通道，使 AI 能在完成回复后主动等待用户的下一条指令，形成"回复 → 等待 → 收到消息 → 继续回复"的永续循环。

## 核心特性

| 特性 | 说明 |
|------|------|
| **同轮多反馈** | AI 完成回复后自动调用 `check_messages` 进入等待状态，用户随时通过浏览器/侧栏补充指令 |
| **多会话隔离** | 左侧边栏管理多个独立会话，每个会话拥有独立的消息队列和等待状态 |
| **消息编辑与分支** | 编辑历史消息后自动清除后续内容并重新处理，支持对话分支 |
| **持久化存储** | 所有会话和消息自动保存到本地 JSON 文件，server 重启后数据不丢失 |
| **草稿自动保存** | 输入框内容自动保存到 localStorage，断线后可恢复 |
| **结构化模板** | 7 种预设模板（实验配置、结果汇报、代码审查、文献讨论、写作润色、记笔记、自由讨论） |
| **lab-notes 集成** | 选择"记笔记"模板时，AI 自动将内容整理后写入项目 `lab-notes/` 目录 |
| **AI 主动提问** | `ask_question` 工具支持单选/多选弹窗，AI 可在任务中向用户寻求决策 |
| **VS Code 侧栏扩展** | 可选的 VS Code/Cursor 扩展，将聊天界面嵌入编辑器侧栏 |
| **零后端依赖** | 纯本地运行，不依赖任何外部服务器或数据库 |

## 系统架构

```
┌─────────────────┐     MCP stdio      ┌─────────────┐
│   Cursor/VS Code │ ◄────────────────► │  gjw MCP    │
│   AI Agent       │   check_messages   │   Server    │
└─────────────────┘   ask_question     └──────┬──────┘
                                              │
                                              │ HTTP
                                              ▼
┌─────────────────────────────────────────────────────┐
│              Browser / VS Code Webview               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 会话列表  │  │ 聊天区域  │  │ 模板栏 + 输入框   │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
                              │
                              ▼
                    ~/.gjw/db.json（持久化）
```

**双通道设计**：
- **MCP stdio**：AI 通过 `check_messages`/`ask_question`/`edit_message` 工具与后端交互
- **HTTP Server**：浏览器通过 REST API 轮询新消息、发送消息、回答问题

## 快速开始

### 环境要求

- Node.js 18+
- Cursor 或 VS Code（支持 MCP）

### 1. 安装依赖

```bash
cd gjw
npm install
```

### 2. 配置 MCP

在项目 `.cursor/mcp.json` 中添加（将路径替换为实际路径）：

```json
{
  "mcpServers": {
    "gjw-mcp": {
      "command": "node",
      "args": ["/path/to/gjw/server.js"]
    }
  }
}
```

**Windows 示例**：
```json
"args": ["E:\\projects\\gjw\\server.js"]
```

**macOS/Linux 示例**：
```json
"args": ["/home/user/projects/gjw/server.js"]
```

### 3. 复制 AI 规则

将以下规则文件复制到你项目的 `.cursor/rules/` 目录：

| 文件 | 作用 |
|------|------|
| `gjw.mdc` | 强制 AI 每轮结束后调用 `check_messages` 等待用户消息 |
| `mcp-messenger.mdc` | MCP 工具调用规范（含 `ask_question`、`edit_message` 等） |
| `lab-notes.mdc`（可选） | lab-notes 知识库写入规范 |

### 4. 启动服务

```bash
node server.js
```

服务启动后会监听两个通道：
- **MCP stdio**：通过标准输入输出与 Cursor AI 通信
- **HTTP**：默认端口 `3456`（可通过环境变量 `GJW_PORT` 修改）

### 5. 打开聊天界面

**方式 A — 浏览器**：
访问 http://localhost:3456

**方式 B — VS Code 扩展**（推荐）：
1. 进入 `vscode-extension/` 目录
2. 按 [扩展 README](vscode-extension/README.md) 安装
3. 按 `Ctrl+Shift+X`（Mac: `Cmd+Shift+X`）打开侧栏面板

## 使用指南

### 基础聊天流程

1. 在 Cursor 中发起一次 Agent 对话
2. AI 完成回复后，规则强制其调用 `check_messages`，浏览器状态变为「等待输入...」
3. 在浏览器/侧栏输入后续指令，点击发送
4. AI 收到消息继续工作，完成后再次调用 `check_messages`
5. 循环往复，无需开启新对话

### 多会话管理

- 点击左侧「➕ 新建会话」创建新会话
- 点击会话名称切换
- 悬停会话显示「✕」删除（默认会话不可删除）
- 每个会话独立存储消息和等待状态

### 消息编辑

1. 将鼠标悬停在自己的消息上
2. 点击「编辑」按钮
3. 修改内容后确认
4. 该消息之后的所有内容自动清除，AI 重新从编辑点处理

### 结构化模板

输入框上方有 7 个模板按钮：

| 模板 | 用途 |
|------|------|
| 💬 自由讨论 | 默认模式，无特殊标记 |
| 🧪 实验配置 | 标记为实验配置消息 |
| 📊 结果汇报 | 标记为结果汇报消息 |
| 💻 代码审查 | 标记为代码审查消息 |
| 📚 文献讨论 | 标记为文献讨论消息 |
| ✍️ 写作润色 | 标记为写作润色消息 |
| 📓 记到 lab-notes | AI 收到专用 prompt，自动整理写入 `lab-notes/` |

### AI 主动提问

当 AI 需要你做选择时（如选框架、选方案），会调用 `ask_question` 工具：
- 浏览器弹出模态框显示问题和选项
- 支持单选和多选
- 每道题均可输入自定义补充文本
- 确认后 AI 根据回答继续工作

## MCP 工具文档

### `check_messages`

**描述**：检查用户新消息。AI 每轮回复完成后必须调用此工具。

**参数**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ai_response` | string | 否 | AI 的回复文本，会显示在聊天界面 |
| `session` | string | 否 | 目标会话 ID，省略则使用 `default` |

**返回值**：用户消息的 prompt 对象（含 text 和 images）。

### `ask_question`

**描述**：向用户提问并等待回答。用于需要用户做决策的场景。

**参数**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `questions` | array | 是 | 问题列表，每项含 `question`、`options`、`allow_multiple` |
| `session` | string | 否 | 目标会话 ID |

**示例**：
```json
{
  "questions": [
    {
      "question": "选择优化策略",
      "options": [
        { "id": "a", "label": "并行化" },
        { "id": "b", "label": "缓存" }
      ],
      "allow_multiple": false
    }
  ]
}
```

### `edit_message`

**描述**：编辑历史消息并触发重新处理。

**参数**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message_id` | number | 是 | 要编辑的消息 ID |
| `new_text` | string | 是 | 新消息文本 |
| `session` | string | 否 | 目标会话 ID |

## HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回聊天界面 HTML |
| GET | `/sessions` | 获取会话列表 |
| POST | `/sessions` | 创建新会话 `{name, projectPath?}` |
| DELETE | `/sessions?id=` | 删除会话 |
| GET | `/messages?session=&after=` | 获取历史消息 |
| GET | `/poll?session=&after=` | 轮询 AI 响应和状态 |
| POST | `/send` | 发送消息 `{session, message, images?, template?}` |
| POST | `/edit` | 编辑消息 `{session, messageId, newText}` |
| POST | `/answer` | 回答问题 `{qid, answers}` |

### Poll 响应格式

```json
{
  "waiting": true,
  "queueLength": 0,
  "responses": [
    { "id": 6, "text": "AI 回复内容", "time": "14:32" }
  ],
  "questions": [
    { "qid": "q-xxx", "questions": [...] }
  ]
}
```

## 数据持久化

所有数据存储在 `~/.gjw/db.json`：

```json
{
  "sessions": [
    { "id": "default", "name": "默认会话", "projectPath": "", "createdAt": "...", "updatedAt": "..." }
  ],
  "messages": [
    { "id": 1, "sessionId": "default", "role": "user", "text": "...", "images": [], "createdAt": "...", "edited": false, "template": null }
  ],
  "nextMessageId": 2,
  "nextQuestionId": 1
}
```

**安全设计**：
- 写入采用临时文件 + 原子 rename，崩溃/断电不丢数据
- 所有写操作通过文件级队列锁串行化，避免并发覆盖
- 请求体大小限制 10MB，防止 DoS
- CORS 仅允许 `localhost`/`127.0.0.1` 来源

## VS Code 扩展

详见 [vscode-extension/README.md](vscode-extension/README.md)。

扩展配置项（`settings.json`）：

```json
{
  "gjw.port": 3456
}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GJW_PORT` | `3456` | HTTP 服务端口 |

## 开发指南

### 项目结构

```
gjw/
├── server.js              # 主程序（MCP + HTTP）
├── package.json           # Node.js 依赖
├── .cursor/
│   ├── mcp.json           # MCP 配置模板
│   └── rules/
│       ├── gjw.mdc        # AI 规则：强制 check_messages
│       ├── mcp-messenger.mdc  # MCP 工具调用规范
│       └── lab-notes.mdc  # lab-notes 写入规范（可选）
├── vscode-extension/      # VS Code 侧栏扩展
│   ├── package.json
│   ├── extension.js
│   └── README.md
└── README.md              # 本文档
```

### 本地开发

```bash
# 启动服务（开发模式）
node server.js

# 修改代码后重启
# Cursor: Ctrl+Shift+P → "Developer: Reload Window"
```

## 安全注意事项

- 本服务仅监听 `localhost`，不对外暴露
- 不要修改 `.cursor/rules/` 中的安全相关条款
- 数据文件 `~/.gjw/db.json` 包含所有聊天记录，注意备份

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| 浏览器显示「未连接」 | server.js 未运行 | `node server.js` |
| AI 回复后浏览器无变化 | AI 未调用 `check_messages` | 确认规则文件已复制到 `.cursor/rules/` |
| 发送消息后 AI 无响应 | MCP 连接断开 | 重启 Cursor / 重新加载窗口 |
| 消息显示重复 | 极罕见竞态 | 刷新浏览器页面 |
| server 重启后消息丢失 | `db.json` 损坏 | 检查 `~/.gjw/` 目录下的 `.tmp` 文件 |

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.0 | 2025-04 | 全面重构：持久化、多会话、消息编辑、模板、lab-notes、VS Code 扩展 |
| v1.x | 2025-03 | 初代实现：基础 check_messages 桥接 |

## 许可证

MIT
