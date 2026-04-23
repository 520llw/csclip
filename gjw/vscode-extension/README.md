# gjw VS Code 扩展

将 gjw 的浏览器聊天界面嵌入到 Cursor/VS Code 侧栏，无需切窗口即可收发消息。

## 功能

- **侧栏嵌入**：gjw 的完整界面内嵌在 Cursor 侧栏，与编辑器并排
- **保留上下文**：切换编辑器标签或隐藏面板后，聊天状态不丢失
- **快捷键**：`Ctrl+Shift+X`（Mac: `Cmd+Shift+X`）快速打开/聚焦面板
- **刷新命令**：`gjw: 刷新面板`（命令面板中搜索）
- **端口可配置**：支持自定义 gjw MCP 服务端口

## 安装方法

### 方式 1：直接加载（推荐，开发调试）

1. 确保 `server.js` 已运行：`node server.js`
2. 在 Cursor/VS Code 中按 `Ctrl+Shift+P` → `Extensions: Install from VSIX...`
3. 或者按 `Ctrl+Shift+P` → `Developer: Install Extension from Location...`
4. 选择本目录 `vscode-extension/`
5. 重启 Cursor/VS Code
6. 左侧活动栏点击"gjw"图标，或按 `Ctrl+Shift+X` 打开面板

### 方式 2：打包成 .vsix（分发）

```bash
# 需要安装 vsce
npm install -g @vscode/vsce

# 进入扩展目录
cd vscode-extension

# 打包
vsce package --allow-missing-repository

# 安装生成的 .vsix 文件
# 在 Cursor 中：Ctrl+Shift+P → Extensions: Install from VSIX...
```

## 配置

在 `settings.json` 中修改端口：

```json
{
  "gjw.port": 3456
}
```

或在设置面板中搜索 "gjw" → 修改 "gjw MCP 服务端口"。

## 故障排查

| 问题 | 解决 |
|------|------|
| 面板显示"正在连接 gjw..." | 确认 `server.js` 已运行（`node server.js`）且端口一致 |
| 面板空白 | 点击命令面板 → `gjw: 刷新面板` |
| 扩展未激活 | 重启 Cursor/VS Code |
| 端口冲突 | 修改 `GJW_PORT` 环境变量启动 server，并在设置中同步修改 `gjw.port` |

## 与原生浏览器的区别

| | 浏览器 | VS Code 扩展 |
|--|--------|-------------|
| 窗口位置 | 独立浏览器窗口 | 嵌入侧栏 |
| 切窗口 | 需要 Alt+Tab | 点击侧栏图标即可 |
| 功能完整性 | ✅ 100% | ✅ 100%（iframe 加载） |
| 多显示器 | 可拖到外屏 | 只能在 Cursor 内 |

推荐：主力使用扩展，需要大屏幕时切到浏览器（http://localhost:3456）。

## 扩展结构

```
vscode-extension/
├── package.json      # 扩展配置（视图、命令、快捷键、配置项）
├── extension.js      # 扩展主代码（WebviewProvider）
└── README.md         # 本文档
```

`extension.js` 创建一个 `WebviewViewProvider`，在侧栏注入 iframe 加载 `http://localhost:${port}`。`retainContextWhenHidden: true` 确保隐藏面板后状态不丢失。
