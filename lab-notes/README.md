# lab-notes · 经验 / 实验 / 文献知识库

> 与 **`.cursor/rules/lab-notes.mdc`** 联动。把研究过程写成**项目内的 Markdown**，换 Chat、换 AI、写论文都能溯源。  
> **gjw 面板**里请用 **🔬 科研 → 📓 记录** 分类下的模板驱动 AI 维护本目录（勿只靠口头说「记一下」）。

---

## 1. 这套东西解决什么问题

| 痛点 | lab-notes 的做法 |
|------|------------------|
| 对话一关，实验细节没了 | 每条实验一条 `experiments/*.md`，带日期与 slug |
| 新开的 AI 不知道你做过什么 | 先更新 `HANDOFF.md`，新会话 `@HANDOFF.md` 或让 AI 读它 |
| 写论文时找不到数字从哪来 | 论文章节草稿要求**标注来源文件路径** |
| 只记成功、不记失败 | `insights/` 专门沉淀教训，规则鼓励写清根因 |

---

## 2. 目录结构（与规则一致）

```
lab-notes/
├── README.md                 # 本文件：人类使用说明
├── INDEX.md                  # 总索引表（新条目在顶部追加一行）
├── HANDOFF.md                # 接手简报（换人或新 Chat 时先读）
├── experiments/              # 实验日志
│   └── YYYY-MM-DD-<slug>.md
│   └── _template.md          # 手工开写时可复制
├── insights/                 # 经验 / 失败 / trick
│   └── YYYY-MM-DD-<slug>.md
│   └── _template.md
└── literature/               # 文献笔记
    └── <year>-<AuthorLast>-<slug>.md
    └── _template.md
```

- **slug**：小写英文 + 连字符，短且可检索，例如 `lr-warmup-sweep`。
- **日期**：文件名里的日期建议用**写入当日**（ISO `YYYY-MM-DD`），与 frontmatter 的 `date` 对齐。

---

## 3. 使用前置条件（否则 AI 建不了文件）

1. 在 **Cursor** 里打开**正确的项目根目录**（根目录下能看到 `lab-notes/` 与 `.cursor/rules/`）。
2. 使用 **Agent 模式**，并已通过 **gjw / 寸止** 正常进入「等待反馈」状态（反馈框可编辑）。
3. 在 **🔬 科研 → 📓 记录** 中选模板，把 `[在此填入…]` 补全后 **提交**；若 AI 未落盘，在对话里明确一句：「请在本项目根目录的 `lab-notes/` 下创建/更新上述文件」。

---

## 4. 推荐工作流（按天）

### 第一次用（初始化）

1. 打开 gjw → **🔬 科研** → 分类 **📓 记录** → 点 **「知识库 · 初始化」**。
2. 在反馈框补全：当前课题方向、是否已有部分实验（可选）。
3. 提交后，AI 应：确认规则、补齐缺失目录/文件、更新 `INDEX.md`（若已有条目则登记，**不编造**不存在的实验）。

### 日常跑实验

| 阶段 | 建议使用的模板 | 产出 |
|------|------------------|------|
| 改配置 / 开跑前 | **知识库 · 新实验日志** | `experiments/YYYY-MM-DD-<slug>.md`，`status: running` |
| 出结果后 | **知识库 · 追加结果** | 在原文件末尾增加 `## 更新 …` 段，更新指标与结论 |
| 踩坑或顿悟 | **知识库 · 经验总结** | `insights/…md`，severity 标高若影响全局 |
| 读完一篇论文 | **知识库 · 文献笔记** | `literature/…md`，不确定引用处标注「待核实」 |

### 换 Cursor 对话 / 换 AI 接手

1. 跑 **「知识库 · 接手简报」**，生成/覆盖 **`HANDOFF.md`**（规则要求保留旧版有价值摘要）。
2. 新对话第一条： `@lab-notes/HANDOFF.md` 或「请先阅读 HANDOFF 再回答」。
3. 需要细节时再 `@lab-notes/experiments/某文件.md`。

### 写论文

1. 跑 **「知识库 · 生成论文章节」**，在模板里填目标章节（Experiments / Discussion / Limitations 等）。
2. 要求输出中**每个关键数字**括号注明来源，如 `(lab-notes/experiments/2026-04-17-xxx.md)`。
3. 日志里没写的数字 = **不允许编造**，应标 `TODO: 需补`。

---

## 5. 各文件职责速查

| 文件 | 谁维护 | 何时读 |
|------|--------|--------|
| `INDEX.md` | AI 每次新增/更新条目后在**顶部**追加一行 | 想浏览全库时 |
| `HANDOFF.md` | 跑「接手简报」时 **覆盖**更新 | **新会话第一件事** |
| `experiments/*` | 每次实验一条；结果用「追加结果」叠代 | 写方法、写消融、对比表格 |
| `insights/*` | 可复用准则、失败根因 | Discussion、Limitations、避坑 |
| `literature/*` | 论文笔记 + BibTeX | Related Work、对比方法 |

---

## 6. Frontmatter 必填字段（摘要）

完整定义见 **`lab-notes.mdc`**。常见类型：

- **experiment**：`date` `type` `exp_id` `status` `tags` `seed` `commit`（可选）
- **insight**：`date` `type` `tags` `related_exps` `severity`
- **literature**：`date` `type` `cite_key` `venue` `tags` `rating` 等

---

## 7. 与 Git、隐私

- **推荐**将 `lab-notes/` 纳入 Git，与代码、论文草稿同库；`commit:` 字段可与某次提交对应。
- 若含**未公开数据、匿名审稿材料**，可单独建 `lab-notes/private/` 并写入 **`.gitignore`**，规则同样适用但勿误提交公开仓库。

---

## 8. 常见问题（FAQ）

**Q：INDEX 表格乱了怎么办？**  
A：以 `lab-notes.mdc` 为准手工修表头；或让 AI「仅按 INDEX 规范重排，不删历史行」。

**Q：HANDOFF 太长？**  
A：模板已限制扫描条数；可要求 AI 只保留最近 30 条实验与 5 条高 severity insight。

**Q：实验做了一半想改标题/slug？**  
A：重命名文件并全局替换引用；或新开一条实验日志说明「承接自某 slug」，旧文件保留作历史。

**Q：可以只用模板、不装 gjw 吗？**  
A：可以。把 📓 记录里模板文字复制到任意 Chat，只要项目里有 `lab-notes/` 与 `lab-notes.mdc`，AI 仍应遵守规则；但 **同轮多反馈省额度** 依赖 gjw 的交互流程。

---

## 9. 与 `research-workflow.mdc` 的关系

- **research-workflow**：约束「回答内容」的学术规范。  
- **lab-notes**：约束「项目里怎么记」。  
二者同时生效：写论文型输出时既要有严谨叙述，也要有可追溯的 lab-notes 来源。

---

## 10. 模板文件（手工开写时）

- `experiments/_template.md`
- `insights/_template.md`
- `literature/_template.md`

复制后改文件名与 frontmatter；更推荐用 gjw **📓 记录** 模板让 AI 生成，减少格式错误。

---

更多关于 **gjw 面板、MCP、额度心智模型**，见项目根目录 **`gjw-使用说明.md`**。
