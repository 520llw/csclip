import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

/* ═══════════════════════════════════════════════════════════════════════════
 * gjw v2.0 — 科研增强版
 * 重构内容：持久化存储 | 草稿保存 | 消息编辑 | 多会话隔离 | 结构化模板
 * ═══════════════════════════════════════════════════════════════════════════ */

const PORT = parseInt(process.env.GJW_PORT || "3456", 10);
const DATA_DIR = path.join(os.homedir(), ".gjw");
const DB_PATH = path.join(DATA_DIR, "db.json");

// ── 内存状态（按 sessionId 隔离）─────────────────────────────────
const waitingResolve = {};   // sessionId -> resolve function | null
const messageQueue = {};     // sessionId -> array of pending payloads
const aiResponses = {};      // sessionId -> array of {id, text, time, questionId?}
const pendingQuestions = {}; // questionId -> { questions, resolve, sessionId, createdAt }

// ── DB 并发锁 ────────────────────────────────────────────────────
let dbPromise = null;
async function withDB(fn) {
  while (dbPromise) {
    try { await dbPromise; } catch {}
  }
  dbPromise = (async () => {
    const db = await loadDB();
    const result = await fn(db);
    await saveDB(db);
    return result;
  })();
  try {
    return await dbPromise;
  } finally {
    dbPromise = null;
  }
}

// ── 数据层（零依赖 JSON 持久化）───────────────────────────────────
async function ensureDataDir() {
  try { await fs.mkdir(DATA_DIR, { recursive: true }); } catch {}
}

async function loadDB() {
  try {
    const raw = await fs.readFile(DB_PATH, "utf-8");
    return JSON.parse(raw);
  } catch {
    return { sessions: [], messages: [], nextMessageId: 1, nextQuestionId: 1 };
  }
}

async function saveDB(db) {
  await ensureDataDir();
  const tmpPath = DB_PATH + ".tmp";
  await fs.writeFile(tmpPath, JSON.stringify(db, null, 2), "utf-8");
  await fs.rename(tmpPath, DB_PATH);
}

async function createSession(name, projectPath) {
  const sess = await withDB((db) => {
    const id = "sess-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 6);
    const s = { id, name: name || "新会话", projectPath: projectPath || "", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    db.sessions.push(s);
    return s;
  });
  // 初始化内存状态
  waitingResolve[sess.id] = null;
  messageQueue[sess.id] = [];
  aiResponses[sess.id] = [];
  return sess;
}

async function getSessions() {
  return withDB((db) => {
    // 如果没有 default 会话，自动创建
    if (!db.sessions.find(s => s.id === "default")) {
      db.sessions.unshift({ id: "default", name: "默认会话", projectPath: "", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() });
      waitingResolve["default"] = null;
      messageQueue["default"] = [];
      aiResponses["default"] = [];
    }
    return [...db.sessions].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  });
}

async function deleteSession(sessionId) {
  if (sessionId === "default") return false;
  await withDB((db) => {
    db.sessions = db.sessions.filter(s => s.id !== sessionId);
    db.messages = db.messages.filter(m => m.sessionId !== sessionId);
  });
  delete waitingResolve[sessionId];
  delete messageQueue[sessionId];
  delete aiResponses[sessionId];
  return true;
}

async function saveMessage(sessionId, role, text, images, extra = {}) {
  return withDB((db) => {
    const id = db.nextMessageId++;
    const msg = {
      id,
      sessionId,
      role,
      text: text || "",
      images: images || [],
      createdAt: new Date().toISOString(),
      edited: false,
      parentId: extra.parentId || null,
      originalId: extra.originalId || null,
      template: extra.template || null,
    };
    db.messages.push(msg);
    // 更新会话 updatedAt
    const sess = db.sessions.find(s => s.id === sessionId);
    if (sess) sess.updatedAt = new Date().toISOString();
    return msg;
  });
}

async function getMessages(sessionId, afterId = 0) {
  const db = await loadDB();
  return db.messages
    .filter(m => m.sessionId === sessionId && m.id > afterId)
    .sort((a, b) => a.id - b.id);
}

async function editMessage(messageId, newText) {
  return withDB((db) => {
    const msg = db.messages.find(m => m.id === messageId);
    if (!msg) return null;
    msg.text = newText;
    msg.edited = true;
    msg.updatedAt = new Date().toISOString();
    return msg;
  });
}

async function deleteMessagesAfter(sessionId, messageId) {
  await withDB((db) => {
    const target = db.messages.find(m => m.id === messageId && m.sessionId === sessionId);
    if (!target) return;
    db.messages = db.messages.filter(m => !(m.sessionId === sessionId && m.id > messageId));
  });
}

// 初始化：确保 default 会话存在
await getSessions();

// ── MCP Server ──────────────────────────────────────────────────
const mcpServer = new McpServer({
  name: "gjw-message",
  version: "2.0.0",
});

function buildPrompt(payload) {
  const text = typeof payload === "string" ? payload : payload?.text || "";
  const images = Array.isArray(payload?.images) ? payload.images : [];
  const template = payload?.template || "";

  let promptText;
  if (template === "记笔记") {
    promptText = `用户要求将以下内容记入 lab-notes/ 知识库：\n\n${text || "(无文字，见下方图片)"}\n\n【你的任务】\n请按项目根目录下 .cursor/rules/lab-notes.mdc 的规范，将上述内容整理后写入 lab-notes/ 目录。\n\n步骤：\n1. 判断内容类型：experiment（实验）/ insight（经验）/ literature（文献）\n2. 生成符合规范的 Markdown 文件，包含正确的 frontmatter\n3. 在 lab-notes/INDEX.md 顶部追加索引行\n4. 如果是 insight，确保 severity 标注准确；如果是 experiment，包含 exp_id 和 status\n\n文件命名规范：\n- experiments/YYYY-MM-DD-<slug>.md\n- insights/YYYY-MM-DD-<slug>.md\n- literature/<year>-<AuthorLast>-<slug>.md\n\n注意：\n- 只写入用户明确提供的信息，不编造数据\n- 不确定的引用标注"（待核实）"\n- 完成后调用 check_messages 继续等待下一条指令`;
  } else if (template) {
    promptText = `[结构化消息 · ${template}]\n\n用户发来新消息:\n\n${text || "(无文字，见下方图片)"}\n\n请根据上述消息（含图片时请结合图片内容）继续工作。完成后再次调用 check_messages 工具，将你的回复内容传入 ai_response 参数。`;
  } else {
    promptText = `用户发来新消息:\n\n${text || "(无文字，见下方图片)"}\n\n请根据上述消息（含图片时请结合图片内容）继续工作。完成后再次调用 check_messages 工具，将你的回复内容传入 ai_response 参数。`;
  }

  const content = [{ type: "text", text: promptText }];
  images.forEach((img) => {
    const mime = img.mime || "image/png";
    let data = img.data || "";
    if (data.startsWith("data:")) {
      data = data.replace(/^data:[^;]+;base64,/, "");
    }
    content.push({ type: "image", data, mimeType: mime });
  });
  return { content };
}

mcpServer.tool(
  "check_messages",
  "Check for new user messages from gjw. Call this after completing every response to wait for the next user instruction.",
  {
    ai_response: z.string().optional().describe("Your response text to display in the chat UI before waiting for the next message"),
    session: z.string().optional().describe("Target session ID. Omit to use 'default'"),
  },
  async ({ ai_response, session }) => {
    const sid = session || "default";
    if (!waitingResolve[sid]) waitingResolve[sid] = null;
    if (!messageQueue[sid]) messageQueue[sid] = [];
    if (!aiResponses[sid]) aiResponses[sid] = [];

    if (ai_response) {
      // 使用 saveMessage 返回的真实消息 ID，确保 server 重启后浏览器仍能正确轮询
      const msg = await saveMessage(sid, "ai", ai_response, []);
      aiResponses[sid].push({ id: msg.id, text: ai_response, time: new Date().toLocaleTimeString() });
    }

    if (messageQueue[sid].length > 0) {
      return buildPrompt(messageQueue[sid].shift());
    }

    const msg = await new Promise((resolve) => {
      waitingResolve[sid] = resolve;
    });
    waitingResolve[sid] = null;
    return buildPrompt(msg);
  }
);

mcpServer.tool(
  "ask_question",
  "Ask the user a multiple-choice or single-choice question and wait for their answer. Use this when you need the user to make a decision.",
  {
    questions: z.array(z.object({
      question: z.string(),
      options: z.array(z.object({ id: z.string(), label: z.string() })),
      allow_multiple: z.boolean().optional(),
    })).describe("List of questions to ask"),
    session: z.string().optional(),
  },
  async ({ questions, session }) => {
    const sid = session || "default";
    const qid = "q-" + Date.now().toString(36);
    pendingQuestions[qid] = { questions, resolve: null, sessionId: sid, createdAt: Date.now() };

    // 推送到 AI responses 让浏览器展示问题
    const text = questions.map((q, i) => {
      const opts = q.options.map(o => `  [${o.id}] ${o.label}`).join("\n");
      return `${i + 1}. ${q.question}${q.allow_multiple ? "（可多选）" : "（单选）"}\n${opts}`;
    }).join("\n\n");

    // 使用 saveMessage 返回的真实消息 ID
    const msg = await saveMessage(sid, "ai", `❓ 等待用户回答:\n\n${text}`, []);
    aiResponses[sid].push({ id: msg.id, text: `❓ 等待用户回答:\n\n${text}`, time: new Date().toLocaleTimeString(), questionId: qid });

    const answer = await new Promise((resolve) => {
      pendingQuestions[qid].resolve = resolve;
    });
    delete pendingQuestions[qid];

    return {
      content: [
        { type: "text", text: `用户回答:\n${JSON.stringify(answer, null, 2)}\n\n请根据回答继续工作。完成后再次调用 check_messages。` },
      ],
    };
  }
);

mcpServer.tool(
  "edit_message",
  "Edit a previously sent user message and trigger re-processing from that point. Useful when the user wants to correct a message.",
  {
    message_id: z.number().describe("The ID of the message to edit"),
    new_text: z.string().describe("The new message text"),
    session: z.string().optional(),
  },
  async ({ message_id, new_text, session }) => {
    const sid = session || "default";
    const msg = await editMessage(message_id, new_text);
    if (!msg) {
      return { content: [{ type: "text", text: "消息不存在或无法编辑。" }] };
    }
    // 删除该消息之后的所有消息（分支对话）
    await deleteMessagesAfter(sid, message_id);
    // 清理内存中的 aiResponses（与 HTTP /edit 保持一致）
    if (aiResponses[sid]) {
      aiResponses[sid] = aiResponses[sid].filter(r => r.id <= message_id);
    }
    // 把编辑后的消息加入队列，让 AI 重新处理
    messageQueue[sid].push({ text: new_text, images: msg.images, template: msg.template });
    return { content: [{ type: "text", text: `消息 ${message_id} 已更新，后续历史已清除，将重新处理。` }] };
  }
);

const transport = new StdioServerTransport();
await mcpServer.connect(transport);
console.error("[gjw-mcp] MCP server connected v2.0");


// ── HTTP Server ─────────────────────────────────────────────────
const httpServer = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  // CORS: 仅允许本地来源
  const origin = req.headers.origin;
  if (origin && (origin.startsWith("http://localhost:") || origin.startsWith("http://127.0.0.1:"))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
  }
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  // GET / — 聊天界面
  if (req.method === "GET" && url.pathname === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
    res.end(HTML);
    return;
  }

  // GET /sessions — 会话列表
  if (req.method === "GET" && url.pathname === "/sessions") {
    const sessions = await getSessions();
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ sessions }));
    return;
  }

  // POST /sessions — 创建会话
  if (req.method === "POST" && url.pathname === "/sessions") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "invalid json" }));
      return;
    }
    const { name, projectPath } = body;
    const sess = await createSession(name, projectPath);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true, session: sess }));
    return;
  }

  // DELETE /sessions — 删除会话
  if (req.method === "DELETE" && url.pathname === "/sessions") {
    const id = url.searchParams.get("id");
    const ok = await deleteSession(id);
    res.writeHead(ok ? 200 : 400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok }));
    return;
  }

  // GET /messages?session=&after= — 历史消息
  if (req.method === "GET" && url.pathname === "/messages") {
    const sid = url.searchParams.get("session") || "default";
    const after = parseInt(url.searchParams.get("after") || "0", 10);
    const messages = await getMessages(sid, after);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ messages }));
    return;
  }

  // GET /poll?session=&after= — 轮询状态
  if (req.method === "GET" && url.pathname === "/poll") {
    const sid = url.searchParams.get("session") || "default";
    const afterId = parseInt(url.searchParams.get("after") || "0", 10);
    if (!aiResponses[sid]) aiResponses[sid] = [];
    const newResponses = aiResponses[sid].filter((r) => r.id > afterId);
    const questions = Object.entries(pendingQuestions)
      .filter(([_, q]) => q.sessionId === sid)
      .map(([qid, q]) => ({ qid, questions: q.questions }));
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      waiting: waitingResolve[sid] !== null && waitingResolve[sid] !== undefined,
      queueLength: messageQueue[sid]?.length || 0,
      responses: newResponses,
      questions,
    }));
    return;
  }

  // POST /send — 发送消息
  if (req.method === "POST" && url.pathname === "/send") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "invalid json" }));
      return;
    }
    const { session, message, images, template } = body;
    const sid = session || "default";
    const hasText = message && String(message).trim();
    const hasImages = Array.isArray(images) && images.length > 0;
    if (!hasText && !hasImages) {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "empty" }));
      return;
    }
    const payload = {
      text: hasText ? String(message).trim() : "",
      images: hasImages ? images : [],
      template: template || null,
    };
    const saved = await saveMessage(sid, "user", payload.text, payload.images, { template: payload.template });
    if (waitingResolve[sid]) {
      waitingResolve[sid](payload);
      waitingResolve[sid] = null;
    } else {
      if (!messageQueue[sid]) messageQueue[sid] = [];
      messageQueue[sid].push(payload);
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true, messageId: saved.id }));
    return;
  }

  // POST /edit — 编辑消息
  if (req.method === "POST" && url.pathname === "/edit") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "invalid json" }));
      return;
    }
    const { session, messageId, newText } = body;
    const sid = session || "default";
    const msg = await editMessage(messageId, newText);
    if (!msg) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "not found" }));
      return;
    }
    await deleteMessagesAfter(sid, messageId);
    // 清理内存中的 aiResponses（只保留编辑点之前的）
    if (aiResponses[sid]) {
      aiResponses[sid] = aiResponses[sid].filter(r => r.id <= messageId);
    }
    // 把编辑后的消息加入队列
    if (!messageQueue[sid]) messageQueue[sid] = [];
    messageQueue[sid].push({ text: newText, images: msg.images, template: msg.template });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true, message: msg }));
    return;
  }

  // POST /answer — 回答问题
  if (req.method === "POST" && url.pathname === "/answer") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "invalid json" }));
      return;
    }
    const { qid, answers } = body;
    const q = pendingQuestions[qid];
    if (q && q.resolve) {
      q.resolve(answers);
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  res.writeHead(404);
  res.end("Not Found");
});

function readBody(req, maxBytes = 10 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let body = "";
    let length = 0;
    req.on("data", (chunk) => {
      length += chunk.length;
      if (length > maxBytes) {
        req.destroy();
        reject(new Error("request body too large"));
        return;
      }
      body += chunk;
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

httpServer.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    const alt = PORT + 1;
    console.error(`[gjw-mcp] Port ${PORT} in use, trying ${alt}`);
    httpServer.listen(alt, () => console.error(`[gjw-mcp] Chat UI ready: http://localhost:${alt}`));
  } else {
    console.error("[gjw-mcp] HTTP error:", err);
  }
});

httpServer.listen(PORT, () => {
  console.error(`[gjw-mcp] Chat UI ready: http://localhost:${PORT}`);
});

// 定期清理超时未回答的问题（30分钟）
const QUESTION_TIMEOUT_MS = 30 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  Object.entries(pendingQuestions).forEach(([qid, q]) => {
    if (q.createdAt && now - q.createdAt > QUESTION_TIMEOUT_MS) {
      if (q.resolve) q.resolve(null);
      delete pendingQuestions[qid];
    }
  });
}, 60 * 1000);


// ── Chat HTML（前端）────────────────────────────────────────────
const HTML = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>gjw · 科研增强版</title>
  <style>
    :root {
      --bg-primary: #0d1117; --bg-secondary: #161b22; --bg-tertiary: #21262d;
      --bg-hover: #30363d; --border: rgba(255,255,255,0.08);
      --text-primary: #e6edf3; --text-secondary: #8b949e; --text-muted: #6e7681;
      --accent: #58a6ff; --accent-hover: #79b8ff; --accent-glow: rgba(88,166,255,0.2);
      --success: #3fb950; --danger: #f85149; --warn: #d29922;
      --user-bubble: linear-gradient(135deg, #238636, #2ea043);
      --ai-bubble: #21262d; --radius: 12px;
      --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    html,body { height:100%; font-family:var(--font); background:var(--bg-primary); color:var(--text-primary); overflow:hidden; }

    /* Layout */
    .app { display:flex; height:100vh; }
    .sidebar {
      width:260px; background:var(--bg-secondary); border-right:1px solid var(--border);
      display:flex; flex-direction:column; flex-shrink:0;
    }
    .sidebar-header {
      padding:16px; border-bottom:1px solid var(--border);
      display:flex; align-items:center; gap:10px;
    }
    .logo {
      width:32px; height:32px; border-radius:8px;
      background:var(--user-bubble); color:#fff;
      display:flex; align-items:center; justify-content:center; font-size:16px;
    }
    .sidebar-header h2 { font-size:14px; font-weight:700; }
    .sidebar-header p { font-size:11px; color:var(--text-muted); }

    .new-session-btn {
      margin:12px 16px; padding:8px 12px; border-radius:8px;
      background:var(--bg-tertiary); border:1px dashed var(--border);
      color:var(--text-secondary); font-size:12px; cursor:pointer;
      display:flex; align-items:center; justify-content:center; gap:6px;
      transition:all .2s;
    }
    .new-session-btn:hover { border-color:var(--accent); color:var(--accent); }

    .session-list { flex:1; overflow-y:auto; padding:0 8px 12px; }
    .session-item {
      display:flex; align-items:center; gap:8px;
      padding:10px 12px; border-radius:8px; cursor:pointer;
      font-size:13px; color:var(--text-secondary); margin-bottom:4px;
      transition:background .15s; position:relative;
    }
    .session-item:hover { background:var(--bg-hover); }
    .session-item.active { background:rgba(88,166,255,0.12); color:var(--accent); }
    .session-item .s-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .session-item .s-del {
      opacity:0; width:22px; height:22px; display:flex; align-items:center; justify-content:center;
      border-radius:4px; font-size:12px; color:var(--danger); cursor:pointer;
    }
    .session-item:hover .s-del { opacity:1; }
    .session-item .s-del:hover { background:rgba(248,81,73,0.15); }

    .main { flex:1; display:flex; flex-direction:column; min-width:0; }

    /* Header */
    .header {
      padding:12px 20px; background:var(--bg-secondary); border-bottom:1px solid var(--border);
      display:flex; align-items:center; justify-content:space-between; flex-shrink:0;
    }
    .header-title { font-size:14px; font-weight:600; }
    .status-pill {
      display:flex; align-items:center; gap:6px; font-size:12px;
      padding:5px 12px; border-radius:20px; background:var(--bg-tertiary);
      color:var(--text-secondary); border:1px solid var(--border);
    }
    .status-dot { width:7px; height:7px; border-radius:50%; background:var(--danger); transition:background .3s; }
    .status-pill.waiting { background:rgba(63,185,80,0.12); border-color:rgba(63,185,80,0.3); color:var(--success); }
    .status-pill.waiting .status-dot { background:var(--success); animation:blink 1.5s infinite; }
    .status-pill.connected .status-dot { background:var(--accent); }
    .status-pill.offline .status-dot { background:var(--text-muted); }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

    /* Templates bar */
    .templates-bar {
      display:flex; gap:6px; padding:8px 20px; background:var(--bg-primary);
      border-bottom:1px solid var(--border); overflow-x:auto; flex-shrink:0;
    }
    .tpl-btn {
      padding:5px 10px; border-radius:6px; font-size:11px; font-weight:500;
      background:var(--bg-tertiary); border:1px solid var(--border);
      color:var(--text-muted); cursor:pointer; white-space:nowrap; transition:all .15s;
    }
    .tpl-btn:hover { color:var(--text-secondary); border-color:var(--border); }
    .tpl-btn.active { background:rgba(88,166,255,0.15); border-color:var(--accent); color:var(--accent); }

    /* Messages */
    #messages {
      flex:1; overflow-y:auto; padding:20px 24px;
      display:flex; flex-direction:column; gap:14px; scroll-behavior:smooth;
    }
    #messages::-webkit-scrollbar { width:6px; }
    #messages::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.12); border-radius:10px; }
    .msg-row { display:flex; gap:10px; animation:fadeIn .3s ease; }
    .msg-row.user { flex-direction:row-reverse; }
    @keyframes fadeIn { from {opacity:0;transform:translateY(8px);} to {opacity:1;transform:translateY(0);} }
    .avatar {
      width:30px; height:30px; border-radius:8px; flex-shrink:0;
      display:flex; align-items:center; justify-content:center; font-size:13px;
    }
    .msg-row.user .avatar { background:var(--user-bubble); color:#fff; }
    .msg-row.ai .avatar { background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-secondary); }
    .bubble-wrap { max-width:75%; display:flex; flex-direction:column; gap:4px; }
    .bubble {
      padding:10px 14px; border-radius:var(--radius); font-size:13.5px; line-height:1.65;
      word-break:break-word; white-space:pre-wrap;
    }
    .msg-row.user .bubble {
      background:var(--user-bubble); color:#fff;
      border-bottom-right-radius:5px;
    }
    .msg-row.ai .bubble {
      background:var(--ai-bubble); color:var(--text-primary);
      border:1px solid var(--border); border-bottom-left-radius:5px;
    }
    .bubble .time { font-size:10px; margin-top:4px; opacity:.55; }
    .bubble-actions {
      display:flex; gap:6px; opacity:0; transition:opacity .15s;
    }
    .msg-row:hover .bubble-actions { opacity:1; }
    .bubble-actions button {
      padding:2px 8px; border-radius:4px; font-size:11px;
      background:var(--bg-hover); border:1px solid var(--border);
      color:var(--text-muted); cursor:pointer;
    }
    .bubble-actions button:hover { color:var(--text-secondary); }

    .empty-state {
      flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;
      gap:14px; color:var(--text-muted); padding:40px;
    }
    .empty-icon { font-size:40px; }
    .empty-state h3 { font-size:15px; font-weight:600; color:var(--text-secondary); }
    .empty-state p { font-size:12.5px; line-height:1.8; text-align:center; max-width:420px; }

    /* Input */
    .input-area {
      padding:12px 20px 16px; background:var(--bg-secondary);
      border-top:1px solid var(--border); flex-shrink:0;
    }
    .input-wrapper {
      display:flex; align-items:flex-end; gap:8px;
      background:var(--bg-primary); border:1px solid var(--border);
      border-radius:14px; padding:5px 5px 5px 14px;
      transition:border-color .2s, box-shadow .2s;
    }
    .input-wrapper:focus-within {
      border-color:rgba(88,166,255,0.45);
      box-shadow:0 0 0 3px var(--accent-glow);
    }
    #input {
      flex:1; padding:9px 0; border:none; background:transparent;
      color:var(--text-primary); font-size:14px; outline:none;
      resize:none; min-height:24px; max-height:120px; font-family:var(--font); line-height:1.5;
    }
    #input::placeholder { color:var(--text-muted); }
    #sendBtn {
      width:36px; height:36px; border-radius:10px;
      background:var(--accent); color:white; border:none;
      cursor:pointer; display:flex; align-items:center; justify-content:center;
      flex-shrink:0; transition:background .2s;
    }
    #sendBtn:hover { background:var(--accent-hover); }
    #sendBtn:disabled { opacity:.4; cursor:not-allowed; }
    .input-tools {
      display:flex; align-items:center; gap:8px; margin-top:8px;
    }
    .input-tools button {
      padding:4px 10px; border-radius:6px; font-size:11px;
      background:transparent; border:1px solid var(--border);
      color:var(--text-muted); cursor:pointer;
    }
    .input-tools button:hover { color:var(--text-secondary); border-color:var(--text-muted); }
    .draft-hint { font-size:11px; color:var(--text-muted); margin-left:auto; }

    /* Image previews */
    #imgPreviews { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
    .img-preview { position:relative; width:64px; height:64px; border-radius:8px; overflow:hidden; }
    .img-preview img { width:100%; height:100%; object-fit:cover; }
    .img-preview .img-remove {
      position:absolute; top:2px; right:2px; width:18px; height:18px;
      border-radius:4px; background:rgba(0,0,0,.6); color:#fff;
      border:none; font-size:12px; cursor:pointer; display:flex;
      align-items:center; justify-content:center;
    }

    /* Question modal */
    .modal-overlay {
      position:fixed; inset:0; background:rgba(0,0,0,.6);
      display:flex; align-items:center; justify-content:center; z-index:1000;
      animation:fadeIn .2s ease;
    }
    .modal-card {
      background:var(--bg-secondary); border:1px solid var(--border);
      border-radius:var(--radius); padding:24px; width:420px; max-width:90vw;
      max-height:80vh; overflow-y:auto;
    }
    .modal-card h3 { font-size:15px; margin-bottom:16px; }
    .q-item { margin-bottom:16px; }
    .q-item p { font-size:13px; margin-bottom:8px; color:var(--text-secondary); }
    .q-option {
      padding:8px 12px; border-radius:6px; margin-bottom:6px;
      background:var(--bg-tertiary); border:1px solid var(--border);
      color:var(--text-secondary); font-size:13px; cursor:pointer;
      display:flex; align-items:center; gap:8px; transition:all .15s;
    }
    .q-option:hover { border-color:var(--accent); }
    .q-option.selected { background:rgba(88,166,255,.15); border-color:var(--accent); color:var(--accent); }
    .modal-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:12px; }
    .modal-actions button {
      padding:7px 16px; border-radius:6px; font-size:13px; cursor:pointer;
    }
    .btn-primary { background:var(--accent); color:#fff; border:none; }
    .btn-primary:hover { background:var(--accent-hover); }

    /* Reconnect banner */
    .reconnect-banner {
      display:none; padding:8px 20px; background:rgba(217,119,6,.12);
      border-bottom:1px solid rgba(217,119,6,.25); color:var(--warn);
      font-size:12px; text-align:center; cursor:pointer;
    }
    .reconnect-banner.show { display:block; }
  </style>
</head>
<body>
  <div class="app">
    <!-- Sidebar -->
    <div class="sidebar">
      <div class="sidebar-header">
        <div class="logo">🍃</div>
        <div>
          <h2>gjw</h2>
          <p>科研增强版 v2.0</p>
        </div>
      </div>
      <button class="new-session-btn" onclick="createSession()">➕ 新建会话</button>
      <div class="session-list" id="sessionList"></div>
    </div>

    <!-- Main -->
    <div class="main">
      <div class="reconnect-banner" id="reconnectBanner" onclick="location.reload()">
        ⚠️ 连接已断开，点击刷新页面重连
      </div>
      <div class="header">
        <div class="header-title" id="headerTitle">默认会话</div>
        <div class="status-pill" id="statusPill">
          <span class="status-dot"></span>
          <span id="statusText">连接中...</span>
        </div>
      </div>
      <div class="templates-bar" id="templatesBar">
        <button class="tpl-btn active" data-tpl="" onclick="selectTemplate(this)">💬 自由讨论</button>
        <button class="tpl-btn" data-tpl="实验配置" onclick="selectTemplate(this)">🧪 实验配置</button>
        <button class="tpl-btn" data-tpl="结果汇报" onclick="selectTemplate(this)">📊 结果汇报</button>
        <button class="tpl-btn" data-tpl="代码审查" onclick="selectTemplate(this)">💻 代码审查</button>
        <button class="tpl-btn" data-tpl="文献讨论" onclick="selectTemplate(this)">📚 文献讨论</button>
        <button class="tpl-btn" data-tpl="写作润色" onclick="selectTemplate(this)">✍️ 写作润色</button>
        <button class="tpl-btn" data-tpl="记笔记" onclick="selectTemplate(this)">📓 记到 lab-notes</button>
      </div>
      <div id="messages">
        <div class="empty-state" id="emptyState">
          <div class="empty-icon">🍃</div>
          <h3>gjw · 科研增强版</h3>
          <p>
            在 Cursor 中发起 Agent 对话，AI 会自动调用 <code>check_messages</code> 进入等待状态。<br>
            然后在这里输入后续指令，实现<strong>同轮多反馈</strong>。
          </p>
        </div>
      </div>
      <div class="input-area">
        <div id="imgPreviews"></div>
        <div class="input-wrapper">
          <textarea id="input" placeholder="输入消息或粘贴图片..." rows="1"></textarea>
          <button id="sendBtn" onclick="send()">➤</button>
        </div>
        <div class="input-tools">
          <button onclick="document.getElementById('imgInput').click()">📎 图片</button>
          <button onclick="clearChat()">🗑 清空</button>
          <span class="draft-hint" id="draftHint"></span>
        </div>
      </div>
    </div>
  </div>
  <input type="file" id="imgInput" accept="image/*" multiple style="display:none">
  <div id="modalContainer"></div>

  <script>

    // ── State ──
    const state = {
      currentSession: 'default',
      sessions: [],
      lastSeenId: 0,
      lastSeenQuestionId: '',
      pendingImages: [],
      selectedTemplate: '',
      isConnected: true,
      messagesLoaded: false,
    };

    const input = document.getElementById('input');
    const messagesEl = document.getElementById('messages');
    const emptyState = document.getElementById('emptyState');
    const statusPill = document.getElementById('statusPill');
    const statusText = document.getElementById('statusText');
    const headerTitle = document.getElementById('headerTitle');
    const sessionList = document.getElementById('sessionList');
    const imgInput = document.getElementById('imgInput');
    const imgPreviews = document.getElementById('imgPreviews');
    const draftHint = document.getElementById('draftHint');
    const reconnectBanner = document.getElementById('reconnectBanner');

    // ── Session Management ──
    async function loadSessions() {
      try {
        const r = await fetch('/sessions');
        const d = await r.json();
        state.sessions = d.sessions || [];
        renderSessions();
      } catch { /* ignore */ }
    }

    function renderSessions() {
      sessionList.innerHTML = '';
      state.sessions.forEach(s => {
        const el = document.createElement('div');
        el.className = 'session-item' + (s.id === state.currentSession ? ' active' : '');
        el.innerHTML = '<span class="s-name">' + escapeHtml(s.name) + '</span>' +
          (s.id !== 'default' ? '<span class="s-del">✕</span>' : '');
        el.onclick = () => switchSession(s.id);
        const delBtn = el.querySelector('.s-del');
        if (delBtn) delBtn.onclick = (e) => { e.stopPropagation(); deleteSession(s.id); };
        sessionList.appendChild(el);
      });
    }

    async function createSession() {
      const name = prompt('会话名称:', '新会话');
      if (!name) return;
      try {
        const r = await fetch('/sessions', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name}) });
        const d = await r.json();
        if (d.ok) {
          await loadSessions();
          switchSession(d.session.id);
        }
      } catch (e) { alert('创建失败: ' + e.message); }
    }

    async function deleteSession(id) {
      if (!confirm('确定删除此会话？历史消息将一并清除。')) return;
      try {
        await fetch('/sessions?id=' + encodeURIComponent(id), { method: 'DELETE' });
        await loadSessions();
        if (state.currentSession === id) switchSession('default');
      } catch {}
    }

    async function switchSession(id) {
      state.currentSession = id;
      state.lastSeenId = 0;
      state.messagesLoaded = false;
      messagesEl.innerHTML = '';
      if (emptyState) {
        messagesEl.appendChild(emptyState);
        emptyState.style.display = 'flex';
      }
      const s = state.sessions.find(x => x.id === id);
      headerTitle.textContent = s ? s.name : id;
      renderSessions();
      await loadHistory();
      restoreDraft();
    }

    // ── Message History ──
    async function loadHistory() {
      try {
        const r = await fetch('/messages?session=' + encodeURIComponent(state.currentSession) + '&after=0');
        const d = await r.json();
        if (d.messages && d.messages.length) {
          if (emptyState) emptyState.style.display = 'none';
          d.messages.forEach(m => renderMessage(m));
          state.lastSeenId = d.messages[d.messages.length - 1].id;
        }
        state.messagesLoaded = true;
      } catch {}
    }

    function renderMessage(m) {
      if (emptyState) emptyState.style.display = 'none';
      const row = document.createElement('div');
      row.className = 'msg-row ' + m.role;
      row.dataset.id = m.id;
      const isUser = m.role === 'user';
      const icon = isUser ? '👤' : '🤖';
      const escaped = escapeHtml(m.text || '');
      // 用 msg-text 包裹文本，方便编辑时精确定位
      let bubbleContent = '<span class="msg-text">' + escaped + '</span>';
      if (m.images && m.images.length) {
        m.images.forEach(img => {
          bubbleContent += '<br><img class="bubble-img" src="' + img.data + '" style="max-width:200px;border-radius:6px;margin-top:6px;">';
        });
      }
      if (m.template) {
        bubbleContent = '<span style="display:inline-block;padding:2px 8px;border-radius:4px;background:rgba(88,166,255,.15);color:var(--accent);font-size:11px;margin-bottom:6px;">' + escapeHtml(m.template) + '</span><br>' + bubbleContent;
      }
      const timeLabel = m.edited ? (formatTime(m.updatedAt || m.createdAt) + ' (已编辑)') : formatTime(m.createdAt);
      const editBtn = isUser ? '<button onclick="startEdit(' + m.id + ')">编辑</button>' : '';
      row.innerHTML =
        '<div class="avatar">' + icon + '</div>' +
        '<div class="bubble-wrap">' +
          '<div class="bubble">' + bubbleContent + '<div class="time">' + timeLabel + '</div></div>' +
          '<div class="bubble-actions">' + editBtn + '</div>' +
        '</div>';
      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function formatTime(iso) {
      if (!iso) return '';
      const d = new Date(iso);
      return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
    }

    // ── Editing ──
    function startEdit(messageId) {
      const msgEl = document.querySelector('.msg-row[data-id="' + messageId + '"]');
      if (!msgEl) return;
      const bubble = msgEl.querySelector('.bubble');
      const textSpan = bubble.querySelector('.msg-text');
      const currentText = textSpan ? textSpan.textContent : '';
      const newText = prompt('编辑消息:', currentText);
      if (newText === null || newText.trim() === currentText.trim()) return;
      fetch('/edit', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ session: state.currentSession, messageId, newText })
      }).then(() => {
        // 清空后续消息并重新加载
        state.lastSeenId = messageId - 1;
        const rows = Array.from(messagesEl.querySelectorAll('.msg-row'));
        const idx = rows.findIndex(r => r.dataset.id == messageId);
        if (idx >= 0) {
          for (let i = rows.length - 1; i > idx; i--) rows[i].remove();
        }
        // 更新当前消息文本
        if (textSpan) textSpan.textContent = newText;
        const timeDiv = bubble.querySelector('.time');
        if (timeDiv) timeDiv.textContent = formatTime(new Date().toISOString()) + ' (已编辑)';
      }).catch(e => alert('编辑失败: ' + e.message));
    }

    // ── Templates ──
    function selectTemplate(btn) {
      document.querySelectorAll('.tpl-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.selectedTemplate = btn.dataset.tpl;
      input.focus();
    }

    // ── Sending ──
    async function send() {
      const msg = input.value.trim();
      const images = state.pendingImages.map(i => ({ data: i.data, mime: i.mime }));
      if (!msg && images.length === 0) return;

      // 捕获当前模板，避免发送后切换模板导致显示不一致
      const currentTemplate = state.selectedTemplate;

      // 本地先渲染，持有 row 引用以便后续补填真实 ID
      const row = addMsg(msg, 'user', images, currentTemplate);
      input.value = ''; input.style.height = 'auto';
      clearPendingImages();
      saveDraft();

      try {
        const r = await fetch('/send', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ session: state.currentSession, message: msg, images, template: currentTemplate })
        });
        const d = await r.json();
        if (d.messageId !== undefined && row && !row.dataset.id) {
          row.dataset.id = d.messageId;
        }
      } catch {
        addMsg('[发送失败，请检查 MCP 服务是否运行]', 'ai');
      }
    }

    function addMsg(text, role, images, template, id) {
      if (emptyState) emptyState.style.display = 'none';
      const row = document.createElement('div');
      row.className = 'msg-row ' + role;
      if (id !== undefined) row.dataset.id = id;
      const icon = role === 'user' ? '👤' : '🤖';
      const escaped = escapeHtml(text || '');
      let bubbleContent = '<span class="msg-text">' + escaped + '</span>';
      if (images && images.length) {
        images.forEach(img => {
          bubbleContent += '<br><img src="' + img.data + '" style="max-width:200px;border-radius:6px;margin-top:6px;">';
        });
      }
      if (role === 'user' && template) {
        bubbleContent = '<span style="display:inline-block;padding:2px 8px;border-radius:4px;background:rgba(88,166,255,.15);color:var(--accent);font-size:11px;margin-bottom:6px;">' + escapeHtml(template) + '</span><br>' + bubbleContent;
      }
      row.innerHTML =
        '<div class="avatar">' + icon + '</div>' +
        '<div class="bubble-wrap">' +
          '<div class="bubble">' + bubbleContent + '<div class="time">' + formatTime(new Date().toISOString()) + '</div></div>' +
        '</div>';
      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return row;
    }

    function clearChat() {
      if (!confirm('确定清空当前会话的显示？（不会删除服务器上的历史）')) return;
      const rows = Array.from(messagesEl.querySelectorAll('.msg-row'));
      const maxId = rows.reduce((max, r) => Math.max(max, parseInt(r.dataset.id || 0)), 0);
      messagesEl.innerHTML = '';
      state.lastSeenId = maxId;
      if (emptyState) {
        messagesEl.appendChild(emptyState);
        emptyState.style.display = 'flex';
      }
    }

    // ── Images ──
    imgInput.addEventListener('change', () => {
      const files = Array.from(imgInput.files || []);
      imgInput.value = '';
      files.forEach(f => addImageFile(f));
    });
    input.addEventListener('paste', e => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          e.preventDefault();
          addImageFile(items[i].getAsFile());
          break;
        }
      }
    });
    function addImageFile(file) {
      const reader = new FileReader();
      reader.onload = () => {
        state.pendingImages.push({ data: reader.result, mime: file.type || 'image/png' });
        renderPreviews();
      };
      reader.readAsDataURL(file);
    }
    function renderPreviews() {
      imgPreviews.innerHTML = '';
      state.pendingImages.forEach((img, i) => {
        const div = document.createElement('div');
        div.className = 'img-preview';
        div.innerHTML = '<img src="' + img.data + '"><button class="img-remove" onclick="removeImage(' + i + ')">×</button>';
        imgPreviews.appendChild(div);
      });
    }
    function removeImage(i) { state.pendingImages.splice(i, 1); renderPreviews(); }
    function clearPendingImages() { state.pendingImages = []; renderPreviews(); }

    // ── Draft Auto-save ──
    const DRAFT_KEY = 'gjw:draft:';
    function saveDraft() {
      const text = input.value;
      if (text) {
        localStorage.setItem(DRAFT_KEY + state.currentSession, text);
        draftHint.textContent = '草稿已保存';
      } else {
        localStorage.removeItem(DRAFT_KEY + state.currentSession);
        draftHint.textContent = '';
      }
    }
    function restoreDraft() {
      const text = localStorage.getItem(DRAFT_KEY + state.currentSession);
      if (text) {
        input.value = text;
        draftHint.textContent = '已恢复草稿';
        input.dispatchEvent(new Event('input'));
      } else {
        draftHint.textContent = '';
      }
    }
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 120) + 'px';
      saveDraft();
    });

    // ── Polling ──
    async function poll() {
      if (!state.messagesLoaded) return;
      try {
        const r = await fetch('/poll?session=' + encodeURIComponent(state.currentSession) + '&after=' + state.lastSeenId);
        const d = await r.json();
        state.isConnected = true;
        reconnectBanner.classList.remove('show');

        // Status
        if (d.waiting) {
          statusText.textContent = '等待输入...';
          statusPill.className = 'status-pill waiting';
        } else {
          statusText.textContent = 'AI 处理中';
          statusPill.className = 'status-pill connected';
        }

        // AI responses
        if (d.responses && d.responses.length > 0) {
          for (const resp of d.responses) {
            addMsg(resp.text, 'ai', undefined, undefined, resp.id);
            state.lastSeenId = Math.max(state.lastSeenId, resp.id);
          }
        }

        // Questions
        if (d.questions && d.questions.length > 0) {
          const q = d.questions[0];
          if (q.qid !== state.lastSeenQuestionId) {
            state.lastSeenQuestionId = q.qid;
            showQuestionModal(q.qid, q.questions);
          }
        }
      } catch {
        state.isConnected = false;
        statusText.textContent = '未连接';
        statusPill.className = 'status-pill offline';
        reconnectBanner.classList.add('show');
      }
    }

    // ── Question Modal ──
    function showQuestionModal(qid, questions) {
      const container = document.getElementById('modalContainer');

      let html = '<div class="modal-overlay" id="qModal"><div class="modal-card"><h3>❓ AI 向你提问</h3>';
      questions.forEach((q, qi) => {
        html += '<div class="q-item"><p>' + escapeHtml(q.question) + (q.allow_multiple ? '（可多选）' : '（单选）') + '</p>';
        q.options.forEach((opt) => {
          html += '<div class="q-option" data-qi="' + qi + '" data-oid="' + opt.id + '" data-multi="' + (q.allow_multiple ? '1' : '') + '">' +
            '<input type="' + (q.allow_multiple ? 'checkbox' : 'radio') + '" name="q' + qi + '"> ' + escapeHtml(opt.label) + '</div>';
        });
        html += '</div>';
      });
      html += '<div class="modal-actions"><button class="btn-primary" id="qSubmitBtn">确认</button></div></div></div>';
      container.innerHTML = html;

      container.querySelectorAll('.q-option').forEach(el => {
        el.onclick = () => toggleOption(el, parseInt(el.dataset.qi), el.dataset.oid, !!el.dataset.multi);
      });
      const submitBtn = container.querySelector('#qSubmitBtn');
      if (submitBtn) submitBtn.onclick = () => submitAnswer(qid);
    }

    function toggleOption(el, qi, oid, allowMultiple) {
      const modal = document.getElementById('qModal');
      if (!allowMultiple) {
        modal.querySelectorAll('.q-option[data-qi="' + qi + '"]').forEach(o => o.classList.remove('selected'));
      }
      el.classList.toggle('selected');
    }

    async function submitAnswer(qid) {
      const modal = document.getElementById('qModal');
      const answers = [];
      modal.querySelectorAll('.q-item').forEach((item, qi) => {
        const selected = Array.from(item.querySelectorAll('.q-option.selected')).map(o => o.dataset.oid);
        answers.push(selected);
      });
      try {
        await fetch('/answer', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ qid, answers })
        });
        modal.remove();
      } catch {}
    }

    // ── Utilities ──
    function escapeHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

    // ── Init ──
    loadSessions();
    loadHistory();
    restoreDraft();
    setInterval(poll, 1500);
    poll();
    input.focus();
  </script>
</body>
</html>`;
