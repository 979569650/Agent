const chatBox = document.getElementById("chatBox");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const refreshNotesBtn = document.getElementById("refreshNotesBtn");
const rebuildBtn = document.getElementById("rebuildBtn");
const notesList = document.getElementById("notesList");
const statusEl = document.getElementById("status");
const chatBanner = document.getElementById("chatBanner");
const securityPanel = document.getElementById("securityPanel");
const securityHint = document.getElementById("securityHint");
const securityKeyInput = document.getElementById("securityKeyInput");
const submitSecurityKeyBtn = document.getElementById("submitSecurityKeyBtn");
const abortAccessTicketBtn = document.getElementById("abortAccessTicketBtn");
const closeSecurityPanelBtn = document.getElementById("closeSecurityPanelBtn");
const helpBtn = document.getElementById("helpBtn");
const configSummaryBtn = document.getElementById("configSummaryBtn");
const metricsBtn = document.getElementById("metricsBtn");
const updateBtn = document.getElementById("updateBtn");
const summaryModel = document.getElementById("summaryModel");
const summaryApiMode = document.getElementById("summaryApiMode");
const summaryConfig = document.getElementById("summaryConfig");
const summaryNotes = document.getElementById("summaryNotes");
const exampleQuestions = document.getElementById("exampleQuestions");
const helpPanel = document.getElementById("helpPanel");
const closeHelpPanelBtn = document.getElementById("closeHelpPanelBtn");
const recentEvents = document.getElementById("recentEvents");
const chatStateBadge = document.getElementById("chatStateBadge");
const rebuildProgressFill = document.getElementById("rebuildProgressFill");
const rebuildProgressLabel = document.getElementById("rebuildProgressLabel");

const tabChat = document.getElementById("tabChat");
const tabNotes = document.getElementById("tabNotes");
const tabConfig = document.getElementById("tabConfig");
const tabSystem = document.getElementById("tabSystem");
const tabAdmin = document.getElementById("tabAdmin");
const chatPage = document.getElementById("chatPage");
const notesPage = document.getElementById("notesPage");
const configPage = document.getElementById("configPage");
const systemPage = document.getElementById("systemPage");
const adminPage = document.getElementById("adminPage");

const themeSelect = document.getElementById("themeSelect");

const noteSelect = document.getElementById("noteSelect");
const loadNoteBtn = document.getElementById("loadNoteBtn");
const noteEditor = document.getElementById("noteEditor");
const newPathInput = document.getElementById("newPathInput");
const createNoteBtn = document.getElementById("createNoteBtn");
const saveNoteBtn = document.getElementById("saveNoteBtn");
const deleteNoteBtn = document.getElementById("deleteNoteBtn");
const renamePathInput = document.getElementById("renamePathInput");
const renameNoteBtn = document.getElementById("renameNoteBtn");
const uploadFileInput = document.getElementById("uploadFileInput");
const uploadBtn = document.getElementById("uploadBtn");
const indexStatusText = document.getElementById("indexStatusText");
const refreshIndexStatusBtn = document.getElementById("refreshIndexStatusBtn");
const manualRebuildBtn = document.getElementById("manualRebuildBtn");
const showRebuildLogsBtn = document.getElementById("showRebuildLogsBtn");

const configStatus = document.getElementById("configStatus");
const loadConfigBtn = document.getElementById("loadConfigBtn");
const saveConfigBtn = document.getElementById("saveConfigBtn");
const testConfigBtn = document.getElementById("testConfigBtn");
const resetConfigBtn = document.getElementById("resetConfigBtn");

const refreshSystemBtn = document.getElementById("refreshSystemBtn");
const systemStatusBox = document.getElementById("systemStatusBox");
const refreshAdminBtn = document.getElementById("refreshAdminBtn");
const adminStatusBox = document.getElementById("adminStatusBox");
const rebuildLogModal = document.getElementById("rebuildLogModal");
const rebuildLogMeta = document.getElementById("rebuildLogMeta");
const rebuildLogBox = document.getElementById("rebuildLogBox");
const refreshRebuildLogsBtn = document.getElementById("refreshRebuildLogsBtn");
const closeRebuildLogModalBtn = document.getElementById("closeRebuildLogModalBtn");
const emptyChat = document.getElementById("emptyChat");

const CFG_KEYS = [
  "BASE_URL", "API_KEY", "MODEL", "API_MODE",
  "REQUEST_TIMEOUT", "REQUEST_MAX_RETRIES", "REQUEST_RETRY_BACKOFF_MS",
  "RERANKER_MODEL", "RERANKER_ENABLED", "ENABLE_IMAGE_OCR", "ENABLE_IMAGE_VLM",
  "VISION_MODEL", "ALLOWED_TOOLS", "OTEL_ENABLED", "LOG_FILE", "RESTRICTED_QUERY_LIMIT_PER_MINUTE",
];

/** @type {{role: "user"|"assistant", content: string}[]} */
const history = [];
const eventStore = [];
let currentTab = "chat";
let currentNotePath = "";
let notesDirty = false;
let rebuildLogPoller = null;
let chatState = "idle";
let pendingTicketId = null;
let pendingTicketMessage = "";
let pendingTicketQuery = "";
let latestHealth = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function pushSystemEvent(text, tone = "") {
  if (!text) return;
  const now = new Date();
  const timeText = now.toTimeString().slice(0, 8);
  eventStore.unshift({ text, tone, timeText });
  if (eventStore.length > 20) {
    eventStore.length = 20;
  }
  if (!recentEvents) return;
  recentEvents.innerHTML = "";
  eventStore.slice(0, 8).forEach((ev) => {
    const li = document.createElement("li");
    li.className = ev.tone || "";
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = `[${ev.timeText}]`;
    const textNode = document.createElement("span");
    textNode.textContent = ev.text;
    li.appendChild(time);
    li.appendChild(textNode);
    recentEvents.appendChild(li);
  });
}

function setChatState(nextState) {
  chatState = nextState;
  const sending = nextState === "sending";
  const awaitingKey = nextState === "awaiting_access_code";
  sendBtn.disabled = sending || awaitingKey;
  chatInput.disabled = awaitingKey;
  submitSecurityKeyBtn.disabled = !awaitingKey;
  abortAccessTicketBtn.disabled = !awaitingKey;
  closeSecurityPanelBtn.disabled = !awaitingKey;
  securityKeyInput.disabled = !awaitingKey;

  if (chatStateBadge) {
    const badgeMap = {
      idle: { text: "状态：就绪", tone: "is-idle" },
      sending: { text: "状态：发送中", tone: "is-sending" },
      awaiting_access_code: { text: "状态：等待口令", tone: "is-awaiting" },
      error: { text: "状态：异常", tone: "is-error" },
    };
    const cfg = badgeMap[nextState] || badgeMap.idle;
    chatStateBadge.textContent = cfg.text;
    chatStateBadge.classList.remove("is-idle", "is-sending", "is-awaiting", "is-error");
    chatStateBadge.classList.add(cfg.tone);
  }
}

function showBanner(text, tone = "") {
  chatBanner.textContent = text || "";
  chatBanner.className = `status-banner${tone ? ` ${tone}` : ""}`;
  chatBanner.classList.toggle("hidden", !text);
}

function clearBanner() {
  showBanner("");
}

function openHelpPanel() {
  helpPanel?.classList.remove("hidden");
}

function closeHelpPanel() {
  helpPanel?.classList.add("hidden");
}

function openSecurityPanel(message, ticketId, queryText = "") {
  pendingTicketId = ticketId;
  pendingTicketMessage = message || "检测到受限查询，请输入访问口令。";
  pendingTicketQuery = queryText || pendingTicketQuery;
  securityHint.textContent = pendingTicketMessage;
  securityPanel.classList.remove("hidden");
  securityKeyInput.value = "";
  setChatState("awaiting_access_code");
  showBanner("检测到受限查询，请先完成访问口令核验。", "warn");
  securityKeyInput.focus();
}

function closeSecurityPanel() {
  pendingTicketId = null;
  pendingTicketMessage = "";
  pendingTicketQuery = "";
  securityPanel.classList.add("hidden");
  securityKeyInput.value = "";
  if (chatState === "awaiting_access_code") {
    setChatState("idle");
  }
}

function setConfigStatus(text) {
  configStatus.textContent = text;
}

function setIndexStatus(text) {
  indexStatusText.textContent = text;
}

function clampProgress(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n)));
}

function updateRebuildProgress(progress = 0, message = "") {
  const p = clampProgress(progress);
  if (rebuildProgressFill) {
    rebuildProgressFill.style.width = `${p}%`;
  }
  if (rebuildProgressLabel) {
    rebuildProgressLabel.textContent = message ? `索引进度：${p}% · ${message}` : `索引进度：${p}%`;
  }
}

function appendMessage(role, content) {
  if (emptyChat) {
    emptyChat.classList.add("hidden");
  }
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const labelMap = {
    user: "你",
    assistant: "助手",
    system: "系统",
    error: "错误",
  };
  div.textContent = `${labelMap[role] || "消息"}: ${content}`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function appendCommandResultCard(type, title, body, meta = {}) {
  if (emptyChat) {
    emptyChat.classList.add("hidden");
  }
  const card = document.createElement("div");
  card.className = `msg ${type || "system"}`;

  const titleEl = document.createElement("div");
  titleEl.className = "command-card-title";
  const titleText = meta.command ? `${title}（${meta.command}）` : title;
  titleEl.textContent = titleText;

  const bodyEl = document.createElement("div");
  bodyEl.textContent = body;

  card.appendChild(titleEl);
  card.appendChild(bodyEl);
  chatBox.appendChild(card);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function appendCommandResult(title, content, tone = "system", meta = {}) {
  appendCommandResultCard(tone, title, content, meta);
}

function formatConfigSummary(values = {}) {
  return [
    `model: ${values.MODEL || latestHealth?.model || "-"}`,
    `api_mode: ${values.API_MODE || latestHealth?.api_mode || "-"}`,
    `reranker_enabled: ${values.RERANKER_ENABLED || "-"}`,
    `image_vlm_enabled: ${values.ENABLE_IMAGE_VLM || "-"}`,
    `image_ocr_enabled: ${values.ENABLE_IMAGE_OCR || "-"}`,
    `allowed_tools: ${values.ALLOWED_TOOLS || "-"}`,
  ].join("\n");
}

function formatMetrics(metrics = {}) {
  return [
    `requests_total: ${metrics.requests_total ?? 0}`,
    `retrieval_hit_total: ${metrics.retrieval_hit_total ?? 0}`,
    `retrieval_miss_total: ${metrics.retrieval_miss_total ?? 0}`,
    `error_total: ${metrics.error_total ?? 0}`,
    `avg_latency_ms: ${metrics.avg_latency_ms ?? 0}`,
    `retrieval_hit_rate: ${metrics.retrieval_hit_rate ?? 0}`,
    `error_rate: ${metrics.error_rate ?? 0}`,
  ].join("\n");
}

async function loadRuntimeSummary() {
  try {
    const [health, system] = await Promise.all([
      requestJson("/api/health"),
      requestJson("/api/system/status"),
    ]);
    latestHealth = health;
    summaryModel.textContent = health.model || "-";
    summaryApiMode.textContent = health.api_mode || "-";
    summaryConfig.textContent = health.config_ready ? "配置完整" : "配置未完成";
    summaryNotes.textContent = String(system.notes_count ?? 0);

    if (!health.config_ready) {
      showBanner("当前配置未完成，请先在配置页补全 BASE_URL 和 API_KEY。", "warn");
      pushSystemEvent("配置未完成：请先补全 BASE_URL 和 API_KEY", "warn");
    }

    const indexState = system?.index_task?.state || "idle";
    if (indexState === "running") {
      showBanner("索引正在构建中，检索结果可能延迟更新。", "warn");
      pushSystemEvent("索引构建中：检索结果可能延迟更新", "warn");
    } else if (indexState === "error") {
      showBanner(`索引状态异常：${system?.index_task?.last_error || "请尝试手动重建"}`, "error");
      pushSystemEvent("索引构建异常：请查看状态或手动重建", "error");
    }

    return { health, system };
  } catch (err) {
    summaryModel.textContent = "读取失败";
    summaryApiMode.textContent = "读取失败";
    summaryConfig.textContent = "服务异常";
    summaryNotes.textContent = "-";
    showBanner(`运行摘要读取失败：${err.message}`, "error");
    pushSystemEvent(`运行摘要读取失败：${err.message}`, "error");
    return null;
  }
}

async function handleLocalCommand(rawText) {
  const text = (rawText || "").trim();
  const cmd = text.toLowerCase();

  if (cmd === "/help") {
    const helpText = [
      "可直接提问，也可以使用本地命令：",
      "- /help：查看帮助",
      "- /files：列出本地笔记",
      "- /config：查看当前配置摘要",
      "- /metrics：查看运行指标",
      "- /update：触发索引重建",
      "示例问题：本地有什么笔记 / 帮我总结某篇笔记 / 不要ocr，看一下这张图",
    ].join("\n");
    openHelpPanel();
    appendCommandResult("帮助", helpText, "system", { command: "/help" });
    showBanner("已打开帮助说明。", "success");
    return true;
  }

  if (cmd === "/files") {
    const data = await requestJson("/api/notes");
    const files = data.files || [];
    appendCommandResult(
      "本地笔记",
      files.length ? files.map((f) => `- ${f}`).join("\n") : "当前没有可用笔记。",
      "system",
      { command: "/files", count: files.length },
    );
    showBanner("笔记列表已更新。", "success");
    return true;
  }

  if (cmd === "/config") {
    const data = await requestJson("/api/config");
    appendCommandResult("当前配置摘要", formatConfigSummary(data.values || {}), "system", { command: "/config" });
    showBanner("配置摘要已展示。", "success");
    return true;
  }

  if (cmd === "/metrics") {
    const data = await requestJson("/api/system/status");
    appendCommandResult("运行指标", formatMetrics(data.metrics || {}), "system", { command: "/metrics" });
    showBanner("运行指标已展示。", "success");
    return true;
  }

  if (cmd === "/update") {
    appendCommandResult("索引重建", "已触发索引重建，请稍候查看状态。", "system", { command: "/update" });
    showBanner("索引重建已开始。", "warn");
    await triggerRebuildAndWait("聊天区命令触发");
    return true;
  }

  return false;
}

async function requestJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data?.detail || `HTTP ${resp.status}`);
  return data;
}

function applyTheme(mode) {
  localStorage.setItem("theme", mode);
  const root = document.documentElement;
  if (mode === "system") {
    root.removeAttribute("data-theme");
    return;
  }
  root.setAttribute("data-theme", mode);
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  clearBanner();
  closeSecurityPanel();
  appendMessage("user", text);
  history.push({ role: "user", content: text });
  chatInput.value = "";
  setStatus("聊天请求中...");
  setChatState("sending");

  try {
    if (await handleLocalCommand(text)) {
      setStatus("命令执行完成");
      setChatState("idle");
      return;
    }

    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history }),
    });

    if (data.status === "awaiting_access_code") {
      appendMessage("system", data.message || "检测到受限查询，请输入访问口令。");
      pushSystemEvent("受限查询触发：等待访问口令", "warn");
      openSecurityPanel(data.message, data.ticket_id, text);
      setStatus("等待访问口令输入");
      return;
    }

    const reply = data.reply || data.message || "";
    appendMessage("assistant", reply);
    history.push({ role: "assistant", content: reply });
    setStatus("聊天完成");
    if (data.status === "verified") {
      showBanner(data.message || "访问口令核验通过。", "success");
      pushSystemEvent("访问口令核验通过", "success");
    } else {
      clearBanner();
      if (/未找到相关本地笔记/.test(reply)) {
        pushSystemEvent("检索未命中：未找到相关本地笔记", "warn");
      } else {
        pushSystemEvent("请求完成：检索命中或直接回答", "success");
      }
    }
  } catch (err) {
    appendMessage("error", `请求失败: ${err.message}`);
    setStatus("聊天失败");
    const friendly = latestHealth?.config_ready
      ? `请求失败：${err.message}。请稍后重试，或检查系统状态页。`
      : `请求失败：配置未完成，请先在配置页补全 BASE_URL 和 API_KEY。`;
    showBanner(friendly, "error");
    pushSystemEvent(friendly, "error");
  } finally {
    if (chatState !== "awaiting_access_code") {
      setChatState("idle");
    }
  }
}

async function submitSecurityKey() {
  const securityKey = (securityKeyInput.value || "").trim();
  if (!pendingTicketId) {
    showBanner("当前没有待核验的受限查询。", "warn");
    return;
  }
  if (!securityKey) {
    showBanner("请输入访问口令。", "warn");
    return;
  }

  setStatus("正在核验访问口令...");
  submitSecurityKeyBtn.disabled = true;
  abortAccessTicketBtn.disabled = true;

  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: pendingTicketQuery || "受限查询核验",
        history,
        ticket_id: pendingTicketId,
        access_code: securityKey,
      }),
    });

    if (data.status === "verification_failed") {
      appendMessage("error", data.message || "访问口令核验失败。");
      showBanner(data.message || "访问口令核验失败。", "error");
      pushSystemEvent(data.message || "访问口令核验失败", "error");
      setStatus("防护核验失败");
      securityKeyInput.focus();
      return;
    }

    const reply = data.reply || "";
    if (reply) {
      appendMessage("assistant", reply);
      history.push({ role: "assistant", content: reply });
    }
    closeSecurityPanel();
    showBanner(data.message || "访问口令核验通过。", "success");
    setStatus("防护核验通过，已返回结果");
  } catch (err) {
    appendMessage("error", `核验失败: ${err.message}`);
    showBanner(`核验失败：${err.message}`, "error");
    setStatus("防护核验失败");
  } finally {
    if (chatState === "awaiting_access_code") {
      submitSecurityKeyBtn.disabled = false;
      abortAccessTicketBtn.disabled = false;
    }
  }
}

async function closeSecurityPanelAction() {
  if (!pendingTicketId) {
    closeSecurityPanel();
    clearBanner();
    setStatus("已关闭访问口令输入弹窗");
    return;
  }
  appendMessage("system", "你已关闭访问口令输入弹窗，可稍后重试该查询。\n提示：该受限查询票据会在一段时间后自动过期。\n");
  pushSystemEvent("已关闭访问口令输入弹窗", "warn");
  showBanner("已关闭访问口令输入弹窗。", "warn");
  closeSecurityPanel();
  setStatus("等待用户后续操作");
}

async function abortAccessTicket() {
  if (!pendingTicketId) {
    return;
  }
  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: pendingTicketQuery || "受限查询中止",
        history,
        ticket_id: pendingTicketId,
        abort_ticket: true,
      }),
    });
    appendMessage("system", data.message || "查询已中止。");
    pushSystemEvent(data.message || "受限查询已中止", "warn");
    showBanner(data.message || "查询已中止。", "warn");
    setStatus("受限查询已中止");
  } catch (err) {
    appendMessage("error", `中止失败: ${err.message}`);
    showBanner(`中止失败：${err.message}`, "error");
    pushSystemEvent(`中止失败：${err.message}`, "error");
  } finally {
    closeSecurityPanel();
  }
}

async function loadNotes() {
  setStatus("加载笔记列表...");
  notesList.innerHTML = "";
  noteSelect.innerHTML = "";
  try {
    const data = await requestJson("/api/notes");
    const files = data.files || [];

    if (!files.length) {
      const li = document.createElement("li");
      li.textContent = "未发现笔记文件";
      notesList.appendChild(li);

      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(无可用笔记)";
      noteSelect.appendChild(opt);
    } else {
      files.forEach((f) => {
        const li = document.createElement("li");
        li.textContent = f;
        notesList.appendChild(li);

        const opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        noteSelect.appendChild(opt);
      });
    }
    setStatus(`笔记已加载（${files.length}）`);
  } catch (err) {
    setStatus(`加载失败：${err.message}`);
  }
}

async function fetchIndexStatus() {
  try {
    const data = await requestJson("/api/index/status");
    const s = data.status || {};
    const err = s.last_error ? ` | error=${s.last_error}` : "";
    setIndexStatus(`state=${s.state} progress=${s.progress}% message=${s.message}${err}`);
    updateRebuildProgress(s.progress ?? 0, s.message || "");
    return s;
  } catch (err) {
    setIndexStatus(`读取状态失败：${err.message}`);
    updateRebuildProgress(0, "状态读取失败");
    return null;
  }
}

async function triggerRebuildAndWait(reason = "manual") {
  setStatus(`正在重建索引（${reason}）...`);
  pushSystemEvent(`索引重建开始：${reason}`, "warn");
  appendMessage("system", `索引重建开始：${reason}`);
  showBanner(`索引重建进行中：${reason}`, "warn");
  setIndexStatus("触发中...");
  updateRebuildProgress(0, "准备开始");
  manualRebuildBtn.disabled = true;
  rebuildBtn.disabled = true;
  updateBtn.disabled = true;
  try {
    await requestJson("/api/index/rebuild", { method: "POST" });
    for (let i = 0; i < 180; i += 1) {
      const s = await fetchIndexStatus();
      if (!s) break;
      if (s.state === "success") {
        updateRebuildProgress(100, s.message || "重建完成");
        appendMessage("system", `索引重建完成：${s.message || "已完成"}`);
        pushSystemEvent("索引重建完成", "success");
        setStatus("索引重建完成");
        showBanner("索引重建完成。", "success");
        notesDirty = false;
        await loadNotes();
        await loadRuntimeSummary();
        return;
      }
      if (s.state === "error") {
        appendMessage("error", `索引重建失败：${s.last_error || s.message || "未知错误"}`);
        pushSystemEvent(`索引重建失败：${s.last_error || s.message || "未知错误"}`, "error");
        setStatus("索引重建失败");
        showBanner(`索引重建失败：${s.last_error || s.message || "未知错误"}`, "error");
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    setStatus("索引重建等待超时，可手动刷新状态");
    showBanner("索引重建等待超时，可稍后手动刷新。", "warn");
    pushSystemEvent("索引重建等待超时：请手动刷新状态", "warn");
  } catch (err) {
    setStatus(`重建失败：${err.message}`);
    showBanner(`索引重建失败：${err.message}`, "error");
    pushSystemEvent(`索引重建失败：${err.message}`, "error");
  } finally {
    manualRebuildBtn.disabled = false;
    rebuildBtn.disabled = false;
    updateBtn.disabled = false;
  }
}

async function loadCurrentNote() {
  const p = noteSelect.value;
  if (!p) {
    currentNotePath = "";
    noteEditor.value = "";
    return;
  }
  if (!p.endsWith(".md") && !p.endsWith(".txt")) {
    currentNotePath = p;
    noteEditor.value = "(当前为图片文件，不支持文本编辑)";
    return;
  }
  try {
    const data = await requestJson(`/api/notes/content?path=${encodeURIComponent(p)}`);
    currentNotePath = data.path;
    noteEditor.value = data.content || "";
    notesDirty = false;
    setStatus(`已加载：${currentNotePath}`);
  } catch (err) {
    setStatus(`加载失败：${err.message}`);
  }
}

async function createNote() {
  const p = (newPathInput.value || "").trim();
  if (!p) return;
  try {
    await requestJson("/api/notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: p, content: "" }),
    });
    await loadNotes();
    noteSelect.value = p;
    currentNotePath = p;
    noteEditor.value = "";
    notesDirty = true;
    setStatus(`已新建：${p}`);
  } catch (err) {
    setStatus(`新建失败：${err.message}`);
  }
}

async function saveNote() {
  const p = currentNotePath || noteSelect.value;
  if (!p) {
    setStatus("请先选择或创建笔记");
    return;
  }
  if (!p.endsWith(".md") && !p.endsWith(".txt")) {
    setStatus("图片文件不支持文本保存");
    return;
  }
  try {
    await requestJson("/api/notes", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: p, content: noteEditor.value || "" }),
    });
    currentNotePath = p;
    notesDirty = true;
    setStatus(`已保存：${p}（待更新索引）`);
  } catch (err) {
    setStatus(`保存失败：${err.message}`);
  }
}

async function deleteNote() {
  const p = currentNotePath || noteSelect.value;
  if (!p) return;
  if (!confirm(`确认删除 ${p} ?`)) return;
  try {
    await requestJson(`/api/notes?path=${encodeURIComponent(p)}`, { method: "DELETE" });
    currentNotePath = "";
    noteEditor.value = "";
    notesDirty = true;
    await loadNotes();
    setStatus(`已删除：${p}（待更新索引）`);
  } catch (err) {
    setStatus(`删除失败：${err.message}`);
  }
}

async function renameNote() {
  const oldPath = currentNotePath || noteSelect.value;
  const newPath = (renamePathInput.value || "").trim();
  if (!oldPath || !newPath) {
    setStatus("请选择笔记并填写新路径");
    return;
  }
  try {
    await requestJson("/api/notes/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ old_path: oldPath, new_path: newPath }),
    });
    currentNotePath = newPath;
    notesDirty = true;
    await loadNotes();
    noteSelect.value = newPath;
    setStatus(`已重命名：${oldPath} -> ${newPath}（待更新索引）`);
  } catch (err) {
    setStatus(`重命名失败：${err.message}`);
  }
}

async function uploadNote() {
  const file = uploadFileInput.files?.[0];
  if (!file) {
    setStatus("请先选择文件");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  try {
    await requestJson("/api/notes/upload", { method: "POST", body: form });
    notesDirty = true;
    await loadNotes();
    setStatus(`上传成功：${file.name}（待更新索引）`);
  } catch (err) {
    setStatus(`上传失败：${err.message}`);
  }
}

function applyButtonVariants() {
  document.querySelectorAll("button").forEach((btn) => {
    btn.classList.add("btn");
    if (
      btn.classList.contains("chip") ||
      btn.closest(".tabs") ||
      btn.classList.contains("icon-btn")
    ) {
      return;
    }
    if (btn.classList.contains("secondary")) {
      btn.classList.add("btn-secondary");
      return;
    }
    btn.classList.add("btn-primary");
  });

  deleteNoteBtn?.classList.remove("btn-primary", "btn-secondary");
  deleteNoteBtn?.classList.add("btn-danger");
  abortAccessTicketBtn?.classList.remove("btn-primary", "btn-secondary");
  abortAccessTicketBtn?.classList.add("btn-danger");
  closeRebuildLogModalBtn?.classList.remove("btn-primary", "btn-secondary", "btn-danger");
  closeRebuildLogModalBtn?.classList.add("btn-ghost");
}

async function loadConfig() {
  setConfigStatus("读取配置中...");
  try {
    const data = await requestJson("/api/config");
    const values = data.values || {};
    CFG_KEYS.forEach((k) => {
      const el = cfgEl(k);
      if (el) el.value = values[k] ?? "";
    });
    setConfigStatus("配置已加载");
  } catch (err) {
    setConfigStatus(`读取失败：${err.message}`);
  }
}

async function saveConfig() {
  setConfigStatus("保存中...");
  const values = {};
  CFG_KEYS.forEach((k) => {
    const el = cfgEl(k);
    if (!el) return;
    values[k] = el.value;
  });
  try {
    const data = await requestJson("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    setConfigStatus(`保存成功：${(data.updated || []).join(", ")}`);
  } catch (err) {
    setConfigStatus(`保存失败：${err.message}`);
  }
}

async function testConfig() {
  setConfigStatus("连接测试中...");
  try {
    const data = await requestJson("/api/config/test", { method: "POST" });
    setConfigStatus(data.ok ? `测试成功：${data.preview || "ok"}` : `测试失败：${data.message}`);
    pushSystemEvent(data.ok ? "配置测试成功" : `配置测试失败：${data.message}`, data.ok ? "success" : "error");
  } catch (err) {
    setConfigStatus(`测试失败：${err.message}`);
    pushSystemEvent(`配置测试失败：${err.message}`, "error");
  }
}

async function resetConfig() {
  if (!confirm("确认恢复默认配置？")) return;
  setConfigStatus("恢复默认中...");
  try {
    await requestJson("/api/config/reset", { method: "POST" });
    await loadConfig();
    setConfigStatus("已恢复默认并加载");
  } catch (err) {
    setConfigStatus(`恢复失败：${err.message}`);
  }
}

async function loadSystemStatus() {
  try {
    const data = await requestJson("/api/system/status");
    systemStatusBox.textContent = JSON.stringify(data, null, 2);
    if (data.status_hint === "config_incomplete") {
      showBanner("当前配置未完成，请先在配置页补全 BASE_URL 和 API_KEY。", "warn");
    }
  } catch (err) {
    systemStatusBox.textContent = `读取失败：${err.message}`;
  }
}

async function loadAdminDiagnostics() {
  try {
    const data = await requestJson("/api/admin/diagnostics");
    const consoleTail = data?.recent_logs?.console_tail || "(暂无控制台输出)";
    const eventsTail = data?.recent_logs?.events_jsonl_tail || "(暂无事件日志)";
    const summary = {
      ok: data.ok,
      paths: data.paths,
      config: data.config,
      runtime: data.runtime,
    };
    adminStatusBox.textContent = [
      "===== 后端运行摘要 =====",
      JSON.stringify(summary, null, 2),
      "",
      "===== 启动/运行控制台输出 =====",
      consoleTail,
      "",
      "===== events.jsonl 结构化日志 =====",
      eventsTail,
    ].join("\n");
  } catch (err) {
    adminStatusBox.textContent = `读取失败：${err.message}`;
  }
}

async function loadRebuildLogs() {
  try {
    const data = await requestJson("/api/admin/console?limit=300");
    const status = data?.index_task || {};
    rebuildLogMeta.textContent = `state=${status.state || "unknown"} progress=${status.progress ?? 0}% message=${status.message || ""}`;
    rebuildLogBox.textContent = data?.console_tail || "(暂无控制台输出)";
    rebuildLogBox.scrollTop = rebuildLogBox.scrollHeight;
  } catch (err) {
    rebuildLogMeta.textContent = "读取详情日志失败";
    rebuildLogBox.textContent = `读取失败：${err.message}`;
  }
}

function startRebuildLogPolling() {
  stopRebuildLogPolling();
  rebuildLogPoller = setInterval(loadRebuildLogs, 1200);
}

function stopRebuildLogPolling() {
  if (rebuildLogPoller) {
    clearInterval(rebuildLogPoller);
    rebuildLogPoller = null;
  }
}

async function openRebuildLogModal() {
  rebuildLogModal.classList.remove("hidden");
  rebuildLogModal.style.display = "flex";
  await loadRebuildLogs();
  startRebuildLogPolling();
}

function closeRebuildLogModal() {
  rebuildLogModal.classList.add("hidden");
  rebuildLogModal.style.display = "none";
  stopRebuildLogPolling();
}

function handleGlobalKeydown(e) {
  if (e.key !== "Escape") return;

  if (!rebuildLogModal.classList.contains("hidden")) {
    closeRebuildLogModal();
    return;
  }

  if (!helpPanel?.classList.contains("hidden")) {
    closeHelpPanel();
    showBanner("已关闭帮助面板。", "warn");
    setStatus("帮助面板已关闭");
    return;
  }

  if (!securityPanel.classList.contains("hidden")) {
    closeSecurityPanelAction();
  }
}

async function switchTab(next) {
  if (next === currentTab) return;

  if (currentTab === "notes" && notesDirty) {
    await triggerRebuildAndWait("离开笔记管理页自动触发");
  }

  currentTab = next;
  const inChat = next === "chat";
  const inNotes = next === "notes";
  const inConfig = next === "config";
  const inSystem = next === "system";
  const inAdmin = next === "admin";

  chatPage.classList.toggle("hidden", !inChat);
  notesPage.classList.toggle("hidden", !inNotes);
  configPage.classList.toggle("hidden", !inConfig);
  systemPage.classList.toggle("hidden", !inSystem);
  adminPage.classList.toggle("hidden", !inAdmin);

  tabChat.classList.toggle("active", inChat);
  tabNotes.classList.toggle("active", inNotes);
  tabConfig.classList.toggle("active", inConfig);
  tabSystem.classList.toggle("active", inSystem);
  tabAdmin.classList.toggle("active", inAdmin);

  if (inChat) {
    openHelpPanel();
    await loadRuntimeSummary();
  }
  if (inNotes) {
    await loadNotes();
    await fetchIndexStatus();
  }
  if (inConfig) {
    await loadConfig();
  }
  if (inSystem) {
    await loadSystemStatus();
  }
  if (inAdmin) {
    await loadAdminDiagnostics();
  }
}

sendBtn.addEventListener("click", sendMessage);
submitSecurityKeyBtn.addEventListener("click", submitSecurityKey);
abortAccessTicketBtn.addEventListener("click", abortAccessTicket);
closeSecurityPanelBtn.addEventListener("click", closeSecurityPanelAction);
helpBtn.addEventListener("click", async () => {
  appendMessage("user", "/help");
  await handleLocalCommand("/help");
});
closeHelpPanelBtn?.addEventListener("click", closeHelpPanel);
configSummaryBtn.addEventListener("click", async () => {
  appendMessage("user", "/config");
  await handleLocalCommand("/config");
});
metricsBtn.addEventListener("click", async () => {
  appendMessage("user", "/metrics");
  await handleLocalCommand("/metrics");
});
updateBtn.addEventListener("click", async () => {
  appendMessage("user", "/update");
  await handleLocalCommand("/update");
});
if (exampleQuestions) {
  exampleQuestions.addEventListener("click", (e) => {
    const target = e.target.closest("[data-question]");
    if (!target) return;
    chatInput.value = target.dataset.question || "";
    chatInput.focus();
  });
}
chatInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  if (e.shiftKey) return;
  e.preventDefault();
  sendMessage();
});
securityKeyInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitSecurityKey();
});
refreshNotesBtn.addEventListener("click", loadNotes);
rebuildBtn.addEventListener("click", () => triggerRebuildAndWait("聊天页手动触发"));

tabChat.addEventListener("click", () => switchTab("chat"));
tabNotes.addEventListener("click", () => switchTab("notes"));
tabConfig.addEventListener("click", () => switchTab("config"));
tabSystem.addEventListener("click", () => switchTab("system"));
tabAdmin.addEventListener("click", () => switchTab("admin"));

loadNoteBtn.addEventListener("click", loadCurrentNote);
createNoteBtn.addEventListener("click", createNote);
saveNoteBtn.addEventListener("click", saveNote);
deleteNoteBtn.addEventListener("click", deleteNote);
renameNoteBtn.addEventListener("click", renameNote);
uploadBtn.addEventListener("click", uploadNote);
refreshIndexStatusBtn.addEventListener("click", fetchIndexStatus);
manualRebuildBtn.addEventListener("click", () => triggerRebuildAndWait("笔记页手动触发"));
showRebuildLogsBtn.addEventListener("click", openRebuildLogModal);
noteEditor.addEventListener("input", () => {
  if (currentTab === "notes") notesDirty = true;
});

loadConfigBtn.addEventListener("click", loadConfig);
saveConfigBtn.addEventListener("click", saveConfig);
testConfigBtn.addEventListener("click", testConfig);
resetConfigBtn.addEventListener("click", resetConfig);
refreshSystemBtn.addEventListener("click", loadSystemStatus);
refreshAdminBtn.addEventListener("click", loadAdminDiagnostics);
refreshRebuildLogsBtn.addEventListener("click", loadRebuildLogs);
closeRebuildLogModalBtn.addEventListener("click", closeRebuildLogModal);
rebuildLogModal.addEventListener("click", (e) => {
  if (e.target === rebuildLogModal) closeRebuildLogModal();
});
document.addEventListener("keydown", handleGlobalKeydown);

themeSelect.addEventListener("change", (e) => applyTheme(e.target.value));

const savedTheme = localStorage.getItem("theme") || "system";
themeSelect.value = savedTheme;
applyTheme(savedTheme);
applyButtonVariants();
setChatState("idle");

loadNotes();
fetchIndexStatus();
loadRuntimeSummary();


