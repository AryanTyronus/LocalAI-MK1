document.addEventListener("DOMContentLoaded", () => {
  const feed = document.getElementById("feed");
  const messagesArea = document.querySelector(".messages-area");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const charCount = document.getElementById("charCount");

  const projectSelect = document.getElementById("projectSelect");
  const modeSelect = document.getElementById("modeSelect");
  const memoryToggle = document.getElementById("memoryToggle");
  const devLogToggle = document.getElementById("devLogToggle");
  const modelDot = document.getElementById("modelDot");
  const modelStatusText = document.getElementById("modelStatusText");
  const devLogPanel = document.getElementById("devLogPanel");

  const newChatBtn = document.getElementById("newChatBtn");
  const settingsBtn = document.getElementById("settingsBtn");
  const workspaceConfig = document.getElementById("workspaceConfig");

  const memoryDrawerBtn = document.getElementById("memoryDrawerBtn");
  const logsDrawerBtn = document.getElementById("logsDrawerBtn");
  const drawer = document.getElementById("rightDrawer");
  const drawerTitle = document.getElementById("drawerTitle");
  const drawerContent = document.getElementById("drawerContent");
  const drawerCloseBtn = document.getElementById("drawerCloseBtn");

  const toolLogs = [];
  const toolRegex = /\b(weather|news|stock|ticker|read file|run python|\/tool|nse|bse|nifty|sensex)\b/i;
  const SYSTEM_CARD_PREFIX = "__SYSTEM_CARD__:";

  const markedRenderer = (text) => marked.parse(escapeHtml(text), { breaks: true, gfm: true });

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function scrollToBottom() {
    if (messagesArea) {
      messagesArea.scrollTop = messagesArea.scrollHeight;
    }
  }

  function nowLabel() {
    return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }

  function updateSendState() {
    sendBtn.disabled = input.value.trim() === "";
    if (charCount) {
      charCount.textContent = `${input.value.length} chars`;
    }
  }

  function setModelStatus(type, text) {
    modelDot.classList.remove("status-dot--idle", "status-dot--active", "status-dot--error");
    modelDot.classList.add(type === "error" ? "status-dot--error" : type === "active" ? "status-dot--active" : "status-dot--idle");
    modelStatusText.textContent = text;
  }

  function addDevLog(message) {
    if (!devLogPanel || !devLogToggle.checked) return;
    const line = document.createElement("div");
    const time = new Date().toLocaleTimeString();
    line.textContent = `[${time}] ${message}`;
    devLogPanel.appendChild(line);
    devLogPanel.scrollTop = devLogPanel.scrollHeight;
  }

  function persistUiState() {
    localStorage.setItem("localai_project", projectSelect.value);
    localStorage.setItem("localai_mode", modeSelect.value);
    localStorage.setItem("localai_memory", memoryToggle.checked ? "1" : "0");
    localStorage.setItem("localai_devlogs", devLogToggle.checked ? "1" : "0");
    localStorage.setItem("localai_workspace_open", workspaceConfig.classList.contains("hidden") ? "0" : "1");
  }

  function restoreUiState() {
    const p = localStorage.getItem("localai_project");
    const m = localStorage.getItem("localai_mode");
    const mem = localStorage.getItem("localai_memory");
    const logs = localStorage.getItem("localai_devlogs");
    const workspaceOpen = localStorage.getItem("localai_workspace_open");

    if (p && projectSelect.querySelector(`option[value="${p}"]`)) projectSelect.value = p;
    if (m && modeSelect.querySelector(`option[value="${m}"]`)) modeSelect.value = m;
    if (mem !== null) memoryToggle.checked = mem === "1";
    if (logs !== null) devLogToggle.checked = logs === "1";
    if (workspaceOpen === "0") workspaceConfig.classList.add("hidden");

    devLogPanel.classList.toggle("hidden", !devLogToggle.checked);
  }

  function runtimeOptions() {
    return {
      project: projectSelect.value,
      memory_enabled: memoryToggle.checked,
      dev_logs: devLogToggle.checked
    };
  }

  function clearWelcome() {
    const welcome = feed.querySelector(".welcome");
    if (welcome) welcome.remove();
  }

  function createMessageRow(role) {
    const row = document.createElement("article");
    row.className = role === "user" ? "msg-row user" : "msg-row";

    const avatar = document.createElement("div");
    avatar.className = `avatar ${role === "user" ? "user" : "ai"}`;
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "YO" : "AI";

    const col = document.createElement("div");
    col.className = "msg-col";

    const bubble = document.createElement("div");
    bubble.className = `bubble ${role === "user" ? "user" : "ai"}`;

    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.textContent = `${role === "user" ? "You" : "LocalAI"} · ${nowLabel()}`;

    col.appendChild(bubble);
    col.appendChild(meta);
    row.appendChild(col);
    row.appendChild(avatar);
    feed.appendChild(row);

    scrollToBottom();
    return { bubble, row };
  }

  function addMessage(text, role, options = {}) {
    clearWelcome();
    const { bubble } = createMessageRow(role);
    bubble.innerHTML = role === "ai" ? markedRenderer(text) : escapeHtml(text);

    if (options.memoryUpdated) {
      const note = document.createElement("div");
      note.className = "memory-note";
      note.textContent = "Memory updated";
      bubble.appendChild(note);
    }

    if (options.toolStatus) {
      const note = document.createElement("div");
      note.className = "tool-status";
      note.textContent = options.toolStatus;
      bubble.appendChild(note);
    }

    return bubble;
  }

  function appendTokenFooter(bubble, tokens) {
    const footer = document.createElement("div");
    footer.className = "token-footer";
    footer.textContent = `Tokens: ${tokens.total_tokens} (Prompt: ${tokens.prompt_tokens}, Completion: ${tokens.completion_tokens})`;
    bubble.appendChild(footer);
  }

  function createStreamingBubble() {
    clearWelcome();
    const { bubble } = createMessageRow("ai");
    bubble.classList.add("streaming");
    return bubble;
  }

  function renderHelpSystemCard(bubble, payload) {
    const sections = Array.isArray(payload.sections) ? payload.sections : [];
    const sectionHtml = sections.map((section) => {
      const groups = Array.isArray(section.groups) ? section.groups : [];
      const groupHtml = groups.map((group) => {
        const items = Array.isArray(group.items) ? group.items : [];
        const itemHtml = items.map((item) => {
          if (section.kind === "commands") {
            return `<div class="memory-row"><strong>${escapeHtml(String(item.name || ""))}</strong>: ${escapeHtml(String(item.description || ""))}</div>`;
          }
          return `<div class="memory-row"><strong>${escapeHtml(String(item.name || item.usage || ""))}</strong>: ${escapeHtml(String(item.description || item.value || ""))}</div>`;
        }).join("");

        return `<div class="memory-card"><h4>${escapeHtml(String(group.category || "General"))}</h4>${itemHtml || "<div class='memory-row'>No entries</div>"}</div>`;
      }).join("");

      return `<section><h4>${escapeHtml(String(section.title || section.kind || "Section"))}</h4>${groupHtml || "<div class='memory-row'>No entries</div>"}</section>`;
    }).join("");

    bubble.classList.remove("bubble", "ai", "streaming");
    bubble.classList.add("bubble", "system");
    bubble.innerHTML = `<div class="memory-card"><h4>${escapeHtml(String(payload.title || "System"))}</h4>${sectionHtml || "<div class='memory-row'>No help entries available.</div>"}</div>`;
  }

  function looksLikeToolMessage(message) {
    return toolRegex.test(message || "");
  }

  function logToolRun(message, elapsedMs, success) {
    const entry = {
      when: new Date().toLocaleTimeString(),
      input: message,
      elapsedMs,
      success
    };
    toolLogs.unshift(entry);
    if (toolLogs.length > 40) toolLogs.pop();
    if (drawer.classList.contains("open") && drawerTitle.textContent === "Tool Logs") {
      renderToolLogs();
    }
  }

  function renderToolLogs() {
    if (!toolLogs.length) {
      drawerContent.innerHTML = `<div class="memory-card"><div class="memory-row">No tool logs yet.</div></div>`;
      return;
    }
    drawerContent.innerHTML = toolLogs.map((item) => {
      return `
        <div class="memory-card">
          <h4>${item.when}</h4>
          <div class="memory-row"><strong>Input:</strong> ${escapeHtml(item.input)}</div>
          <div class="memory-row"><strong>Status:</strong> ${item.success ? "completed" : "failed"}</div>
          <div class="memory-row"><strong>Duration:</strong> ${item.elapsedMs} ms</div>
        </div>
      `;
    }).join("");
  }

  function normalizeMemoryData(data) {
    if (!data || typeof data !== "object") return null;
    if (data.user || data.preferences || data.system_state) return data;
    if (data.content) {
      try {
        return JSON.parse(data.content);
      } catch (_e) {
        return null;
      }
    }
    return null;
  }

  function extractToolPayload(text) {
    const start = text.indexOf("{");
    if (start < 0) return null;
    const raw = text.slice(start);
    try {
      return JSON.parse(raw);
    } catch (_e) {
      return null;
    }
  }

  async function fetchStructuredMemory() {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "/tool read file structured_memory.json",
        mode: modeSelect.value,
        options: runtimeOptions()
      })
    });
    const payload = await response.json();
    if (!payload || !payload.response) return null;
    const toolResult = extractToolPayload(payload.response);
    if (!toolResult || !toolResult.ok) return null;
    return normalizeMemoryData(toolResult);
  }

  function renderMemoryCards(memoryData) {
    if (!memoryData) {
      drawerContent.innerHTML = `<div class="memory-card"><div class="memory-row">Memory data unavailable.</div></div>`;
      return;
    }
    const user = memoryData.user || {};
    const prefs = memoryData.preferences || {};
    const state = memoryData.system_state || {};
    const difficulties = Array.isArray(state.difficulties) ? state.difficulties : [];

    const prefRows = Object.keys(prefs).length
      ? Object.entries(prefs).map(([k, v]) => `<div class="memory-row"><strong>${escapeHtml(k)}:</strong> ${escapeHtml(String(v))}</div>`).join("")
      : `<div class="memory-row">No preferences saved.</div>`;

    const diffRows = difficulties.length
      ? difficulties.map((d) => `<div class="memory-row">- ${escapeHtml(String(d))}</div>`).join("")
      : `<div class="memory-row">No difficulty subjects saved.</div>`;

    drawerContent.innerHTML = `
      <div class="memory-card">
        <h4>User</h4>
        <div class="memory-row"><strong>Name:</strong> ${escapeHtml(String(user.name || "Not set"))}</div>
        <div class="memory-row"><strong>Birth Year:</strong> ${escapeHtml(String(user.birth_year || "Not set"))}</div>
        <div class="memory-row"><strong>Age:</strong> ${escapeHtml(String(user.age || "Not set"))}</div>
      </div>
      <div class="memory-card">
        <h4>Preferences</h4>
        ${prefRows}
      </div>
      <div class="memory-card">
        <h4>Difficulties</h4>
        ${diffRows}
      </div>
    `;
  }

  async function openMemoryDrawer() {
    drawer.classList.add("open");
    drawerTitle.textContent = "Memory";
    drawerContent.innerHTML = `<div class="memory-card"><div class="memory-row">Loading memory...</div></div>`;
    try {
      const memoryData = await fetchStructuredMemory();
      renderMemoryCards(memoryData);
    } catch (_err) {
      drawerContent.innerHTML = `<div class="memory-card"><div class="memory-row">Failed to load memory snapshot.</div></div>`;
    }
  }

  function openLogsDrawer() {
    drawer.classList.add("open");
    drawerTitle.textContent = "Tool Logs";
    renderToolLogs();
  }

  async function sendMessageStream() {
    const message = input.value.trim();
    if (!message) return;

    input.value = "";
    input.style.height = "auto";
    updateSendState();
    setModelStatus("active", "Generating...");
    addMessage(message, "user");

    const mode = modeSelect.value;
    const options = runtimeOptions();
    const runTool = looksLikeToolMessage(message);
    const start = Date.now();
    let toolStatusBubble = null;

    if (runTool) {
      toolStatusBubble = addMessage("Working...", "ai", { toolStatus: "Running tool..." });
    }

    const streamBubble = createStreamingBubble();
    let fullText = "";

    try {
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, mode, options })
      });
      addDevLog(`stream status=${response.status}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let doneTokens = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (!data) continue;

          if (data.startsWith("__ERROR__:")) {
            throw new Error(data.slice(10));
          }

          if (data.startsWith("__TOKENS__:")) {
            doneTokens = JSON.parse(data.slice(11));
            continue;
          }

          if (data.startsWith(SYSTEM_CARD_PREFIX)) {
            try {
              const payload = JSON.parse(data.slice(SYSTEM_CARD_PREFIX.length));
              renderHelpSystemCard(streamBubble, payload);
              continue;
            } catch (_err) {
              // Ignore malformed payload and continue text rendering.
            }
          }

          fullText += data;
          streamBubble.innerHTML = markedRenderer(fullText);
          scrollToBottom();
        }
      }

      streamBubble.classList.remove("streaming");
      if (doneTokens) {
        appendTokenFooter(streamBubble, doneTokens);
        if (doneTokens.memory_updated) {
          const mem = document.createElement("div");
          mem.className = "memory-note";
          mem.textContent = "Memory updated";
          streamBubble.appendChild(mem);
        }
        if (doneTokens.runtime) {
          addDevLog(`runtime ${JSON.stringify(doneTokens.runtime)}`);
        }
      }

      if (runTool && toolStatusBubble) {
        const elapsed = Date.now() - start;
        logToolRun(message, elapsed, true);
        const toolStatus = toolStatusBubble.querySelector(".tool-status");
        if (toolStatus) toolStatus.textContent = `Tool completed in ${elapsed} ms`;
      }

      setModelStatus("active", "Ready");
    } catch (err) {
      setModelStatus("error", "Error");
      streamBubble.classList.remove("streaming");
      streamBubble.innerHTML = markedRenderer(`Error: ${String(err)}`);

      if (runTool && toolStatusBubble) {
        const elapsed = Date.now() - start;
        logToolRun(message, elapsed, false);
        const toolStatus = toolStatusBubble.querySelector(".tool-status");
        if (toolStatus) toolStatus.textContent = `Tool failed in ${elapsed} ms`;
      }

      addDevLog(`stream error=${String(err)}`);
    }
  }

  function clearConversation() {
    feed.innerHTML = `
      <div class="welcome">
        <div class="welcome-glyph" aria-hidden="true">◈</div>
        <h2>Welcome to LocalAI</h2>
        <p>Your private assistant running locally. Ask coding, research, or tool-powered questions.</p>
      </div>
    `;
    setModelStatus("idle", "Idle");
  }

  input.addEventListener("input", () => {
    updateSendState();
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessageStream();
    }
  });

  sendBtn.addEventListener("click", sendMessageStream);
  newChatBtn.addEventListener("click", clearConversation);

  projectSelect.addEventListener("change", persistUiState);
  modeSelect.addEventListener("change", persistUiState);
  memoryToggle.addEventListener("change", persistUiState);
  devLogToggle.addEventListener("change", () => {
    devLogPanel.classList.toggle("hidden", !devLogToggle.checked);
    persistUiState();
  });

  settingsBtn.addEventListener("click", () => {
    workspaceConfig.classList.toggle("hidden");
    persistUiState();
  });

  memoryDrawerBtn.addEventListener("click", openMemoryDrawer);
  logsDrawerBtn.addEventListener("click", openLogsDrawer);
  drawerCloseBtn.addEventListener("click", () => drawer.classList.remove("open"));

  restoreUiState();
  updateSendState();
  setModelStatus("idle", "Idle");
});
