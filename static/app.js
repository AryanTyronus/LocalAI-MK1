document.addEventListener("DOMContentLoaded", () => {

  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const feed = document.getElementById("feed");

  // Configure marked for safe markdown rendering
  marked.setOptions({
    breaks: true,
    gfm: true
  });

  function updateButtonState() {
    sendBtn.disabled = input.value.trim() === "";
  }

  /**
   * Escape HTML to prevent XSS
   */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Render markdown safely
   */
  function renderMarkdown(text) {
    // First escape any raw HTML
    const escaped = escapeHtml(text);
    // Then parse markdown
    return marked.parse(escaped);
  }

  /**
   * Add a user message to the feed
   */
  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg-row msg-row--user";

    const bubble = document.createElement("div");
    bubble.className = "bubble bubble--user";
    bubble.textContent = text;

    row.appendChild(bubble);
    feed.appendChild(row);

    feed.scrollTop = feed.scrollHeight;
  }

  /**
   * Add a typing indicator bubble
   */
  function addTypingIndicator() {
    const row = document.createElement("div");
    row.className = "msg-row msg-row--ai";
    row.id = "typing-indicator";

    const bubble = document.createElement("div");
    bubble.className = "bubble bubble--ai typing-indicator";
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    row.appendChild(bubble);
    feed.appendChild(row);

    feed.scrollTop = feed.scrollHeight;
    return row;
  }

  /**
   * Remove typing indicator and return the row for replacement
   */
  function removeTypingIndicator() {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) {
      indicator.remove();
    }
  }

  /**
   * Add an AI message bubble (with streaming support)
   */
  function addAiMessageStart() {
    const row = document.createElement("div");
    row.className = "msg-row msg-row--ai";

    const bubble = document.createElement("div");
    bubble.className = "bubble bubble--ai";
    bubble.id = "current-ai-message";

    row.appendChild(bubble);
    feed.appendChild(row);

    feed.scrollTop = feed.scrollHeight;
    return { row, bubble };
  }

  /**
   * Append content to the current AI message
   */
  function appendToAiMessage(content) {
    const bubble = document.getElementById("current-ai-message");
    if (bubble) {
      // Render markdown for the full content
      bubble.innerHTML = renderMarkdown(content);
      feed.scrollTop = feed.scrollHeight;
    }
  }

  /**
   * Add token usage footer
   */
  function addTokenFooter(promptTokens, completionTokens, totalTokens) {
    const bubble = document.getElementById("current-ai-message");
    if (bubble) {
      const footer = document.createElement("div");
      footer.className = "token-footer";
      footer.textContent = `Tokens: ${totalTokens} (Prompt: ${promptTokens}, Completion: ${completionTokens})`;
      bubble.appendChild(footer);
    }
  }

  /**
   * Send message using streaming endpoint
   */
  async function sendMessageStream() {
    const message = input.value.trim();
    if (!message) return;

    sendBtn.disabled = true;
    input.value = "";

    // Add user message
    addUserMessage(message);

    // Add typing indicator
    const typingRow = addTypingIndicator();

    try {
      const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });

      console.log("Stream Status:", response.status);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      // Remove typing indicator and create AI message bubble
      typingRow.remove();
      const { bubble } = addAiMessageStart();

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        
        // Parse SSE data
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            // Check for error
            if (data.startsWith('__ERROR__:')) {
              const errorMsg = data.slice(10);
              fullContent += '\n\n' + errorMsg;
              bubble.innerHTML = renderMarkdown(fullContent);
              continue;
            }
            
            // Check for token info (final chunk)
            if (data.startsWith('__TOKENS__:')) {
              const tokenInfo = JSON.parse(data.slice(11));
              console.log("Token info:", tokenInfo);
              addTokenFooter(
                tokenInfo.prompt_tokens,
                tokenInfo.completion_tokens,
                tokenInfo.total_tokens
              );
              continue;
            }
            
            // Regular content chunk
            if (data) {
              fullContent += data;
              bubble.innerHTML = renderMarkdown(fullContent);
              feed.scrollTop = feed.scrollHeight;
            }
          }
        }
      }

    } catch (error) {
      console.error("Frontend Stream Error:", error);
      typingRow.remove();
      
      const row = document.createElement("div");
      row.className = "msg-row msg-row--ai";
      
      const bubble = document.createElement("div");
      bubble.className = "bubble bubble--ai";
      bubble.textContent = "Error connecting to server.";
      
      row.appendChild(bubble);
      feed.appendChild(row);
    }

    updateButtonState();
  }

  /**
   * Send message using regular fetch (fallback)
   */
  async function sendMessage() {
    const message = input.value.trim();
    if (!message) return;

    sendBtn.disabled = true;

    addUserMessage(message);
    input.value = "";

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });

      console.log("Status:", response.status);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const rawText = await response.text();
      console.log("Raw response:", rawText);

      let data;

      try {
        data = JSON.parse(rawText);
      } catch (parseError) {
        console.error("JSON Parse Error:", parseError);
        addMessage("Invalid JSON received from server.", "ai");
        return;
      }

      console.log("Parsed JSON:", data);

      if (!data.response) {
        addMessage("No 'response' key found in backend reply.", "ai");
        return;
      }

      addMessage(data.response, "ai");

    } catch (error) {
      console.error("Frontend Error:", error);
      addMessage("Error connecting to server.", "ai");
    }

    updateButtonState();
  }

  function addMessage(text, role) {
    const row = document.createElement("div");
    row.className = `msg-row msg-row--${role}`;

    const bubble = document.createElement("div");
    bubble.className = `bubble bubble--${role}`;
    bubble.textContent = text;

    row.appendChild(bubble);
    feed.appendChild(row);

    feed.scrollTop = feed.scrollHeight;
  }

  input.addEventListener("input", updateButtonState);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessageStream();
    }
  });

  sendBtn.addEventListener("click", sendMessageStream);

});

