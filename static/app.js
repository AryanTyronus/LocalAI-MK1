document.addEventListener("DOMContentLoaded", () => {

  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const feed = document.getElementById("feed");

  function updateButtonState() {
    sendBtn.disabled = input.value.trim() === "";
  }

  async function sendMessage() {
    const message = input.value.trim();
    if (!message) return;

    sendBtn.disabled = true;

    addMessage(message, "user");
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

      // Read raw text first (safer than directly calling .json())
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
      sendMessage();
    }
  });

  sendBtn.addEventListener("click", sendMessage);

});