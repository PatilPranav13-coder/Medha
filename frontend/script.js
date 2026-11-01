const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const modeSelect = document.querySelector("#mode-select");
const platformSelect = document.querySelector("#platform-select");
const publicSources = document.querySelector("#public-sources");

function addMessage(text, role, isTyping = false) {
  welcome.hidden = true;
  const article = document.createElement("article");
  article.className = "message " + role;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "M";

  const content = document.createElement("div");
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isTyping ? " typing" : "");
  bubble.textContent = text;
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = role === "user" ? "You" : "Medha";
  content.append(bubble, label);
  article.append(avatar, content);
  messages.append(article);
  article.scrollIntoView({ block: "end", behavior: "smooth" });
  return bubble;
}

function addSources(bubble, sources) {
  if (!Array.isArray(sources) || sources.length === 0) return;
  const list = document.createElement("div");
  list.className = "sources";
  const heading = document.createElement("strong");
  heading.textContent = "Public sources";
  list.append(heading);

  sources.forEach(source => {
    if (!source || typeof source.url !== "string" || !source.url.startsWith("https://")) return;
    const link = document.createElement("a");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = source.title || source.url;
    list.append(link);
    if (source.attribution || source.license) {
      const note = document.createElement("span");
      note.textContent = [source.attribution, source.license].filter(Boolean).join(" · ");
      list.append(note);
    }
  });

  if (list.querySelector("a")) bubble.after(list);
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message) return;

  addMessage(message, "user");
  input.value = "";
  sendButton.disabled = true;
  const pending = addMessage("Medha is thinking…", "assistant", true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        mode: modeSelect.value,
        platform: platformSelect.value,
        use_public_sources: publicSources.checked
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");
    pending.classList.remove("typing");
    pending.textContent = data.response;
    addSources(pending, data.sources);
  } catch (error) {
    pending.classList.remove("typing");
    pending.textContent = error.message || "Unable to contact Medha. Check Ollama and the FastAPI server.";
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  sendMessage(input.value);
});

document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => {
    if (button.dataset.mode) modeSelect.value = button.dataset.mode;
    if (button.dataset.platform) platformSelect.value = button.dataset.platform;
    sendMessage(button.dataset.prompt);
  });
});

document.querySelector("#new-chat").addEventListener("click", () => {
  messages.innerHTML = "";
  welcome.hidden = false;
  input.focus();
});
