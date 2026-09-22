const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");

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
      body: JSON.stringify({ message })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");
    pending.classList.remove("typing");
    pending.textContent = data.response;
  } catch (error) {
    pending.classList.remove("typing");
    pending.textContent = "Unable to contact Medha. Please check that the FastAPI server is running.";
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
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

document.querySelector("#new-chat").addEventListener("click", () => {
  messages.innerHTML = "";
  welcome.hidden = false;
  input.focus();
});
