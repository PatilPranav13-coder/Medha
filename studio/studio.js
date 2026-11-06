const idea = document.querySelector("#idea");
const plan = document.querySelector("#plan");
const toast = document.querySelector("#toast");
const projectName = document.querySelector("#project-name");
let toastTimer;
function notify(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2600);
}
document.querySelectorAll("[data-idea]").forEach(button => {
  button.addEventListener("click", () => {
    idea.value = button.dataset.idea;
    idea.focus();
  });
});
document.querySelector("#new-project").addEventListener("click", () => {
  idea.value = "";
  idea.focus();
  notify("Ready for a new project brief.");
});
document.querySelector("#create").addEventListener("click", async () => {
  const brief = idea.value.trim();
  if (!brief) {
    notify("Describe your project first.");
    idea.focus();
    return;
  }
  const button = document.querySelector("#create");
  button.disabled = true;
  button.textContent = "Creating plan…";
  try {
    const response = await fetch("/api/projects/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ brief })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not create plan.");
    projectName.textContent = data.project.name;
    plan.className = "plan";
    plan.innerHTML = "";
    const heading = document.createElement("strong");
    heading.textContent = data.project.name;
    const summary = document.createElement("p");
    summary.textContent = data.project.summary;
    const list = document.createElement("ol");
    data.project.steps.forEach(step => {
      const item = document.createElement("li");
      item.textContent = step;
      list.append(item);
    });
    plan.append(heading, summary, list);
    notify("Project plan created. File generation is the next step.");
  } catch (error) {
    notify(error.message || "Unable to reach the Medha API.");
  } finally {
    button.disabled = false;
    button.innerHTML = 'Generate project plan <span>→</span>';
  }
});