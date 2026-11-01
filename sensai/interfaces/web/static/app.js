"use strict";

const $ = (id) => document.getElementById(id);
const main = $("main");
const conversation = $("conversation");
const scroller = $("scroller");
const input = $("input");
const send = $("send");

let state = null;
let answering = false;
// The tokens of the answer being generated: Ollama only gives the exact count at the end.
let liveTokens = 0;
let toastTimer = null;

async function post(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : `The server answered ${response.status}`);
  }
  return response;
}

function toast(message) {
  // An open dialog covers the rest of the page.
  ($("settings").open ? $("settings") : document.body).append($("toast"));
  $("toast").textContent = message;
  $("toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("visible"), 5000);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Everything is escaped first: what the model writes can come from a leaked document, it must never run in the page.
function escapeHtml(text) {
  const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return text.replace(/[&<>"']/g, (character) => entities[character]);
}

function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*\n]*)\*/g, "$1<em>$2</em>")
    .replace(/\[(\d+(?:\s*[,;]\s*\d+)*)\]/g, '<span class="cite">$1</span>');
}

function markdown(text) {
  const html = [];
  let paragraph = [];
  let list = null;
  let fence = null;

  const closeParagraph = () => {
    if (paragraph.length) html.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (list) html.push(`</${list}>`);
    list = null;
  };
  const addItem = (tag, content) => {
    closeParagraph();
    if (list !== tag) {
      closeList();
      html.push(`<${tag}>`);
      list = tag;
    }
    html.push(`<li>${inline(content)}</li>`);
  };

  for (const line of text.split("\n")) {
    if (fence) {
      if (line.trim().startsWith("```")) {
        html.push(`<pre><code>${escapeHtml(fence.join("\n"))}</code></pre>`);
        fence = null;
      } else {
        fence.push(line);
      }
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (line.trim().startsWith("```")) {
      closeParagraph();
      closeList();
      fence = [];
    } else if (heading) {
      closeParagraph();
      closeList();
      html.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
    } else if (bullet) {
      addItem("ul", bullet[1]);
    } else if (numbered) {
      addItem("ol", numbered[1]);
    } else if (!line.trim()) {
      closeParagraph();
      closeList();
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  closeParagraph();
  closeList();
  // While the answer is streamed, a block of code is not closed yet.
  if (fence) html.push(`<pre><code>${escapeHtml(fence.join("\n"))}</code></pre>`);
  return html.join("");
}

function nearBottom() {
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
}

function scrollDown() {
  scroller.scrollTop = scroller.scrollHeight;
}

function addSources(parent, sources) {
  if (!sources.length) return;
  const list = element("div", "sources");
  for (const source of sources) list.append(element("span", "", `[${source.number}] ${source.citation}`));
  parent.append(list);
}

function addTurn(turn) {
  conversation.append(element("div", "user", turn.user));
  const assistant = element("div", "assistant");
  const activity = element("div", "activity");
  for (const tool of turn.tools) activity.append(element("div", "step", tool));
  const answer = element("div", "answer");
  answer.innerHTML = markdown(turn.assistant);
  assistant.append(activity, answer);
  addSources(assistant, turn.sources);
  conversation.append(assistant);
}

function showConversation() {
  conversation.replaceChildren();
  state.turns.forEach(addTurn);
  if (state.condensed) {
    conversation.prepend(element("div", "note", "Older messages are condensed into a summary the model still has."));
  }
  main.classList.toggle("empty", !state.turns.length);
  scrollDown();
}

function showUsage() {
  const { used, limit, condensed_from: condensedFrom } = state.context;
  const tokens = used + liveTokens;
  $("meter-fill").style.width = `${Math.min(100, (tokens / limit) * 100)}%`;
  $("usage-text").textContent = `${tokens.toLocaleString("en-US")} / ${limit.toLocaleString("en-US")} tokens`;
  $("usage").classList.toggle("full", tokens >= condensedFrom);
}

function showSessions() {
  const list = $("sessions");
  list.replaceChildren();
  if (!state.sessions.length) list.append(element("div", "none", "No saved conversation yet."));
  for (const session of state.sessions) {
    const button = element("button", session.id === state.session ? "active" : "", session.title);
    button.type = "button";
    button.title = `${session.title} (${session.updated})`;
    button.addEventListener("click", () => openSession(session.id));
    list.append(button);
  }
}

function showPersonas() {
  const select = $("persona");
  select.replaceChildren(new Option("Default", ""));
  for (const persona of state.personas) {
    const option = new Option(persona.name, persona.id);
    option.title = persona.role;
    select.append(option);
  }
  select.value = state.persona ?? "";
}

function showFeatures() {
  const list = $("features");
  list.replaceChildren();
  for (const feature of state.features) {
    const label = element("label", "feature");
    const text = element("div");
    text.append(element("strong", "", feature.name), element("span", "", feature.description));
    const toggle = element("input");
    toggle.type = "checkbox";
    toggle.checked = feature.on;
    toggle.addEventListener("change", () => setFeature(feature.name, toggle));
    label.append(text, toggle);
    list.append(label);
  }
}

function applyState(next) {
  state = next;
  $("model").textContent = state.model;
  showUsage();
  showSessions();
  showPersonas();
  showFeatures();
}

function setAnswering(value) {
  answering = value;
  document.body.classList.toggle("busy", value);
  $("persona").disabled = value;
  send.setAttribute("aria-label", value ? "Stop" : "Send");
  updateSend();
}

function updateSend() {
  send.disabled = !answering && !input.value.trim();
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

function askApproval(parent, event) {
  const card = element("div", "approval");
  const name = element("strong", "", `Allow ${event.tool}?`);
  const reason = element("p", "", event.reason);
  const actions = element("div", "actions");
  const choices = [["allow", "Allow", "allow"], ["always", "Always this session", ""], ["deny", "Deny", ""]];
  for (const [choice, text, className] of choices) {
    const button = element("button", className, text);
    button.type = "button";
    button.addEventListener("click", async () => {
      card.remove();
      await post("/api/approval", { choice }).catch((error) => toast(error.message));
    });
    actions.append(button);
  }
  card.append(name, reason, actions);
  parent.append(card);
}

function askQuestion(parent, event) {
  const card = element("form", "approval question");
  const reply = async (answer) => {
    card.replaceWith(element("div", "step", `[ask_user] ${event.question}`));
    await post("/api/question", { answer }).catch((error) => toast(error.message));
  };
  const options = element("div", "actions");
  for (const option of event.options) {
    const button = element("button", "", option);
    button.type = "button";
    button.addEventListener("click", () => reply(option));
    options.append(button);
  }
  const field = element("input");
  field.placeholder = event.options.length ? "Or in your own words" : "Your answer";
  field.setAttribute("aria-label", "Your answer");
  const skip = element("button", "", "Don't answer");
  skip.type = "button";
  skip.addEventListener("click", () => reply(null));
  const own = element("div", "actions");
  own.append(field, element("button", "allow", "Send"), skip);
  card.addEventListener("submit", (submitted) => {
    submitted.preventDefault();
    if (field.value.trim()) reply(field.value.trim());
  });
  card.append(element("strong", "", event.question), options, own);
  parent.append(card);
  field.focus();
}

// One answer on the page, filled in by the events of the server.
function startAnswer() {
  const assistant = element("div", "assistant");
  const activity = element("div", "activity");
  const answer = element("div", "answer");
  const thinking = element("div", "thinking");
  assistant.append(activity, answer, thinking);
  conversation.append(assistant);
  let text = "";
  let drawing = false;

  const stopThinking = () => thinking.remove();
  const draw = () => {
    drawing = false;
    const follow = nearBottom();
    answer.innerHTML = markdown(text);
    if (follow) scrollDown();
  };

  return (event) => {
    const follow = nearBottom();
    if (event.type === "token") {
      stopThinking();
      text += event.text;
      liveTokens += 1;
      showUsage();
      if (!drawing) {
        drawing = true;
        requestAnimationFrame(draw);
      }
      return;
    }
    if (event.type === "activity") {
      activity.append(element("div", event.warning ? "step warning" : "step", event.text));
    } else if (event.type === "approval") {
      askApproval(activity, event);
    } else if (event.type === "question") {
      askQuestion(activity, event);
    } else if (event.type === "done") {
      stopThinking();
      addSources(assistant, event.sources);
    } else if (event.type === "stopped") {
      stopThinking();
      const kept = "Stopped. What was written is kept: your next message can redirect it.";
      assistant.append(element("div", "note", event.kept ? kept : "Stopped."));
    } else if (event.type === "error") {
      stopThinking();
      assistant.append(element("div", "note error", event.text));
    } else if (event.type === "state") {
      stopThinking();
      activity.querySelector(".approval")?.remove();
      liveTokens = 0;
      applyState(event);
    }
    if (follow) scrollDown();
  };
}

async function readEvents(response, handle) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();
    for (const block of blocks) {
      if (block.startsWith("data: ")) handle(JSON.parse(block.slice(6)));
    }
  }
}

async function ask(message) {
  input.value = "";
  resizeInput();
  main.classList.remove("empty");
  conversation.append(element("div", "user", message));
  const handle = startAnswer();
  scrollDown();
  setAnswering(true);
  liveTokens = 0;
  try {
    await readEvents(await post("/api/chat", { message }), handle);
  } catch (error) {
    handle({ type: "error", text: error.message });
  }
  setAnswering(false);
  input.focus();
}

async function act(path, body) {
  try {
    const response = await post(path, body);
    applyState(await response.json());
    showConversation();
    return true;
  } catch (error) {
    toast(error.message);
    return false;
  }
}

async function openSession(id) {
  document.body.classList.remove("open");
  if (id !== state.session) await act(`/api/sessions/${encodeURIComponent(id)}`);
}

async function setFeature(name, toggle) {
  try {
    applyState(await (await post(`/api/features/${encodeURIComponent(name)}`, { on: toggle.checked })).json());
  } catch (error) {
    toggle.checked = !toggle.checked;
    toast(error.message);
  }
}

async function stop() {
  send.disabled = true;
  await post("/api/stop").catch((error) => toast(error.message));
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  if (answering) {
    stop();
  } else if (input.value.trim()) {
    ask(input.value.trim());
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!answering) $("composer").requestSubmit();
  }
});

input.addEventListener("input", () => {
  resizeInput();
  updateSend();
});

$("new-chat").addEventListener("click", async () => {
  document.body.classList.remove("open");
  if (await act("/api/new")) input.focus();
});

$("persona").addEventListener("change", async (event) => {
  const previous = state.persona ?? "";
  if (!(await act("/api/persona", { id: event.target.value || null }))) event.target.value = previous;
});

$("open-settings").addEventListener("click", () => $("settings").showModal());
$("toggle-sidebar").addEventListener("click", () => document.body.classList.toggle("open"));

fetch("/api/state")
  .then((response) => response.json())
  .then((loaded) => {
    applyState(loaded);
    showConversation();
  })
  .catch(() => toast("Cannot reach Sensai: is python -m sensai web still running?"));
