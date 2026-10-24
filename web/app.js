const $ = (id) => document.getElementById(id);

const ICONS = {
  pdf: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M7 3.5h7.2L19 8.2V20a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 6 20V5A1.5 1.5 0 0 1 7.5 3.5H7z"/><path d="M14 3.8V8.2H18.4"/><path d="M8.5 12.5h7M8.5 16h5" stroke-linecap="round"/></svg>',
  image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="4" y="5" width="16" height="14" rx="2"/><path d="M4 15.5l4.2-3.6 3.2 2.6 2.4-2 6.2 4.2"/><circle cx="9" cy="9.2" r="1.1" fill="currentColor" stroke="none"/></svg>',
  sheet: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="3.5" width="14" height="17" rx="1.6"/><path d="M5 9h14M5 13.5h14M5 18h14M10 3.5v17M15 9v12"/></svg>',
  video: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 7.5v9l8-4.5-8-4.5z"/></svg>',
  audio: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M5 10v4M9 7v10M13 4.5v15M17 8.5v7M21 11v2"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M7 3.5h7.2L19 8.2V20a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 6 20V5A1.5 1.5 0 0 1 7.5 3.5z"/><path d="M14 3.8V8.2H18.4"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 7.2v9.6l8.2-4.8L9 7.2z"/></svg>',
};

const queryEl = $("query");
const hintEl = $("hint");
let showingResult = false;
let pollTimer = 0;
let hfToken = false;
let openKey = "";

function autosize(el) {
  el.style.height = "auto";
  el.style.height = `${el.scrollHeight + 6}px`;
}

function stripQuotes(value) {
  let text = value.trim();
  if (text.startsWith("„") || text.startsWith('"') || text.startsWith("'")) text = text.slice(1).trimStart();
  if (text.endsWith("“") || text.endsWith('"') || text.endsWith("'")) text = text.slice(0, -1).trimEnd();
  return text;
}

function withQuotes(value) {
  const text = stripQuotes(value);
  return text ? `„${text}“` : "";
}

function setHint(text, kind) {
  hintEl.hidden = !text;
  hintEl.textContent = text || "";
  hintEl.classList.toggle("is-error", kind === "error");
  hintEl.classList.toggle("is-warn", kind === "warn");
}

function errText(data, fallback) {
  if (data && typeof data.detail === "string") return data.detail;
  return fallback;
}

function icon(name) {
  const span = document.createElement("span");
  span.innerHTML = ICONS[name] || ICONS.file;
  return span;
}

function meter(percent) {
  const wrap = document.createElement("div");
  wrap.className = "meter";
  wrap.style.setProperty("--p", String(Math.max(0, Math.min(100, percent || 0))));
  wrap.innerHTML = '<div class="track"><div class="fill"></div><div class="knob"></div></div>';
  const label = document.createElement("div");
  label.className = "pct";
  label.textContent = `${percent} %`;
  wrap.appendChild(label);
  return wrap;
}

function bindOpen(button, item) {
  button.addEventListener("click", () => openDetail(item, button));
}

function evidenceCard(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ev";
  button.dataset.source = item.source;
  if (item.modality === "video") {
    const thumb = document.createElement("div");
    thumb.className = "thumb";
    if (item.thumb_url) {
      const img = document.createElement("img");
      img.alt = "";
      img.src = item.thumb_url;
      thumb.appendChild(img);
    }
    const play = document.createElement("span");
    play.className = "play";
    play.innerHTML = ICONS.play;
    thumb.appendChild(play);
    if (item.duration_label) {
      const dur = document.createElement("span");
      dur.className = "dur";
      dur.textContent = item.duration_label;
      thumb.appendChild(dur);
    }
    button.appendChild(thumb);
  } else {
    const box = document.createElement("span");
    box.className = "icon-box";
    box.appendChild(icon(item.icon));
    button.appendChild(box);
  }
  const copy = document.createElement("span");
  copy.className = "ev-copy";
  const title = document.createElement("span");
  title.className = "ev-title";
  title.textContent = item.title || item.source;
  const sub = document.createElement("span");
  sub.className = "ev-sub";
  sub.textContent = item.subtitle || "";
  copy.append(title, sub);
  const pct = document.createElement("span");
  pct.className = "ev-pct";
  pct.textContent = `${item.percent} %`;
  button.append(copy, pct);
  bindOpen(button, item);
  return button;
}

function fileCard(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "file-card";
  button.dataset.source = item.source;
  const head = document.createElement("div");
  head.className = "file-head";
  const ico = document.createElement("span");
  ico.className = "file-ico";
  ico.appendChild(icon(item.icon || (item.modality === "video" ? "video" : "pdf")));
  const name = document.createElement("span");
  name.className = "fname";
  name.textContent = item.title || item.source;
  head.append(ico, name);
  button.append(head, meter(item.percent));
  bindOpen(button, item);
  return button;
}

function render(data, setQuery) {
  if (setQuery && data.query) {
    queryEl.value = data.query;
    autosize(queryEl);
  }
  const answer = $("answer");
  const text = $("answer-text");
  const chips = $("chips");
  chips.innerHTML = "";
  if (data.answer) {
    answer.hidden = false;
    text.textContent = data.answer;
    for (const chip of data.chips || []) {
      const span = document.createElement("span");
      span.className = "chip";
      span.textContent = chip;
      chips.appendChild(span);
    }
  } else {
    answer.hidden = true;
  }
  const evidence = $("evidence");
  evidence.innerHTML = "";
  for (const item of data.evidence || []) evidence.appendChild(evidenceCard(item));
  const files = data.files || [];
  $("files-wrap").hidden = files.length === 0;
  const rail = $("files");
  rail.innerHTML = "";
  for (const item of files) rail.appendChild(fileCard(item));
  closeDetail();
  showingResult = Boolean(data.answer || (data.evidence || []).length || files.length);
  if (showingResult && !hintEl.classList.contains("is-warn") && !hintEl.classList.contains("is-error")) {
    setHint("");
  }
}

function closeDetail() {
  openKey = "";
  $("detail").hidden = true;
  $("detail").innerHTML = "";
  document.querySelectorAll(".is-on").forEach((el) => el.classList.remove("is-on"));
}

function openDetail(item, button) {
  const key = `${item.source}:${item.start ?? ""}:${item.page ?? ""}`;
  if (openKey === key) {
    closeDetail();
    return;
  }
  closeDetail();
  openKey = key;
  button.classList.add("is-on");
  const detail = $("detail");
  detail.hidden = false;
  const head = document.createElement("div");
  head.className = "detail-head";
  const copy = document.createElement("div");
  copy.className = "ev-copy";
  const title = document.createElement("div");
  title.className = "ev-title";
  title.textContent = item.title || item.source;
  const sub = document.createElement("div");
  sub.className = "ev-sub";
  sub.textContent = item.subtitle || "";
  copy.append(title, sub);
  const pct = document.createElement("div");
  pct.className = "ev-pct";
  pct.textContent = `${item.percent} %`;
  head.append(copy, pct);
  const passage = document.createElement("p");
  passage.className = "passage";
  passage.textContent = item.text || "Kein Textauszug für diese Datei.";
  detail.append(head, passage);
  if (item.media_url && (item.modality === "video" || item.modality === "audio")) {
    const player = document.createElement("div");
    player.className = "player";
    const media = document.createElement(item.modality === "video" ? "video" : "audio");
    media.controls = true;
    media.preload = "metadata";
    media.src = item.start ? `${item.media_url}#t=${item.start}` : item.media_url;
    player.appendChild(media);
    detail.appendChild(player);
  } else if (item.media_url) {
    const player = document.createElement("div");
    player.className = "player";
    const link = document.createElement("a");
    link.href = item.media_url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "PDF öffnen";
    player.appendChild(link);
    detail.appendChild(player);
  }
  detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderLibrary(files) {
  const list = $("library");
  list.innerHTML = "";
  $("library-empty").hidden = files.length > 0;
  for (const file of files) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = file.name;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = file.modality;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "x";
    remove.setAttribute("aria-label", `${file.name} entfernen`);
    remove.textContent = "×";
    remove.addEventListener("click", async () => {
      await fetch(`/api/files/${encodeURIComponent(file.name)}`, { method: "DELETE" });
      refresh();
    });
    li.append(name, tag, remove);
    list.appendChild(li);
  }
  $("build").disabled = files.length === 0;
}

function applyStatus(status) {
  renderLibrary(status.files || []);
  const running = status.state === "running";
  $("build").disabled = running || (status.files || []).length === 0;
  $("build-box").hidden = !(running || status.message);
  $("build-fill").style.width = `${Math.round((status.progress || 0) * 100)}%`;
  $("build-msg").textContent = status.message || "";
  $("build-log").textContent = (status.log || []).join("\n");
  if (status.settings) applySettings(status.settings);
  if (!showingResult) {
    if (status.state === "error" && status.message) setHint(status.message, "error");
    else if (status.stale) setHint("Quellen haben sich geändert. Index neu bauen.", "warn");
    else if (status.ready) setHint("Eingabetaste sucht in den indexierten Quellen.");
    else setHint("Quellen öffnen, Dateien ablegen und den Index bauen.");
  } else if (status.stale) {
    setHint("Quellen haben sich geändert. Index neu bauen.", "warn");
  }
  hfToken = Boolean(status.hf_token);
  syncBackend();
  if (running) schedulePoll(400);
}

function applySettings(settings) {
  if (document.activeElement && $("index-form").contains(document.activeElement)) return;
  if (settings.backend) $("backend").value = settings.backend;
  if (settings.variant) $("variant").value = settings.variant;
  if (settings.dim) $("dim").value = String(settings.dim);
  if (settings.whisper_id) $("whisper").value = settings.whisper_id;
  $("captions").checked = settings.use_captions !== false;
  for (const [id, key] of [["chunk", "chunk_words"], ["window", "window_s"], ["frame", "frame_every"], ["maxframes", "max_frames"]]) {
    if (settings[key] != null) {
      $(id).value = settings[key];
      $(`${id}-out`).value = settings[key];
    }
  }
}

function schedulePoll(ms) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(refresh, ms);
}

async function refresh() {
  try {
    const res = await fetch("/api/status");
    applyStatus(await res.json());
  } catch {
    setHint("Keine Verbindung zum Server.", "error");
  }
}

function syncBackend() {
  const hf = $("backend").value.startsWith("Transformers");
  $("variant").disabled = hf;
  $("hf-warn").hidden = !(hf && !hfToken);
}

function collectSettings() {
  return {
    backend: $("backend").value,
    variant: $("variant").value,
    dim: Number($("dim").value),
    whisper_id: $("whisper").value,
    use_captions: $("captions").checked,
    chunk_words: Number($("chunk").value),
    window_s: Number($("window").value),
    frame_every: Number($("frame").value),
    max_frames: Number($("maxframes").value),
  };
}

async function search() {
  const raw = queryEl.value.trim();
  if (!raw) return;
  setHint("Suche …");
  const modalities = [...document.querySelectorAll(".mod:checked")].map((el) => el.value);
  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: stripQuotes(raw), modalities }),
    });
    const data = await res.json();
    if (!res.ok) {
      setHint(errText(data, "Suche fehlgeschlagen."), "error");
      return;
    }
    queryEl.value = withQuotes(raw);
    autosize(queryEl);
    render(data, false);
    if (data.stale) setHint("Quellen haben sich geändert. Index neu bauen.", "warn");
  } catch {
    setHint("Keine Verbindung zum Server.", "error");
  }
}

function openDrawer() {
  $("drawer").hidden = false;
  $("scrim").hidden = false;
}

function closeDrawer() {
  $("drawer").hidden = true;
  $("scrim").hidden = true;
}

async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const body = new FormData();
  for (const file of fileList) body.append("files", file, file.name);
  const res = await fetch("/api/files", { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) setHint(errText(data, "Hochladen fehlgeschlagen."), "error");
  refresh();
}

async function loadBeispiel() {
  const res = await fetch("/api/beispiel");
  const data = await res.json();
  render(data, true);
  closeDrawer();
}

queryEl.addEventListener("input", () => autosize(queryEl));
queryEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    search();
  }
});

$("open-sources").addEventListener("click", openDrawer);
$("close-sources").addEventListener("click", closeDrawer);
$("scrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeDrawer();
    closeDetail();
  }
});

$("backend").addEventListener("change", syncBackend);
for (const id of ["chunk", "window", "frame", "maxframes"]) {
  $(id).addEventListener("input", () => {
    $(`${id}-out`).value = $(id).value;
  });
}

$("index-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("build").disabled = true;
  setHint("Index wird gebaut …");
  const res = await fetch("/api/index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectSettings()),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    setHint(errText(data, "Index konnte nicht gestartet werden."), "error");
    $("build").disabled = false;
    return;
  }
  closeDrawer();
  refresh();
});

$("clear").addEventListener("click", async () => {
  await fetch("/api/clear", { method: "POST" });
  showingResult = false;
  render({ answer: "", evidence: [], files: [] }, false);
  refresh();
});

$("beispiel").addEventListener("click", loadBeispiel);

const drop = $("drop");
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("is-over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("is-over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("is-over");
  uploadFiles(event.dataTransfer.files);
});
$("file-input").addEventListener("change", (event) => {
  uploadFiles(event.target.files);
  event.target.value = "";
});

autosize(queryEl);
refresh();
if (window.__BEISPIEL) render(window.__BEISPIEL, true);
else if (new URLSearchParams(location.search).has("beispiel")) loadBeispiel();
