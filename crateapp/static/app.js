/* Crate — the interface.
 *
 * Two rules govern everything here:
 *  1. The sorting panel is ALWAYS in the page. Only its motion is conditional,
 *     because a panel that appears and disappears makes the layout jump every
 *     time a run starts or stops.
 *  2. Never render a number the API did not return. No rounding for effect, no
 *     re-derivation, no invented commentary about what the model is "thinking".
 *     The panel exists so a 0.81 vs 0.79 near-tie looks like the coin-flip it is.
 */

const api = {
  async get(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  },
};

const state = {
  crates: [],
  uncertain: 0,
  unsorted: 0,
  view: "__review__",
  folders: [],
  progress: null,
  lastRenderedTrack: null,
  error: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

/* ---------------------------------------------------------------- shell */

function shell() {
  const app = $("#app");
  app.innerHTML = "";

  const header = el("header", "app-header");
  header.append(el("h1", null, "Crate"));
  const stats = el("div", "header-stats");
  stats.id = "header-stats";
  header.append(stats);
  app.append(header);

  const banner = el("div", "error-banner hidden");
  banner.id = "error-banner";
  app.append(banner);

  app.append(toolbar());

  const body = el("div", "app-body");
  const side = el("nav", "sidebar");
  side.id = "sidebar";
  const content = el("main", "content");
  content.id = "content";
  body.append(side, content);
  app.append(body);

  app.append(sortingPanel());
}

function toolbar() {
  const bar = el("div", "toolbar");

  const form = el("form");
  const input = el("input");
  input.type = "text";
  input.placeholder = "Path to a music folder…";
  input.id = "folder-input";
  const add = el("button", null, "Add folder");
  add.type = "submit";
  form.append(input, add);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const folder = input.value.trim();
    if (!folder) return;
    add.disabled = true;
    try {
      const res = await api.post("/api/scan", { folder });
      if (!state.folders.includes(folder)) state.folders.push(folder);
      input.value = "";
      setStatus(`${res.added} added · ${res.unchanged} already known` +
                (res.missing ? ` · ${res.missing} missing` : ""));
      await refresh();
    } catch (err) {
      showError(err.message);
    } finally {
      add.disabled = false;
    }
  });
  bar.append(form);

  bar.append(el("div", "sep"));

  const start = el("button", "primary", "Start sorting");
  start.id = "start-btn";
  start.addEventListener("click", async () => {
    start.disabled = true;
    try {
      await api.post("/api/start", { folders: state.folders });
      setStatus("Sorting started");
      pollProgress();
    } catch (err) {
      showError(err.message);
      start.disabled = false;
    }
  });
  bar.append(start);

  const exportXml = el("button", null, "Export to Rekordbox");
  exportXml.addEventListener("click", () => doExport("rekordbox"));
  const exportFolders = el("button", null, "Export folders");
  exportFolders.addEventListener("click", () => doExport("folders"));
  bar.append(exportXml, exportFolders);

  const status = el("div", "toolbar-status");
  status.id = "toolbar-status";
  bar.append(status);

  const chips = el("div", "folder-chips");
  chips.id = "folder-chips";
  bar.append(chips);

  return bar;
}

async function doExport(kind) {
  const suggested = kind === "rekordbox"
    ? "~/Desktop/crate_export/rekordbox.xml"
    : "~/Desktop/crate_export";
  const dest = prompt(`Export ${kind} to:`, suggested);
  if (!dest) return;
  setStatus("Exporting…");
  try {
    const res = await api.post("/api/export", { dest, kind });
    const n = res.exported;
    setStatus(typeof n === "object"
      ? `Exported ${Object.values(n).reduce((a, b) => a + b, 0)} tracks into ${Object.keys(n).length} folders`
      : `Exported ${n} tracks`);
  } catch (err) {
    showError(err.message);
  }
}

function setStatus(msg) {
  const n = $("#toolbar-status");
  if (n) n.textContent = msg;
}

function showError(msg) {
  const b = $("#error-banner");
  if (!b) return;
  b.textContent = msg;
  b.classList.remove("hidden");
  setTimeout(() => b.classList.add("hidden"), 6000);
}

/* --------------------------------------------------------------- sidebar */

function renderSidebar() {
  const side = $("#sidebar");
  side.innerHTML = "";

  const reviewCount = state.uncertain + state.unsorted;
  side.append(sidebarItem("Needs review", reviewCount, "__review__", true));

  if (state.crates.length) side.append(el("div", "sidebar-divider"));

  if (!state.crates.length) {
    side.append(el("div", "sidebar-empty", "No crates yet"));
  } else {
    for (const c of state.crates) {
      side.append(sidebarItem(c.name, c.count, c.name, false));
    }
  }

  const total = state.crates.reduce((a, c) => a + c.count, 0);
  const stats = $("#header-stats");
  stats.innerHTML = "";
  const strong = el("strong", null, total.toLocaleString());
  stats.append(strong, document.createTextNode(
    ` track${total === 1 ? "" : "s"} · ${reviewCount} to review`));
}

function sidebarItem(label, count, key, isReview) {
  const b = el("button", "sidebar-item" + (isReview ? " review-item" : ""));
  if (state.view === key) b.classList.add("active");
  b.append(el("span", "label", label), el("span", "count tnum", String(count)));
  b.addEventListener("click", () => {
    state.view = key;
    renderSidebar();
    renderContent();
  });
  return b;
}

/* --------------------------------------------------------------- content */

async function renderContent() {
  const c = $("#content");
  c.innerHTML = "";
  if (state.view === "__review__") return renderReview(c);
  return renderCrate(c, state.view);
}

async function renderCrate(root, name) {
  root.append(el("h2", null, name));
  let data;
  try {
    data = await api.get("/api/crate/" + encodeURIComponent(name));
  } catch (err) {
    return showError(err.message);
  }
  if (!data.tracks.length) {
    root.append(el("p", "quiet-line", "Nothing in this crate yet."));
    return;
  }
  root.append(el("p", "content-sub",
    `${data.tracks.length} track${data.tracks.length === 1 ? "" : "s"}`));
  root.append(trackTable(data.tracks));
}

function trackTable(tracks) {
  const t = el("table", "track-table");
  const thead = el("thead");
  const hr = el("tr");
  ["Track", "BPM", "Key"].forEach((h) => hr.append(el("th", null, h)));
  thead.append(hr);
  const tb = el("tbody");
  for (const tr of tracks) {
    const row = el("tr");
    const nameCell = el("td", "col-name");
    nameCell.append(document.createTextNode(tr.filename));
    if (tr.band === "uncertain") {
      const dot = el("span", "uncertain-dot");
      dot.title = "Filed, but it was a close call";
      nameCell.append(dot);
    }
    row.append(nameCell);
    row.append(el("td", "col-bpm", tr.bpm == null ? "—" : tr.bpm.toFixed(1)));
    row.append(el("td", "col-key", tr.camelot || "—"));
    tb.append(row);
  }
  t.append(thead, tb);
  return t;
}

/* ---------------------------------------------------------------- review */

/* Three different things live in this queue and must not look alike:
 *   uncertain — filed, but a close call. Already a member of its crate.
 *   unknown   — matched no crate: a genre never played. A real discovery.
 *   failed    — could not be analysed. A problem, not a discovery.
 * Presenting a broken file as an exciting new genre would be a lie. */
function classify(track, kind) {
  if (kind === "uncertain") {
    return { cls: "uncertain", badge: "Close call",
             reason: "Filed, but two crates were nearly tied — confirm or change it." };
  }
  if (track.error) {
    return { cls: "failed", badge: "Couldn't read",
             reason: track.error };
  }
  return { cls: "unknown", badge: "New sound",
           reason: "Matches none of your crates. If this is a genre you've started playing, give it one." };
}

async function renderReview(root) {
  root.append(el("h2", null, "Needs review"));
  let data;
  try {
    data = await api.get("/api/review");
  } catch (err) {
    return showError(err.message);
  }

  const rows = [
    ...data.uncertain.map((t) => ({ t, kind: "uncertain" })),
    ...data.unsorted.map((t) => ({ t, kind: "unsorted" })),
  ];
  if (!rows.length) {
    root.append(el("p", "quiet-line", "Nothing to review."));
    return;
  }

  const groups = { uncertain: [], unknown: [], failed: [] };
  for (const r of rows) groups[classify(r.t, r.kind).cls].push(r);

  const sections = [
    ["uncertain", "Close calls", "Already filed. Confirm the crate, or move it."],
    ["unknown", "New sounds", "Matched none of your crates — these may be genres worth adding."],
    ["failed", "Couldn't read", "These files could not be analysed. Not a genre problem."],
  ];

  for (const [key, title, note] of sections) {
    if (!groups[key].length) continue;
    const sec = el("section", "review-section");
    const h = el("h3", null, title);
    h.append(el("span", "badge badge-" + key, String(groups[key].length)));
    sec.append(h);
    sec.append(el("p", "review-section-note", note));
    for (const { t, kind } of groups[key]) sec.append(reviewRow(t, kind));
    root.append(sec);
  }
}

function reviewRow(track, kind) {
  const info = classify(track, kind);
  const row = el("div", "review-row");

  row.append(el("div", "rr-name", track.filename));
  const meta = [track.bpm == null ? null : track.bpm.toFixed(1) + " bpm", track.camelot]
    .filter(Boolean).join("  ");
  row.append(el("div", "rr-meta tnum", meta || "—"));
  row.append(el("div", "rr-reason reason-" + info.cls, info.reason));

  const actions = el("div", "rr-actions");

  const select = el("select");
  for (const c of state.crates) select.append(new Option(c.name, c.name));
  select.append(new Option("New crate…", "__new__"));

  const newName = el("input");
  newName.type = "text";
  newName.placeholder = "Name the crate";
  newName.classList.add("hidden");
  select.addEventListener("change", () => {
    newName.classList.toggle("hidden", select.value !== "__new__");
    if (select.value === "__new__") newName.focus();
  });

  /* A move always counts as the model getting it wrong — no prompt.
   * An also-add asks, because it can mean either "right, and it belongs here
   * too" or "wrong, but leave it". Counting every also-add as a success would
   * inflate the measured accuracy exactly when the DJ is being lenient. */
  const toggle = el("div", "also-add-toggle");
  const seg = el("div", "seg");
  const wasRight = el("button", "on", "Pick was right");
  const wasWrong = el("button", null, "Pick was wrong");
  wasRight.addEventListener("click", () => {
    wasRight.classList.add("on"); wasWrong.classList.remove("on");
  });
  wasWrong.addEventListener("click", () => {
    wasWrong.classList.add("on"); wasRight.classList.remove("on");
  });
  seg.append(wasRight, wasWrong);
  toggle.append(seg);

  const move = el("button", null, "Move");
  const add = el("button", null, "Also add");

  const chosen = () => select.value === "__new__" ? newName.value.trim() : select.value;

  move.addEventListener("click", () => submit("move"));
  add.addEventListener("mouseenter", () => toggle.classList.add("visible"));
  add.addEventListener("click", () => {
    if (!toggle.classList.contains("visible")) {
      toggle.classList.add("visible");
      return;                       // first click reveals the question
    }
    submit("add", !wasRight.classList.contains("on"));
  });

  async function submit(mode, wasError) {
    const to = chosen();
    if (!to) { newName.focus(); return; }
    move.disabled = add.disabled = true;
    try {
      await api.post("/api/correct", {
        track_id: track.id, to_crate: to, mode,
        was_error: mode === "move" ? true : !!wasError,
      });
      row.style.opacity = "0";
      setTimeout(async () => { await refresh(); }, 180);
    } catch (err) {
      showError(err.message);
      move.disabled = add.disabled = false;
    }
  }

  actions.append(select, newName, move, add, toggle);
  row.append(actions);

  const hint = el("div", "rr-hint",
    "A move is recorded as the model getting it wrong. An also-add asks you.");
  row.append(hint);
  return row;
}

/* --------------------------------------------------- the sorting panel */

function sortingPanel() {
  const p = el("section", "sorting-panel at-rest");
  p.id = "sorting-panel";

  const head = el("div", "sp-head");
  head.append(el("span", "sp-status-dot"));
  const title = el("div", "sp-title");
  title.id = "sp-title";
  title.textContent = "Not sorting";
  head.append(title);
  const meta = el("div", "sp-meta tnum");
  meta.id = "sp-meta";
  head.append(meta);
  p.append(head);

  const body = el("div", "sp-body");
  body.id = "sp-body";
  p.append(body);

  return p;
}

function renderProgress() {
  const p = $("#sorting-panel");
  const body = $("#sp-body");
  const prog = state.progress;
  if (!p || !body) return;

  const running = !!(prog && prog.running);
  p.classList.toggle("running", running);
  p.classList.toggle("at-rest", !running);

  $("#sp-title").textContent = running ? "Sorting" : "Last sorted";
  $("#sp-meta").textContent = prog && prog.total
    ? `${prog.done} of ${prog.total}` + (prog.errors ? ` · ${prog.errors} failed` : "")
    : "";

  const cur = prog && prog.current;
  if (!cur) {
    body.innerHTML = "";
    body.append(el("div", "sp-empty",
      "Add a folder above and press Start sorting."));
    state.lastRenderedTrack = null;
    return;
  }

  // Same track already drawn — leave it alone rather than re-animating.
  if (state.lastRenderedTrack === cur.filename && body.querySelector(".sp-live")) return;

  const prev = body.querySelector(".sp-live");
  const live = buildLive(cur);
  if (prev && running) {
    prev.classList.add("dissolving");
    setTimeout(() => prev.remove(), 200);
    body.append(live);
  } else {
    body.innerHTML = "";
    body.append(live);
  }
  state.lastRenderedTrack = cur.filename;
}

function buildLive(cur) {
  const live = el("div", "sp-live");

  const idc = el("div", "sp-curve-wrap");
  idc.append(el("div", "sp-filename", cur.filename));
  const bits = [
    cur.bpm == null ? null : cur.bpm.toFixed(2) + " bpm",
    cur.camelot,
  ].filter(Boolean).join("   ");
  idc.append(el("div", "sp-meta tnum", bits));
  idc.append(curveSvg(cur.energy || []));
  live.append(idc);

  live.append(bars(cur));
  return live;
}

function curveSvg(energy) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "sp-curve-svg");
  svg.setAttribute("preserveAspectRatio", "none");
  const W = 300, H = 76;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!energy.length) return svg;

  const max = Math.max(...energy, 0.0001);
  const pt = (i, v) => [
    (i / Math.max(energy.length - 1, 1)) * W,
    H - (v / max) * (H - 6) - 3,
  ];
  const d = energy.map((v, i) => {
    const [x, y] = pt(i, v);
    return `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");

  const fill = document.createElementNS(NS, "path");
  fill.setAttribute("class", "sp-curve-fill");
  fill.setAttribute("d", `${d} L${W},${H} L0,${H} Z`);
  svg.append(fill);

  const line = document.createElementNS(NS, "path");
  line.setAttribute("class", "sp-curve-line");
  line.setAttribute("d", d);
  svg.append(line);
  return svg;
}

function bars(cur) {
  const wrap = el("div", "sp-bars");
  const sims = cur.similarities || [];
  const SHOWN = 4;

  sims.forEach((s, i) => {
    const row = el("div", "sp-bar-row");
    if (i >= SHOWN) row.classList.add("hidden", "sp-extra");
    row.append(el("div", "sp-bar-name", s.crate));
    const track = el("div", "sp-bar-track");
    const fill = el("div", "sp-bar-fill" + (i === 0 ? " is-winner" : ""));
    fill.style.width = "0%";
    track.append(fill);
    row.append(track);
    // The value is printed exactly as the classifier returned it.
    row.append(el("div", "sp-bar-value tnum", s.similarity.toFixed(3)));
    wrap.append(row);
    requestAnimationFrame(() => {
      fill.style.width = Math.max(0, Math.min(1, s.similarity)) * 100 + "%";
    });
  });

  if (sims.length > SHOWN) {
    const more = el("button", "sp-more-toggle", `Show all ${sims.length}`);
    let open = false;
    more.addEventListener("click", () => {
      open = !open;
      wrap.querySelectorAll(".sp-extra").forEach((r) => r.classList.toggle("hidden", !open));
      more.textContent = open ? "Show fewer" : `Show all ${sims.length}`;
    });
    wrap.append(more);
  }

  const verdict = el("div", "sp-verdict band-" + (cur.band || "unknown"));
  const label = cur.crate ? cur.crate : "No crate matched";
  verdict.append(el("span", "tag", label));
  verdict.append(document.createTextNode(
    `  ${cur.band}` + (cur.margin != null ? `  ·  margin ${cur.margin.toFixed(3)}` : "")));
  wrap.append(verdict);

  return wrap;
}

/* ------------------------------------------------------------- polling */

let pollTimer = null;
let lastContentRefresh = 0;

async function pollProgress() {
  clearTimeout(pollTimer);
  try {
    state.progress = await api.get("/api/progress");
    renderProgress();
    const running = state.progress && state.progress.running;
    const btn = $("#start-btn");
    if (btn) btn.disabled = !!running;
    if (running) {
      // Crate counts move as tracks are filed.
      const s = await api.get("/api/state");
      Object.assign(state, s);
      renderSidebar();
      // The list underneath goes stale mid-run too — a track can sit in the
      // review queue for the moment between being analysed and being filed,
      // which reads as the app contradicting itself. Refresh it, but not on
      // every 500ms tick, or the list flickers under the cursor.
      const now = Date.now();
      if (now - lastContentRefresh > 2500) {
        lastContentRefresh = now;
        await renderContent();
      }
    }
    pollTimer = setTimeout(pollProgress, running ? 500 : 3000);
  } catch (err) {
    pollTimer = setTimeout(pollProgress, 5000);
  }
}

async function refresh() {
  try {
    const s = await api.get("/api/state");
    Object.assign(state, s);
  } catch (err) {
    showError(err.message);
    return;
  }
  renderSidebar();
  await renderContent();
}

async function boot() {
  shell();
  await refresh();
  pollProgress();
}

boot();
