/* Crate — the interface.
 *
 * Three rules:
 *  1. Never render a number the API did not return. No rounding for effect, no
 *     invented commentary. The panel exists so a 0.81 vs 0.79 near-tie looks
 *     like the coin-flip it is.
 *  2. Never clear the view before the data arrives. Clearing then awaiting is
 *     what made the screen go black during a fetch, and stay black on error.
 *  3. Never rebuild a list the user might be clicking. Nodes are updated in
 *     place; a rebuild only happens when the set of things actually changes.
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
  crates: [], uncertain: 0, unsorted: 0,
  view: "__review__", folders: [], progress: null,
  panelTrack: null, shownKey: null, showAllBars: false,
  // set builder — kept across view switches so a built set survives a
  // detour into a crate and back
  sets: { crate: "", seed: null, length: 8, mode: "balanced", arc: "steady",
          pool: null, poolStamp: null, result: null, query: "" },
};

/* A DJ reads 3:02, never 182.7s. */
function clock(sec) {
  if (sec == null) return "—";
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

/* ------------------------------------------------------------------ shell */

function shell() {
  const app = $("#app");
  app.innerHTML = "";

  const head = el("header", "app-header");
  head.append(el("h1", null, "Crate"));
  const stats = el("div", "header-stats"); stats.id = "stats";
  head.append(stats);
  app.append(head);

  const err = el("div", "error-banner hidden"); err.id = "err";
  app.append(err);

  app.append(toolbar());

  const body = el("div", "app-body");
  const side = el("nav", "sidebar"); side.id = "sidebar";
  const content = el("main", "content"); content.id = "content";
  body.append(side, content);
  app.append(body);

  app.append(panel());
}

function toolbar() {
  const bar = el("div", "toolbar");

  const form = el("form");
  const input = el("input"); input.type = "text";
  input.placeholder = "Path to a music folder…"; input.id = "folder";
  const add = el("button", null, "Add"); add.type = "submit";
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
      status(`${res.added} added · ${res.unchanged} known${res.missing ? ` · ${res.missing} missing` : ""}`);
      await refresh();
    } catch (e2) { fail(e2.message); } finally { add.disabled = false; }
  });
  bar.append(form, el("div", "sep"));

  const start = el("button", "primary", "Start sorting"); start.id = "start";
  start.addEventListener("click", async () => {
    start.disabled = true;
    try {
      await api.post("/api/start", { folders: state.folders });
      status("Sorting started"); poll();
    } catch (e) { fail(e.message); start.disabled = false; }
  });
  bar.append(start);

  const x1 = el("button", null, "Export XML");
  x1.addEventListener("click", () => doExport("rekordbox"));
  const x2 = el("button", null, "Export folders");
  x2.addEventListener("click", () => doExport("folders"));
  bar.append(x1, x2);

  const st = el("div", "toolbar-status"); st.id = "status";
  bar.append(st);
  return bar;
}

async function doExport(kind) {
  const dest = prompt(`Export ${kind} to:`,
    kind === "rekordbox" ? "~/Desktop/crate_export/rekordbox.xml" : "~/Desktop/crate_export");
  if (!dest) return;
  status("Exporting…");
  try {
    const r = await api.post("/api/export", { dest, kind });
    const n = r.exported;
    status(typeof n === "object"
      ? `Exported ${Object.values(n).reduce((a, b) => a + b, 0)} tracks · ${Object.keys(n).length} folders`
      : `Exported ${n} tracks`);
  } catch (e) { fail(e.message); }
}

const status = (m) => { const n = $("#status"); if (n) n.textContent = m; };
function fail(m) {
  const b = $("#err"); if (!b) return;
  b.textContent = m; b.classList.remove("hidden");
  clearTimeout(fail._t);
  fail._t = setTimeout(() => b.classList.add("hidden"), 6000);
}

/* ---------------------------------------------------------------- sidebar */

function renderSidebar() {
  const side = $("#sidebar");
  const review = state.uncertain + state.unsorted;
  const sig = ["__review__", "__sets__",
               ...state.crates.map((c) => c.name)].join("\u0000");

  // In place when the set is unchanged. Rebuilding twice a second during a run
  // threw buttons away mid-click, so clicking a crate while sorting did nothing.
  if (side.dataset.sig === sig) {
    const counts = { __review__: review };
    for (const c of state.crates) counts[c.name] = c.count;
    for (const b of side.querySelectorAll(".sidebar-item")) {
      const n = counts[b.dataset.key];
      if (n !== undefined) b.querySelector(".count").textContent = String(n);
      b.classList.toggle("active", state.view === b.dataset.key);
    }
    return headerStats(review);
  }

  side.dataset.sig = sig;
  side.innerHTML = "";
  side.append(item("Needs review", review, "__review__", true));
  side.append(item("Set builder", "", "__sets__", false, "sets-item"));
  side.append(el("div", "sidebar-heading", "Crates"));
  if (!state.crates.length) side.append(el("div", "sidebar-empty", "None yet"));
  for (const c of state.crates) side.append(item(c.name, c.count, c.name, false));
  headerStats(review);
}

function headerStats(review) {
  const total = state.crates.reduce((a, c) => a + c.count, 0);
  const n = $("#stats");
  n.innerHTML = "";
  n.append(el("strong", null, total.toLocaleString()),
    document.createTextNode(` sorted · ${review} to review`));
}

function item(label, count, key, isReview, extra) {
  const b = el("button", "sidebar-item" + (isReview ? " review-item" : "")
                       + (extra ? " " + extra : ""));
  b.dataset.key = key;
  if (state.view === key) b.classList.add("active");
  b.append(el("span", "label", label), el("span", "count", String(count)));
  b.addEventListener("click", () => {
    if (state.view === key) return;
    state.view = key;
    renderSidebar();
    renderContent();
  });
  return b;
}

/* Click selects (and shows in the panel), double-click plays. Both gestures
 * land on the same row, so the single-click handler must be harmless to run
 * twice - selecting an already-selected track is a no-op. */
function makePlayable(node, track) {
  if (track.id == null) return node;
  node.dataset.trackId = String(track.id);
  node.classList.add("clickable");
  node.title = `${track.filename}\n\nClick to inspect \u00b7 double-click to play`;
  node.addEventListener("click", () => selectTrack(track.id));
  node.addEventListener("dblclick", (e) => {
    e.preventDefault();
    selectTrack(track.id);
    player.play(track.id);
  });
  return node;
}

/* ---------------------------------------------------------------- content */

let contentToken = 0;

async function renderContent() {
  const token = ++contentToken;
  preview.stop();        // never leave a crossfade running behind a view change
  const root = $("#content");
  let node;
  try {
    // Build the whole view BEFORE touching the DOM. Clearing first and then
    // awaiting is what made the screen go black mid-fetch and stay black if
    // the fetch failed.
    node = state.view === "__review__" ? await buildReview()
         : state.view === "__sets__"    ? await buildSets()
         : await buildCrate(state.view);
  } catch (e) {
    fail(e.message);
    return;                       // leave what is on screen; do not blank it
  }
  if (token !== contentToken) return;   // a newer click already won
  root.replaceChildren(node);
  // The rows are new nodes, so re-mark whichever one is on the deck.
  player.paint();
}

async function buildCrate(name) {
  const data = await api.get("/api/crate/" + encodeURIComponent(name));
  const frag = document.createDocumentFragment();

  const head = el("div", "content-head");
  head.append(el("h2", null, name),
              el("span", "sub", `${data.tracks.length} track${data.tracks.length === 1 ? "" : "s"}`));
  frag.append(head);

  if (!data.tracks.length) {
    frag.append(el("p", "quiet-line", "Nothing here yet."));
    return frag;
  }

  const list = el("div", "tracks");
  const hr = el("div", "track-head");
  hr.append(el("div", null, "Track"), el("div", null, "BPM"), el("div", null, "Key"));
  list.append(hr);
  for (const t of data.tracks) list.append(trackRow(t));
  frag.append(list);
  return frag;
}

function trackRow(t) {
  const r = el("div", "track-row" + (t.band === "uncertain" ? " is-uncertain" : ""));
  r.append(el("div", "t-name", t.filename),
           el("div", "t-bpm", t.bpm == null ? "—" : t.bpm.toFixed(1)),
           el("div", "t-key", t.camelot || "—"));
  return makePlayable(r, t);
}

/* ------------------------------------------------------------------ review */

/* Three different things live here and must not look alike: a close call
 * (already filed), a genre never played (a discovery), and a file that could
 * not be read (a problem). Showing a decode failure as a new genre is a lie. */
function kindOf(track, group) {
  if (group === "uncertain") {
    // The group header already explains what a close call is; repeating that
    // sentence on every row is noise. Show where it landed instead - that
    // differs per row and is the thing being confirmed.
    return { k: "uncertain", label: "Close calls",
             why: track.crate ? `filed in ${track.crate}` : "filed provisionally" };
  }
  if (track.error) {
    return { k: "failed", label: "Couldn’t read", why: track.error };
  }
  return { k: "unknown", label: "New sounds",
           why: "Matches none of your crates." };
}

async function buildReview() {
  const data = await api.get("/api/review");
  const frag = document.createDocumentFragment();
  const rows = [
    ...data.uncertain.map((t) => ({ t, g: "uncertain" })),
    ...data.unsorted.map((t) => ({ t, g: "unsorted" })),
  ];

  const head = el("div", "content-head");
  head.append(el("h2", null, "Needs review"),
              el("span", "sub", rows.length ? `${rows.length} waiting` : "all clear"));
  frag.append(head);

  if (!rows.length) {
    frag.append(el("p", "quiet-line", "Nothing to review."));
    return frag;
  }

  const groups = { uncertain: [], unknown: [], failed: [] };
  for (const r of rows) groups[kindOf(r.t, r.g).k].push(r);

  const notes = {
    uncertain: "Already filed. Confirm the crate, or move it.",
    unknown: "Matched none of your crates — these may be worth a new one.",
    failed: "These files could not be analysed. Not a genre problem.",
  };
  const titles = { uncertain: "Close calls", unknown: "New sounds", failed: "Couldn’t read" };

  for (const k of ["uncertain", "unknown", "failed"]) {
    if (!groups[k].length) continue;
    const g = el("div", "review-group");
    const h = el("h3", null, titles[k]);
    h.append(el("span", "pill " + k, String(groups[k].length)));
    g.append(h, el("p", "note", notes[k]));
    for (const { t, g: grp } of groups[k]) g.append(reviewRow(t, grp));
    frag.append(g);
  }
  return frag;
}

function reviewRow(track, group) {
  const info = kindOf(track, group);
  const row = el("div", "review-row");
  row.append(el("div", "r-name", track.filename));
  const meta = [track.bpm == null ? null : track.bpm.toFixed(1), track.camelot]
    .filter(Boolean).join("  ");
  row.append(el("div", "r-meta", meta || "—"));
  row.append(el("div", "r-why " + info.k, info.why));

  const act = el("div", "r-actions");
  const sel = el("select");
  for (const c of state.crates) sel.append(new Option(c.name, c.name));
  sel.append(new Option("New crate…", "__new__"));
  // Default to where the track already is, never the alphabetically-first
  // crate - otherwise "Move" quietly files it somewhere the DJ never chose.
  if (track.crate) sel.value = track.crate;
  const fresh = el("input"); fresh.type = "text"; fresh.placeholder = "Crate name";
  fresh.classList.add("hidden");
  sel.addEventListener("change", () => {
    fresh.classList.toggle("hidden", sel.value !== "__new__");
    if (sel.value === "__new__") fresh.focus();
  });

  /* A move is always the model being wrong — no prompt. An also-add asks,
   * because counting every also-add as a success would inflate the accuracy
   * exactly when the DJ is being lenient. */
  const ask = el("div", "ask");
  const seg = el("div", "seg");
  const yes = el("button", "sel", "was right");
  const no = el("button", null, "was wrong");
  yes.addEventListener("click", () => { yes.classList.add("sel"); no.classList.remove("sel"); });
  no.addEventListener("click", () => { no.classList.add("sel"); yes.classList.remove("sel"); });
  seg.append(yes, no);
  ask.append(el("span", null, "Its pick"), seg);

  const move = el("button", null, "Move");
  const also = el("button", null, "Also add");
  const target = () => (sel.value === "__new__" ? fresh.value.trim() : sel.value);

  move.addEventListener("click", () => send("move", true));
  also.addEventListener("click", () => {
    if (!ask.classList.contains("on")) { ask.classList.add("on"); return; }
    send("add", !yes.classList.contains("sel"));
  });

  async function send(mode, wasError) {
    const to = target();
    if (!to) { fresh.focus(); return; }
    move.disabled = also.disabled = true;
    try {
      await api.post("/api/correct",
        { track_id: track.id, to_crate: to, mode, was_error: !!wasError });
      row.style.opacity = "0";
      setTimeout(refresh, 160);
    } catch (e) {
      fail(e.message);
      move.disabled = also.disabled = false;
    }
  }

  act.append(sel, fresh, ask, move, also);
  // The controls live inside the row; without this, choosing a crate or
  // pressing Move would also fire the row's select/play handlers.
  act.addEventListener("click", (e) => e.stopPropagation());
  act.addEventListener("dblclick", (e) => e.stopPropagation());
  row.append(act);
  return makePlayable(row, track);
}

/* -------------------------------------------------------- the set builder */

/* Two questions a DJ actually asks: "build me something from here" and
 * "what goes next". Same engine, two shapes.
 *
 * Every suggestion shows its reasons and its mix points. A number on its own
 * ("0.87") is not actionable — "tempo matches, same key, mix out at 3:02"
 * is. */

const MODES = [["safe", "Safe"], ["balanced", "Balanced"], ["adventurous", "Adventurous"]];
const ARCS = [["steady", "Hold"], ["build", "Build"], ["cool", "Cool down"]];

async function setPool() {
  const s = state.sets;
  // Keyed on the analysed count as well as the crate: sorting adds mixable
  // tracks while the DJ works, and a pool cached before they arrived would
  // never offer them.
  const stamp = `${s.crate}|${state.progress ? state.progress.done : 0}`;
  if (s.pool && s.poolStamp === stamp) return s.pool;
  s.poolStamp = stamp;
  const q = s.crate ? "?crate=" + encodeURIComponent(s.crate) : "";
  const previous = s.seed && s.pool
    ? s.pool.find((t) => t.id === s.seed) : null;
  s.pool = (await api.get("/api/setpool" + q)).tracks;
  // Narrowing the crate can drop the chosen starting track. Clearing it
  // silently leaves the DJ staring at an empty field wondering what they did,
  // so say what happened.
  if (s.seed && !s.pool.some((t) => t.id === s.seed)) {
    s.seed = null;
    s.result = null;
    if (previous) {
      status(`“${previous.filename}” isn’t in ${s.crate || "this pool"} — pick another start.`);
    }
  }
  return s.pool;
}

async function buildSets() {
  const s = state.sets;
  const pool = await setPool();
  const frag = document.createDocumentFragment();

  const head = el("div", "content-head");
  head.append(el("h2", null, "Set builder"),
    el("span", "sub", `${pool.length} mixable track${pool.length === 1 ? "" : "s"}`));
  frag.append(head);

  frag.append(setControls(pool));

  const out = el("div", "set-out"); out.id = "set-out";
  frag.append(out);
  queueMicrotask(() => renderSetResult());

  if (!pool.length) {
    out.append(el("p", "quiet-line",
      "No analysed tracks yet. Sort some music first."));
  }
  return frag;
}

function setControls(pool) {
  const s = state.sets;
  const box = el("div", "set-controls");

  // --- crate -------------------------------------------------------------
  const crateSel = el("select");
  crateSel.append(new Option("All crates", ""));
  for (const c of state.crates) crateSel.append(new Option(c.name, c.name));
  crateSel.value = s.crate;
  crateSel.addEventListener("change", async () => {
    s.crate = crateSel.value; s.pool = null;
    await renderContent();
  });
  box.append(field("From", crateSel));

  // --- seed --------------------------------------------------------------
  // A datalist rather than a 700-option <select>: the DJ knows the track name
  // and typing it is faster than scrolling to it.
  const seedWrap = el("div", "seed-pick");
  const seed = el("input"); seed.type = "text";
  seed.placeholder = "Type a track name…";
  seed.setAttribute("list", "seed-list");
  const list = el("datalist"); list.id = "seed-list";
  for (const t of pool.slice(0, 900)) {
    const o = new Option(t.filename, String(t.id));
    o.label = t.filename;
    list.append(o);
  }
  const current = pool.find((t) => t.id === s.seed);
  if (current) seed.value = current.filename;
  const byName = new Map(pool.map((t) => [t.filename, t.id]));
  seed.addEventListener("change", () => {
    s.seed = byName.get(seed.value) ?? null;
  });
  seedWrap.append(seed, list);

  const dice = el("button", "ghost", "Surprise me");
  dice.title = "Pick a random starting track";
  dice.addEventListener("click", () => {
    if (!pool.length) return;
    const t = pool[Math.floor(Math.random() * pool.length)];
    s.seed = t.id; seed.value = t.filename;
  });
  seedWrap.append(dice);
  box.append(field("Start from", seedWrap));

  // --- length / mode / arc ------------------------------------------------
  const len = el("input"); len.type = "number";
  len.min = "2"; len.max = "40"; len.value = String(s.length);
  len.addEventListener("change", () => {
    s.length = Math.max(2, Math.min(40, Number(len.value) || 8));
    len.value = String(s.length);
  });
  box.append(field("Tracks", len));

  box.append(field("Risk", segmented(MODES, s.mode, (v) => { s.mode = v; })));
  box.append(field("Energy", segmented(ARCS, s.arc, (v) => { s.arc = v; })));

  // --- actions ------------------------------------------------------------
  const acts = el("div", "set-actions");
  const go = el("button", "primary", "Build set");
  const nxt = el("button", null, "What plays next?");
  acts.append(go, nxt);
  box.append(acts);

  const run = async (kind) => {
    if (!s.seed) { fail("Pick a track to start from."); seed.focus(); return; }
    go.disabled = nxt.disabled = true;
    status("Working out the set…");
    try {
      s.result = kind === "set"
        ? { kind, data: await api.post("/api/set", {
            seed_id: s.seed, length: s.length, mode: s.mode,
            arc: s.arc, crate: s.crate || undefined }) }
        : { kind, data: await api.post("/api/next", {
            track_id: s.seed, mode: s.mode, arc: s.arc,
            limit: 12, crate: s.crate || undefined }),
            from: pool.find((t) => t.id === s.seed) };
      status("");
      renderSetResult();
    } catch (e) { fail(e.message); status(""); }
    finally { go.disabled = nxt.disabled = false; }
  };
  go.addEventListener("click", () => run("set"));
  nxt.addEventListener("click", () => run("next"));

  return box;
}

function field(label, control) {
  const f = el("label", "set-field");
  f.append(el("span", "set-label", label), control);
  return f;
}

function segmented(options, value, onPick) {
  const seg = el("div", "seg seg-wide");
  for (const [v, label] of options) {
    const b = el("button", v === value ? "sel" : null, label);
    b.addEventListener("click", () => {
      for (const o of seg.children) o.classList.remove("sel");
      b.classList.add("sel");
      onPick(v);
    });
    seg.append(b);
  }
  return seg;
}

function renderSetResult() {
  const out = $("#set-out");
  const r = state.sets.result;
  if (!out) return;
  if (!r) {
    out.replaceChildren(el("p", "quiet-line",
      "Pick a starting track, then build a set or ask what plays next."));
    return;
  }
  out.replaceChildren(r.kind === "set" ? setList(r.data) : nextList(r.data, r.from));
}

/* Score bar and label. The number is printed as returned; the word is a
 * reading of it, and the two sit together so the reading can be checked. */
function scoreChip(score) {
  const word = score >= 0.85 ? "easy" : score >= 0.7 ? "works"
             : score >= 0.5 ? "tricky" : "hard";
  const c = el("span", "score-chip s-" + word);
  c.append(el("b", null, score.toFixed(2)), el("i", null, word));
  return c;
}

function reasonList(reasons) {
  const w = el("div", "reasons");
  for (const x of reasons) w.append(el("span", "reason", x));
  return w;
}

function mixLine(mix) {
  const w = el("div", "mixline");
  w.append(el("span", "mx-k", "out"), el("span", "mx-v tnum", clock(mix.out_at)));
  if (mix.out_window && mix.out_window[1] != null) {
    w.append(el("span", "mx-win tnum",
      `${clock(mix.out_window[0])}–${clock(mix.out_window[1])}`));
  }
  w.append(el("span", "mx-arrow", "→"));
  w.append(el("span", "mx-k", "in"), el("span", "mx-v tnum", clock(mix.in_at)));
  if (mix.in_window && mix.in_window[1] != null) {
    w.append(el("span", "mx-win tnum",
      `${clock(mix.in_window[0])}–${clock(mix.in_window[1])}`));
  }
  if (mix.in_note) w.append(el("span", "mx-note", mix.in_note));
  return w;
}

function setList(data) {
  const frag = document.createDocumentFragment();

  const bar = el("div", "set-summary");
  bar.append(el("span", "ss-n", `${data.tracks.length} tracks`));
  if (data.average_score != null) {
    bar.append(el("span", "ss-sep", "·"),
      el("span", null, "average "), scoreChip(data.average_score));
  }
  if (data.weakest) {
    bar.append(el("span", "ss-sep", "·"),
      el("span", "ss-weak", `weakest link into #${data.weakest.position + 1}`));
  }
  frag.append(bar);

  const list = el("ol", "set-list");
  for (const t of data.tracks) {
    if (t.from_previous) list.append(transitionRow(t.from_previous, data.weakest, t.position));
    const li = el("li", "set-track");
    li.append(el("span", "st-pos tnum", String(t.position + 1)));
    const main = el("div", "st-main");
    main.append(el("div", "st-name", t.filename));
    const meta = el("div", "st-meta tnum");
    meta.append(el("span", null, t.bpm == null ? "—" : t.bpm.toFixed(1) + " BPM"),
                el("span", "st-key", t.camelot || "—"),
                el("span", null, clock(t.duration)));
    if (t.duplicates) {
      // duplicates is the number of OTHER copies folded in, so the library
      // holds one more file than that.
      const total = t.duplicates + 1;
      const d = el("span", "st-dupe", `${total} copies in library`);
      d.title = `${t.duplicates} other cop${t.duplicates > 1 ? "ies" : "y"} of this same recording were folded in`;
      meta.append(d);
    }
    main.append(meta);
    li.append(main, energyPip(t.energy));
    list.append(makePlayable(li, t));
  }
  frag.append(list);
  return frag;
}

function energyPip(e) {
  const w = el("div", "st-energy");
  w.title = e == null ? "energy unknown" : `energy ${e.toFixed(2)}`;
  const f = el("div", "st-energy-fill");
  f.style.height = e == null ? "0%" : Math.round(Math.max(0, Math.min(1, e)) * 100) + "%";
  w.append(f);
  return w;
}

/* ---- hearing the transition ----------------------------------------------
 *
 * Web Audio rather than two <audio> elements: a crossfade wants sample-accurate
 * gain ramps on a shared clock, and media elements give neither. The server
 * hands back short WAV segments because browsers cannot decode AIFF, which is
 * most of this library.
 */

const PREVIEW_LEAD = 8;     // seconds of the outgoing track before the blend
const PREVIEW_TAIL = 10;    // seconds of the incoming track after it

const preview = {
  ctx: null,
  playing: null,            // { stop(), button }
  cache: new Map(),

  context() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    return this.ctx;
  },

  async segment(trackId, at, len) {
    const key = `${trackId}@${at.toFixed(2)}+${len}`;
    if (this.cache.has(key)) return this.cache.get(key);
    const r = await fetch(`/api/preview/${trackId}?at=${at.toFixed(2)}&len=${len}`);
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.error || `preview failed (${r.status})`);
    }
    const buf = await this.context().decodeAudioData(await r.arrayBuffer());
    // Decoded audio is megabytes per segment; keep only a working set.
    if (this.cache.size > 12) this.cache.delete(this.cache.keys().next().value);
    this.cache.set(key, buf);
    return buf;
  },

  stop() {
    if (!this.playing) return;
    const { stop, button } = this.playing;
    this.playing = null;
    try { stop(); } catch (e) { /* already ended */ }
    if (button) button.textContent = button.dataset.idle;
  },

  /* Play the end of A, ramping into the start of B across `blend` seconds —
   * which is what the DJ will actually do, so it is what they should hear. */
  async play(button, aId, aAt, bId, bAt, blend) {
    if (this.playing && this.playing.button === button) { this.stop(); return; }
    this.stop();

    button.disabled = true;
    button.textContent = "Loading…";
    let aBuf, bBuf;
    try {
      const lead = Math.max(0, aAt - PREVIEW_LEAD);
      [aBuf, bBuf] = await Promise.all([
        this.segment(aId, lead, PREVIEW_LEAD + blend + 1),
        this.segment(bId, Math.max(0, bAt), blend + PREVIEW_TAIL),
      ]);
    } catch (e) {
      fail(e.message);
      button.disabled = false;
      button.textContent = button.dataset.idle;
      return;
    }
    button.disabled = false;

    const ctx = this.context();
    if (ctx.state === "suspended") await ctx.resume();
    const t0 = ctx.currentTime + 0.08;
    const blendAt = t0 + Math.min(PREVIEW_LEAD, aBuf.duration);

    const src = (buf, gain) => {
      const s = ctx.createBufferSource(); s.buffer = buf;
      s.connect(gain).connect(ctx.destination);
      return s;
    };
    const gA = ctx.createGain(), gB = ctx.createGain();
    gA.gain.setValueAtTime(1, t0);
    gA.gain.setValueAtTime(1, blendAt);
    gA.gain.linearRampToValueAtTime(0, blendAt + blend);
    gB.gain.setValueAtTime(0, t0);
    gB.gain.setValueAtTime(0, blendAt);
    gB.gain.linearRampToValueAtTime(1, blendAt + blend);

    const sA = src(aBuf, gA), sB = src(bBuf, gB);
    sA.start(t0);
    sB.start(blendAt);

    button.textContent = "Stop";
    const stop = () => { try { sA.stop(); } catch (e) {} try { sB.stop(); } catch (e) {} };
    sB.addEventListener("ended", () => {
      if (this.playing && this.playing.button === button) {
        this.playing = null;
        button.textContent = button.dataset.idle;
      }
    });
    this.playing = { stop, button };
  },
};

function previewButton(step, label) {
  const b = el("button", "ghost preview-btn", label);
  b.dataset.idle = label;
  const m = step.mix;
  const ok = m && m.out_at != null && m.in_at != null
    && step.a_id != null && step.b_id != null;
  if (!ok) {
    b.disabled = true;
    b.title = "No mix point for this pair";
    return b;
  }
  b.addEventListener("click", () =>
    preview.play(b, step.a_id, m.out_at, step.b_id, m.in_at, 8));
  return b;
}

function transitionRow(step, weakest, position) {
  const isWeak = weakest && weakest.position === position;
  const row = el("div", "transition" + (isWeak ? " is-weak" : ""));
  const head = el("div", "tr-head");
  head.append(scoreChip(step.score));
  if (Math.abs(step.stretch_pct) >= 0.5) {
    head.append(el("span", "tr-stretch tnum",
      `${step.stretch_pct > 0 ? "+" : ""}${step.stretch_pct.toFixed(1)}%`));
  }
  if (isWeak) head.append(el("span", "tr-weak", "weakest link"));
  head.append(previewButton(step, "Hear it"));
  row.append(head, reasonList(step.reasons), mixLine(step.mix));
  return row;
}

function nextList(data, from) {
  const frag = document.createDocumentFragment();
  const head = el("div", "set-summary");
  head.append(el("span", "ss-n", "After "),
              el("strong", null, from ? from.filename : "this track"));
  frag.append(head);

  if (!data.next.length) {
    frag.append(el("p", "quiet-line", "Nothing else in this pool to mix into."));
    return frag;
  }

  const list = el("ol", "next-list");
  for (const n of data.next) {
    const li = el("li", "next-item");
    const top = el("div", "ni-top");
    top.append(scoreChip(n.score), el("div", "ni-name", n.track.filename));
    const meta = el("div", "ni-meta tnum");
    meta.append(el("span", null, n.track.bpm == null ? "—" : n.track.bpm.toFixed(1)),
                el("span", "st-key", n.track.camelot || "—"));
    top.append(meta);
    li.append(top, reasonList(n.reasons), mixLine(n.mix));

    const acts = el("div", "ni-acts");
    acts.append(previewButton(n, "Hear it"));
    const use = el("button", "ghost", "Start from here");
    use.addEventListener("click", async () => {
      state.sets.seed = n.track.id;
      state.sets.result = null;
      await renderContent();
    });
    acts.append(use);
    acts.addEventListener("click", (e) => e.stopPropagation());
    acts.addEventListener("dblclick", (e) => e.stopPropagation());
    li.append(acts);
    list.append(makePlayable(li, n.track));
  }
  frag.append(list);
  return frag;
}

/* ------------------------------------------------ the panel and the player
 *
 * One panel, two sources: the track the analyser is working on, or a track the
 * DJ clicked. Same renderer for both, which is why /api/track returns the same
 * shape the runner reports. A live run wins - watching the sort happen is the
 * point of the panel - and a clicked track takes the stage once it is idle.
 */

const player = {
  audio: null,
  trackId: null,
  raf: 0,

  element() {
    if (!this.audio) {
      const a = new Audio();
      a.preload = "metadata";
      a.addEventListener("ended", () => this.paint());
      a.addEventListener("pause", () => this.paint());
      a.addEventListener("play", () => this.tick());
      a.addEventListener("error", () => {
        if (this.trackId != null) fail("Could not play that file.");
      });
      this.audio = a;
    }
    return this.audio;
  },

  /* Playing a track and showing it are the same gesture: you cannot listen to
   * one waveform while looking at another. */
  async play(trackId) {
    const a = this.element();
    if (this.trackId === trackId && !a.paused) { a.pause(); return; }
    if (this.trackId !== trackId) {
      this.trackId = trackId;
      a.src = `/api/audio/${trackId}`;
    }
    preview.stop();                 // one thing making sound at a time
    try { await a.play(); } catch (e) { fail("Could not play that file."); }
    this.paint();
  },

  stop() {
    if (!this.audio) return;
    this.audio.pause();
    this.trackId = null;
    this.audio.removeAttribute("src");
    this.paint();
  },

  isPlaying(trackId) {
    return this.audio && this.trackId === trackId && !this.audio.paused;
  },

  position() {
    return this.audio ? this.audio.currentTime : 0;
  },

  seek(sec) {
    if (this.audio && this.trackId != null && isFinite(sec)) {
      this.audio.currentTime = Math.max(0, sec);
      this.paint();
    }
  },

  /* Only while actually playing: a rAF loop that runs on a paused deck is a
   * battery leak for no picture. */
  tick() {
    cancelAnimationFrame(this.raf);
    const step = () => {
      if (!this.audio || this.audio.paused) return this.paint();
      this.paint();
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  },

  paint() {
    const t = shownTrack();
    if (t && t.id === this.trackId) drawWave(t, false);
    const b = $("#sp-play");
    if (b) {
      const live = t && this.isPlaying(t.id);
      b.textContent = live ? "Pause" : "Play";
      b.disabled = !t || t.id == null;
    }
    const c = $("#sp-clock");
    if (c) {
      const t2 = shownTrack();
      c.textContent = (t2 && this.trackId === t2.id)
        ? `${clock(this.position())} / ${clock(t2.duration)}`
        : "";
    }
    for (const row of document.querySelectorAll("[data-track-id]")) {
      row.classList.toggle("is-playing",
        this.isPlaying(Number(row.dataset.trackId)));
    }
  },
};

/* What the panel is currently showing. A live run wins; otherwise the clicked
 * track. */
function shownTrack() {
  const prog = state.progress;
  if (prog && prog.running && prog.current) return prog.current;
  return state.panelTrack || (prog && prog.current) || null;
}

/* Clicking a track selects it. If nothing is being analysed it takes over the
 * panel; during a run the live track keeps the stage and this one waits. */
async function selectTrack(trackId) {
  if (trackId == null) return;
  if (state.panelTrack && state.panelTrack.id === trackId) {
    renderPanel();
    return;
  }
  try {
    state.panelTrack = await api.get("/api/track/" + trackId);
  } catch (e) { fail(e.message); return; }
  state.showAllBars = false;
  renderPanel();
}

function panel() {
  const p = el("section", "sorting at-rest"); p.id = "panel";

  const left = el("div", "sp-left");
  const st = el("div", "sp-status");
  st.append(el("span", "sp-dot"));
  const label = el("span", null, "Not sorting"); label.id = "sp-label";
  const count = el("span", "sp-count"); count.id = "sp-count";
  st.append(label, count);

  const play = el("button", "sp-play", "Play"); play.id = "sp-play";
  play.disabled = true;
  play.addEventListener("click", () => {
    const t = shownTrack();
    if (t && t.id != null) player.play(t.id);
  });
  const cl = el("span", "sp-clock tnum"); cl.id = "sp-clock";
  st.append(play, cl);
  left.append(st);

  const name = el("div", "sp-name"); name.id = "sp-name"; name.textContent = "\u2014";
  const sub = el("div", "sp-sub"); sub.id = "sp-sub";
  left.append(name, sub);

  const wrap = el("div", "sp-canvas-wrap");
  const cv = el("canvas", "sp-canvas"); cv.id = "sp-canvas";
  const empty = el("div", "sp-canvas-empty"); empty.id = "sp-empty";
  empty.textContent = "Click a track to see it, or press Start sorting";
  wrap.append(cv, empty);

  // Click the waveform to move the playhead - the markers are where a DJ
  // wants to jump to, so they should be reachable by pointing at them.
  cv.addEventListener("click", (e) => {
    const t = shownTrack();
    if (!t || t.id == null || !t.duration) return;
    const r = cv.getBoundingClientRect();
    const k = (e.clientX - r.left - WAVE_PAD) / (r.width - WAVE_PAD * 2);
    const at = Math.max(0, Math.min(1, k)) * t.duration;
    if (player.trackId !== t.id) player.play(t.id).then(() => player.seek(at));
    else player.seek(at);
  });
  left.append(wrap);

  const right = el("div", "sp-right"); right.id = "sp-right";
  p.append(left, right);
  return p;
}

function renderPanel() {
  const p = $("#panel");
  const prog = state.progress;
  const running = !!(prog && prog.running);
  p.classList.toggle("running", running);
  p.classList.toggle("at-rest", !running);

  $("#sp-label").textContent = running ? "Sorting"
    : state.panelTrack ? "Selected" : "Last sorted";
  $("#sp-count").textContent = running && prog.total
    ? `${prog.done} / ${prog.total}` + (prog.errors ? `  \u00b7  ${prog.errors} failed` : "")
    : "";

  const cur = shownTrack();
  if (!cur) {
    $("#sp-name").textContent = "\u2014";
    $("#sp-sub").textContent = "";
    $("#sp-empty").classList.remove("hidden");
    $("#sp-right").innerHTML = "";
    return;
  }

  const changed = state.shownKey !== panelKey(cur);
  state.shownKey = panelKey(cur);

  $("#sp-name").textContent = cur.filename;
  $("#sp-sub").textContent = [
    cur.bpm == null ? null : cur.bpm.toFixed(2) + " BPM",
    cur.camelot,
    cur.duration ? clock(cur.duration) : null,
    cur.analysed === false ? "not analysed yet" : null,
  ].filter(Boolean).join("   \u00b7   ");
  $("#sp-empty").classList.toggle("hidden", !!(cur.energy && cur.energy.length));

  drawWave(cur, running && changed);
  if (changed) state.showAllBars = false;
  drawBars(cur);
  player.paint();
}

const panelKey = (t) => `${t.id != null ? t.id : ""}|${t.filename}`;

/* File the selected track by hand.
 *
 * These live in the panel rather than on every row: the row list is a
 * filename at the height of a filename, and the decision is better made with
 * the model's reasoning visible next to it, which is exactly what the panel
 * already shows. It also means one set of controls works from the crate list,
 * the review queue and the set builder alike. */
function panelActions(track) {
  const box = el("div", "sp-actions");

  const sel = el("select");
  for (const c of state.crates) sel.append(new Option(c.name, c.name));
  sel.append(new Option("New crate…", "__new__"));
  // Default to where it already is, never the alphabetically-first crate -
  // otherwise Move quietly files it somewhere never chosen.
  const filed = track.crates && track.crates.length ? track.crates[0].name : null;
  if (filed) sel.value = filed;

  const fresh = el("input"); fresh.type = "text";
  fresh.placeholder = "Crate name"; fresh.classList.add("hidden");
  sel.addEventListener("change", () => {
    fresh.classList.toggle("hidden", sel.value !== "__new__");
    if (sel.value === "__new__") fresh.focus();
  });

  const move = el("button", null, "Move here");
  const add = el("button", null, "Also add");
  move.title = "This track belongs in that crate instead";
  add.title = "It belongs in that crate as well as where it is";

  const target = () => (sel.value === "__new__" ? fresh.value.trim() : sel.value);

  async function send(mode) {
    const to = target();
    if (!to) { fresh.focus(); return; }
    if (mode === "move" && filed === to) {
      status(`Already in ${to}.`);
      return;
    }
    move.disabled = add.disabled = true;
    try {
      await api.post("/api/correct", {
        track_id: track.id, to_crate: to, mode,
        // A move says the model was wrong. An also-add does not: the DJ is
        // saying it fits both, which is not the model failing.
        was_error: mode === "move",
      });
      status(mode === "move" ? `Moved to ${to}.` : `Also added to ${to}.`);
      state.panelTrack = await api.get("/api/track/" + track.id);
      await refresh();
      renderPanel();
    } catch (e) {
      fail(e.message);
    } finally {
      move.disabled = add.disabled = false;
    }
  }
  move.addEventListener("click", () => send("move"));
  add.addEventListener("click", () => send("add"));

  box.append(el("span", "v-key", "File it"), sel, fresh, move, add);
  return box;
}

/* ---- the waveform ------------------------------------------------------- */

const WAVE_PAD = 10;

/* A mirrored waveform, because it is what a DJ expects to see and it reads at
 * a glance. Canvas because this animates.
 *
 * On top of the bars: the structure the analyser found, drawn where it found
 * it. A cue list in a table is data; a marker sitting over the drop is the
 * same fact you can actually aim at. */
const MOMENT_COLOURS = {
  drop:      "#6ecf9b",
  breakdown: "#6e8fff",
  buildup:   "#d3a04e",
  intro:     "#59616e",
  outro:     "#59616e",
  phrase:    "#3b4250",
  "main section": "#5c8cc4",
};

function drawWave(track, animate) {
  const energy = (track && track.energy) || [];
  const moments = (track && track.moments) || [];
  const cv = $("#sp-canvas");
  if (!cv) return;
  const wrap = cv.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (!W || !H) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  if (!energy.length) return;

  const mid = H / 2;
  const max = Math.max(...energy, 1e-6);
  const usable = W - WAVE_PAD * 2;
  const n = energy.length;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dur = track && track.duration;
  const xOf = (frac) => WAVE_PAD + frac * usable;

  const paint = (upTo) => {
    g.clearRect(0, 0, W, H);

    // centre line
    g.strokeStyle = "rgba(255,255,255,0.05)";
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(WAVE_PAD, mid); g.lineTo(W - WAVE_PAD, mid); g.stroke();

    const barW = Math.max(1, usable / n - 0.6);
    for (let i = 0; i < upTo; i++) {
      const x = xOf(i / n);
      const h = (energy[i] / max) * (mid - 14);
      const grad = g.createLinearGradient(0, mid - h, 0, mid + h);
      grad.addColorStop(0, "rgba(110,143,255,0.85)");
      grad.addColorStop(0.5, "rgba(92,140,196,0.55)");
      grad.addColorStop(1, "rgba(43,111,122,0.85)");
      g.fillStyle = grad;
      g.fillRect(x, mid - h, barW, h * 2);
      // A brighter cap on the loudest bars, so the drops stand off the page
      // instead of the whole curve reading as one block of blue.
      if (energy[i] / max > 0.78) {
        g.fillStyle = "rgba(190,214,255,0.5)";
        g.fillRect(x, mid - h, barW, 1.5);
        g.fillRect(x, mid + h - 1.5, barW, 1.5);
      }
    }

    // structure markers, only once the bars behind them are drawn
    if (upTo >= n && dur) {
      g.font = "9px -apple-system, system-ui, sans-serif";
      // Two label lanes, each tracking how far right it has been filled. On a
      // six-minute track the cues bunch up and single-lane labels overprinted
      // each other into unreadable mush ("Buil|Bu|Drop|wn2"). A tick with no
      // label still marks the spot; overlapping text marks nothing.
      const laneEnd = [-Infinity, -Infinity];
      for (const m of moments) {
        if (m.time == null) continue;
        const x = xOf(Math.max(0, Math.min(1, m.time / dur)));
        const c = MOMENT_COLOURS[m.type] || "#59616e";
        const minor = m.type === "phrase";
        g.strokeStyle = c;
        g.globalAlpha = minor ? 0.35 : 0.85;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(x, minor ? mid - 12 : 4);
        g.lineTo(x, minor ? mid + 12 : H - 4);
        g.stroke();
        g.globalAlpha = 1;

        if (minor || !m.label) continue;
        const tw = g.measureText(m.label).width;
        // Flip inside the canvas rather than let it clip off the right edge.
        const lx = x + 3 + tw > W - WAVE_PAD ? x - 3 - tw : x + 3;
        const lane = laneEnd.findIndex((endX) => lx > endX + 4);
        if (lane === -1) continue;              // no room: keep the tick only
        laneEnd[lane] = lx + tw;
        const ly = lane === 0 ? 11 : 21;
        // A backing plate, because the marker ticks and the bars run behind
        // the text and were cutting through the letters.
        g.fillStyle = "rgba(8,9,11,0.82)";
        g.fillRect(lx - 2, ly - 8, tw + 4, 10);
        g.fillStyle = c;
        g.fillText(m.label, lx, ly);
      }
    }

    // leading edge while drawing in
    if (upTo < n) {
      g.fillStyle = "rgba(110,143,255,0.9)";
      g.fillRect(xOf(upTo / n), 6, 1, H - 12);
    }

    // the playhead, when this is the track on the deck
    if (upTo >= n && dur && track.id != null && player.trackId === track.id) {
      const x = xOf(Math.max(0, Math.min(1, player.position() / dur)));
      g.fillStyle = "rgba(255,255,255,0.10)";
      g.fillRect(WAVE_PAD, 0, x - WAVE_PAD, H);
      g.fillStyle = "#e6e9ef";
      g.fillRect(x, 0, 1.5, H);
      g.beginPath(); g.arc(x + 0.75, 4, 3, 0, Math.PI * 2); g.fill();
    }
  };

  if (!animate || reduce) { paint(n); return; }

  cancelAnimationFrame(drawWave._raf);
  const t0 = performance.now();
  const total = 520;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / total);
    paint(Math.max(1, Math.floor(k * n)));
    if (k < 1) drawWave._raf = requestAnimationFrame(step);
  };
  drawWave._raf = requestAnimationFrame(step);
}

function drawBars(cur) {
  const root = $("#sp-right");
  const sims = cur.similarities || [];
  const SHOW = state.showAllBars ? sims.length : Math.min(5, sims.length);

  root.innerHTML = "";
  sims.slice(0, SHOW).forEach((s, i) => {
    const row = el("div", "bar-row" + (i === 0 ? " win" : ""));
    row.append(el("div", "bar-name", s.crate));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    track.append(fill);
    // Printed exactly as the classifier returned it.
    row.append(track, el("div", "bar-val", s.similarity.toFixed(3)));
    root.append(row);
    requestAnimationFrame(() => {
      fill.style.width = Math.max(0, Math.min(1, s.similarity)) * 100 + "%";
    });
  });

  if (sims.length > 5) {
    const more = el("button", "sp-more",
      state.showAllBars ? "Show fewer" : `Show all ${sims.length}`);
    more.addEventListener("click", () => {
      state.showAllBars = !state.showAllBars;
      drawBars(cur);
    });
    root.append(more);
  }

  /* Two different facts, kept apart. Where the track is filed is one thing;
   * what the model thinks of it now is another, and the margin belongs to the
   * second. Printed as one line they read as a single claim - "tech confident
   * · margin 0.023" on a track the model actually scored highest as house. */
  if (cur.id != null) root.append(panelActions(cur));

  const v = el("div", "sp-verdict band-" + (cur.band || "unknown"));
  const filed = cur.crates && cur.crates.length ? cur.crates : null;
  if (filed) {
    v.append(el("span", "v-key", "Filed in"));
    for (const a of filed) {
      const c = el("span", "crate", a.name);
      c.title = a.source === "human" ? "your call" : "sorted automatically";
      if (a.source === "human") c.classList.add("by-hand");
      v.append(c);
    }
  } else {
    v.append(el("span", "crate", cur.crate || "Not filed yet"));
  }

  if (sims.length) {
    const m = el("div", "sp-verdict-model");
    m.append(el("span", "v-key", "Model now"),
             el("strong", null, sims[0].crate),
             el("span", "v-num tnum", sims[0].similarity.toFixed(3)));
    if (cur.margin != null) {
      m.append(el("span", "v-num tnum", `margin ${cur.margin.toFixed(3)}`));
    }
    // Worth pointing out rather than leaving the DJ to compare two lists.
    if (filed && !filed.some((a) => a.name === sims[0].crate)) {
      m.append(el("span", "v-disagree", "disagrees"));
    }
    root.append(v, m);
    return;
  }
  root.append(v);
}

/* ------------------------------------------------------------------ polling */

let timer = null, lastReview = 0;

async function poll() {
  clearTimeout(timer);
  try {
    state.progress = await api.get("/api/progress");
    renderPanel();
    const running = state.progress && state.progress.running;
    const b = $("#start"); if (b) b.disabled = !!running;
    if (running) {
      Object.assign(state, await api.get("/api/state"));
      renderSidebar();
      // Only the review queue moves meaningfully mid-run; re-rendering a crate
      // the DJ is reading would yank the list away under them.
      if (state.view === "__review__" && Date.now() - lastReview > 3000) {
        lastReview = Date.now();
        await renderContent();
      }
    }
    timer = setTimeout(poll, running ? 600 : 3000);
  } catch (e) {
    timer = setTimeout(poll, 5000);
  }
}

async function refresh() {
  try { Object.assign(state, await api.get("/api/state")); }
  catch (e) { fail(e.message); return; }
  renderSidebar();
  await renderContent();
}

window.addEventListener("resize", () => {
  const t = shownTrack();
  if (t) drawWave(t, false);
});

(async function boot() {
  shell();
  await refresh();
  poll();
})();
