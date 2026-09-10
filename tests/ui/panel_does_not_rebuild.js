/* Does the panel's genre column still rebuild itself on every poll tick?
 *
 * That rebuild is the bug the DJ reported: the numbers "reset and go back".
 * Verified here rather than in a browser, because Chrome in this environment
 * will not hold a connection to the local server.
 *
 * Enough of a DOM to run drawBars() and nothing more.
 */
const fs = require("fs");

function makeNode(tag) {
  const n = {
    tagName: tag, className: "", textContent: "", title: "", disabled: false,
    children: [], dataset: {}, style: {}, _listeners: {},
    classList: {
      _s: new Set(),
      add(...c) { c.forEach((x) => this._s.add(x)); },
      remove(...c) { c.forEach((x) => this._s.delete(x)); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    append(...kids) { kids.forEach((k) => this.children.push(k)); },
    appendChild(k) { this.children.push(k); return k; },
    addEventListener(ev, fn) { (this._listeners[ev] ||= []).push(fn); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    replaceChildren(...kids) { this.children = kids; },
    getBoundingClientRect() { return { left: 0, width: 800 }; },
  };
  Object.defineProperty(n, "innerHTML", {
    get() { return ""; },
    set(v) { if (v === "") this.children = []; },
  });
  return n;
}

const spRight = makeNode("div");
const nodes = { "#sp-right": spRight };

global.window = {
  matchMedia: () => ({ matches: false }),
  devicePixelRatio: 1,
  addEventListener() {},
  AudioContext: function () {},
};
global.document = {
  createElement: makeNode,
  querySelector: (sel) => nodes[sel] || null,
  querySelectorAll: () => [],
  addEventListener() {},
  body: makeNode("body"),
};
global.requestAnimationFrame = (fn) => { fn(0); return 1; };
global.cancelAnimationFrame = () => {};
global.fetch = () => new Promise(() => {});
global.Option = function (t, v) { const o = makeNode("option"); o.textContent = t; o.value = v; return o; };
global.performance = { now: () => 0 };
global.Audio = function () { return makeNode("audio"); };

// Load the app without letting boot() run.
let src = fs.readFileSync(process.argv[2], "utf8");
src = src.replace(/\(async function boot\(\)[\s\S]*$/, "");
const load = new Function(src + "\n;return { drawBars, state, panelKey };");
const app = load();

const track = {
  id: 7, filename: "x.aiff", band: "uncertain",
  crates: [{ name: "house", source: "auto" }],
  similarities: [
    { crate: "house", similarity: 0.61 },
    { crate: "tech", similarity: 0.22 },
    { crate: "pop", similarity: 0.09 },
  ],
  margin: 0.39,
};

app.drawBars(track);
const first = spRight.children.slice();
const firstBar = first[0];
const key1 = spRight.dataset.key;

// Eleven poll ticks' worth of redraws, exactly as poll() would produce.
for (let i = 0; i < 11; i++) app.drawBars(track);
const after = spRight.children.slice();

const stable = firstBar === after[0] && first.length === after.length;
console.log("rebuilds on an unchanged track:", stable ? "NO (fixed)" : "YES (bug)");
console.log("  key set:", JSON.stringify(key1));
console.log("  same DOM nodes after 11 redraws:", stable);

// A genuinely different track must still rebuild.
const other = { ...track, id: 8, filename: "y.aiff" };
app.drawBars(other);
const rebuilt = spRight.children[0] !== firstBar;
console.log("rebuilds when the track changes:", rebuilt ? "YES (correct)" : "NO (broken)");

// So must a band/crate change on the same track - that is what makes the
// Correct button disappear after confirming.
app.drawBars(track);
const beforeConfirm = spRight.children[0];
app.drawBars({ ...track, band: "confident",
               crates: [{ name: "house", source: "human" }] });
const afterConfirm = spRight.children[0] !== beforeConfirm;
console.log("rebuilds when the track is confirmed:", afterConfirm ? "YES (correct)" : "NO (broken)");

process.exit(stable && rebuilt && afterConfirm ? 0 : 1);
