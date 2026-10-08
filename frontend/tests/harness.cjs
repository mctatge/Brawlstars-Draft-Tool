const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

const frontend = path.resolve(__dirname, "..");
const compiledSources = new Map();
const reference = {
  brawlers: [11, 22, 33].map(id => ({ id, name: `Brawler ${id}`, cls: "Tank", rarity: "Rare", image_url: "" })),
  maps: [101, 202].map(id => ({ id, name: `Map ${id}`, mode: "Brawl Ball", image_url: "", games: 1000 - id })),
  modes: ["Brawl Ball"], brackets: ["Diamond", "Masters"], boosted: [],
};
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const response = (id, phase = "pick") => ({
  phase, picks: phase === "pick" ? [{ brawler_id: id, name: `Brawler ${id}`, score: .6, gaps: [], breakdown: {} }] : [],
  bans: phase === "ban" ? [{ brawler_id: id, name: `Brawler ${id}`, threat: .6 }] : [], warnings: [], composition: {}, game_plan: null,
});
const rank = (tag, bracket = "Masters") => ({ found: true, tag, tier: 16, tier_label: bracket, bracket, source: "live" });
const roster = (tag) => ({ loaded: true, tag, name: tag, owned: [{ id: 11, mastery: .6, power: 11, gaps: [], owned_star_powers: [], owned_gadgets: [], owned_gears: [] }] });

function createHarness(overrides = {}, saved = {}) {
  const hooks = [], effectQueue = [], modules = new Map(), timers = new Map(), events = new Map();
  let cursor = 0, time = 0, timerId = 0, queued = false, disposed = false, tree, Component;
  const storage = new Map(Object.entries(saved));
  const requests = { recommend: [], top: [], rank: [], roster: [] };
  const network = name => (body, options = {}) => {
    const d = deferred(); requests[name].push({ body, options, ...d }); return d.promise;
  };
  const api = {
    getReference: async () => reference,
    getHealth: async () => ({ status: "ok", model: true, matches: 1000 }),
    getMeta: async () => ({ shifted: false, shifts: [], new_brawlers: [], note: "" }),
    getRank: network("rank"), getRoster: network("roster"),
    recommend: network("recommend"), getTopPicks: network("top"),
    getLoadout: async () => ({}), warmPersonal: () => {}, ...overrides,
  };
  const addEventListener = (name, fn) => {
    if (!events.has(name)) events.set(name, new Set()); events.get(name).add(fn);
  };
  const removeEventListener = (name, fn) => events.get(name)?.delete(fn);
  const document = { visibilityState: "visible", activeElement: null, addEventListener, removeEventListener };
  const schedule = () => {
    if (queued || disposed) return;
    queued = true;
    queueMicrotask(() => { queued = false; if (!disposed) render(); });
  };
  const react = {
    useState(initial) {
      const i = cursor++;
      if (!(i in hooks)) hooks[i] = typeof initial === "function" ? initial() : initial;
      return [hooks[i], value => {
        const next = typeof value === "function" ? value(hooks[i]) : value;
        if (!Object.is(next, hooks[i])) { hooks[i] = next; schedule(); }
      }];
    },
    useRef(initial) { const i = cursor++; if (!(i in hooks)) hooks[i] = { current: initial }; return hooks[i]; },
    useMemo(fn, deps) {
      const i = cursor++, prev = hooks[i];
      if (!prev || deps.some((d, j) => !Object.is(d, prev.deps[j]))) hooks[i] = { value: fn(), deps };
      return hooks[i].value;
    },
    useEffect(fn, deps) {
      const i = cursor++, prev = hooks[i];
      if (!prev || !deps || deps.some((d, j) => !Object.is(d, prev.deps?.[j]))) {
        hooks[i] = { deps, cleanup: prev?.cleanup };
        effectQueue.push(() => { hooks[i].cleanup?.(); hooks[i].cleanup = fn(); });
      }
    },
  };
  const jsx = (type, props, key) => ({ type, props: props || {}, key });
  const setTimer = (fn, delay = 0, interval = 0) => {
    const id = ++timerId; timers.set(id, { fn, at: time + delay, interval }); return id;
  };
  const globals = {
    AbortController, DOMException, URL, URLSearchParams, Promise, console,
    process: { env: {} }, queueMicrotask,
    Date: class extends Date { static now() { return time; } },
    setTimeout: (fn, ms) => setTimer(fn, ms), clearTimeout: id => timers.delete(id),
    setInterval: (fn, ms) => setTimer(fn, ms, ms), clearInterval: id => timers.delete(id),
    window: { addEventListener, removeEventListener }, document,
    localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    fetch: (...args) => harness.fetch(...args),
  };
  function loadTs(file) {
    file = path.resolve(file);
    if (modules.has(file)) return modules.get(file);
    const exports = {}, module = { exports };
    modules.set(file, exports);
    if (!compiledSources.has(file)) compiledSources.set(file, ts.transpileModule(fs.readFileSync(file, "utf8"), {
      compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
    }).outputText);
    const source = compiledSources.get(file);
    const requireModule = name => {
      if (name === "react") return react;
      if (name === "react/jsx-runtime") return { jsx, jsxs: jsx };
      if (name === "@/lib/api") return api;
      if (name.startsWith("@/components/")) return { default: function Stub() {} };
      const target = name.startsWith("@/") ? path.join(frontend, name.slice(2)) : path.resolve(path.dirname(file), name);
      return loadTs(target + ".ts");
    };
    vm.runInNewContext(source, { ...globals, require: requireModule, module, exports }, { filename: file });
    modules.set(file, module.exports);
    return module.exports;
  }
  function render() {
    cursor = 0;
    tree = Component();
    const pending = effectQueue.splice(0); pending.forEach(fn => fn());
  }
  const nodes = () => {
    const out = [];
    const visit = value => {
      if (Array.isArray(value)) value.forEach(visit);
      else if (value && typeof value === "object" && "props" in value) { out.push(value); visit(value.props.children); }
    };
    visit(tree); return out;
  };
  const harness = {
    api, requests, storage, document, reference,
    fetch: () => { throw new Error("Unexpected network request"); },
    start() { Component = loadTs(path.join(frontend, "components/DraftBoard.tsx")).default; render(); },
    async flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); },
    async advance(ms) {
      const target = time + ms;
      while (true) {
        const next = [...timers].filter(([, t]) => t.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        const [id, t] = next; time = t.at;
        if (t.interval) t.at += t.interval; else timers.delete(id);
        t.fn(); await this.flush();
      }
      time = target; await this.flush();
    },
    find(name) { return nodes().find(n => typeof n.type === "string" ? n.type === name : n.type?.name === name); },
    all(name) { return nodes().filter(n => typeof n.type === "string" ? n.type === name : n.type?.name === name); },
    text() { return JSON.stringify(tree, (key, value) => typeof value === "function" ? undefined : value); },
    event(name, event = {}) { for (const fn of [...(events.get(name) || [])]) fn({ preventDefault() {}, ...event }); },
    skipBans() { this.event("keydown", { key: "Tab" }); },
    changeMap(id) { this.find("select").props.onChange({ target: { value: String(id) } }); },
    requestState() { return loadTs(path.join(frontend, "lib/request-state.ts")); },
    actualApi() { return loadTs(path.join(frontend, "lib/api.ts")); },
    dispose() { disposed = true; for (const h of hooks) h?.cleanup?.(); timers.clear(); },
  };
  return harness;
}
module.exports = { createHarness, deferred, response, rank, roster };
