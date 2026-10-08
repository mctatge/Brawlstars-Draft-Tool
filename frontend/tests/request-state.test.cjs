const test = require("node:test");
const assert = require("node:assert/strict");
const { createHarness } = require("./harness.cjs");

test("API deadlines cover a stalled response body and abort the transport", async t => {
  const h = createHarness(); t.after(() => h.dispose());
  let signal;
  h.fetch = async (_url, init) => { signal = init.signal; return { ok: true, json: () => new Promise(() => {}) }; };
  const pending = h.actualApi().getRoster("AAA", { timeoutMs: 25 });
  const rejected = assert.rejects(pending, error => error.name === "TimeoutError" && error.message.includes("roster"));
  await h.advance(25); await rejected;
  assert.equal(signal.aborted, true);
});

test("caller cancellation handles an already-aborted signal without starting a fetch", async t => {
  const h = createHarness(); t.after(() => h.dispose());
  const controller = new AbortController(); controller.abort();
  let fetched = false; h.fetch = () => { fetched = true; return new Promise(() => {}); };
  await assert.rejects(h.actualApi().recommend({}, { signal: controller.signal }), error => error.name === "AbortError");
  assert.equal(fetched, false);
});

test("a new roster success clears stale state, and failures cannot import another account", t => {
  const h = createHarness(); t.after(() => h.dispose());
  const { preserveRoster } = h.requestState();
  const previous = { loaded: true, tag: "AAA", name: "A", owned: [{ id: 11 }], stale: true, error: "timeout" };
  const good = preserveRoster(previous, { ...previous, stale: false, error: null }, "#AAA");
  assert.equal(good.stale, false); assert.equal(good.error, null);
  const bad = preserveRoster(previous, { loaded: false, tag: "BBB", name: "", owned: [], error: "offline" }, "BBB");
  assert.equal(bad.loaded, false); assert.equal(bad.owned.length, 0); assert.equal(bad.tag, "BBB");
});
