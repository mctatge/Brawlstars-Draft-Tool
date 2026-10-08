const test = require("node:test");
const assert = require("node:assert/strict");
const { createHarness, response, rank, roster } = require("./harness.cjs");

async function board(t, overrides, saved) {
  const h = createHarness(overrides, saved); t.after(() => h.dispose()); h.start(); await h.flush(); return h;
}
async function personalized(t, bracket = "Masters") {
  const h = await board(t, {}, { "bsdraft.tag": "#AAA" });
  h.requests.rank[0].resolve(rank("AAA", bracket)); await h.flush();
  h.requests.roster[0].resolve(roster("AAA")); await h.flush();
  h.skipBans(); await h.flush(); await h.advance(120); return h;
}

test("rapid map changes reject late recommendation, warning, error, and top-meta writes", async t => {
  const h = await board(t); h.skipBans(); await h.flush(); await h.advance(120);
  const old = h.requests.recommend[0], oldTop = h.requests.top[0];
  old.resolve(response(11)); await h.flush();
  assert.equal(h.find("TheCall").props.r.brawler_id, 11);
  h.changeMap(202); await h.flush();
  assert.equal(h.find("TheCall"), undefined, "previous map advice must disappear before the debounce");
  assert.equal(old.options.signal.aborted, true);
  await h.advance(120);
  const latest = h.requests.recommend.at(-1), latestTop = h.requests.top.at(-1);
  latest.resolve({ ...response(22), warnings: [{ text: "current warning", severity: "warn" }] });
  latestTop.resolve({ picks: [{ brawler_id: 22 }] }); await h.flush();
  oldTop.resolve({ picks: [{ brawler_id: 11 }] }); await h.flush();
  assert.equal(h.find("TheCall").props.r.brawler_id, 22);
  assert.equal(h.find("TopMetaStrip").props.picks[0].brawler_id, 22);
  // A second obsolete request rejects after its replacement succeeds.
  h.changeMap(101); await h.flush(); await h.advance(120);
  const failedOld = h.requests.recommend.at(-1);
  h.changeMap(202); await h.flush(); await h.advance(120);
  h.requests.recommend.at(-1).resolve(response(33)); await h.flush();
  failedOld.reject(new Error("obsolete failure")); await h.flush();
  assert.equal(h.find("TheCall").props.r.brawler_id, 33);
  assert.ok(!h.text().includes("obsolete failure"));
});

test("late general and personal responses cannot replace the newer blind-pick board", async t => {
  const h = await personalized(t, "Diamond");
  const old = [...h.requests.recommend];
  h.changeMap(202); await h.flush(); await h.advance(120);
  const newer = h.requests.recommend.filter(r => r.body.map_id === 202);
  newer.find(r => !r.body.personalize).resolve(response(33));
  newer.find(r => r.body.personalize).resolve(response(11)); await h.flush();
  old.forEach(r => r.resolve(response(22))); await h.flush();
  const columns = h.find("PickColumns").props;
  assert.equal(columns.general[0].brawler_id, 33);
  assert.equal(columns.personal[0].brawler_id, 11);
  h.find("RankWidget").props.onClear(); await h.flush();
  assert.equal(h.find("PickColumns"), undefined);
});

test("choosing the player's seat hides old advice and placement enforces fieldability", async t => {
  const h = await personalized(t);
  h.requests.recommend.at(-1).resolve(response(22));
  h.requests.top.at(-1).resolve({ picks: [{ brawler_id: 22 }] }); await h.flush();
  assert.equal(h.find("TheCall").props.r.brawler_id, 22);
  h.all("SeatCheck")[0].props.onToggle(); await h.flush();
  assert.equal(h.find("TheCall"), undefined);
  // The meta rail remains general. Its placement handler must enforce the current seat's roster.
  h.find("TopMetaStrip").props.onPick(22); await h.flush(); await h.advance(120);
  const own = h.requests.recommend.at(-1).body;
  assert.equal(own.personalize, true);
  assert.deepEqual(Array.from(own.our_team), []);
  assert.equal(own.roster[0].power, 11);
  h.find("TopMetaStrip").props.onPick(11); await h.flush(); await h.advance(120);
  assert.deepEqual(Array.from(h.requests.recommend.at(-1).body.our_team), [11]);
});

test("clearing a player invalidates the saved-tag lookup and local storage writes", async t => {
  const h = await board(t, {}, { "bsdraft.tag": "#AAA" });
  const old = h.requests.rank[0];
  h.find("RankWidget").props.onClear(); await h.flush();
  old.resolve(rank("AAA")); await h.flush();
  assert.equal(old.options.signal.aborted, true);
  assert.equal(h.find("RankWidget").props.tag, "");
  assert.equal(h.find("RankWidget").props.rankInfo, null);
  assert.equal(h.storage.has("bsdraft.tag"), false);
  assert.equal(h.requests.roster.length, 0);
});

test("a late old-account map rank lookup cannot overwrite the newly loaded account", async t => {
  const h = await personalized(t);
  h.changeMap(202); await h.flush();
  const old = h.requests.rank.at(-1);
  h.find("RankWidget").props.setTag("#BBB"); await h.flush();
  h.find("RankWidget").props.onCheck(); await h.flush();
  const next = h.requests.rank.at(-1);
  next.resolve(rank("BBB", "Diamond")); await h.flush();
  old.resolve(rank("AAA")); await h.flush();
  assert.equal(old.options.signal.aborted, true);
  assert.equal(h.find("RankWidget").props.rankInfo.tag, "BBB");
  assert.equal(h.storage.get("bsdraft.tag"), "BBB");
  assert.equal(h.requests.roster.at(-1).body, "BBB");
});

test("a definitive live unplaced answer clears the old bracket after a season reset", async t => {
  const h = await personalized(t);
  h.changeMap(202); await h.flush();
  h.requests.rank.at(-1).resolve({ found: false, tag: "AAA", bracket: null, source: "live", error: "no Ranked games yet" });
  await h.flush(); await h.advance(120);
  assert.equal(h.find("RankWidget").props.rankInfo.found, false);
  assert.equal(h.requests.recommend.at(-1).body.rank_bracket, null);
});

test("loaded:false roster refresh retains only the same account's last good roster visibly", async t => {
  const h = await personalized(t);
  h.all("SeatCheck")[0].props.onToggle(); await h.flush();
  await h.advance(5 * 60 * 1000);
  h.requests.roster.at(-1).resolve({ loaded: false, tag: "AAA", name: "", owned: [], error: "HTTP 403" });
  await h.flush(); await h.advance(120);
  assert.ok(h.text().includes("Roster refresh failed"));
  assert.equal(h.find("SeatHint").props.ready, true);
  assert.equal(h.requests.recommend.at(-1).body.personalize, true);
  h.find("RankWidget").props.setTag("#BBB"); await h.flush();
  assert.ok(!h.text().includes("Roster refresh failed"));
  assert.equal(h.find("SeatHint").props.ready, false);
});

test("an unanswered reference fetch ends in a retryable boot error within the total budget", async t => {
  const h = createHarness(); t.after(() => h.dispose());
  const actualApi = h.actualApi();
  h.fetch = () => new Promise(() => {});
  h.api.getReference = actualApi.getReference;
  h.start(); await h.flush(); await h.advance(90_000);
  assert.ok(h.find("BootScreen").props.error);
  assert.ok(h.find("BootScreen").props.error.includes("timed out") || h.find("BootScreen").props.error.includes("90 seconds"));
});


test("late ban advice cannot replace pick advice after placing a ban and skipping ahead", async t => {
  const h = await board(t); await h.advance(120);
  h.requests.recommend.at(-1).resolve(response(11, "ban")); await h.flush();
  h.find("TheCall").props.onPlace(); await h.flush();
  assert.equal(h.find("TheCall"), undefined);
  await h.advance(120);
  const pendingBan = h.requests.recommend.at(-1);
  assert.deepEqual(Array.from(pendingBan.body.bans), [11]);
  h.skipBans(); await h.flush(); await h.advance(120);
  h.requests.recommend.at(-1).resolve(response(33)); await h.flush();
  pendingBan.resolve(response(22, "ban")); await h.flush();
  assert.equal(pendingBan.options.signal.aborted, true);
  assert.equal(h.find("TheCall").props.kind, "pick");
  assert.equal(h.find("TheCall").props.r.brawler_id, 33);
});

test("a failed roster transport preserves personalization and a successful refresh clears the warning", async t => {
  const h = await personalized(t);
  await h.advance(5 * 60 * 1000);
  h.requests.roster.at(-1).reject(new Error("roster: 503")); await h.flush();
  assert.equal(h.find("SeatHint").props.ready, true);
  assert.ok(h.text().includes("Roster refresh failed"));
  await h.advance(5 * 60 * 1000);
  h.requests.roster.at(-1).resolve(roster("AAA")); await h.flush();
  assert.ok(!h.text().includes("Roster refresh failed"));
  // A response already in flight when the account is cleared must not resurrect its roster.
  await h.advance(5 * 60 * 1000);
  const old = h.requests.roster.at(-1);
  h.find("RankWidget").props.onClear(); await h.flush();
  old.resolve(roster("AAA")); await h.flush();
  assert.equal(old.options.signal.aborted, true);
  assert.equal(h.find("SeatHint").props.ready, false);
});
