"use client";

import { useState } from "react";
import example from "@/data/demo-draft.json";

const percent = (n: number) => `${(100 * n).toFixed(1)}%`;

export default function DemoDraft() {
  const [index, setIndex] = useState(0);
  const step = example.steps[index];
  return (
    <div className="max-w-4xl mx-auto py-8">
      <p className="label mb-3 text-[var(--gold)]">Recorded example · {example.recorded_at.slice(0, 10)}</p>
      <h1 className="display text-3xl md:text-5xl mb-3">See the picks change.</h1>
      <p className="text-[var(--muted)] max-w-2xl mb-6">Walk through a draft on {example.map.name}. This saved example works without a player tag or a live connection. It shows a past snapshot, so use the live board for current advice.</p>
      <div className="flex flex-wrap gap-2 mb-6" role="group" aria-label="Example draft steps">
        {example.steps.map((s, i) => <button key={s.title} className="seg px-3 py-2" aria-pressed={i === index}
          style={i === index ? { borderColor: "var(--blue)", color: "var(--text)" } : undefined}
          onClick={() => setIndex(i)}>{i + 1}. {s.title}</button>)}
      </div>
      <section className="panel p-4 md:p-6" aria-live="polite" aria-atomic="true">
        <div className="flex flex-wrap justify-between gap-2 mb-5"><h2 className="text-xl font-bold">{step.title}</h2>
          <span className="label">{example.map.mode} · Mythic · {example.map.name}</span></div>
        <p className="text-[var(--muted)] mb-5">{step.note}</p>
        <div className="grid sm:grid-cols-2 gap-3 mb-6">
          {([{ label: "Your team", picks: step.allies, color: "var(--blue)" }, { label: "Enemy team", picks: step.enemies, color: "var(--red)" }]).map(team =>
            <div key={team.label} className="border border-[var(--line)] p-3">
              <div className="label mb-2" style={{ color: team.color }}>{team.label}</div>
              <div className="flex gap-2 flex-wrap">{[0, 1, 2].map(i => <span key={i} className="mono text-sm px-3 py-2 bg-[var(--panel2)]">{team.picks[i] ?? "Open slot"}</span>)}</div>
            </div>)}
        </div>
        <h3 className="label mb-3">Recorded recommendations</h3>
        <div className="grid sm:grid-cols-2 gap-3">
          {step.picks.map((pick, i) => <div key={pick.brawler_id} className="border p-4" style={{ borderColor: i === 0 ? "var(--blue)" : "var(--line)" }}>
            <div className="flex justify-between items-baseline gap-3"><span className="text-xl font-bold">{pick.name}</span><span className="mono text-[var(--green)]">{percent(pick.score)}</span></div>
            <div className="label mt-1 mb-3">{pick.cls} · draft score</div>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-sm text-[var(--muted)]">
              <dt>Map win rate</dt><dd>{percent(pick.map_winrate)}</dd>
              <dt>Model estimate</dt><dd>{pick.win_prob == null ? "Unavailable" : percent(pick.win_prob)}</dd>
              <dt>Sample confidence</dt><dd>{percent(pick.confidence)}</dd>
            </dl>
          </div>)}
        </div>
        <p className="text-xs text-[var(--muted)] mt-5">The draft score blends several signals. It is not a calibrated probability of winning, and this example does not measure a benefit from following the suggestions.</p>
      </section>
      <div className="flex flex-wrap justify-between items-center gap-3 mt-5">
        <button className="seg px-4 py-2" disabled={index === example.steps.length - 1} onClick={() => setIndex(i => i + 1)}>Next step →</button>
        <a href="/" className="seg px-4 py-2">Open the live beta →</a>
      </div>
      <details className="mt-6 text-xs text-[var(--muted)] break-all"><summary className="cursor-pointer">Example source</summary>
        <p className="mt-2">{example.source_note} Sample: {example.stats_matches.toLocaleString()} matches. Balance era: {example.era_id}. Model SHA-256: {example.model_sha256}. Aggregate statistics SHA-256: {example.stats_sha256}.</p>
      </details>
    </div>
  );
}
