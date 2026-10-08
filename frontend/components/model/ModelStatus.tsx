"use client";
import { useEffect, useState } from "react";
import { requestJson } from "@/lib/request-state";

type Metrics = {
  trained_at: string; source_commit: string; evaluation_kind: string;
  n_total: number; n_train: number; n_selection: number; n_test: number;
  test_start_ts: number; data_through_ts: number;
  embedding: { logloss: number; acc: number; auc: number; ece: number };
  model: { parameters: number };
  publication_gate: { delta: number; max_full_delta: number; incumbent_test_overlap: string };
};
type Status = {
  available: boolean; analysis_era_id: string; release_id: string | null;
  sha256: string | null; published_at: string | null; metrics: Metrics | null;
  evaluation_status: string; note: string | null;
};
export default function ModelStatus() {
  const [status, setStatus] = useState<Status | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    requestJson<Status>(`${process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"}/api/model`,
      "model statistics", {}, { signal: controller.signal, timeoutMs: 10_000 })
      .then(value => { if (!controller.signal.aborted) setStatus(value); })
      .catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, []);
  const metrics = status?.evaluation_status === "verified_bundle" ? status.metrics : null;
  return <div className="panel" aria-live="polite">
    <h3 className="mb-3">Served model statistics</h3>
    {!status && <p>{failed ? "The live model report could not be reached. No current performance figures are shown." : "Checking the live model report…"}</p>}
    {status && <p>Neural model: <strong>{status.available ? "available" : "unavailable; the board uses empirical statistics"}</strong>. Balance era: {status.analysis_era_id || "unknown"}.</p>}
    {status && !metrics && <p>{status.note || "No evaluation verified against the served weights is available."}</p>}
    {metrics && <>
      <div className="overflow-x-auto"><table className="w-full text-left text-sm my-4"><thead><tr><th>Final test</th><th>Log loss ↓</th><th>Accuracy ↑</th><th>AUC ↑</th><th>ECE ↓</th></tr></thead>
        <tbody><tr><td>{metrics.n_test.toLocaleString()} matches</td><td>{metrics.embedding.logloss.toFixed(4)}</td><td>{(100 * metrics.embedding.acc).toFixed(1)}%</td><td>{metrics.embedding.auc.toFixed(3)}</td><td>{metrics.embedding.ece.toFixed(3)}</td></tr></tbody></table></div>
      <p>{metrics.n_total.toLocaleString()} matches: {metrics.n_train.toLocaleString()} training, {metrics.n_selection.toLocaleString()} selection, {metrics.n_test.toLocaleString()} final test. {metrics.model.parameters.toLocaleString()} learned parameters.</p>
      <p>Evaluation: {metrics.evaluation_kind}. Test window: {new Date(metrics.test_start_ts * 1000).toISOString().slice(0, 10)} through {new Date(metrics.data_through_ts * 1000).toISOString().slice(0, 10)}. Incumbent overlap: {metrics.publication_gate.incumbent_test_overlap}.</p>
      <p>Paired log-loss change against the released incumbent: {metrics.publication_gate.delta.toFixed(6)}; allowed regression: {metrics.publication_gate.max_full_delta.toFixed(4)}. Calibration and single-pick baselines are diagnostics, not publication gates.</p>
    </>}
    {status?.sha256 && <details className="text-xs break-all mt-3"><summary>Artifact identity</summary><p>Release: {status.release_id || "legacy"}. SHA-256: {status.sha256}. Published: {status.published_at || "unverified"}.</p>{metrics && <p>Trained: {metrics.trained_at}. Source commit: {metrics.source_commit}.</p>}</details>}
  </div>;
}
