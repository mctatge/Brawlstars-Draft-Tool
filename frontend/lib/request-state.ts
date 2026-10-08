// Request identity is independent of whether the transport honors cancellation.
export type KeyedResult<T> = { key: string; data: T };

export function currentResult<T>(result: KeyedResult<T> | null, key: string): T | null {
  return result?.key === key ? result.data : null;
}

export function normalizePlayerTag(tag: string): string {
  return tag.trim().replace(/^#+/, "").toUpperCase();
}

export class LatestRequest {
  private controller: AbortController | null = null;

  start() {
    this.cancel();
    const controller = new AbortController();
    this.controller = controller;
    return {
      signal: controller.signal,
      current: () => this.controller === controller && !controller.signal.aborted,
      cancel: () => {
        controller.abort();
        if (this.controller === controller) this.controller = null;
      },
    };
  }

  cancel() {
    this.controller?.abort();
    this.controller = null;
  }
}

export type RequestOptions = { signal?: AbortSignal; timeoutMs?: number };

// Bound both the connection and reading the response body. The explicit race also handles a
// transport that never settles after abort, so loading always has an application deadline.
export async function requestJson<T>(
  url: string, label: string, init: RequestInit = {}, options: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  const forwardAbort = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener("abort", forwardAbort, { once: true });
  if (options.signal?.aborted) forwardAbort();
  const timeoutMs = options.timeoutMs ?? 20_000;
  let rejectAbort: (reason: unknown) => void = () => {};
  const onAbort = () => rejectAbort(controller.signal.reason || new DOMException("Aborted", "AbortError"));
  const aborted = new Promise<never>((_, reject) => {
    rejectAbort = reject;
    controller.signal.addEventListener("abort", onAbort, { once: true });
    if (controller.signal.aborted) onAbort();
  });
  const timer = setTimeout(() => controller.abort(
    new DOMException(`${label}: request timed out`, "TimeoutError"),
  ), timeoutMs);
  try {
    return await Promise.race([
      (async () => {
        controller.signal.throwIfAborted();
        const res = await fetch(url, { ...init, signal: controller.signal });
        if (!res.ok) throw new Error(`${label}: ${res.status}`);
        return await res.json() as T;
      })(),
      aborted,
    ]);
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", forwardAbort);
    controller.signal.removeEventListener("abort", onAbort);
  }
}

export function preserveRoster<T extends {
  loaded: boolean; tag: string; name: string; owned: unknown[]; error?: string | null; stale?: boolean;
}>(current: T | null, incoming: T, tag: string): T {
  const sameTag = (value: string) => normalizePlayerTag(value) === normalizePlayerTag(tag);
  if (incoming.loaded && sameTag(incoming.tag)) return { ...incoming, stale: false, error: null };
  const error = incoming.error || "couldn't refresh your roster";
  if (current?.loaded && sameTag(current.tag)) return { ...current, stale: true, error };
  return { ...incoming, loaded: false, tag, name: "", owned: [], stale: false, error };
}
