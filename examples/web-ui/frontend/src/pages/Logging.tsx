import { useState } from "react";
import { getJson } from "../api";
import { Feature } from "../components/Feature";
import type { TraceEvent } from "../types";

function summarize(e: TraceEvent): string {
  const r = e as Record<string, unknown>;
  if (e.event === "llm_request") {
    return `model=${String(r.model)} · messages=${String(r.message_count)} · tools=${String(r.tool_count)}`;
  }
  if (e.event === "llm_response") {
    const u = (r.usage ?? {}) as { input_tokens?: number; output_tokens?: number };
    return `stop=${String(r.stop_reason)} · in=${u.input_tokens ?? 0} · out=${u.output_tokens ?? 0} · text_len=${String(r.text_len)}`;
  }
  const { event, trace_id, ...rest } = r;
  void event;
  void trace_id;
  return JSON.stringify(rest);
}

export default function Logging() {
  const [events, setEvents] = useState<TraceEvent[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setBusy(true);
    setError(null);
    try {
      setEvents(await getJson<TraceEvent[]>("/api/logs"));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Feature
      title="Logging & Traces"
      blurb="Every model call emits a structured trace event — correlated, inspectable, non-invasive."
      how={
        <>
          <p>
            The model is wrapped with <code>.traced(tracer)</code>, so it emits an{" "}
            <code>LlmRequest</code> before each call and an <code>LlmResponse</code>{" "}
            (or <code>LlmError</code>) after — for both <code>chat</code> and{" "}
            <code>stream</code> — without touching the provider adapter. Events
            correlate by <code>trace_id</code>. In production, swap the recorder for{" "}
            <code>TracingTracer</code> to feed the <code>tracing</code> ecosystem.
          </p>
          <pre>{`let model = base_model.traced(Arc::new(tracer)); // via ChatModelExt`}</pre>
          <p>Use the other pages first to generate some events, then refresh.</p>
        </>
      }
    >
      <div className="row">
        <button onClick={load} disabled={busy}>
          {busy ? "Loading…" : "Refresh logs"}
        </button>
      </div>
      {error && <div className="error">⚠ {error}</div>}
      {events && (
        <div className="result">
          <p className="meta">{events.length} event(s)</p>
          <table className="logs">
            <thead>
              <tr>
                <th>trace</th>
                <th>event</th>
                <th>detail</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i} className={e.event}>
                  <td>{e.trace_id}</td>
                  <td>
                    <code>{e.event}</code>
                  </td>
                  <td>{summarize(e)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Feature>
  );
}
