import { useState } from "react";
import { getJson } from "../api";
import { Feature } from "../components/Feature";
import type { Message } from "../types";

function text(m: Message): string {
  return m.content
    .map((b) => (b.type === "text" ? (b as { text: string }).text : ""))
    .join("");
}

export default function Memory() {
  const [session, setSession] = useState("chat");
  const [history, setHistory] = useState<Message[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setBusy(true);
    setError(null);
    try {
      setHistory(await getJson<Message[]>(`/api/history/${encodeURIComponent(session)}`));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Feature
      title="Memory"
      blurb="Conversations are remembered per session and replayed into each request."
      how={
        <>
          <p>
            Each turn the backend loads the session's <code>ChatHistory</code> from a{" "}
            <code>ChatStore</code>, appends the new message, calls the model, records
            the reply, and saves it back. The store is pluggable: the built-in{" "}
            <code>InMemoryChatStore</code>, or a SQLite one (run the backend with{" "}
            <code>STORE=sqlite</code>) that survives restarts.
          </p>
          <pre>{`let mut hist = ChatHistory::from_messages(store.load(&session).await?);
hist.user(prompt);
let resp = model.chat(hist.to_request(MODEL).build()?).await?;
hist.record_response(&resp);
store.save(&session, hist.into_messages()).await?;`}</pre>
          <p>
            Chat on the <strong>Streaming Chat</strong> page (session{" "}
            <code>"chat"</code>), then load it here.
          </p>
        </>
      }
    >
      <div className="row">
        <input
          value={session}
          onChange={(e) => setSession(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && load()}
          placeholder="session id"
          disabled={busy}
        />
        <button onClick={load} disabled={busy}>
          {busy ? "Loading…" : "Load history"}
        </button>
      </div>
      {error && <div className="error">⚠ {error}</div>}
      {history && (
        <div className="result">
          <p className="meta">
            session <code>{session}</code> · {history.length} stored message(s)
          </p>
          {history.length === 0 ? (
            <p className="hint">Nothing stored yet — chat under this session first.</p>
          ) : (
            <div className="log">
              {history.map((m, i) => (
                <div key={i} className={`msg ${m.role}`}>
                  <span className="who">{m.role}</span>
                  <span className="text">{text(m)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Feature>
  );
}
