import { useState } from "react";
import { streamChat } from "../api";
import { Feature } from "../components/Feature";

type Msg = { role: "user" | "assistant"; text: string };

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send() {
    const prompt = input.trim();
    if (!prompt || busy) return;
    setInput("");
    setError(null);
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: prompt }, { role: "assistant", text: "" }]);
    const append = (t: string) =>
      setMessages((m) => {
        const copy = m.slice();
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = { role: "assistant", text: last.text + t };
        return copy;
      });
    try {
      await streamChat({ prompt, session: "chat" }, append, setError);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Feature
      title="Streaming Chat"
      blurb="Responses stream into the page token-by-token as the model generates them."
      how={
        <>
          <p>
            The backend calls <code>model.stream(request)</code> and forwards each{" "}
            <code>StreamEvent::TextDelta</code> as a Server-Sent Event; the browser
            reads the response body and appends tokens live. No <code>futures</code>{" "}
            crate is needed — <code>ChatStream::next()</code> is an inherent method.
          </p>
          <pre>{`let mut events = model.stream(request).await?;
while let Some(ev) = events.next().await {
    if let Ok(StreamEvent::TextDelta(t)) = ev {
        // emit SSE: data: {"text": t}
    }
}`}</pre>
        </>
      }
    >
      <div className="log">
        {messages.length === 0 && <p className="hint">Ask the model something…</p>}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <span className="who">{m.role === "user" ? "you" : "ai"}</span>
            <span className="text">
              {m.text || (busy && i === messages.length - 1 ? "…" : "")}
            </span>
          </div>
        ))}
      </div>
      {error && <div className="error">⚠ {error}</div>}
      <div className="row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Type a message and hit Enter"
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !input.trim()}>
          {busy ? "…" : "Send"}
        </button>
      </div>
    </Feature>
  );
}
