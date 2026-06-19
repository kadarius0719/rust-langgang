import { Link } from "react-router-dom";

const FEATURES: [string, string, string][] = [
  ["/chat", "Streaming Chat", "Token-by-token responses over SSE via ChatModel::stream."],
  ["/agent", "Agent & Tools", "A tool-calling loop: the model calls FnTools and their results feed back."],
  ["/structured", "Structured Output", "Deserialize the model's answer straight into a typed Rust struct."],
  ["/memory", "Memory", "Per-session conversation history via ChatHistory + ChatStore."],
  ["/logging", "Logging & Traces", "Structured TraceEvents from a non-invasive Tracer."],
];

export default function Overview() {
  return (
    <div className="feature">
      <h1>ai-core — feature showcase</h1>
      <p className="blurb">
        <code>ai-core</code> is a small, composable, provider-agnostic AI layer for
        Rust. This app drives a local model (Ollama <code>llama3.2:1b</code>) through
        a tiny axum backend that embeds the crate — fully offline. Each page below
        exercises one capability and explains how it maps to the library.
      </p>
      <div className="cards">
        {FEATURES.map(([to, title, desc]) => (
          <Link key={to} to={to} className="card">
            <h3>{title}</h3>
            <p>{desc}</p>
          </Link>
        ))}
      </div>
      <p className="blurb" style={{ marginTop: "1.5rem" }}>
        Architecture: <code>React → axum (REST/SSE) → ai-core → Ollama /v1</code>.
        The same <code>ChatRequest</code>/<code>ChatResponse</code> types flow through
        every layer, and swapping Ollama for a hosted provider is a one-line change.
      </p>
    </div>
  );
}
