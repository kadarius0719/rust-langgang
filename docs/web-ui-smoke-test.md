# End-to-end smoke test: `ai-core` + a React chat UI (offline, via Ollama)

This guide stands up a complete, **fully offline** chat application on top of
`ai-core` and verifies it end to end — including conversation **memory**,
structured **logging**, and a **pluggable storage backend** (in-process or a
SQLite database):

```
React (Vite, :5173)  ──HTTP/SSE──▶  Rust backend (axum, :8080)  ──▶  ai-core  ──▶  Ollama /v1 (:11434)  ──▶  llama3.2:1b
   browser chat UI                    REST + token streaming          OpenAI-compatible adapter           local model
                                      + memory + trace logging
                                              │
                                              ▼
                                   InMemoryChatStore  ◀─or─▶  SQLite (webui.db)
```

`ai-core` is a *library*, not a server, so the React UI talks to a small Rust
HTTP service that embeds `ai-core`. The backend points `ai-core`'s
OpenAI-compatible adapter at Ollama's `/v1` endpoint — no API keys, no network.

Every code block below has been compiled/typechecked and run against a live
`llama3.2:1b`; the `curl` outputs shown are real.

> **Why the OpenAI adapter and not a "native Ollama" one?** `ai-core`'s
> `OpenAiClient::local(...)` already speaks to Ollama's OpenAI-compatible `/v1`
> surface, which covers chat, streaming, and tool-calling. A native Ollama
> (NDJSON `/api/chat`) adapter is on the roadmap but only adds value for
> Ollama-specific knobs; see [Swapping providers later](#swapping-providers-later).

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Ollama: pull a model](#2-ollama-pull-a-model)
3. [The Rust backend](#3-the-rust-backend)
4. [The React frontend](#4-the-react-frontend)
5. [Run the end-to-end smoke test](#5-run-the-end-to-end-smoke-test)
6. [Checking memory and logging](#6-checking-memory-and-logging)
7. [Switching to a SQLite database](#7-switching-to-a-sqlite-database)
8. [How it maps to `ai-core`](#8-how-it-maps-to-ai-core)
9. [Swapping providers later](#swapping-providers-later)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Tool | Version used | Install |
| --- | --- | --- |
| Rust | 1.82+ (tested 1.94) | <https://rustup.rs> |
| Node.js | 20+ (tested 23) | <https://nodejs.org> |
| Ollama | 0.30+ | `brew install ollama` / <https://ollama.com/download> |

The SQLite backend needs **nothing extra** — `sqlx` bundles SQLite and writes a
local file. You'll run three processes (Ollama, the Rust backend, the Vite dev
server), so keep three terminals handy.

---

## 2. Ollama: pull a model

```sh
ollama serve            # leave running in its own terminal
ollama pull llama3.2:1b # ~1.3 GB; small + fast, good enough for a smoke test
```

Verify it's up:

```sh
curl -s http://localhost:11434/api/tags | head
# {"models":[{"name":"llama3.2:1b", ...}]}
```

`llama3.2:1b` is the default the backend below uses. To use a different model,
change the `MODEL` constant in `src/main.rs`.

---

## 3. The Rust backend

An axum service that embeds `ai-core`. It has conversation memory (per-session
history), structured trace logging, and a storage backend chosen at startup:

| Method & path | Purpose |
| --- | --- |
| `GET /api/health` | liveness probe |
| `POST /api/chat` | non-streaming chat — `{prompt, session?}` → `{text, usage}` |
| `POST /api/chat/stream` | **SSE** token stream — `{prompt, session?}` → `data: {"text": "…"}` |
| `GET /api/history/{session}` | the stored conversation for a session (memory) |
| `GET /api/logs` | the recorded trace events (logging) |

### 3.1 Create the project

```sh
cargo new --bin webui-backend
cd webui-backend
```

### 3.2 `Cargo.toml`

Point the `ai-core` path at your checkout (or use a `git`/`version` dependency).

```toml
[package]
name = "webui-backend"
version = "0.1.0"
edition = "2021"

[dependencies]
ai-core = { path = "../rust_lang/crates/ai-core", features = ["openai"] }
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.6", features = ["cors"] }
async-stream = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sqlx = { version = "0.8", default-features = false, features = ["runtime-tokio", "sqlite"] }
```

### 3.3 `src/main.rs`

```rust
//! A tiny HTTP backend that puts `ai-core` behind a REST + SSE API for a web UI,
//! with conversation memory (per-session history), logging (structured trace
//! events), and a pluggable memory backend: in-process by default, or SQLite
//! when run with `STORE=sqlite`. Fully offline via Ollama.

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;

use ai_core::{
    ChatHistory, ChatModel, ChatModelExt, ChatStore, DynChatStore, InMemoryChatStore, Message,
    OpenAiClient, OpenAiModel, RecordingTracer, StreamEvent, TraceEvent, Traced, Tracer,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tower_http::cors::CorsLayer;

const OLLAMA_V1: &str = "http://localhost:11434/v1";
const MODEL: &str = "llama3.2:1b";
const SYSTEM: &str = "You are a concise, helpful assistant.";

// --- logging: a tracer that prints each event AND records it for /api/logs ---
#[derive(Clone)]
struct LogTracer {
    rec: RecordingTracer,
}

impl Tracer for LogTracer {
    fn record(&self, event: TraceEvent) {
        eprintln!("[trace] {}", serde_json::to_string(&event).unwrap_or_default());
        self.rec.record(event);
    }
}

// --- memory backend #2: a ChatStore backed by SQLite (one row per message) ---
#[derive(Clone)]
struct SqliteChatStore {
    pool: SqlitePool,
}

impl SqliteChatStore {
    async fn connect(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts).await?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS messages (
                 id      INTEGER PRIMARY KEY AUTOINCREMENT,
                 session TEXT NOT NULL,
                 payload TEXT NOT NULL)",
        )
        .execute(&pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session, id)")
            .execute(&pool)
            .await?;
        Ok(Self { pool })
    }
}

/// sqlx error -> ai-core error (via the `Box<dyn Error>` escape hatch).
fn db_err(err: sqlx::Error) -> ai_core::Error {
    ai_core::Error::from(Box::new(err) as Box<dyn std::error::Error + Send + Sync>)
}

// Implementing ai-core's `ChatStore` is all it takes to plug in a new backend;
// `serde` (de)serializes each provider-neutral `Message` to a JSON payload.
impl ChatStore for SqliteChatStore {
    fn load(&self, session: &str) -> impl Future<Output = ai_core::Result<Vec<Message>>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            let rows: Vec<String> =
                sqlx::query_scalar("SELECT payload FROM messages WHERE session = ? ORDER BY id")
                    .bind(&session)
                    .fetch_all(&pool)
                    .await
                    .map_err(db_err)?;
            let mut messages = Vec::with_capacity(rows.len());
            for row in rows {
                messages.push(serde_json::from_str(&row)?);
            }
            Ok(messages)
        }
    }

    fn save(
        &self,
        session: &str,
        messages: Vec<Message>,
    ) -> impl Future<Output = ai_core::Result<()>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            let mut tx = pool.begin().await.map_err(db_err)?;
            sqlx::query("DELETE FROM messages WHERE session = ?")
                .bind(&session)
                .execute(&mut *tx)
                .await
                .map_err(db_err)?;
            for message in &messages {
                let payload = serde_json::to_string(message)?;
                sqlx::query("INSERT INTO messages (session, payload) VALUES (?, ?)")
                    .bind(&session)
                    .bind(payload)
                    .execute(&mut *tx)
                    .await
                    .map_err(db_err)?;
            }
            tx.commit().await.map_err(db_err)?;
            Ok(())
        }
    }

    fn append(
        &self,
        session: &str,
        message: Message,
    ) -> impl Future<Output = ai_core::Result<()>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            let payload = serde_json::to_string(&message)?;
            sqlx::query("INSERT INTO messages (session, payload) VALUES (?, ?)")
                .bind(&session)
                .bind(payload)
                .execute(&pool)
                .await
                .map_err(db_err)?;
            Ok(())
        }
    }

    fn clear(&self, session: &str) -> impl Future<Output = ai_core::Result<()>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            sqlx::query("DELETE FROM messages WHERE session = ?")
                .bind(&session)
                .execute(&pool)
                .await
                .map_err(db_err)?;
            Ok(())
        }
    }
}

#[derive(Clone)]
struct AppState {
    /// The model, wrapped so every call emits trace events.
    model: Arc<Traced<OpenAiModel>>,
    /// A read handle onto recorded trace events (shares the buffer).
    logs: RecordingTracer,
    /// Conversation memory — runtime-selected backend behind ai-core's facade.
    store: Arc<dyn DynChatStore>,
    model_id: String,
}

#[derive(Deserialize)]
struct ChatIn {
    prompt: String,
    #[serde(default = "default_session")]
    session: String,
}

fn default_session() -> String {
    "default".to_string()
}

#[derive(Serialize)]
struct ChatOut {
    text: String,
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Serialize)]
struct Token {
    text: String,
}

/// Maps an ai-core error into a 500 with its message.
struct AppError(ai_core::Error);

impl From<ai_core::Error> for AppError {
    fn from(err: ai_core::Error) -> Self {
        Self(err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = OpenAiClient::local(OLLAMA_V1).chat_model(MODEL);
    let logs = RecordingTracer::new();
    // Attach the tracer non-invasively: the model now emits trace events.
    let model = Arc::new(base.traced(Arc::new(LogTracer { rec: logs.clone() })));

    // Pick the conversation-memory backend at runtime. Both implement ai-core's
    // `ChatStore`; `DynChatStore` erases them to one type behind an `Arc`, so the
    // handlers below are identical regardless of which one is chosen.
    let store: Arc<dyn DynChatStore> = match std::env::var("STORE").as_deref() {
        Ok("sqlite") => {
            println!("memory: SQLite (./webui.db — survives restarts)");
            Arc::new(SqliteChatStore::connect("webui.db").await?)
        }
        _ => {
            println!("memory: in-process InMemoryChatStore (cleared on restart)");
            Arc::new(InMemoryChatStore::new())
        }
    };

    let state = AppState {
        model,
        logs,
        store,
        model_id: MODEL.to_string(),
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(chat_stream))
        .route("/api/history/{session}", get(history))
        .route("/api/logs", get(get_logs))
        .with_state(state)
        // Permissive CORS so the Vite dev server (a different origin) can call us.
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    println!("backend listening on http://127.0.0.1:8080 (model: {MODEL})");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

/// Non-streaming chat that remembers the conversation per session.
async fn chat(
    State(state): State<AppState>,
    Json(body): Json<ChatIn>,
) -> Result<Json<ChatOut>, AppError> {
    // Recall prior turns, append the new user message, ask the model.
    let mut hist = ChatHistory::from_messages(state.store.load_boxed(&body.session).await?);
    hist.user(body.prompt);
    let request = hist.to_request(&state.model_id).system(SYSTEM).build()?;
    let response = state.model.chat(request).await?;
    // Remember the answer and persist the updated transcript.
    hist.record_response(&response);
    state
        .store
        .save_boxed(&body.session, hist.into_messages())
        .await?;

    Ok(Json(ChatOut {
        text: response.text(),
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
    }))
}

/// Streaming chat with the same per-session memory.
async fn chat_stream(State(state): State<AppState>, Json(body): Json<ChatIn>) -> impl IntoResponse {
    let model = state.model.clone();
    let store = state.store.clone();
    let model_id = state.model_id.clone();

    let stream = async_stream::stream! {
        let prior = match store.load_boxed(&body.session).await {
            Ok(prior) => prior,
            Err(err) => { yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string())); return; }
        };
        let mut hist = ChatHistory::from_messages(prior);
        hist.user(body.prompt);

        let request = match hist.to_request(&model_id).system(SYSTEM).build() {
            Ok(request) => request,
            Err(err) => { yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string())); return; }
        };

        match model.stream(request).await {
            Ok(mut events) => {
                let mut full = String::new();
                // `next()` is ai-core's inherent method — no `futures::StreamExt` needed.
                while let Some(event) = events.next().await {
                    match event {
                        Ok(StreamEvent::TextDelta(text)) => {
                            full.push_str(&text);
                            if let Ok(ev) = Event::default().json_data(Token { text }) {
                                yield Ok::<_, Infallible>(ev);
                            }
                        }
                        Err(err) => {
                            yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string()));
                            return;
                        }
                        _ => {}
                    }
                }
                // Persist the assistant turn so the next request remembers it.
                hist.assistant(full);
                let _ = store.save_boxed(&body.session, hist.into_messages()).await;
                yield Ok::<_, Infallible>(Event::default().event("done").data("[DONE]"));
            }
            Err(err) => {
                yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string()));
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Inspect a session's stored conversation (memory).
async fn history(
    State(state): State<AppState>,
    Path(session): Path<String>,
) -> Result<Json<Vec<Message>>, AppError> {
    Ok(Json(state.store.load_boxed(&session).await?))
}

/// Inspect the recorded trace events (logging).
async fn get_logs(State(state): State<AppState>) -> Json<Vec<TraceEvent>> {
    Json(state.logs.events())
}
```

Design notes:

- **Tokens are JSON-encoded per SSE event** (`data: {"text":"…"}`). LLM tokens can
  be newlines, and raw `data:` text would get mangled by SSE's line-framing.
- **`events.next()` is `ai-core`'s inherent `ChatStream::next`**, so the backend
  drives the stream without importing `futures::StreamExt`.
- **The store is chosen at runtime.** `InMemoryChatStore` (built in) and
  `SqliteChatStore` (≈40 lines, written here) both implement `ChatStore`;
  `Arc<dyn DynChatStore>` erases the choice so the handlers never change.

### 3.4 Run it and smoke the API

```sh
cargo run        # defaults to the in-process store
# memory: in-process InMemoryChatStore (cleared on restart)
# backend listening on http://127.0.0.1:8080 (model: llama3.2:1b)
```

In another terminal:

```sh
curl -s http://127.0.0.1:8080/api/health
# ok

curl -s -X POST http://127.0.0.1:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"prompt":"Say hello in exactly three words."}'
# {"text":"Hello there again.","input_tokens":40,"output_tokens":5}

curl -s -N -X POST http://127.0.0.1:8080/api/chat/stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"Count one to five, words only."}'
# data: {"text":"One"}
#
# data: {"text":"\n"}
# ...
# event: done
# data: [DONE]
```

---

## 4. The React frontend

A Vite + React + TypeScript app that streams tokens into a chat log by reading
the SSE response body. (Because the backend now keeps per-session memory and the
UI omits a `session`, it defaults to `"default"` — so the chat is automatically
multi-turn.)

### 4.1 Scaffold

```sh
npm create vite@latest webui-frontend -- --template react-ts
cd webui-frontend
npm install
```

### 4.2 Replace `src/App.tsx`

```tsx
import { useEffect, useRef, useState } from "react";
import "./App.css";

// The Rust backend (see webui-backend). Permissive CORS lets this dev-server
// origin (http://localhost:5173) call it directly.
const API = "http://localhost:8080";

type Msg = { role: "user" | "assistant"; text: string };

/** Parse one SSE frame ("event:"/"data:" lines) and dispatch it. */
function handleFrame(
  frame: string,
  onToken: (t: string) => void,
  onError: (m: string) => void,
) {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  const data = dataLines.join("\n");
  if (event === "error") return onError(data);
  if (event === "done") return;
  try {
    onToken((JSON.parse(data) as { text: string }).text);
  } catch {
    /* ignore keep-alive / non-JSON frames */
  }
}

/** POST a prompt and stream tokens out of the SSE response body. */
async function streamChat(
  prompt: string,
  onToken: (t: string) => void,
  onError: (m: string) => void,
) {
  const res = await fetch(`${API}/api/chat/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok || !res.body) return onError(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      handleFrame(buf.slice(0, idx), onToken, onError);
      buf = buf.slice(idx + 2);
    }
  }
}

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
      await streamChat(prompt, append, setError);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <h1>ai-core chat</h1>
      <div className="log">
        {messages.length === 0 && <p className="hint">Ask the local model something…</p>}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <span className="who">{m.role === "user" ? "you" : "ai"}</span>
            <span className="text">
              {m.text || (busy && i === messages.length - 1 ? "…" : "")}
            </span>
          </div>
        ))}
        <div ref={endRef} />
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
    </div>
  );
}
```

### 4.3 Replace `src/App.css`

```css
.app {
  max-width: 640px;
  margin: 2rem auto;
  padding: 0 1rem;
  font-family: system-ui, sans-serif;
}

h1 {
  font-size: 1.25rem;
}

.log {
  min-height: 50vh;
  max-height: 60vh;
  overflow-y: auto;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.hint {
  color: #888;
}

.msg {
  display: flex;
  gap: 0.5rem;
  line-height: 1.4;
}

.msg .who {
  flex: 0 0 2rem;
  font-size: 0.7rem;
  text-transform: uppercase;
  color: #999;
  padding-top: 0.15rem;
}

.msg.user .text {
  font-weight: 600;
}

.msg .text {
  white-space: pre-wrap;
}

.error {
  color: #b00020;
  margin: 0.5rem 0;
}

.row {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.row input {
  flex: 1;
  padding: 0.5rem 0.75rem;
  border: 1px solid #ccc;
  border-radius: 8px;
  font-size: 1rem;
}

.row button {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 8px;
  background: #2563eb;
  color: white;
  font-size: 1rem;
  cursor: pointer;
}

.row button:disabled {
  opacity: 0.5;
  cursor: default;
}
```

### 4.4 Run it

```sh
npm run dev
#   ➜  Local:   http://localhost:5173/
```

Open <http://localhost:5173>.

---

## 5. Run the end-to-end smoke test

With all three processes running:

| Terminal | Command | Expectation |
| --- | --- | --- |
| 1 | `ollama serve` | stays up; logs requests |
| 2 | `cargo run` (in `webui-backend`) | `backend listening on …:8080` |
| 3 | `npm run dev` (in `webui-frontend`) | `Local: http://localhost:5173/` |

In the browser:

1. Type **"Write a haiku about Rust."** and press Enter → tokens **stream in
   one at a time** (not all at once).
2. Ask a follow-up like **"Now make it funnier."** → it remembers the previous
   haiku, because the backend replays the session's history each turn.

**What "pass" looks like:** tokens render incrementally, follow-ups are
context-aware, and there are no errors in the browser console or the backend
terminal.

---

## 6. Checking memory and logging

`ai-core` makes both first-class, and the backend exposes each for inspection.

### Memory — `ChatHistory` + `ChatStore`

The handlers keep per-session conversation state with this loop:

```rust
let mut hist = ChatHistory::from_messages(store.load(&session).await?); // recall
hist.user(prompt);
let request = hist.to_request(MODEL).system(SYSTEM).build()?;
let response = model.chat(request).await?;
hist.record_response(&response);                                        // remember
store.save(&session, hist.into_messages()).await?;
```

**Check it** with two turns on one session, then read it back:

```sh
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"session":"demo","prompt":"My name is Sam and my favorite color is teal. Reply OK."}'
# {"text":"Thanks for sharing that information about yourself, Sam!", ...}

curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"session":"demo","prompt":"What is my name and favorite color?"}'
# {"text":"Your name is Sam, and your favorite color is teal.", "input_tokens":77, ...}

curl -s localhost:8080/api/history/demo
# [ {"role":"user",...}, {"role":"assistant",...}, {"role":"user",...}, {"role":"assistant",...} ]
```

The second turn recalls "Sam / teal", and `input_tokens` grows (49 → 77) because
the prior turns are replayed.

### Logging — `Tracer` + `TraceEvent`

Wrapping the model with `.traced(tracer)` (via `ChatModelExt`) makes it emit a
structured `LlmRequest` before each call and an `LlmResponse`/`LlmError` after —
for both `chat` and `stream` — without changing the adapter. The demo's
`LogTracer` prints each event **and** records it for `/api/logs`:

```sh
curl -s localhost:8080/api/logs
```
```json
[
  {"event":"llm_request","trace_id":1,"model":"llama3.2:1b","system":true,"message_count":1,"tool_count":0},
  {"event":"llm_response","trace_id":1,"stop_reason":"end_turn","usage":{"input_tokens":49,"output_tokens":11},"text_len":56},
  {"event":"llm_request","trace_id":2,"model":"llama3.2:1b","system":true,"message_count":3,"tool_count":0},
  {"event":"llm_response","trace_id":2,"stop_reason":"end_turn","usage":{"input_tokens":77,"output_tokens":21},"text_len":88}
]
```

Events correlate by `trace_id`, and you can watch `message_count` climb `1 → 3`
as memory feeds back into each request. The same tracer also captures
`ToolSelected` / `ToolExecuted` / `AgentStep`, so an `Agent` run is fully
auditable. The backend terminal shows the same lines live (`[trace] {…}`).

> **Production logging:** swap `LogTracer` for `ai-core`'s `TracingTracer`
> (the `tracing` feature) to bridge events into the
> [`tracing`](https://docs.rs/tracing) ecosystem — `tracing-subscriber` for JSON
> logs, OpenTelemetry export, etc. To persist trace events, implement
> `ai-core`'s `TraceStore` (same shape as `ChatStore`) or use
> `persist_recording(&tracer, &store)`.

---

## 7. Switching to a SQLite database

The in-process store is cleared on restart. Because `SqliteChatStore` implements
the same `ChatStore` trait, switching is just an environment variable — **no code
change**:

```sh
STORE=sqlite cargo run
# memory: SQLite (./webui.db — survives restarts)
```

**Prove it persists across a restart:**

```sh
# turn 1 + 2 under a session
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"session":"persist","prompt":"Remember the code is BLUE-42. Reply OK."}' >/dev/null
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"session":"persist","prompt":"What is the code?"}' >/dev/null

curl -s localhost:8080/api/history/persist | jq length    # => 4

# stop the backend (Ctrl-C), then start it again with STORE=sqlite, and:
curl -s localhost:8080/api/history/persist | jq length    # => 4  (still there!)
ls -la webui.db                                           # the database file
```

With the default in-process store the same sequence returns `0` after a restart;
with SQLite the conversation is still there.

**Going further — same trait, real databases:**

- **Postgres / MySQL:** point `sqlx` at a server (e.g. `features = ["postgres"]`)
  and implement `ChatStore` the same way — `load`/`save`/`append`/`clear`.
- **Redis / DynamoDB / a REST API:** any async backend works; `ChatStore` is an
  async, object-safe trait (`DynChatStore`), so the handlers never change.
- The schema here is deliberately minimal (one JSON row per `Message`); a real
  app might index by `session`, paginate, or store structured columns.

---

## 8. How it maps to `ai-core`

| In the backend | `ai-core` API | What it does |
| --- | --- | --- |
| `OpenAiClient::local(OLLAMA_V1).chat_model(MODEL)` | `OpenAiClient` / `ChatClient` | OpenAI-compatible client + model handle, no auth |
| `base.traced(Arc::new(tracer))` | `ChatModelExt::traced` → `Traced` | non-invasive structured logging |
| `RecordingTracer` / `TraceEvent` | `trace` | record + inspect LLM decisions |
| `ChatHistory` (`user`/`record_response`/`to_request`/`into_messages`) | `history` | per-turn conversation state |
| `InMemoryChatStore` / `SqliteChatStore: ChatStore` | `history::ChatStore` | pluggable, async, per-session persistence |
| `Arc<dyn DynChatStore>` | `history::DynChatStore` | runtime-selected store, one type |
| `model.chat(request)` / `model.stream(request)` | `ChatModel` | one-shot response / token stream |
| `events.next().await` | `ChatStream::next` | drive the stream without `futures::StreamExt` |

Further `ai-core` features to layer on the same way: tools + an `Agent` loop
(register a `FnTool` in a `ToolBox`), and structured output (`schema` feature,
`model.structured::<T>(request)`).

---

## Swapping providers later

The backend is provider-agnostic except for one line. To target the hosted
OpenAI API instead of Ollama:

```rust
// was: OpenAiClient::local("http://localhost:11434/v1").chat_model("llama3.2:1b")
let model = OpenAiClient::new(std::env::var("OPENAI_API_KEY")?).chat_model("gpt-4o-mini");
```

When the **native Ollama (NDJSON) adapter** lands, it will expose Ollama-specific
knobs (the `options` block — `num_ctx`, `repeat_penalty`, mirostat — plus
`keep_alive` and native eval timings) that the `/v1` surface hides. Because both
implement the same `ChatModel`/`ChatStream` contracts, switching will again be a
one-line change to how `model` is constructed; the handlers stay identical.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Backend: `error sending request` / connection refused | Ollama isn't running. `ollama serve`, then `curl localhost:11434/api/tags`. |
| `model "…" not found` | `ollama pull llama3.2:1b` (or set `MODEL` to one you have via `ollama list`). |
| Browser console: CORS error | The backend must run with `CorsLayer::permissive()` (it does) and be reachable at `http://localhost:8080`. |
| UI shows the whole reply at once | Hit `/api/chat/stream` (not `/api/chat`); don't put a buffering proxy in front. |
| Memory empty after restart | Expected with the default in-process store — run `STORE=sqlite cargo run` to persist. |
| `where is the database?` | `./webui.db`, created next to where you run the backend; delete it to reset. |
| First token is slow | Cold model load; the first request after `ollama serve` pays a one-time cost. |
| `cargo` can't find `ai-core` | Fix the `path` in `Cargo.toml`, or use `git = "https://github.com/kadarius0719/rust-langgang"`. |

---

## Project layout recap

```
webui-backend/          # Rust + axum + ai-core  → :8080
├── Cargo.toml
├── src/main.rs
└── webui.db            # created only when run with STORE=sqlite

webui-frontend/         # Vite + React + TS       → :5173
├── package.json
└── src/
    ├── App.tsx
    └── App.css
```

Both were compiled/typechecked and run against a live `llama3.2:1b` (in-process
and SQLite stores, with a restart to confirm persistence) while writing this guide.
```
