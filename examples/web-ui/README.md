# web-ui — an `ai-core` feature showcase

A small, fully offline web app that drives a local model through `ai-core` and
demonstrates each of the crate's capabilities on its own page:

| Page | Shows | Endpoint |
| --- | --- | --- |
| Streaming Chat | token-by-token `ChatModel::stream` over SSE | `POST /api/chat/stream` |
| Agent & Tools | an `Agent` tool-calling loop + the decision transcript | `POST /api/agent` |
| Structured Output | `model.structured::<T>()` into a typed Rust struct | `POST /api/structured` |
| MCP Tools | an `Agent` whose tools come from an [MCP](../mcp) server (the backend is also an MCP client) | `GET /api/mcp/tools`, `POST /api/mcp/agent` |
| Memory | per-session `ChatHistory` + `ChatStore` (in-memory or SQLite) | `GET /api/history/{session}` |
| Logging & Traces | structured `TraceEvent`s from a non-invasive `Tracer` | `GET /api/logs` |

```
React (Vite, :5173)  ──HTTP/SSE──▶  axum backend (:8080)  ──▶  ai-core  ──▶  Ollama /v1 (:11434)
```

The backend is a standalone crate (not part of the workspace) with a path
dependency on `../../../crates/ai-core`, so it builds straight from a clone.

## Prerequisites

- **Rust** 1.82+ and **Node** 20+
- **Ollama** with a model: `ollama serve` then `ollama pull llama3.2:1b`
  (set a different `MODEL` in `backend/src/main.rs` to use another).

## Run it (three terminals)

```sh
# 1. model
ollama serve

# 2. backend  (add STORE=sqlite to persist memory to ./webui.db across restarts)
cd examples/web-ui/backend
cargo run

# 3. frontend
cd examples/web-ui/frontend
npm install
npm run dev      # http://localhost:5173
```

Open <http://localhost:5173> and click through the sidebar.

The **MCP Tools** page additionally needs the MCP server from the sibling example:

```sh
cd examples/mcp && cargo run --bin mcp-server   # serves tools on :9000
```

## Quick API smoke (no UI)

```sh
curl -s localhost:8080/api/health
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"prompt":"Say hi in three words."}'
curl -s -X POST localhost:8080/api/agent -H 'content-type: application/json' \
  -d '{"prompt":"What is the weather in Paris?"}'
```

## Learn more

A step-by-step walkthrough of how this is built — including the SSE wiring, the
pluggable `ChatStore`, and the tracing setup — is in
[`docs/web-ui-smoke-test.md`](../../docs/web-ui-smoke-test.md).

## Notes

- Fully offline; no API keys. To target a hosted provider instead, change one
  line in `backend/src/main.rs` (`OpenAiClient::local(...)` → `OpenAiClient::new(key)`).
- `backend/webui.db` (created only with `STORE=sqlite`) and `target/` /
  `node_modules/` are gitignored.
