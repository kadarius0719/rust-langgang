# ai

A small, composable, provider-agnostic **AI layer for Rust** — chat, tools,
structured output, and streaming over many model providers behind one API. It
takes LangChain's genuinely good ideas (a composable core, a clean provider
abstraction, structured output, tools) and leaves its bloat behind (no LCEL
DSL, no output-parser zoo, no `AgentExecutor`, no memory-class hierarchy).

Built to be owned and maintained in-house, with best-practice Rust: a lean
dependency tree, feature-gated providers, and unit + integration tests.

## Workspace

| Crate | What it is |
|-------|------------|
| [`ai-core`](crates/ai-core) | The crate you depend on: domain model, `ChatModel` trait, provider adapters (feature-gated), middleware/composition, tracing. |

(Planned: `ai-macros` for a `#[tool]` derive, `ai-retrieval` wrapping
[`swiftide`](https://github.com/bosun-ai/swiftide) for RAG, `ai-test-util` for
downstream mocks.)

## Status

**Phase 0 complete** — workspace, provider-neutral domain model, normalized
error model, request builder, and the transport-agnostic `ChatModel` trait
(with its object-safe `DynChatModel` bridge).

**First provider live** — an OpenAI-compatible adapter (`OpenAiClient`) that
covers your offline path: point `base_url` at llama.cpp's `llama-server`, LM
Studio, or Ollama's `/v1`. Non-streaming chat, SSE streaming, tool calls, structured
output (`structured::<T>()`), and normalized errors, with pluggable `Auth`.
(Native Ollama adapter next.)

**Agent loop** — `Agent` runs the call → run-tools → repeat loop over any model
with a `ToolBox`, a `max_steps` cap, and an inspectable per-step trace; unknown
or failing tools are fed back to the model rather than crashing the run.

**Conversation memory** — `ChatHistory` owns the transcript; `ChatStore`
(`InMemoryChatStore` built in) persists it for save/resume by session id, with
the same async, object-safe shape as `TraceStore`. All domain types are
`serde`-serializable, so you can also bring your own storage
(Postgres/Dynamo/files) with no trait to implement — see [USAGE.md](USAGE.md).

**Composition (Runnable)** — compose functions, models, and parsers into
reusable pipelines: typed `.then()`, `parallel`/`parallel_map` fan-out, `Branch`
routing, `with_fallback`, concurrent `batch`, and a `ModelRunnable` bridge. Typed
and erased (`DynRunnable`) paths. (`cargo run --example pipeline`)

**Traceability + memory complete** — structured `TraceEvent`s for LLM decisions
(tool selection, stop reason, token usage) via a non-invasive `Traced<M>`
decorator and a pluggable `Tracer`, plus a `TraceStore` persistence layer
(`InMemoryTraceStore` built in; bridge to the `tracing` crate behind a feature).

**Extensible without forking** — customize in your own crate: decorate any model
with `ChatModelExt` combinators (`map_request`, `map_response`, `with_fallback`,
`traced`), implement `ChatModel` for your own wrapper, pass closures as hooks
(`FnTool`, `fn_tracer`), add params via `ChatRequest.extra`, or add methods via
your own extension traits. See [EXTENDING.md](EXTENDING.md).

`clippy -D warnings` clean on default and full feature sets.

## Documentation

- **[USAGE.md](USAGE.md)** — install, quickstart, streaming, tools, structured output, tracing, offline setup.
- **[EXTENDING.md](EXTENDING.md)** — customize and extend in your own crate, without forking.
- **[PLAN.md](PLAN.md)** — architecture and roadmap.
- Runnable examples in [`crates/ai-core/examples`](crates/ai-core/examples).

## Tracing LLM decisions

```rust
use std::sync::Arc;
use ai_core::{ChatModelExt, RecordingTracer, InMemoryTraceStore, memory::persist_recording};

let tracer = RecordingTracer::new();
let model = some_model.traced(Arc::new(tracer.clone()));   // non-invasive wrapper

let _ = model.chat(request).await?;                        // request + response recorded

for event in tracer.events() {                             // inspect decisions
    println!("{event:?}");
}
persist_recording(&tracer, &InMemoryTraceStore::new()).await?; // or persist them
```

## Quickstart

Point it at a local/offline model (Ollama, llama.cpp `llama-server`, LM Studio):

```rust
use ai_core::{ChatModel, ChatRequest, OpenAiClient};

#[tokio::main]
async fn main() -> ai_core::Result<()> {
    let model = OpenAiClient::local("http://localhost:11434/v1").chat_model("llama3.1");
    let request = ChatRequest::builder("llama3.1").user("Say hi.").build()?;
    println!("{}", model.chat(request).await?.text());
    Ok(())
}
```

Full guide: **[USAGE.md](USAGE.md)**. Runnable: `cargo run --example chat --features openai`.

## Developing

```sh
cargo test                              # default features (no providers)
cargo test --features full              # everything
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

## License

MIT OR Apache-2.0.
