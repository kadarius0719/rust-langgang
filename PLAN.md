# ai — architecture & roadmap

A from-scratch, idiomatic-Rust framework inspired by LangChain, owned and
maintained in-house. This file is the source of truth for *why* the crate is
shaped the way it is and *what* is built vs. planned.

## Goals & non-goals

**Goals:** plug an AI layer into a Rust app (chat, structured output, tool use,
agents) over many providers/models behind one abstraction; best-practice Rust;
a lean dependency tree; unit + integration tests; no duplication; no bloat.

**Non-goals (explicitly rejected LangChain bloat):** no LCEL pipe-operator DSL
as the primary surface; no output-parser zoo (structured output is a model-layer
feature); no `AgentExecutor` (an agent is a small, explicit, inspectable
tool-call loop); no memory-class hierarchy (history is owned `Vec<Message>` +
an optional store trait); no loader/splitter/toolkit zoo; no bespoke
callback-manager (use the `tracing` crate).

## Decisions locked (from project kickoff)

- **Providers must span offline, AWS (Bedrock), and hosted APIs — start with
  offline.** "Offline" = a **local HTTP runner** (Ollama / llama.cpp
  `llama-server` / LM Studio), not in-process inference. This rides on the
  OpenAI-compatible adapter + a native Ollama adapter, reusing the HTTP stack.
  The `ChatModel` trait is transport-agnostic, so an in-process engine, Bedrock,
  or a hosted API can be added later without changing the trait.
- **Build the Runnable composition layer** (typed `.then()` pipeline + erased
  `dyn` path + combinators) — not deferred.
- **Async-first**, with a thin `blocking` convenience behind a feature.
- **RAG via [`swiftide`](https://github.com/bosun-ai/swiftide)** (active, MIT) in
  a separate `ai-retrieval` crate rather than reinventing an ingestion pipeline.

## Architecture

**Identity:** a provider-neutral domain model (`Message`, `ContentBlock`,
`ToolDef`, `Usage`, `StopReason`, `StreamEvent`) + the `ChatModel` trait, with
per-provider serde adapters translating to/from each wire format. Wire DTOs stay
*separate* from domain types; the domain serde is our own canonical format (for
persistence/tests), never derived onto one provider's JSON.

**Async/dyn strategy:** native `async fn`-in-traits (RPITIT) on the hot path
(zero-cost), plus an object-safe `DynChatModel`/`DynTool` boxed-future facade
(blanket-impl'd) for runtime provider selection. `Box<dyn DynChatModel>` is
itself a `ChatModel`, so erased models compose back into typed code.

**Streaming is the primitive:** `ChatModel::stream` returns a `ChatStream`
(normalized `StreamEvent`s, `Item = Result<…>` so mid-stream errors are
first-class); `ChatModel::chat` defaults to accumulating the stream.

**Traceability for LLM decisions:** every model call and the decisions around it
(tool selection, stop reason, token usage; and — wired in later phases — agent
steps, retries, fallbacks) are captured as structured `TraceEvent`s correlated
by a `TraceId`. Capture is **non-invasive**: a `Traced<M>` decorator (via
`model.traced(tracer)`) wraps any `ChatModel` and emits to a `Tracer`, so
provider adapters stay oblivious. Sinks: `RecordingTracer` (in-memory),
`TracingTracer` (the `tracing` crate, behind the `tracing` feature), or a
`NoopTracer`. **Memory/persistence scaffolding:** a `TraceStore` trait
(async, object-safe via `DynTraceStore`) with an `InMemoryTraceStore` default
persists those events beyond a run; `persist_recording` flushes a recorder into
any store. This is the same async-append-and-load shape the conversation-memory
layer (`ChatStore`) will take in Phase 2.

**Extensibility without forking:** users customize entirely in their own crate.
Behavior lives behind traits they implement (`ChatModel`, `Tool`, `Tracer`,
`TraceStore`); they compose decorators via the `ChatModelExt` combinators
(`map_request`, `map_response`, `with_fallback`, `traced`), add their own
behavior by implementing `ChatModel` for a wrapper struct, add params via
`ChatRequest.extra`, add convenience methods via their own extension traits, and
pass closures as hooks (`FnTool`, `fn_tracer`/`FnTracer`). `Arc<M>` is itself a
`ChatModel` for cheap sharing. See `EXTENDING.md`. (Timed `with_retry` + a
pluggable HTTP `Auth`/client land in Phases 2/1 respectively.)

**Feature philosophy (deliberate deviation from the research synthesis):** only
feature-gate things that pull *real dependencies* — HTTP providers (`http` +
per-provider features), `schemars` (`schema`), `tokio` blocking (`blocking`).
Pure-Rust layers (Runnable, agent, history, prompt) stay always-compiled. This
keeps the feature matrix small and additive. Default features = **zero
providers**; the user opts in.

### Workspace layout

```
ai/
├─ crates/
│  ├─ ai-core/            # the crate users depend on
│  │  └─ src/
│  │     ├─ message.rs    # Role, Message, ContentBlock, ImageSource          [done]
│  │     ├─ error.rs      # Error, ApiErrorKind (non_exhaustive, normalized)  [done]
│  │     ├─ request.rs    # ChatRequest + builder                             [done]
│  │     ├─ response.rs   # ChatResponse, Usage, StopReason, ToolUseRef       [done]
│  │     ├─ stream.rs     # ChatStream, StreamEvent, accumulator              [done]
│  │     ├─ tool.rs       # Tool/DynTool, ToolDef, ToolChoice                 [done]
│  │     ├─ structured.rs # ResponseFormat (+ schemars helper under `schema`) [done]
│  │     ├─ model.rs      # ChatModel, DynChatModel, ChatClient               [done]
│  │     ├─ trace.rs      # Traceability: TraceEvent, Tracer, Traced<M>       [done]
│  │     ├─ memory.rs     # TraceStore + InMemoryTraceStore (persistence)     [done]
│  │     ├─ middleware.rs # ChatModelExt decorators (map_request/response/fallback) [done]
│  │     ├─ auth.rs       # pluggable Auth (Bearer/ApiKey/NoAuth)            [done]
│  │     ├─ providers/    # openai [done]; ollama/anthropic/bedrock/gemini   [phase 1+]
│  │     ├─ runnable.rs   # Runnable + combinators (then/parallel/branch/...)  [done]
│  │     ├─ agent.rs      # explicit tool-call loop (Agent, ToolBox)          [done]
│  │     ├─ history.rs    # ChatHistory + ChatStore (conversation memory)     [done]
│  │     ├─ template.rs   # tiny prompt interpolation                        [phase 2]
│  │     └─ providers/    # openai, ollama, anthropic, bedrock, gemini       [phase 1+]
│  ├─ ai-macros/          # #[tool] derive                                   [phase 4]
│  ├─ ai-retrieval/       # wraps swiftide                                   [phase 4]
│  └─ ai-test-util/       # MockChatModel for downstream tests              [phase 4]
```

### Reuse vs. build

- **Reuse:** `reqwest` (HTTP), `tokio` (runtime, behind `blocking`), `serde`/
  `serde_json`, `futures` + `async-stream` + `eventsource-stream` (streaming),
  `thiserror` (error enum shape is ours), `schemars` (tool/structured schemas),
  `swiftide` (RAG), `tracing` (observability).
- **Build/own:** the domain model, the `ChatModel`/provider abstraction, every
  provider adapter (the Messages/Chat-Completions surfaces are small and stable
  — implement directly on `reqwest`), the Runnable layer, the agent loop.
- **Don't depend on:** `langchain-rust` (stale), `llm-chain`/`rustformers-llm`
  (abandoned), `clust`/`anthropic-rs` (dormant). Read `rig`/`genai` for ideas.

## Roadmap

- **Phase 0 — foundation. ✅ DONE.** Workspace, domain model, error model,
  request builder, `ChatModel`/`DynChatModel` traits, `ChatStream`/`StreamEvent`
  + accumulator. Clippy clean (default + full).
- **Traceability + memory scaffolding. ✅ DONE.** `TraceEvent`/`TraceId`,
  `Tracer` (+ `Noop`/`Recording`/`Tracing` impls), the `Traced<M>` decorator,
  and the `TraceStore`/`InMemoryTraceStore` persistence layer with
  `persist_recording`.
- **Extensibility / middleware layer. ✅ DONE.** `ChatModelExt` decorators
  (`map_request`/`map_response`/`with_fallback`/`traced`), `Arc<M>` composition,
  closures-as-hooks (`FnTool`, `FnTracer`/`fn_tracer`), and `EXTENDING.md`.
  21 tests + 4 doc-tests green across default and full.
- **Phase 1 — offline MVP. ✅ DONE.** Built: the OpenAI-compatible
  adapter (`OpenAiClient::new`/`local` + `with_base_url`/`with_auth`/
  `with_http_client`, `OpenAiModel`) — non-streaming chat, SSE streaming folded
  into `StreamEvent`, tool binding + tool-call parsing, normalized error mapping,
  and request-shape verification, all wiremock-tested — plus the pluggable
  `Auth` trait (`Bearer`/`ApiKey`/`NoAuth`), the `structured::<T>()` typed helper
  (`StructuredExt`, `schema` feature), a streaming tool-call test, runnable
  examples (`chat`/`streaming`), and the `USAGE.md` integration guide. The
  native Ollama (NDJSON) adapter is **deferred** — the OpenAI-compatible adapter
  already reaches Ollama via `/v1`; add it only if Ollama-specific knobs
  (`options`/`keep_alive`, `/api/embeddings`) are needed (Phase 4). Original
  scope below:
  OpenAI-compatible adapter (configurable
  `base_url`/auth → covers local llama.cpp/LM Studio + hosted gateways) and a
  native Ollama adapter; SSE + NDJSON stream parsing folded into `StreamEvent`;
  `bind tools` on the request + tool-call parsing (incl. streamed partial-JSON
  assembly); `structured::<T>()` via schemars (native JSON mode + tool-call
  fallback); tiny prompt/template helper. Gated by `wiremock` integration tests
  + `insta` request-body snapshots; live tests behind an `#[ignore]`/env-key
  `live` path.
- **Phase 2 — composition + agents. ✅ DONE.** The explicit agent tool-call loop
  (`Agent`/`AgentOutcome`/`StopCause`, the `ToolBox` registry, `max_steps`, and an
  inspectable step trace via `AgentStep`/`ToolSelected`/`ToolExecuted`;
  unknown/failing tools fed back as error results, not fatal); conversation memory
  — `ChatHistory` + the `ChatStore` persistence trait (`InMemoryChatStore`,
  object-safe `DynChatStore`); and the `Runnable` composition layer —
  `then`/`map_out`/`parallel`/`Branch`/`with_fallback`/`erase`/`parallel_map`/
  `model_runnable`, with typed and erased (`DynRunnable`) paths. The Runnable
  layer was adversarially reviewed (multi-agent): a `Box<dyn DynRunnable>`
  Send-across-`tokio::spawn` defect was found, fixed, and regression-tested, and
  `parallel`/`parallel_map` were made genuinely fail-fast (`try_join_all`).
  ⬜ Deferred by design: timed `with_retry` (needs a backoff-timer dep) and
  `RunConfig`.
- **Phase 3 — Anthropic + Bedrock + conformance.** Anthropic Messages adapter
  (content blocks, top-level system, `tool_use`/`tool_result` by id, block-framed
  SSE → same `StreamEvent`); AWS Bedrock adapter; cross-provider conformance
  suite (same `ChatRequest` → equivalent normalized `ChatResponse`); `tracing`
  spans + usage hooks.
- **Phase 4 — breadth.** Gemini adapter; `#[tool]` proc-macro (`ai-macros`);
  `ai-retrieval` (swiftide); `ai-test-util` (`MockChatModel`); optional `blocking`
  surface polish.

## Open items / risks

- Structured-output JSON-Schema subset divergence across providers (OpenAI
  strict needs `additionalProperties:false` + all-required; Gemini subset;
  Anthropic quirks) → per-adapter down-conversion + tool-call fallback.
- Gemini tool calls are name-keyed (no call id) → adapter tracks name→call with
  positional disambiguation; document the parallel-call limitation.
- Provider wire drift → `#[serde(default)]` + catch-alls, `StopReason::Other`,
  `ChatRequest.extra` passthrough, `#[non_exhaustive]` enums, `insta` snapshots.
- `dyn`-compat tax: two trait flavors (RPITIT + boxed facade) until
  `async fn` in `dyn` traits stabilizes; the blanket impl bridges them.
- Keep the Runnable layer from creeping into LCEL: cap the combinator set, lead
  with typed `.then()` + plain `async`/`?`.
