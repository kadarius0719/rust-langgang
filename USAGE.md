# Using `ai-core`

A practical guide to adding an AI layer to your Rust app — chat, streaming,
tools, structured output, and decision tracing — over local/offline or hosted
models behind one trait.

- New here? Start with [Install](#install) → [Quickstart](#quickstart).
- Customizing behavior? See [EXTENDING.md](EXTENDING.md).
- Architecture & roadmap? See [PLAN.md](PLAN.md).

---

## Mental model

- **`ChatModel`** is the one trait everything goes through. A provider (OpenAI,
  a local runner, …) implements it; you call `chat()` or `stream()`.
- **`ChatRequest`** is provider-neutral. You build it once; the adapter
  translates it to each provider's wire format.
- **Features** turn providers on. The crate is lean by default — you opt into
  exactly what you use.

---

## Install

```toml
[dependencies]
ai-core = { version = "0.1", features = ["openai"] }
tokio = { version = "1", features = ["full"] }
```

Add features for what you need:

| Feature | Enables |
|---------|---------|
| `openai` | OpenAI-compatible adapter — OpenAI **and** offline/local runners (llama.cpp `llama-server`, LM Studio, Ollama `/v1`) and gateways (OpenRouter, Together, Groq, Azure). |
| `schema` | `structured::<T>()` typed output + tool schemas from Rust types (`schemars`). |
| `tracing` | Bridge decision traces to the [`tracing`](https://docs.rs/tracing) crate. |
| `blocking` | A blocking convenience over the async API. |
| `full` | Everything. |

Default (no features) gives you the domain types and `ChatRequest` builder with
a tiny dependency tree — useful for crates that pass requests around without
talking to a provider directly.

---

## Quickstart

### Offline / local model

Run a local OpenAI-compatible server (e.g. `ollama serve` then
`ollama pull llama3.1`, or llama.cpp's `llama-server`), then:

```rust
use ai_core::{ChatModel, ChatRequest, OpenAiClient};

#[tokio::main]
async fn main() -> ai_core::Result<()> {
    let model = OpenAiClient::local("http://localhost:11434/v1") // Ollama's OpenAI endpoint
        .chat_model("llama3.1");

    let request = ChatRequest::builder("llama3.1")
        .system("You are concise.")
        .user("Name three things Rust is good for.")
        .max_tokens(200)
        .build()?;

    let response = model.chat(request).await?;
    println!("{}", response.text());
    Ok(())
}
```

### Hosted API

```rust
use ai_core::OpenAiClient;

let client = OpenAiClient::new(std::env::var("OPENAI_API_KEY")?); // or any gateway via .with_base_url(...)
let model = client.chat_model("gpt-4o-mini");
```

Runnable versions live in [`crates/ai-core/examples`](crates/ai-core/examples):
`cargo run --example chat --features openai`.

---

## Streaming

`stream()` yields normalized [`StreamEvent`]s; `chat()` is just `stream()`
accumulated for you.

```rust
use ai_core::{ChatModel, ChatRequest, StreamEvent};
use futures::StreamExt;
use std::io::Write;

let mut stream = model.stream(ChatRequest::builder("llama3.1")
    .user("Write a haiku about Rust.").build()?).await?;

while let Some(event) = stream.next().await {
    if let StreamEvent::TextDelta(text) = event? {
        print!("{text}");
        std::io::stdout().flush().ok();
    }
}
```

Mid-stream errors arrive as `Err(_)` items, so handle them in the loop. To turn
a stream back into a single response, call `.collect_response().await`.

---

## Tools

Declare tools on the request; read back the calls the model decided to make.

```rust
use ai_core::{ChatModel, ChatRequest, ContentBlock, Message, ToolDef};

let request = ChatRequest::builder("llama3.1")
    .user("What's the weather in Paris?")
    .tool(ToolDef::new(
        "get_weather",
        "Get the current weather for a city.",
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    ))
    .build()?;

let response = model.chat(request).await?;
for call in response.tool_uses() {
    println!("model wants {}({})", call.name, call.args);
    // ... run the tool, then feed the result back as a tool-result message:
    // Message::tool_result(call.id, "72°F and sunny", false)
}
```

With the `schema` feature you can derive the schema from a Rust type instead of
writing JSON by hand (`ai_core::tool_def::<MyArgs>("name", "desc")`), or wrap a
closure as a tool with `FnTool::new(def, |args| async move { ... })`.

### Agent loop

`Agent` runs the loop for you: call the model → run any requested tools via its
`ToolBox` → feed the results back → repeat until a final answer or the
`max_steps` cap.

```rust
use ai_core::{Agent, FnTool, ToolDef};

let agent = Agent::new(model)
    .system("You are a helpful assistant.")
    .tool(FnTool::new(
        ToolDef::new(
            "get_weather",
            "Get the weather for a city.",
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        ),
        |args: serde_json::Value| async move {
            let city = args["city"].as_str().unwrap_or("?");
            Ok::<_, ai_core::Error>(serde_json::json!({ "city": city, "temp": "72F" }))
        },
    ))
    .max_steps(5);

let outcome = agent.run("What's the weather in Paris?").await?;
println!("{}", outcome.text());     // final answer
// outcome.messages — full transcript; outcome.steps / outcome.stopped describe the run.
```

Attach `.tracer(Arc::new(RecordingTracer::new()))` to capture each step, tool
selection, and tool execution as `TraceEvent`s. Unknown or failing tools are fed
back to the model as error tool-results (not fatal), so it can recover.

---

## Structured output

With the `schema` feature, get output deserialized straight into a Rust type.
`structured::<T>()` derives a JSON schema from `T`, asks the provider to honor
it, and parses the result.

```rust
use ai_core::{ChatRequest, StructuredExt};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Recipe {
    title: String,
    steps: Vec<String>,
}

let recipe: Recipe = model
    .structured(ChatRequest::builder("llama3.1")
        .user("Give me a recipe for pancakes.")
        .build()?)
    .await?;
```

---

## Conversation memory

`ChatHistory` is an owned transcript with ergonomic helpers — `user`/`assistant`,
`record_response` (append the model's reply), and `to_request(model)` (a builder
pre-loaded with the messages).

```rust
use ai_core::{ChatHistory, ChatModel};

let mut history = ChatHistory::new();
history.user("Remember my name is Ada.");
let response = model.chat(history.to_request("llama3.1").build()?).await?;
history.record_response(&response);
```

## Persistence: store data your way

The crate **never owns your data store** — it hands you serde-serializable values
and takes them straight back. Every domain type (`Message`, `ChatHistory`,
`ChatResponse`, `TraceEvent`, …) is `Serialize`/`Deserialize`, so you persist
them in *any* backend (Postgres JSONB, DynamoDB, a graph node, a file) and reload
to resume. This works on default features — your storage choice is fully
decoupled from the provider/HTTP/schema features. There are two ways; pick per
use case.

### Option A — bring your own storage (DIY serde)

**When:** you already have a database/ORM and want full control of the schema and
queries, with no trait to implement.

```rust
// Save — serialize and store however you like.
let blob = serde_json::to_string(&history)?;            // the whole transcript ...
// let blob = serde_json::to_string(history.messages())?;  // ... or a plain Message array
my_db.put("session-42", &blob).await?;                  // Postgres / Dynamo / graph / file

// Resume in a later run — load, deserialize, continue.
let mut history: ChatHistory = serde_json::from_str(&my_db.get("session-42").await?)?;
history.user("continue where we left off");
let response = model.chat(history.to_request("llama3.1").build()?).await?;
history.record_response(&response);
```

**Why:** zero coupling. The crate imposes nothing about *where* or *how* you
store — it just round-trips plain JSON (or any serde format) through your code.

### Option B — the `ChatStore` trait

**When:** you want a uniform `load`/`save`/`append`/`clear` interface across the
app, the ability to swap backends without touching call sites, and an in-memory
default for tests.

```rust
use ai_core::{ChatHistory, ChatModel, ChatStore, InMemoryChatStore};

let store = InMemoryChatStore::new();              // swap for your backend impl
let mut history = ChatHistory::from_messages(store.load("user-42").await?);  // resume
history.user("What did we discuss last time?");
let response = model.chat(history.to_request("llama3.1").build()?).await?;
history.record_response(&response);
store.save("user-42", history.into_messages()).await?;  // persist
```

Implementing it for a real backend is a thin wrapper over Option A:

```rust
struct PgChatStore { /* connection pool */ }

impl ChatStore for PgChatStore {
    async fn load(&self, session: &str) -> ai_core::Result<Vec<ai_core::Message>> {
        let json: String = /* SELECT messages FROM sessions WHERE id = $1 */;
        Ok(serde_json::from_str(&json)?)
    }
    async fn save(&self, session: &str, messages: Vec<ai_core::Message>) -> ai_core::Result<()> {
        let json = serde_json::to_string(&messages)?;
        /* UPSERT sessions (id, messages) VALUES ($1, $2) */
        Ok(())
    }
    async fn append(&self, session: &str, message: ai_core::Message) -> ai_core::Result<()> {
        /* append to the row */ Ok(())
    }
    async fn clear(&self, session: &str) -> ai_core::Result<()> {
        /* DELETE FROM sessions WHERE id = $1 */ Ok(())
    }
}
```

**Why:** the trait is a thin, swappable interface over Option A — useful when
several parts of the app persist conversations and you want one seam (and a
mockable `InMemoryChatStore` in tests).

### Agent transcripts

An agent run returns the full transcript — *including tool calls and results* —
as `outcome.messages`. Persist it like any `Vec<Message>` and resume with
`run_messages`:

```rust
let outcome = agent.run("Plan my week").await?;
my_db.put("agent-7", &serde_json::to_string(&outcome.messages)?).await?;

// later, in a new process:
let prior: Vec<ai_core::Message> = serde_json::from_str(&my_db.get("agent-7").await?)?;
let outcome = agent.run_messages(prior).await?;   // continues — tools and all
```

**Why:** lossless resume of multi-step agent conversations; tool calls/results
are `ContentBlock` variants that round-trip cleanly.

### Decision traces

`TraceEvent` is serde too. Either `drain()` a `RecordingTracer` into your own
store, or use a `TraceStore` (`InMemoryTraceStore` built in) with
`persist_recording`:

```rust
let events = tracer.drain();                       // Vec<TraceEvent>, serde-serializable
my_db.put("trace-9", &serde_json::to_string(&events)?).await?;   // your audit log
```

**Why:** a durable, queryable audit of every LLM decision, in whatever store you
already run.

### Which should I use?

- **DIY serde (A)** for the simplest path and total control — you own the schema.
- **`ChatStore` / `TraceStore` traits (B)** when you want a uniform, swappable
  interface or are sharing the pattern across the app.

They're the same underneath; the traits add an interface, not capability.

## Composing pipelines (Runnable)

For multi-step flows, the `Runnable` layer composes steps — functions, models,
parsers — into reusable, typed pipelines with `invoke` and concurrent `batch`.

```rust
use ai_core::{Runnable, RunnableExt};
use ai_core::runnable::{from_fn, model_runnable};

// build prompt -> model -> extract text
let pipeline = from_fn(build_request)
    .then(model_runnable(model))
    .map_out(|response| response.text());
let answer = pipeline.invoke("summarize this".to_string()).await?;
```

- **Fan-out:** `parallel(vec![a, b, c]).invoke(input)` runs same-typed branches
  concurrently into a `Vec<Out>` (e.g. race/compare model handles). Fail-fast.
- **Heterogeneous keyed fan-out:** `parallel_map(vec![("k", Box::new(a.erase())
  as Box<dyn DynRunnable>), …])` → a JSON object keyed by branch name.
- **Routing:** `Branch::new(default).when(predicate, branch)` (first match wins).
- **Recovery:** `step.with_fallback(other)`.
- **Dynamic/heterogeneous graphs:** `.erase()` a typed step to a `DynRunnable`
  over JSON; `Box<dyn DynRunnable>` is itself a `Runnable`, so it composes back
  into the typed combinators.

The layer is request/response oriented — to stream tokens, call `model.stream(…)`
on the model directly. (Timed `with_retry`/backoff is intentionally deferred.)
Runnable example: `cargo run --example pipeline`.

## Tracing LLM decisions

Wrap any model with `.traced(...)` to capture a structured record of every
request, response, tool selection, and error — then inspect or persist it.

```rust
use std::sync::Arc;
use ai_core::{ChatModelExt, RecordingTracer, InMemoryTraceStore, memory::persist_recording};

let tracer = RecordingTracer::new();
let model = model.traced(Arc::new(tracer.clone()));

let _ = model.chat(request).await?;

for event in tracer.events() {
    println!("{event:?}");                 // audit decisions in-memory
}
persist_recording(&tracer, &InMemoryTraceStore::new()).await?; // or persist them
```

With the `tracing` feature, use `TracingTracer` to emit to the `tracing` crate
(logs / OpenTelemetry) instead. Implement `Tracer`/`TraceStore` yourself for any
custom sink or backend.

---

## Customizing behavior

You extend in your own crate — no fork. Wrap any model with combinators, or
implement `ChatModel` for your own wrapper:

```rust
use ai_core::ChatModelExt;

let model = base_model
    .map_request(|req| { req.max_tokens.get_or_insert(512); }) // inject params
    .with_fallback(secondary_model)                            // recover on error
    .traced(tracer);                                           // observe
```

Provider-specific request fields the crate doesn't model go through
`ChatRequest`'s `extra` passthrough:

```rust
let request = ChatRequest::builder("model")
    .user("hi")
    .extra("seed", 42)
    .extra("frequency_penalty", 0.5)
    .build()?;
```

Custom transport/auth: `OpenAiClient` accepts `.with_base_url(...)`,
`.with_http_client(reqwest::Client)`, and `.with_auth(Arc<dyn Auth>)` (implement
`Auth` for SigV4, OAuth, signed headers). Full guide: [EXTENDING.md](EXTENDING.md).

---

## Error handling

Everything returns `ai_core::Result<T>`. Provider failures are normalized so you
branch on *kind*, not provider-specific strings:

```rust
use ai_core::{ApiErrorKind, Error};

match model.chat(request).await {
    Ok(response) => println!("{}", response.text()),
    Err(Error::Provider { kind: ApiErrorKind::RateLimited, .. }) => { /* back off */ }
    Err(Error::Provider { kind: ApiErrorKind::InvalidAuth, .. }) => { /* fix creds */ }
    Err(e) => eprintln!("error: {e}"),
}
```

---

## Choosing & running a provider

| You want… | Do this |
|-----------|---------|
| Fully offline / on-device | `OpenAiClient::local("http://localhost:11434/v1")` (Ollama) or your llama.cpp/LM Studio URL |
| Hosted OpenAI | `OpenAiClient::new(api_key)` |
| A gateway (OpenRouter/Together/Groq/Azure) | `OpenAiClient::new(key).with_base_url("https://…")` |
| Custom auth (e.g. Bedrock SigV4) | `OpenAiClient::new("").with_auth(Arc::new(MyAuth))` |

All of them implement the same `ChatModel`, so the rest of your code doesn't
change when you switch.
