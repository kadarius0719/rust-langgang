# Extending `ai-core`

Rust has no inheritance — you don't subclass a type and override its methods.
Instead you extend by **implementing traits for your own types** and **composing
wrappers**. Everything below happens in *your* crate; you never fork or redeploy
`ai-core` to customize it.

## 1. Implement a trait (add a backend or capability)

| Trait | Implement it to… |
|-------|------------------|
| `ChatModel` | Add a new provider/backend (offline engine, gateway, custom transport). `stream` is required; `chat` defaults to accumulating the stream. |
| `Tool` / `DynTool` | Add a callable tool. For a one-off, use `FnTool::new(def, closure)` instead of a struct. |
| `Tracer` / `TraceStore` | Add custom observability or persistence. For a one-off tracer, use `fn_tracer(closure)`. |
| `Embeddings`, `ChatClient` | Add embeddings or a model factory. |

To implement one of our traits on a type you **don't own** (the orphan rule
forbids it directly), wrap it in a newtype: `struct MyWrap(TheirType);` then
`impl ChatModel for MyWrap`.

## 2. Override behavior by decorating

Wrap any `ChatModel` with the `ChatModelExt` combinators (import the trait):

```rust
use std::sync::Arc;
use ai_core::{ChatModelExt, RecordingTracer};

let model = base_model
    .map_request(|req| { req.system.get_or_insert_with(|| HOUSE_PROMPT.into()); }) // add/override params
    .map_response(|resp| post_process(resp))                                       // transform output
    .with_fallback(secondary_model)                                                // override reliability
    .traced(Arc::new(RecordingTracer::new()));                                     // observe decisions
```

Stacks are fully typed and zero-cost. Erase to `Box<dyn DynChatModel>` only when
you need to choose a provider at runtime. `Arc<M>` is itself a `ChatModel`, so a
model can be cheaply shared into multiple decorators or tasks.

To add behavior we don't ship (caching, rate-limiting, redaction, custom
retry/backoff), write your own wrapper:

```rust
struct Cached<M> { inner: M, /* cache */ }
impl<M: ChatModel> ChatModel for Cached<M> {
    async fn chat(&self, req: ChatRequest) -> ai_core::Result<ChatResponse> {
        // check cache, else self.inner.chat(req).await, then store
    }
    async fn stream(&self, req: ChatRequest) -> ai_core::Result<ChatStream> {
        self.inner.stream(req).await
    }
}
```

## 3. Add parameters without a crate change

- **`ChatRequest.extra`** (`serde_json::Map`) is merged into the wire body by
  every provider adapter — set provider-specific knobs there. Use the builder's
  `.extra("key", value)`.
- **`ChatResponse.raw`** exposes the provider-native payload for fields the
  domain model doesn't surface.

## 4. Add methods to our types (extension traits)

Give every `ChatModel` your own convenience methods, in your crate:

```rust
use ai_core::{ChatModel, ChatRequest, Result};

trait Ask: ChatModel {
    async fn ask(&self, q: &str) -> Result<String> {
        let req = ChatRequest::builder("…").user(q).build()?;
        Ok(self.chat(req).await?.text())
    }
}
impl<M: ChatModel> Ask for M {}
// now: any_model.ask("hi").await
```

## 5. Override default trait methods

Trait methods with default bodies (e.g. `ChatModel::chat`) are overridable in
your `impl` — provide a cheaper non-streaming path, or different accumulation,
while keeping the rest of the trait's behavior.

---

**Coming with Phase 1 (HTTP providers):** a pluggable `Auth` trait, an
injectable `reqwest::Client`, and `base_url` / header / error-classification
hooks — so transport behavior is customizable from your code too. A timed
`with_retry` (backoff) arrives with the Phase 2 async-timer work.
