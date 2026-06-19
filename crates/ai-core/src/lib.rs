//! `ai-core` — a small, composable, provider-agnostic AI layer for Rust.
//!
//! Plug an AI layer into your app — chat, streaming, tools, and structured
//! output — over many providers behind one trait. It keeps LangChain's good
//! ideas (a composable core, a clean provider abstraction, structured output,
//! tools) and drops the bloat (no LCEL DSL, no parser zoo, no `AgentExecutor`,
//! no memory-class hierarchy). The same [`ChatRequest`]/[`ChatResponse`] flows
//! through an offline local runner, AWS Bedrock, or a hosted API unchanged.
//!
//! # Quickstart
//!
//! Add it with a provider feature. The OpenAI-compatible adapter also reaches
//! **offline/local** runners (llama.cpp `llama-server`, LM Studio, Ollama `/v1`):
//!
//! ```toml
//! [dependencies]
//! ai-core = { version = "0.1", features = ["openai"] }
//! tokio = { version = "1", features = ["full"] }
//! ```
//!
//! ```ignore
//! use ai_core::{ChatModel, ChatRequest, OpenAiClient};
//!
//! // Point at a local model (offline) or use `OpenAiClient::new(api_key)`.
//! let model = OpenAiClient::local("http://localhost:11434/v1").chat_model("llama3.1");
//! let request = ChatRequest::builder("llama3.1").user("Say hi.").build()?;
//! let response = model.chat(request).await?;
//! println!("{}", response.text());
//! ```
//!
//! The request builder works with no provider feature enabled:
//!
//! ```
//! use ai_core::ChatRequest;
//!
//! let request = ChatRequest::builder("gpt-4o-mini")
//!     .system("You are concise.")
//!     .user("Say hi.")
//!     .max_tokens(64)
//!     .build()
//!     .unwrap();
//! assert_eq!(request.messages.len(), 1);
//! ```
//!
//! # Guides
//!
//! - **Usage & integration:** `USAGE.md` (streaming, tools, structured output,
//!   tracing, offline setup).
//! - **Extending without forking:** `EXTENDING.md`.
//! - **Architecture & roadmap:** `PLAN.md`.
//! - **Examples:** `cargo run --example pipeline` (Runnable layer), or `chat` /
//!   `streaming` with `--features openai`.
//!
//! # Feature flags
//!
//! - `openai` — OpenAI-compatible adapter (OpenAI + offline/local + gateways).
//! - `ollama`, `anthropic`, `bedrock` — additional providers (in progress).
//! - `schema` — [`StructuredExt::structured`] and tool schemas via `schemars`.
//! - `tracing` — bridge [`TraceEvent`]s to the `tracing` crate.
//! - `blocking` — a blocking convenience over the async API.
//! - `full` — everything.
#![forbid(unsafe_code)]

pub mod agent;
#[cfg(feature = "http")]
pub mod auth;
pub mod error;
pub mod history;
pub mod memory;
pub mod message;
pub mod middleware;
pub mod model;
pub mod providers;
pub mod request;
pub mod response;
pub mod runnable;
pub mod stream;
pub mod structured;
pub mod tool;
pub mod trace;

/// A boxed, `Send` future — the return type of the object-safe trait facades.
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

pub use agent::{Agent, AgentOutcome, StopCause};
#[cfg(feature = "http")]
pub use auth::{ApiKeyAuth, Auth, BearerAuth, NoAuth};
pub use error::{ApiErrorKind, Error, Result};
pub use history::{ChatHistory, ChatStore, DynChatStore, InMemoryChatStore};
pub use memory::{DynTraceStore, InMemoryTraceStore, TraceStore};
pub use message::{ContentBlock, ImageSource, Message, Role};
pub use middleware::{ChatModelExt, MapRequest, MapResponse, WithFallback};
pub use model::{ChatClient, ChatModel, DynChatModel};
#[cfg(feature = "openai")]
pub use providers::openai::{OpenAiClient, OpenAiModel};
pub use request::{ChatRequest, ChatRequestBuilder};
pub use response::{ChatResponse, StopReason, ToolUseRef, Usage};
pub use runnable::{
    from_fn, model_runnable, parallel, parallel_map, Branch, DynRunnable, Runnable, RunnableExt,
};
pub use stream::{ChatStream, StreamEvent};
#[cfg(feature = "schema")]
pub use structured::StructuredExt;
pub use structured::{ResponseFormat, ResponseFormatKind};
pub use tool::{DynTool, FnTool, Tool, ToolBox, ToolChoice, ToolDef};
#[cfg(feature = "tracing")]
pub use trace::TracingTracer;
pub use trace::{
    fn_tracer, FnTracer, NoopTracer, RecordingTracer, ToolCall, TraceEvent, TraceId, Traced, Tracer,
};

/// The common imports for typical use.
pub mod prelude {
    #[cfg(feature = "schema")]
    pub use crate::StructuredExt;
    pub use crate::{
        from_fn, parallel, Agent, ChatHistory, ChatModel, ChatModelExt, ChatRequest, ChatResponse,
        ChatStore, ChatStream, ContentBlock, Error, FnTool, InMemoryChatStore, Message, Result,
        Role, Runnable, RunnableExt, StopReason, StreamEvent, Tool, ToolBox, ToolDef, TraceEvent,
        Traced, Tracer, Usage,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_messages() {
        let err = ChatRequest::builder("m").build().unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn builder_rejects_empty_model() {
        let err = ChatRequest::builder("   ").user("hi").build().unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
    }

    #[test]
    fn builder_builds_minimal_request() {
        let req = ChatRequest::builder("gpt-4o-mini")
            .system("be brief")
            .user("hello")
            .max_tokens(32)
            .build()
            .unwrap();
        assert_eq!(req.model, "gpt-4o-mini");
        assert_eq!(req.system.as_deref(), Some("be brief"));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].text(), "hello");
        assert_eq!(req.max_tokens, Some(32));
    }

    #[test]
    fn request_serde_round_trip_omits_empty_fields() {
        let req = ChatRequest::builder("m").user("hi").build().unwrap();
        let json = serde_json::to_value(&req).unwrap();
        // Optional/empty fields are skipped, keeping wire bodies clean.
        assert!(json.get("temperature").is_none());
        assert!(json.get("tools").is_none());
        assert!(json.get("extra").is_none());
        let back: ChatRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn stop_reason_preserves_unknown_values() {
        assert_eq!(StopReason::from_wire("tool_use"), StopReason::ToolUse);
        let other = StopReason::from_wire("pause_turn");
        assert_eq!(other, StopReason::Other("pause_turn".to_string()));
        assert_eq!(other.as_str(), "pause_turn");
        // Round-trips through JSON as a bare string.
        let json = serde_json::to_string(&other).unwrap();
        assert_eq!(json, "\"pause_turn\"");
        let back: StopReason = serde_json::from_str(&json).unwrap();
        assert_eq!(back, other);
    }

    #[test]
    fn content_block_text_concatenation() {
        let msg = Message::new(
            Role::Assistant,
            vec![ContentBlock::text("a"), ContentBlock::text("b")],
        );
        assert_eq!(msg.text(), "ab");
    }

    #[test]
    fn api_error_kind_from_status() {
        assert_eq!(ApiErrorKind::from_status(429), ApiErrorKind::RateLimited);
        assert_eq!(ApiErrorKind::from_status(401), ApiErrorKind::InvalidAuth);
        assert_eq!(ApiErrorKind::from_status(503), ApiErrorKind::ServerError);
    }
}
