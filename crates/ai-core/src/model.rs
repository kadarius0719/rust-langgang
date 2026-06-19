//! The core model traits.
//!
//! [`ChatModel`] is the transport-agnostic heart of the crate. It makes **no**
//! assumption about HTTP — an in-process engine, a local HTTP runner, AWS
//! Bedrock, or a hosted API all implement the same trait. `stream` is the
//! primitive; `chat` defaults to accumulating the stream.
//!
//! Native `async fn`-in-traits (RPITIT) keeps the hot path allocation-free, but
//! is not yet `dyn`-compatible, so [`DynChatModel`] provides an object-safe
//! facade (with a blanket impl) for runtime provider selection — and
//! `Box<dyn DynChatModel>` is itself a [`ChatModel`], so erased models compose
//! back into typed code.

use std::future::Future;
use std::sync::Arc;

use crate::{
    error::Result, request::ChatRequest, response::ChatResponse, stream::ChatStream, BoxFuture,
};

/// A provider-agnostic chat model.
///
/// Implement `stream`; `chat` is provided by accumulating the stream (override
/// it if a provider has a cheaper non-streaming path).
pub trait ChatModel: Send + Sync {
    /// Stream a response as a sequence of normalized [`crate::StreamEvent`]s.
    fn stream(&self, request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send;

    /// Produce a complete response.
    ///
    /// Defaults to `stream(request).collect_response()`.
    fn chat(&self, request: ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send {
        async move { self.stream(request).await?.collect_response().await }
    }
}

/// Object-safe facade over [`ChatModel`], for `Box<dyn …>` storage and runtime
/// provider selection. Blanket-implemented for every `ChatModel`.
pub trait DynChatModel: Send + Sync {
    /// See [`ChatModel::stream`].
    fn stream_boxed<'a>(&'a self, request: ChatRequest) -> BoxFuture<'a, Result<ChatStream>>;

    /// See [`ChatModel::chat`].
    fn chat_boxed<'a>(&'a self, request: ChatRequest) -> BoxFuture<'a, Result<ChatResponse>>;
}

impl<T: ChatModel> DynChatModel for T {
    fn stream_boxed<'a>(&'a self, request: ChatRequest) -> BoxFuture<'a, Result<ChatStream>> {
        Box::pin(self.stream(request))
    }

    fn chat_boxed<'a>(&'a self, request: ChatRequest) -> BoxFuture<'a, Result<ChatResponse>> {
        Box::pin(self.chat(request))
    }
}

/// Bridge erased models back into the typed trait so `Box<dyn DynChatModel>`
/// can be used anywhere a `ChatModel` is expected.
impl ChatModel for Box<dyn DynChatModel> {
    fn stream(&self, request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        self.as_ref().stream_boxed(request)
    }

    fn chat(&self, request: ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send {
        self.as_ref().chat_boxed(request)
    }
}

/// Sharing a model across decorators and tasks: an `Arc<M>` is itself a
/// [`ChatModel`], so a single model can be cheaply cloned into a middleware
/// stack or spawned onto many tasks.
impl<M: ChatModel> ChatModel for Arc<M> {
    fn stream(&self, request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        (**self).stream(request)
    }

    fn chat(&self, request: ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send {
        (**self).chat(request)
    }
}

/// A factory that builds configured [`ChatModel`]s by name.
///
/// Kept deliberately small and transport-neutral. HTTP-specific configuration
/// (base URL, custom `reqwest::Client`, auth) belongs on the concrete client
/// builders, not here, so non-HTTP backends can implement this too.
pub trait ChatClient: Send + Sync {
    /// The concrete model type this client produces.
    type Model: ChatModel;

    /// Build a model for the given model identifier.
    fn chat_model(&self, model: impl Into<String>) -> Self::Model;
}
