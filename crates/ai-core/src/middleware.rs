//! Composable middleware for extending model behavior without forking the crate.
//!
//! Rust has no inheritance, so you customize a [`ChatModel`] by *wrapping* it in
//! a decorator that also implements `ChatModel` and changes behavior before or
//! after delegating. The [`ChatModelExt`] combinators build these wrappers
//! fluently:
//!
//! ```
//! use std::sync::Arc;
//! use ai_core::{ChatModelExt, RecordingTracer};
//! # fn demo<M: ai_core::ChatModel>(model: M, fallback: M) {
//! let tracer = RecordingTracer::new();
//! let _customized = model
//!     .map_request(|req| {
//!         req.max_tokens.get_or_insert(512);
//!     })
//!     .with_fallback(fallback)
//!     .traced(Arc::new(tracer));
//! # }
//! ```
//!
//! Each combinator returns a concrete `ChatModel`, so the stack is fully typed
//! and zero-cost; erase to `Box<dyn DynChatModel>` only when you need runtime
//! selection. To add behavior the crate doesn't ship (a cache, a rate limiter,
//! redaction), implement `ChatModel` for your own wrapper struct — the same
//! pattern, in your own crate. See `EXTENDING.md`.

use std::sync::Arc;

use crate::{
    error::Result,
    model::ChatModel,
    request::ChatRequest,
    response::ChatResponse,
    stream::ChatStream,
    trace::{Traced, Tracer},
};

/// Fluent combinators available on every [`ChatModel`].
pub trait ChatModelExt: ChatModel + Sized {
    /// Mutate every outgoing request before it reaches the model — inject a
    /// system prompt, default parameters, or provider-specific `extra` fields.
    fn map_request<F>(self, f: F) -> MapRequest<Self, F>
    where
        F: Fn(&mut ChatRequest) + Send + Sync,
    {
        MapRequest { inner: self, f }
    }

    /// Transform the completed response from [`ChatModel::chat`].
    ///
    /// Streaming ([`ChatModel::stream`]) is passed through unchanged — transform
    /// streamed events with your own [`ChatStream`] wrapper if you need that.
    fn map_response<F>(self, f: F) -> MapResponse<Self, F>
    where
        F: Fn(ChatResponse) -> ChatResponse + Send + Sync,
    {
        MapResponse { inner: self, f }
    }

    /// Fall back to `fallback` if this model returns an error.
    fn with_fallback<N: ChatModel>(self, fallback: N) -> WithFallback<Self, N> {
        WithFallback {
            primary: self,
            fallback,
        }
    }

    /// Wrap this model so its calls emit trace events to `tracer`.
    fn traced(self, tracer: Arc<dyn Tracer>) -> Traced<Self> {
        Traced::new(self, tracer)
    }
}

impl<M: ChatModel> ChatModelExt for M {}

/// A model wrapper that mutates each request before delegating. See
/// [`ChatModelExt::map_request`].
pub struct MapRequest<M, F> {
    inner: M,
    f: F,
}

impl<M, F> ChatModel for MapRequest<M, F>
where
    M: ChatModel,
    F: Fn(&mut ChatRequest) + Send + Sync,
{
    async fn stream(&self, mut request: ChatRequest) -> Result<ChatStream> {
        (self.f)(&mut request);
        self.inner.stream(request).await
    }

    async fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse> {
        (self.f)(&mut request);
        self.inner.chat(request).await
    }
}

/// A model wrapper that transforms the completed response. See
/// [`ChatModelExt::map_response`].
pub struct MapResponse<M, F> {
    inner: M,
    f: F,
}

impl<M, F> ChatModel for MapResponse<M, F>
where
    M: ChatModel,
    F: Fn(ChatResponse) -> ChatResponse + Send + Sync,
{
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        self.inner.stream(request).await
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let response = self.inner.chat(request).await?;
        Ok((self.f)(response))
    }
}

/// A model wrapper that falls back to a second model on error. See
/// [`ChatModelExt::with_fallback`].
pub struct WithFallback<M, N> {
    primary: M,
    fallback: N,
}

impl<M, N> ChatModel for WithFallback<M, N>
where
    M: ChatModel,
    N: ChatModel,
{
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        match self.primary.stream(request.clone()).await {
            Ok(stream) => Ok(stream),
            Err(_) => self.fallback.stream(request).await,
        }
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        match self.primary.chat(request.clone()).await {
            Ok(response) => Ok(response),
            Err(_) => self.fallback.chat(request).await,
        }
    }
}
