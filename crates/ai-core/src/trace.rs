//! Traceability for LLM decisions.
//!
//! Every model call — and the decisions around it (which tools it chose, why it
//! stopped, how many tokens it spent, and later: agent steps, retries,
//! fallbacks) — is captured as a structured [`TraceEvent`] and handed to a
//! [`Tracer`].
//!
//! Tracing is **non-invasive**: wrap any [`ChatModel`] in [`Traced`] (or call
//! `model.traced(tracer)` via [`ChatModelExt`]), and provider adapters stay
//! oblivious. Captured events can be inspected in memory ([`RecordingTracer`]),
//! forwarded to the `tracing` crate ([`TracingTracer`], behind the `tracing`
//! feature), or persisted via a [`crate::memory::TraceStore`].
//!
//! ```
//! use std::sync::Arc;
//! use ai_core::{ChatModelExt, RecordingTracer};
//!
//! let tracer = RecordingTracer::new();
//! // let model = some_model.traced(Arc::new(tracer.clone()));
//! // ... after calls, inspect or persist:
//! assert!(tracer.is_empty());
//! # let _ = Arc::new(tracer);
//! ```

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::stream::Stream;
use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    model::ChatModel,
    request::ChatRequest,
    response::{ChatResponse, StopReason, Usage},
    stream::{ChatStream, StreamEvent},
    structured::ResponseFormatKind,
    tool::ToolChoice,
};

/// A correlation id grouping every event emitted by a single model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TraceId(pub u64);

impl TraceId {
    /// Allocate the next process-unique trace id.
    pub fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        TraceId(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trace-{}", self.0)
    }
}

/// A tool call the model decided to make.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id.
    pub id: String,
    /// Tool name.
    pub name: String,
}

/// A structured record of an LLM decision or lifecycle event.
///
/// `#[non_exhaustive]` so new event kinds (agent steps, retries, fallbacks as
/// they get wired into later phases) are not breaking changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TraceEvent {
    /// A request was issued to a model.
    LlmRequest {
        /// Correlation id.
        trace_id: TraceId,
        /// Target model.
        model: String,
        /// Whether a system prompt was present.
        system: bool,
        /// Number of messages in the request.
        message_count: usize,
        /// Number of tools advertised.
        tool_count: usize,
        /// Tool-usage constraint, if any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_choice: Option<ToolChoice>,
        /// Requested output format, if any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_format: Option<ResponseFormatKind>,
        /// Output token cap, if set.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_tokens: Option<u32>,
        /// Sampling temperature, if set.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temperature: Option<f32>,
    },
    /// A model produced a completed response.
    LlmResponse {
        /// Correlation id.
        trace_id: TraceId,
        /// Model that produced the response, if reported.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Why generation stopped.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<StopReason>,
        /// Token accounting.
        usage: Usage,
        /// Tool calls the model decided to make.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        /// Total characters of assistant text produced.
        text_len: usize,
    },
    /// A request or stream failed.
    LlmError {
        /// Correlation id.
        trace_id: TraceId,
        /// Error detail.
        message: String,
    },
    /// The model selected a tool to call (a routing decision).
    ToolSelected {
        /// Correlation id.
        trace_id: TraceId,
        /// Call id.
        id: String,
        /// Tool name.
        name: String,
        /// Arguments the model produced.
        args: serde_json::Value,
    },
    /// A selected tool was executed (e.g. inside an agent loop).
    ToolExecuted {
        /// Correlation id.
        trace_id: TraceId,
        /// Call id.
        id: String,
        /// Tool name.
        name: String,
        /// Whether execution succeeded.
        ok: bool,
        /// Length of the tool's output payload.
        output_len: usize,
    },
    /// A retry was attempted.
    Retry {
        /// Correlation id.
        trace_id: TraceId,
        /// 1-based attempt number.
        attempt: u32,
        /// Why the retry happened.
        reason: String,
    },
    /// Execution fell back from one model/strategy to another.
    Fallback {
        /// Correlation id.
        trace_id: TraceId,
        /// What was tried first.
        from: String,
        /// What it fell back to.
        to: String,
        /// Why the fallback happened.
        reason: String,
    },
    /// One step of an agent loop.
    AgentStep {
        /// Correlation id.
        trace_id: TraceId,
        /// 1-based step number.
        step: u32,
        /// A short description of the action taken.
        action: String,
    },
    /// A free-form decision annotation.
    Note {
        /// Correlation id.
        trace_id: TraceId,
        /// The note.
        message: String,
    },
}

impl TraceEvent {
    /// The correlation id this event belongs to.
    pub fn trace_id(&self) -> TraceId {
        match self {
            Self::LlmRequest { trace_id, .. }
            | Self::LlmResponse { trace_id, .. }
            | Self::LlmError { trace_id, .. }
            | Self::ToolSelected { trace_id, .. }
            | Self::ToolExecuted { trace_id, .. }
            | Self::Retry { trace_id, .. }
            | Self::Fallback { trace_id, .. }
            | Self::AgentStep { trace_id, .. }
            | Self::Note { trace_id, .. } => *trace_id,
        }
    }

    fn from_request(trace_id: TraceId, request: &ChatRequest) -> Self {
        Self::LlmRequest {
            trace_id,
            model: request.model.clone(),
            system: request.system.is_some(),
            message_count: request.messages.len(),
            tool_count: request.tools.len(),
            tool_choice: request.tool_choice.clone(),
            response_format: request.response_format.as_ref().map(|r| r.kind()),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        }
    }

    fn from_response(trace_id: TraceId, response: &ChatResponse) -> Self {
        Self::LlmResponse {
            trace_id,
            model: response.model.clone(),
            stop_reason: response.stop_reason.clone(),
            usage: response.usage.clone(),
            tool_calls: response
                .tool_uses()
                .iter()
                .map(|t| ToolCall {
                    id: t.id.to_string(),
                    name: t.name.to_string(),
                })
                .collect(),
            text_len: response.text().len(),
        }
    }
}

/// Receives [`TraceEvent`]s. Implementations must be cheap and non-blocking.
pub trait Tracer: Send + Sync {
    /// Record one event.
    fn record(&self, event: TraceEvent);
}

impl<T: Tracer + ?Sized> Tracer for Arc<T> {
    fn record(&self, event: TraceEvent) {
        (**self).record(event);
    }
}

/// Wraps a closure as a [`Tracer`], so a closure can be a sink without a
/// dedicated struct: `model.traced(Arc::new(fn_tracer(|e| println!("{e:?}"))))`.
pub struct FnTracer<F>(pub F);

impl<F: Fn(TraceEvent) + Send + Sync> Tracer for FnTracer<F> {
    fn record(&self, event: TraceEvent) {
        (self.0)(event);
    }
}

/// Convenience constructor for [`FnTracer`].
pub fn fn_tracer<F: Fn(TraceEvent) + Send + Sync>(f: F) -> FnTracer<F> {
    FnTracer(f)
}

/// A tracer that discards everything (the default).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTracer;

impl Tracer for NoopTracer {
    fn record(&self, _event: TraceEvent) {}
}

/// An in-memory tracer that records events for inspection or later persistence.
///
/// Cloning shares the same underlying buffer, so you can keep one handle to read
/// from while another is handed to a [`Traced`] model.
#[derive(Clone, Default)]
pub struct RecordingTracer {
    events: Arc<Mutex<Vec<TraceEvent>>>,
}

impl RecordingTracer {
    /// Create an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// A snapshot copy of recorded events, in order.
    pub fn events(&self) -> Vec<TraceEvent> {
        self.events.lock().expect("tracer mutex poisoned").clone()
    }

    /// Events for a single correlation id, in order.
    pub fn events_for(&self, trace_id: TraceId) -> Vec<TraceEvent> {
        self.events
            .lock()
            .expect("tracer mutex poisoned")
            .iter()
            .filter(|e| e.trace_id() == trace_id)
            .cloned()
            .collect()
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.lock().expect("tracer mutex poisoned").len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.events
            .lock()
            .expect("tracer mutex poisoned")
            .is_empty()
    }

    /// Take all recorded events, leaving the recorder empty.
    pub fn drain(&self) -> Vec<TraceEvent> {
        std::mem::take(&mut self.events.lock().expect("tracer mutex poisoned"))
    }
}

impl Tracer for RecordingTracer {
    fn record(&self, event: TraceEvent) {
        self.events
            .lock()
            .expect("tracer mutex poisoned")
            .push(event);
    }
}

impl std::fmt::Debug for RecordingTracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingTracer")
            .field("len", &self.len())
            .finish()
    }
}

/// Bridges [`TraceEvent`]s to the [`tracing`](https://docs.rs/tracing) crate.
#[cfg(feature = "tracing")]
#[derive(Debug, Clone, Copy, Default)]
pub struct TracingTracer;

#[cfg(feature = "tracing")]
impl Tracer for TracingTracer {
    fn record(&self, event: TraceEvent) {
        match &event {
            TraceEvent::LlmRequest {
                trace_id,
                model,
                message_count,
                tool_count,
                ..
            } => tracing::info!(
                target: "ai::llm",
                trace_id = trace_id.0,
                model = %model,
                messages = *message_count,
                tools = *tool_count,
                "llm request"
            ),
            TraceEvent::LlmResponse {
                trace_id,
                stop_reason,
                usage,
                tool_calls,
                text_len,
                ..
            } => tracing::info!(
                target: "ai::llm",
                trace_id = trace_id.0,
                stop = ?stop_reason,
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                tool_calls = tool_calls.len(),
                text_len = *text_len,
                "llm response"
            ),
            TraceEvent::LlmError { trace_id, message } => tracing::error!(
                target: "ai::llm",
                trace_id = trace_id.0,
                error = %message,
                "llm error"
            ),
            other => tracing::debug!(
                target: "ai::llm",
                trace_id = other.trace_id().0,
                event = ?other,
                "llm event"
            ),
        }
    }
}

/// A [`ChatModel`] decorator that emits [`TraceEvent`]s to a [`Tracer`].
///
/// Records an [`TraceEvent::LlmRequest`] before each call and an
/// [`TraceEvent::LlmResponse`] (or [`TraceEvent::LlmError`]) after — for both
/// `chat` and `stream` (the streamed summary is emitted when the stream ends).
pub struct Traced<M> {
    inner: M,
    tracer: Arc<dyn Tracer>,
}

impl<M> Traced<M> {
    /// Wrap `inner`, sending trace events to `tracer`.
    pub fn new(inner: M, tracer: Arc<dyn Tracer>) -> Self {
        Self { inner, tracer }
    }

    /// Borrow the wrapped model.
    pub fn inner(&self) -> &M {
        &self.inner
    }
}

impl<M: ChatModel> ChatModel for Traced<M> {
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let trace_id = TraceId::next();
        self.tracer
            .record(TraceEvent::from_request(trace_id, &request));
        match self.inner.stream(request).await {
            Ok(stream) => Ok(ChatStream::new(TracedStream::new(
                stream,
                self.tracer.clone(),
                trace_id,
            ))),
            Err(e) => {
                self.tracer.record(TraceEvent::LlmError {
                    trace_id,
                    message: e.to_string(),
                });
                Err(e)
            }
        }
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let trace_id = TraceId::next();
        self.tracer
            .record(TraceEvent::from_request(trace_id, &request));
        match self.inner.chat(request).await {
            Ok(response) => {
                self.tracer
                    .record(TraceEvent::from_response(trace_id, &response));
                Ok(response)
            }
            Err(e) => {
                self.tracer.record(TraceEvent::LlmError {
                    trace_id,
                    message: e.to_string(),
                });
                Err(e)
            }
        }
    }
}

/// Wraps a [`ChatStream`] to record a response summary when it completes.
struct TracedStream {
    inner: ChatStream,
    tracer: Arc<dyn Tracer>,
    trace_id: TraceId,
    done: bool,
    text_len: usize,
    usage: Usage,
    stop_reason: Option<StopReason>,
    tools: Vec<ToolCall>,
}

impl TracedStream {
    fn new(inner: ChatStream, tracer: Arc<dyn Tracer>, trace_id: TraceId) -> Self {
        Self {
            inner,
            tracer,
            trace_id,
            done: false,
            text_len: 0,
            usage: Usage::default(),
            stop_reason: None,
            tools: Vec::new(),
        }
    }

    fn observe(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::TextDelta(s) => self.text_len += s.len(),
            StreamEvent::Usage(u) => self.usage = u.clone(),
            StreamEvent::Stop(r) => self.stop_reason = Some(r.clone()),
            StreamEvent::ToolCallStart { id, name, .. } => self.tools.push(ToolCall {
                id: id.clone(),
                name: name.clone(),
            }),
            _ => {}
        }
    }

    fn summary(&self) -> TraceEvent {
        TraceEvent::LlmResponse {
            trace_id: self.trace_id,
            model: None,
            stop_reason: self.stop_reason.clone(),
            usage: self.usage.clone(),
            tool_calls: self.tools.clone(),
            text_len: self.text_len,
        }
    }
}

impl Stream for TracedStream {
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => {
                this.observe(&event);
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(e))) => {
                if !this.done {
                    this.done = true;
                    let message = e.to_string();
                    this.tracer.record(TraceEvent::LlmError {
                        trace_id: this.trace_id,
                        message,
                    });
                }
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if !this.done {
                    this.done = true;
                    this.tracer.record(this.summary());
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
