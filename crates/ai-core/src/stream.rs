//! Streaming primitives.
//!
//! [`ChatStream`] is the public streaming type — a pinned, boxed stream of
//! [`StreamEvent`]s normalized across every provider. It is the *primitive*:
//! [`crate::ChatModel::chat`] is implemented by accumulating a `ChatStream` into
//! a [`ChatResponse`]. Mid-stream errors are first-class (`Item = Result<…>`).

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::stream::{Stream, StreamExt};

use crate::{
    error::Result,
    message::{ContentBlock, Message, Role},
    response::{ChatResponse, StopReason, Usage},
};

/// A normalized streaming event.
///
/// Adapters fold each provider's wire events (OpenAI homogeneous chunks,
/// Anthropic block frames, Ollama NDJSON, …) into this single vocabulary.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StreamEvent {
    /// The response message has begun.
    MessageStart,
    /// A chunk of assistant text.
    TextDelta(String),
    /// A chunk of thinking/reasoning text.
    ThinkingDelta(String),
    /// A tool call begins at `index` with its id and name.
    ToolCallStart {
        /// Position of this call in the response.
        index: usize,
        /// Provider-assigned call id.
        id: String,
        /// Tool name.
        name: String,
    },
    /// A chunk of the JSON arguments for the tool call at `index`.
    ToolCallArgsDelta {
        /// Position of the call these args belong to.
        index: usize,
        /// Raw JSON fragment to append.
        delta: String,
    },
    /// Token usage (may arrive mid- or end-of-stream).
    Usage(Usage),
    /// The model's stop reason.
    Stop(StopReason),
    /// The response message has ended.
    MessageStop,
}

/// A stream of [`StreamEvent`]s from a model.
pub struct ChatStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>,
}

impl ChatStream {
    /// Wrap any `Send` stream of events.
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamEvent>> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Drain the stream and accumulate it into a complete [`ChatResponse`].
    ///
    /// This is how [`crate::ChatModel::chat`] is built atop `stream`.
    pub async fn collect_response(mut self) -> Result<ChatResponse> {
        let mut acc = ResponseAccumulator::default();
        while let Some(event) = self.inner.next().await {
            acc.push(event?);
        }
        Ok(acc.finish())
    }
}

impl std::fmt::Debug for ChatStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatStream").finish_non_exhaustive()
    }
}

impl Stream for ChatStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// Folds streamed events into a [`ChatResponse`].
#[derive(Default)]
struct ResponseAccumulator {
    thinking: String,
    thinking_signature: Option<String>,
    text: String,
    tools: Vec<PartialTool>,
    usage: Usage,
    stop_reason: Option<StopReason>,
}

struct PartialTool {
    index: usize,
    id: String,
    name: String,
    args: String,
}

impl ResponseAccumulator {
    fn push(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::TextDelta(s) => self.text.push_str(&s),
            StreamEvent::ThinkingDelta(s) => self.thinking.push_str(&s),
            StreamEvent::ToolCallStart { index, id, name } => self.tools.push(PartialTool {
                index,
                id,
                name,
                args: String::new(),
            }),
            StreamEvent::ToolCallArgsDelta { index, delta } => {
                if let Some(tool) = self.tools.iter_mut().find(|t| t.index == index) {
                    tool.args.push_str(&delta);
                }
            }
            StreamEvent::Usage(usage) => self.usage = usage,
            StreamEvent::Stop(reason) => self.stop_reason = Some(reason),
            StreamEvent::MessageStart | StreamEvent::MessageStop => {}
        }
    }

    fn finish(self) -> ChatResponse {
        let mut content = Vec::new();
        if !self.thinking.is_empty() {
            content.push(ContentBlock::Thinking {
                text: self.thinking,
                signature: self.thinking_signature,
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        for tool in self.tools {
            let args = serde_json::from_str(&tool.args)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
            content.push(ContentBlock::ToolUse {
                id: tool.id,
                name: tool.name,
                args,
            });
        }

        ChatResponse {
            message: Message {
                role: Role::Assistant,
                content,
            },
            usage: self.usage,
            stop_reason: self.stop_reason,
            model: None,
            raw: serde_json::Value::Null,
        }
    }
}
