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

    /// A stream of a single event.
    ///
    /// ```
    /// use ai_core::{ChatStream, StreamEvent};
    /// let stream = ChatStream::once(StreamEvent::TextDelta("hi".into()));
    /// ```
    pub fn once(event: StreamEvent) -> Self {
        Self::from_events([event])
    }

    /// A stream replaying a fixed sequence of events.
    ///
    /// The caller needs no `futures` dependency to build a stream — handy for
    /// tests and for custom/in-process [`ChatModel`](crate::ChatModel)s. For a
    /// mid-stream error, use [`new`](Self::new) with your own fallible stream.
    ///
    /// ```
    /// use ai_core::{ChatStream, StreamEvent};
    /// let stream = ChatStream::from_events([
    ///     StreamEvent::MessageStart,
    ///     StreamEvent::TextDelta("hello".into()),
    ///     StreamEvent::MessageStop,
    /// ]);
    /// ```
    pub fn from_events<I>(events: I) -> Self
    where
        I: IntoIterator<Item = StreamEvent>,
        I::IntoIter: Send + 'static,
    {
        Self::new(futures::stream::iter(
            events.into_iter().map(|event| -> Result<StreamEvent> { Ok(event) }),
        ))
    }

    /// Replay a complete [`ChatResponse`] as a stream of events.
    ///
    /// The inverse of [`collect_response`](Self::collect_response): a backend
    /// that only produces a finished response can still satisfy
    /// [`ChatModel::stream`](crate::ChatModel::stream) by replaying it, and
    /// `from_response(r).collect_response()` reproduces `r`.
    pub fn from_response(response: ChatResponse) -> Self {
        let mut events = vec![StreamEvent::MessageStart];
        let mut tool_index = 0usize;
        for block in response.message.content {
            match block {
                ContentBlock::Text { text } => events.push(StreamEvent::TextDelta(text)),
                ContentBlock::Thinking { text, .. } => {
                    events.push(StreamEvent::ThinkingDelta(text));
                }
                ContentBlock::ToolUse { id, name, args } => {
                    events.push(StreamEvent::ToolCallStart {
                        index: tool_index,
                        id,
                        name,
                    });
                    events.push(StreamEvent::ToolCallArgsDelta {
                        index: tool_index,
                        delta: args.to_string(),
                    });
                    tool_index += 1;
                }
                _ => {}
            }
        }
        if response.usage != Usage::default() {
            events.push(StreamEvent::Usage(response.usage));
        }
        if let Some(reason) = response.stop_reason {
            events.push(StreamEvent::Stop(reason));
        }
        events.push(StreamEvent::MessageStop);
        Self::from_events(events)
    }

    /// Pull the next event, or `None` at end of stream.
    ///
    /// An inherent convenience so callers can drive a stream token-by-token
    /// without importing `futures::StreamExt`. ([`ChatStream`] also implements
    /// [`futures::Stream`], so the full `StreamExt` combinator set remains
    /// available to callers who want it.)
    ///
    /// ```ignore
    /// let mut stream = model.stream(request).await?;
    /// while let Some(event) = stream.next().await {
    ///     if let StreamEvent::TextDelta(text) = event? {
    ///         print!("{text}");
    ///     }
    /// }
    /// ```
    pub async fn next(&mut self) -> Option<Result<StreamEvent>> {
        self.inner.next().await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn from_events_needs_no_futures_dep() {
        let response = ChatStream::from_events([
            StreamEvent::MessageStart,
            StreamEvent::TextDelta("hel".into()),
            StreamEvent::TextDelta("lo".into()),
            StreamEvent::MessageStop,
        ])
        .collect_response()
        .await
        .unwrap();
        assert_eq!(response.text(), "hello");
    }

    #[tokio::test]
    async fn next_drives_stream_without_streamext_import() {
        // Note: this module does not import `futures::StreamExt`, so `next`
        // here resolves to the inherent method.
        let mut stream = ChatStream::from_events([
            StreamEvent::TextDelta("a".into()),
            StreamEvent::TextDelta("b".into()),
        ]);
        let mut seen = String::new();
        while let Some(event) = stream.next().await {
            if let StreamEvent::TextDelta(s) = event.unwrap() {
                seen.push_str(&s);
            }
        }
        assert_eq!(seen, "ab");
    }

    #[tokio::test]
    async fn once_yields_single_event() {
        let response = ChatStream::once(StreamEvent::TextDelta("hi".into()))
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text(), "hi");
    }

    #[tokio::test]
    async fn from_response_round_trips_through_collect() {
        let original = ChatResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::text("the answer is"),
                    ContentBlock::ToolUse {
                        id: "call_1".into(),
                        name: "get_weather".into(),
                        args: serde_json::json!({"city": "Paris"}),
                    },
                ],
            },
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Usage::default()
            },
            stop_reason: Some(StopReason::ToolUse),
            model: None,
            raw: serde_json::Value::Null,
        };

        let replayed = ChatStream::from_response(original.clone())
            .collect_response()
            .await
            .unwrap();

        assert_eq!(replayed, original);
    }
}
