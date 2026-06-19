//! Conversation memory: owned chat state plus pluggable persistence.
//!
//! [`ChatHistory`] is just an owned `Vec<Message>` with ergonomic helpers — you
//! own it, there is no memory-class hierarchy. [`ChatStore`] is the optional
//! persistence trait (save/resume conversations by session id), mirroring
//! [`TraceStore`](crate::memory::TraceStore): async, object-safe via
//! [`DynChatStore`], with [`InMemoryChatStore`] as the built-in backend.
//!
//! ```
//! use ai_core::ChatHistory;
//!
//! let mut history = ChatHistory::new();
//! history.user("Hi!");
//! // let response = model.chat(history.to_request("llama3.1").build()?).await?;
//! // history.record_response(&response);
//! assert_eq!(history.len(), 1);
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    message::Message,
    request::{ChatRequest, ChatRequestBuilder},
    response::ChatResponse,
    BoxFuture,
};

/// An owned, ordered conversation transcript.
///
/// It is `serde`-serializable, so you can persist it directly to any store you
/// choose (or serialize [`messages`](Self::messages) for a plain array). See the
/// "Persistence" section of `USAGE.md` for the bring-your-own-storage patterns.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatHistory {
    messages: Vec<Message>,
}

impl ChatHistory {
    /// An empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// A history seeded from existing messages (e.g. loaded from a [`ChatStore`]).
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Append a message.
    pub fn push(&mut self, message: Message) -> &mut Self {
        self.messages.push(message);
        self
    }

    /// Append a user text message.
    pub fn user(&mut self, text: impl Into<String>) -> &mut Self {
        self.push(Message::user(text))
    }

    /// Append an assistant text message.
    pub fn assistant(&mut self, text: impl Into<String>) -> &mut Self {
        self.push(Message::assistant(text))
    }

    /// Append several messages.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) -> &mut Self {
        self.messages.extend(messages);
        self
    }

    /// Append the assistant message from a model response.
    pub fn record_response(&mut self, response: &ChatResponse) -> &mut Self {
        self.push(response.message.clone())
    }

    /// Start a [`ChatRequest`] for `model`, pre-loaded with this history's
    /// messages. Add `system`/`max_tokens`/etc., then `build()`.
    pub fn to_request(&self, model: impl Into<String>) -> ChatRequestBuilder {
        ChatRequest::builder(model).messages(self.messages.iter().cloned())
    }

    /// The messages, in order.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Consume the history, returning its messages.
    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }

    /// The most recent message, if any.
    pub fn last(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// Remove all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Pluggable persistence for conversations, keyed by session id.
///
/// Hot-path trait using native `async fn`-in-traits (RPITIT); for boxed,
/// runtime-selected backends use [`DynChatStore`].
pub trait ChatStore: Send + Sync {
    /// Load a session's messages (empty if the session is unknown).
    fn load(&self, session: &str) -> impl Future<Output = Result<Vec<Message>>> + Send;

    /// Replace a session's messages.
    fn save(
        &self,
        session: &str,
        messages: Vec<Message>,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Append one message to a session.
    fn append(&self, session: &str, message: Message) -> impl Future<Output = Result<()>> + Send;

    /// Append several messages. Defaults to appending one at a time; backends
    /// should override for a single batched write.
    fn append_many(
        &self,
        session: &str,
        messages: Vec<Message>,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            for message in messages {
                self.append(session, message).await?;
            }
            Ok(())
        }
    }

    /// Delete a session.
    fn clear(&self, session: &str) -> impl Future<Output = Result<()>> + Send;
}

/// Object-safe facade over [`ChatStore`] for boxed/pluggable backends.
pub trait DynChatStore: Send + Sync {
    /// See [`ChatStore::load`].
    fn load_boxed<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<Vec<Message>>>;
    /// See [`ChatStore::save`].
    fn save_boxed<'a>(
        &'a self,
        session: &'a str,
        messages: Vec<Message>,
    ) -> BoxFuture<'a, Result<()>>;
    /// See [`ChatStore::append`].
    fn append_boxed<'a>(&'a self, session: &'a str, message: Message) -> BoxFuture<'a, Result<()>>;
    /// See [`ChatStore::clear`].
    fn clear_boxed<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<()>>;
}

impl<T: ChatStore> DynChatStore for T {
    fn load_boxed<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<Vec<Message>>> {
        Box::pin(self.load(session))
    }

    fn save_boxed<'a>(
        &'a self,
        session: &'a str,
        messages: Vec<Message>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.save(session, messages))
    }

    fn append_boxed<'a>(&'a self, session: &'a str, message: Message) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.append(session, message))
    }

    fn clear_boxed<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.clear(session))
    }
}

/// An in-memory [`ChatStore`]. Cloning shares the same sessions.
#[derive(Clone, Default, Debug)]
pub struct InMemoryChatStore {
    sessions: Arc<Mutex<HashMap<String, Vec<Message>>>>,
}

impl InMemoryChatStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored sessions.
    pub fn session_count(&self) -> usize {
        self.sessions
            .lock()
            .expect("chat store mutex poisoned")
            .len()
    }
}

impl ChatStore for InMemoryChatStore {
    fn load(&self, session: &str) -> impl Future<Output = Result<Vec<Message>>> + Send {
        let sessions = self.sessions.clone();
        let session = session.to_string();
        async move {
            Ok(sessions
                .lock()
                .expect("chat store mutex poisoned")
                .get(&session)
                .cloned()
                .unwrap_or_default())
        }
    }

    fn save(
        &self,
        session: &str,
        messages: Vec<Message>,
    ) -> impl Future<Output = Result<()>> + Send {
        let sessions = self.sessions.clone();
        let session = session.to_string();
        async move {
            sessions
                .lock()
                .expect("chat store mutex poisoned")
                .insert(session, messages);
            Ok(())
        }
    }

    fn append(&self, session: &str, message: Message) -> impl Future<Output = Result<()>> + Send {
        let sessions = self.sessions.clone();
        let session = session.to_string();
        async move {
            sessions
                .lock()
                .expect("chat store mutex poisoned")
                .entry(session)
                .or_default()
                .push(message);
            Ok(())
        }
    }

    fn append_many(
        &self,
        session: &str,
        messages: Vec<Message>,
    ) -> impl Future<Output = Result<()>> + Send {
        let sessions = self.sessions.clone();
        let session = session.to_string();
        async move {
            sessions
                .lock()
                .expect("chat store mutex poisoned")
                .entry(session)
                .or_default()
                .extend(messages);
            Ok(())
        }
    }

    fn clear(&self, session: &str) -> impl Future<Output = Result<()>> + Send {
        let sessions = self.sessions.clone();
        let session = session.to_string();
        async move {
            sessions
                .lock()
                .expect("chat store mutex poisoned")
                .remove(&session);
            Ok(())
        }
    }
}
