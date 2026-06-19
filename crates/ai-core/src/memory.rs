//! Persistence scaffolding ("memory") for traceability.
//!
//! A [`Tracer`](crate::Tracer) captures LLM decisions *during* a run; a
//! [`TraceStore`] persists them *beyond* it. [`InMemoryTraceStore`] is the
//! built-in backend; downstream code implements [`TraceStore`] for
//! Redis/Postgres/files/etc. [`persist_recording`] flushes a
//! [`RecordingTracer`] into any store.
//!
//! This is the same shape the conversation-memory layer (`ChatStore`) will take
//! in a later phase: an async, pluggable, object-safe append-and-load trait with
//! an in-memory default.

use std::future::Future;
use std::sync::{Arc, Mutex};

use crate::{
    error::Result,
    trace::{RecordingTracer, TraceEvent, TraceId},
    BoxFuture,
};

/// A durable sink for [`TraceEvent`]s.
///
/// Hot-path trait using native `async fn`-in-traits (RPITIT); for boxed,
/// runtime-selected backends use [`DynTraceStore`].
pub trait TraceStore: Send + Sync {
    /// Append a single event.
    fn append(&self, event: TraceEvent) -> impl Future<Output = Result<()>> + Send;

    /// Append many events. Defaults to appending one at a time; backends should
    /// override for a single batched write.
    fn append_batch(&self, events: Vec<TraceEvent>) -> impl Future<Output = Result<()>> + Send {
        async move {
            for event in events {
                self.append(event).await?;
            }
            Ok(())
        }
    }

    /// Load all events for a correlation id, in insertion order.
    fn load(&self, trace_id: TraceId) -> impl Future<Output = Result<Vec<TraceEvent>>> + Send;

    /// Load every stored event, in insertion order.
    fn all(&self) -> impl Future<Output = Result<Vec<TraceEvent>>> + Send;
}

/// Object-safe facade over [`TraceStore`] for boxed/pluggable backends.
pub trait DynTraceStore: Send + Sync {
    /// See [`TraceStore::append`].
    fn append_boxed<'a>(&'a self, event: TraceEvent) -> BoxFuture<'a, Result<()>>;
    /// See [`TraceStore::load`].
    fn load_boxed<'a>(&'a self, trace_id: TraceId) -> BoxFuture<'a, Result<Vec<TraceEvent>>>;
    /// See [`TraceStore::all`].
    fn all_boxed<'a>(&'a self) -> BoxFuture<'a, Result<Vec<TraceEvent>>>;
}

impl<T: TraceStore> DynTraceStore for T {
    fn append_boxed<'a>(&'a self, event: TraceEvent) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.append(event))
    }

    fn load_boxed<'a>(&'a self, trace_id: TraceId) -> BoxFuture<'a, Result<Vec<TraceEvent>>> {
        Box::pin(self.load(trace_id))
    }

    fn all_boxed<'a>(&'a self) -> BoxFuture<'a, Result<Vec<TraceEvent>>> {
        Box::pin(self.all())
    }
}

/// An in-memory [`TraceStore`]. Cloning shares the same buffer.
#[derive(Clone, Default, Debug)]
pub struct InMemoryTraceStore {
    events: Arc<Mutex<Vec<TraceEvent>>>,
}

impl InMemoryTraceStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored events.
    pub fn len(&self) -> usize {
        self.events
            .lock()
            .expect("trace store mutex poisoned")
            .len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.events
            .lock()
            .expect("trace store mutex poisoned")
            .is_empty()
    }

    /// A synchronous snapshot copy of every stored event.
    pub fn snapshot(&self) -> Vec<TraceEvent> {
        self.events
            .lock()
            .expect("trace store mutex poisoned")
            .clone()
    }
}

impl TraceStore for InMemoryTraceStore {
    fn append(&self, event: TraceEvent) -> impl Future<Output = Result<()>> + Send {
        let events = self.events.clone();
        async move {
            events
                .lock()
                .expect("trace store mutex poisoned")
                .push(event);
            Ok(())
        }
    }

    fn append_batch(&self, batch: Vec<TraceEvent>) -> impl Future<Output = Result<()>> + Send {
        let events = self.events.clone();
        async move {
            events
                .lock()
                .expect("trace store mutex poisoned")
                .extend(batch);
            Ok(())
        }
    }

    fn load(&self, trace_id: TraceId) -> impl Future<Output = Result<Vec<TraceEvent>>> + Send {
        let events = self.events.clone();
        async move {
            let guard = events.lock().expect("trace store mutex poisoned");
            Ok(guard
                .iter()
                .filter(|e| e.trace_id() == trace_id)
                .cloned()
                .collect())
        }
    }

    fn all(&self) -> impl Future<Output = Result<Vec<TraceEvent>>> + Send {
        let events = self.events.clone();
        async move { Ok(events.lock().expect("trace store mutex poisoned").clone()) }
    }
}

/// Flush everything a [`RecordingTracer`] has captured into a [`TraceStore`],
/// draining the recorder.
pub async fn persist_recording(tracer: &RecordingTracer, store: &impl TraceStore) -> Result<()> {
    store.append_batch(tracer.drain()).await
}
