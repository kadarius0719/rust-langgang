//! Integration tests for LLM-decision traceability and the memory store.

use std::future::Future;
use std::sync::Arc;

use ai_core::{
    ApiErrorKind, ChatModel, ChatModelExt, ChatRequest, ChatStream, Error, InMemoryTraceStore,
    RecordingTracer, Result, StopReason, StreamEvent, TraceEvent, TraceStore, Usage,
};

#[derive(Clone)]
struct MockModel {
    events: Vec<StreamEvent>,
}

impl ChatModel for MockModel {
    fn stream(&self, _request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        let events = self.events.clone();
        async move {
            Ok(ChatStream::new(futures::stream::iter(
                events.into_iter().map(Ok::<_, Error>),
            )))
        }
    }
}

#[derive(Clone)]
struct FailingModel;

impl ChatModel for FailingModel {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream> {
        Err(Error::provider("mock", ApiErrorKind::ServerError, "boom"))
    }
}

fn scripted() -> MockModel {
    MockModel {
        events: vec![
            StreamEvent::TextDelta("Hi ".into()),
            StreamEvent::TextDelta("there".into()),
            StreamEvent::ToolCallStart {
                index: 0,
                id: "c1".into(),
                name: "lookup".into(),
            },
            StreamEvent::Usage(Usage {
                input_tokens: 7,
                output_tokens: 3,
                ..Default::default()
            }),
            StreamEvent::Stop(StopReason::ToolUse),
        ],
    }
}

fn request() -> ChatRequest {
    ChatRequest::builder("mock")
        .system("s")
        .user("hi")
        .build()
        .unwrap()
}

#[tokio::test]
async fn chat_records_request_and_response() {
    let tracer = RecordingTracer::new();
    let model = scripted().traced(Arc::new(tracer.clone()));

    let response = model.chat(request()).await.unwrap();
    assert_eq!(response.text(), "Hi there");

    let events = tracer.events();
    assert_eq!(events.len(), 2);
    assert!(matches!(
        events[0],
        TraceEvent::LlmRequest {
            system: true,
            message_count: 1,
            ..
        }
    ));
    match &events[1] {
        TraceEvent::LlmResponse {
            stop_reason,
            usage,
            tool_calls,
            text_len,
            ..
        } => {
            assert_eq!(*stop_reason, Some(StopReason::ToolUse));
            assert_eq!(usage.input_tokens, 7);
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].name, "lookup");
            assert_eq!(*text_len, "Hi there".len());
        }
        other => panic!("expected LlmResponse, got {other:?}"),
    }
    // Request and response are correlated by the same trace id.
    assert_eq!(events[0].trace_id(), events[1].trace_id());
}

#[tokio::test]
async fn stream_records_summary_on_completion() {
    use futures::StreamExt;

    let tracer = RecordingTracer::new();
    let model = scripted().traced(Arc::new(tracer.clone()));

    let stream = model.stream(request()).await.unwrap();
    let collected: Vec<_> = stream.collect().await;
    assert!(collected.iter().all(|e| e.is_ok()));

    let events = tracer.events();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], TraceEvent::LlmRequest { .. }));
    match &events[1] {
        TraceEvent::LlmResponse {
            stop_reason,
            tool_calls,
            text_len,
            ..
        } => {
            assert_eq!(*stop_reason, Some(StopReason::ToolUse));
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(*text_len, "Hi there".len());
        }
        other => panic!("expected LlmResponse summary, got {other:?}"),
    }
}

#[tokio::test]
async fn errors_are_traced() {
    let tracer = RecordingTracer::new();
    let model = FailingModel.traced(Arc::new(tracer.clone()));

    assert!(model.chat(request()).await.is_err());

    let events = tracer.events();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], TraceEvent::LlmRequest { .. }));
    assert!(matches!(events[1], TraceEvent::LlmError { .. }));
}

#[tokio::test]
async fn recording_flushes_to_store() {
    let tracer = RecordingTracer::new();
    let model = scripted().traced(Arc::new(tracer.clone()));
    let _ = model.chat(request()).await.unwrap();

    let store = InMemoryTraceStore::new();
    ai_core::memory::persist_recording(&tracer, &store)
        .await
        .unwrap();

    assert_eq!(store.len(), 2);
    assert!(tracer.is_empty(), "recorder should be drained after flush");

    let all = store.all().await.unwrap();
    let trace_id = all[0].trace_id();
    let by_id = store.load(trace_id).await.unwrap();
    assert_eq!(by_id.len(), 2);
}
