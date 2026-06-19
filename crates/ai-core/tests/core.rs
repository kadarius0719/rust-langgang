//! Integration tests exercising the trait surface with an in-memory mock model.
//!
//! The mock implements [`ChatModel`] by replaying a scripted list of
//! [`StreamEvent`]s — the same path every real adapter will use — so these
//! tests pin down stream accumulation, tool-call assembly, and the
//! erased/typed `DynChatModel` bridge without any network.

use std::future::Future;

use ai_core::{
    ChatModel, ChatRequest, ChatStream, DynChatModel, Error, Result, StopReason, StreamEvent, Usage,
};

#[derive(Clone)]
struct MockModel {
    events: Vec<StreamEvent>,
}

impl ChatModel for MockModel {
    fn stream(&self, _request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        let events = self.events.clone();
        async move {
            let stream = futures::stream::iter(events.into_iter().map(Ok::<StreamEvent, Error>));
            Ok(ChatStream::new(stream))
        }
    }
}

fn scripted() -> MockModel {
    MockModel {
        events: vec![
            StreamEvent::MessageStart,
            StreamEvent::ThinkingDelta("hmm ".into()),
            StreamEvent::TextDelta("Hello, ".into()),
            StreamEvent::TextDelta("world".into()),
            StreamEvent::ToolCallStart {
                index: 0,
                id: "call_1".into(),
                name: "get_weather".into(),
            },
            StreamEvent::ToolCallArgsDelta {
                index: 0,
                delta: "{\"city\":".into(),
            },
            StreamEvent::ToolCallArgsDelta {
                index: 0,
                delta: "\"Paris\"}".into(),
            },
            StreamEvent::Usage(Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            }),
            StreamEvent::Stop(StopReason::ToolUse),
            StreamEvent::MessageStop,
        ],
    }
}

#[tokio::test]
async fn chat_accumulates_text_thinking_tools_and_usage() {
    let model = scripted();
    let request = ChatRequest::builder("mock").user("hi").build().unwrap();

    let response = model.chat(request).await.unwrap();

    assert_eq!(response.text(), "Hello, world");
    assert_eq!(response.usage.input_tokens, 10);
    assert_eq!(response.usage.output_tokens, 5);
    assert_eq!(response.usage.total(), 15);
    assert_eq!(response.stop_reason, Some(StopReason::ToolUse));

    let tools = response.tool_uses();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].id, "call_1");
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(tools[0].args["city"], "Paris");
    assert!(response.has_tool_calls());
}

#[tokio::test]
async fn stream_yields_events_in_order() {
    use futures::StreamExt;

    let model = scripted();
    let request = ChatRequest::builder("mock").user("hi").build().unwrap();

    let stream = model.stream(request).await.unwrap();
    let events: Vec<StreamEvent> = stream.map(|e| e.unwrap()).collect().await;

    assert_eq!(events.first(), Some(&StreamEvent::MessageStart));
    assert_eq!(events.last(), Some(&StreamEvent::MessageStop));
    assert!(events.contains(&StreamEvent::TextDelta("world".into())));
}

#[tokio::test]
async fn dyn_model_bridges_back_to_typed() {
    // Erase to a trait object, then use it through the `ChatModel` impl for
    // `Box<dyn DynChatModel>` — the runtime-provider-selection path.
    let boxed: Box<dyn DynChatModel> = Box::new(scripted());
    let request = ChatRequest::builder("mock").user("hi").build().unwrap();

    let response = boxed.chat(request).await.unwrap();
    assert_eq!(response.text(), "Hello, world");
    assert_eq!(response.tool_uses().len(), 1);
}

#[tokio::test]
async fn mid_stream_error_surfaces() {
    use futures::StreamExt;

    let events = vec![
        Ok(StreamEvent::TextDelta("partial".into())),
        Err(Error::stream("connection reset")),
    ];
    let stream = ChatStream::new(futures::stream::iter(events));

    let collected: Vec<Result<StreamEvent>> = stream.collect().await;
    assert!(matches!(collected.last(), Some(Err(Error::Stream(_)))));
}
