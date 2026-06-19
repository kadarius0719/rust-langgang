//! Integration tests for the extensibility/middleware surface.

use std::future::Future;
use std::sync::{Arc, Mutex};

use ai_core::{
    ApiErrorKind, ChatModel, ChatModelExt, ChatRequest, ChatStream, Error, FnTool, Message, Result,
    StreamEvent, Tool, ToolDef,
};

/// Echoes the request's model + max_tokens back as the response text, so tests
/// can observe how middleware reshaped the request.
#[derive(Clone)]
struct EchoModel;

impl ChatModel for EchoModel {
    fn stream(&self, request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        let text = format!("model={} max={:?}", request.model, request.max_tokens);
        async move {
            Ok(ChatStream::new(futures::stream::iter(vec![
                Ok::<_, Error>(StreamEvent::TextDelta(text)),
                Ok(StreamEvent::MessageStop),
            ])))
        }
    }
}

#[derive(Clone)]
struct FailingModel;

impl ChatModel for FailingModel {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream> {
        Err(Error::provider("x", ApiErrorKind::ServerError, "down"))
    }
}

fn req() -> ChatRequest {
    ChatRequest::builder("m").user("hi").build().unwrap()
}

#[tokio::test]
async fn map_request_injects_params() {
    let model = EchoModel.map_request(|r| {
        r.max_tokens.get_or_insert(512);
    });
    let response = model.chat(req()).await.unwrap();
    assert!(
        response.text().contains("max=Some(512)"),
        "{}",
        response.text()
    );
}

#[tokio::test]
async fn map_response_transforms_output() {
    let model = EchoModel.map_response(|r| {
        let upper = r.text().to_uppercase();
        let mut r = r;
        r.message = Message::assistant(upper);
        r
    });
    let response = model.chat(req()).await.unwrap();
    assert!(
        response.text().starts_with("MODEL=M"),
        "{}",
        response.text()
    );
}

#[tokio::test]
async fn with_fallback_recovers_from_error() {
    let model = FailingModel.with_fallback(EchoModel);
    let response = model.chat(req()).await.unwrap();
    assert!(response.text().contains("model=m"), "{}", response.text());
}

#[tokio::test]
async fn combinators_stack() {
    // Compose several, including the tracer, and confirm the stack is a ChatModel.
    let model = FailingModel.with_fallback(EchoModel).map_request(|r| {
        r.max_tokens.get_or_insert(8);
    });
    let response = model.chat(req()).await.unwrap();
    assert!(
        response.text().contains("max=Some(8)"),
        "{}",
        response.text()
    );
}

#[tokio::test]
async fn arc_model_is_a_chat_model_and_shareable() {
    let model = Arc::new(EchoModel);
    let response = model.chat(req()).await.unwrap();
    assert!(response.text().contains("model=m"));

    let clone = model.clone(); // cheap share into another task/decorator
    let wrapped = clone.with_fallback(EchoModel);
    let _ = wrapped.chat(req()).await.unwrap();
}

#[tokio::test]
async fn fn_tool_from_closure() {
    let calls = Arc::new(Mutex::new(0));
    let counter = calls.clone();
    let tool = FnTool::new(
        ToolDef::new(
            "noop",
            "does nothing",
            serde_json::json!({"type": "object"}),
        ),
        move |args: serde_json::Value| {
            let counter = counter.clone();
            async move {
                *counter.lock().unwrap() += 1;
                Ok(args)
            }
        },
    );

    assert_eq!(tool.def().name, "noop");
    let out = tool.invoke(serde_json::json!({"k": 1})).await.unwrap();
    assert_eq!(out["k"], 1);
    assert_eq!(*calls.lock().unwrap(), 1);
}
