//! End-to-end "consumer journey" smoke test.
//!
//! Mirrors how a first-time downstream user wires ai-core together, asserting
//! each surface works offline: the request builder, the Runnable layer
//! (`from_fn` + `parallel`), conversation history + a `ChatStore`, a `FnTool` in
//! a `ToolBox`, and a mock `ChatModel` driving `chat`, token-by-token streaming,
//! an `Agent` tool-loop, and `structured` output.
//!
//! It deliberately uses the consumer-facing constructors `ChatStream::from_events`
//! and `ChatStream::next` (no `futures` dependency, no `StreamExt` import), so a
//! regression in that ergonomics would fail here. Requires the `schema` feature
//! for the structured-output step.

use std::future::Future;

use ai_core::{
    from_fn, parallel, Agent, ChatHistory, ChatModel, ChatRequest, ChatResponse, ChatStore,
    ChatStream, Error, FnTool, InMemoryChatStore, Message, Result, Role, Runnable, RunnableExt,
    StopReason, StreamEvent, StructuredExt, ToolBox, ToolDef,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// A mock model built with the consumer-facing `ChatStream::from_events` (no
/// `futures` dependency). Emits a tool call when tools are advertised and no
/// tool result is present yet; otherwise streams its fixed `reply`.
struct MockModel {
    reply: String,
}

impl ChatModel for MockModel {
    fn stream(&self, request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        let reply = self.reply.clone();
        async move {
            let wants_tool =
                !request.tools.is_empty() && !request.messages.iter().any(|m| m.role == Role::Tool);
            let events = if wants_tool {
                let name = request.tools[0].name.clone();
                vec![
                    StreamEvent::MessageStart,
                    StreamEvent::ToolCallStart {
                        index: 0,
                        id: "call_1".into(),
                        name,
                    },
                    StreamEvent::ToolCallArgsDelta {
                        index: 0,
                        delta: r#"{"city":"Paris"}"#.into(),
                    },
                    StreamEvent::Stop(StopReason::ToolUse),
                    StreamEvent::MessageStop,
                ]
            } else {
                vec![
                    StreamEvent::MessageStart,
                    StreamEvent::TextDelta(reply),
                    StreamEvent::Stop(StopReason::EndTurn),
                    StreamEvent::MessageStop,
                ]
            };
            Ok(ChatStream::from_events(events))
        }
    }
}

/// A `Runnable` multiplying its input by `factor`; one `impl Runnable` type so
/// several instances can share a `Vec` for `parallel`.
fn scaler(factor: i64) -> impl Runnable<In = i64, Out = i64> {
    from_fn(move |n: i64| async move { Ok::<_, Error>(n * factor) })
}

/// A small derived type for the structured-output step.
#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

#[test]
fn request_builder_validates_and_builds() {
    let request = ChatRequest::builder("demo-model")
        .system("You are concise.")
        .user("Say hi.")
        .max_tokens(64)
        .build()
        .unwrap();
    assert_eq!(request.model, "demo-model");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.system.as_deref(), Some("You are concise."));
    assert_eq!(request.max_tokens, Some(64));

    // The two real invariants are enforced.
    assert!(ChatRequest::builder("m").build().is_err());
    assert!(ChatRequest::builder("   ").user("hi").build().is_err());
}

#[tokio::test]
async fn runnable_pipeline_and_parallel_fan_out() {
    // from_fn (parse) -> then -> from_fn (double)
    let pipeline = from_fn(|s: String| async move {
        s.trim()
            .parse::<i64>()
            .map_err(|e| Error::invalid_request(e.to_string()))
    })
    .then(from_fn(|n: i64| async move { Ok::<_, Error>(n * 2) }));
    assert_eq!(pipeline.invoke("21".to_string()).await.unwrap(), 42);

    // homogeneous parallel fan-out
    let fan = parallel(vec![scaler(2), scaler(3), scaler(10)]);
    assert_eq!(fan.invoke(7).await.unwrap(), vec![14, 21, 70]);
}

#[tokio::test]
async fn history_and_store_round_trip() {
    let mut history = ChatHistory::new();
    history.user("What is 2 + 2?");
    let answer = ChatResponse::from_message(Message::assistant("4"));
    history.record_response(&answer);

    assert_eq!(history.len(), 2);
    assert_eq!(
        history
            .to_request("demo-model")
            .build()
            .unwrap()
            .messages
            .len(),
        2
    );

    let store = InMemoryChatStore::new();
    store
        .save("session-1", history.messages().to_vec())
        .await
        .unwrap();
    let loaded = store.load("session-1").await.unwrap();
    assert_eq!(store.session_count(), 1);
    assert_eq!(loaded, history.messages());
}

#[tokio::test]
async fn toolbox_invoke_by_name() {
    let mut tools = ToolBox::new();
    tools.add(FnTool::new(
        ToolDef::new(
            "add",
            "Add two integers a and b.",
            json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}),
        ),
        |args: serde_json::Value| async move {
            let a = args["a"].as_i64().unwrap_or(0);
            let b = args["b"].as_i64().unwrap_or(0);
            Ok::<_, Error>(json!({ "sum": a + b }))
        },
    ));

    assert_eq!(tools.len(), 1);
    let out = tools.invoke("add", json!({"a": 2, "b": 3})).await.unwrap();
    assert_eq!(out, json!({"sum": 5}));
    assert!(tools.invoke("missing", json!({})).await.is_err());
}

#[tokio::test]
async fn mock_chat_collects_reply() {
    let model = MockModel {
        reply: "Hello from the mock model!".into(),
    };
    let req = ChatRequest::builder("mock")
        .user("Say hi.")
        .build()
        .unwrap();
    let response = model.chat(req).await.unwrap();
    assert_eq!(response.text(), "Hello from the mock model!");
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
}

#[tokio::test]
async fn stream_token_by_token_via_inherent_next() {
    // No `use futures::StreamExt` here — `next` resolves to the inherent method.
    let model = MockModel {
        reply: "stream me".into(),
    };
    let req = ChatRequest::builder("mock").user("hi").build().unwrap();
    let mut stream = model.stream(req).await.unwrap();
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::TextDelta(s) = event.unwrap() {
            text.push_str(&s);
        }
    }
    assert_eq!(text, "stream me");
}

#[tokio::test]
async fn agent_drives_tool_loop() {
    let weather = FnTool::new(
        ToolDef::new(
            "get_weather",
            "Get the weather for a city.",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        ),
        |args: serde_json::Value| async move {
            let city = args["city"].as_str().unwrap_or("?");
            Ok::<_, Error>(json!({ "city": city, "temp_f": 72 }))
        },
    );
    let agent = Agent::new(MockModel {
        reply: "It's 72F in Paris.".into(),
    })
    .system("You are a helpful weather assistant.")
    .tool(weather)
    .max_steps(4);

    let outcome = agent.run("What's the weather in Paris?").await.unwrap();
    assert_eq!(outcome.text(), "It's 72F in Paris.");
    assert_eq!(outcome.steps, 2);
    assert_eq!(
        outcome
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .count(),
        1,
        "model -> tool -> model loop should record one tool result",
    );
}

#[tokio::test]
async fn structured_output_into_derived_type() {
    let model = MockModel {
        reply: r#"{"x": 3, "y": 4}"#.into(),
    };
    let req = ChatRequest::builder("mock")
        .user("Give me a point.")
        .build()
        .unwrap();
    let point: Point = model.structured::<Point>(req).await.unwrap();
    assert_eq!(point, Point { x: 3, y: 4 });
}
