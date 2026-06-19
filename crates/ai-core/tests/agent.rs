//! Integration tests for the explicit agent tool-calling loop.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};

use ai_core::{
    Agent, ChatModel, ChatRequest, ChatStream, Error, FnTool, RecordingTracer, Result, StopCause,
    StopReason, StreamEvent, ToolDef, TraceEvent,
};

/// A model that replays one scripted list of events per call.
#[derive(Clone)]
struct ScriptModel {
    responses: Arc<Mutex<VecDeque<Vec<StreamEvent>>>>,
}

impl ScriptModel {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(scripts.into())),
        }
    }
}

impl ChatModel for ScriptModel {
    fn stream(&self, _request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        let events = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default();
        async move {
            Ok(ChatStream::new(futures::stream::iter(
                events.into_iter().map(Ok::<_, Error>),
            )))
        }
    }
}

fn weather_tool() -> impl ai_core::Tool {
    FnTool::new(
        ToolDef::new(
            "get_weather",
            "Get the weather for a city.",
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        ),
        |args: serde_json::Value| async move {
            let city = args.get("city").and_then(|c| c.as_str()).unwrap_or("?");
            Ok::<_, Error>(serde_json::json!({ "city": city, "temp": "72F" }))
        },
    )
}

fn tool_call_events() -> Vec<StreamEvent> {
    vec![
        StreamEvent::ToolCallStart {
            index: 0,
            id: "c1".into(),
            name: "get_weather".into(),
        },
        StreamEvent::ToolCallArgsDelta {
            index: 0,
            delta: "{\"city\":\"Paris\"}".into(),
        },
        StreamEvent::Stop(StopReason::ToolUse),
    ]
}

fn final_events() -> Vec<StreamEvent> {
    vec![
        StreamEvent::TextDelta("It is 72F in Paris.".into()),
        StreamEvent::Stop(StopReason::EndTurn),
    ]
}

#[tokio::test]
async fn executes_tool_then_finishes() {
    let model = ScriptModel::new(vec![tool_call_events(), final_events()]);
    let agent = Agent::new(model).tool(weather_tool()).max_steps(5);

    let outcome = agent.run("weather in Paris?").await.unwrap();

    assert_eq!(outcome.stopped, StopCause::Final);
    assert_eq!(outcome.steps, 2);
    assert!(outcome.text().contains("72F"), "{}", outcome.text());
    // user, assistant(tool call), tool(result), assistant(final)
    assert_eq!(outcome.messages.len(), 4);
}

#[tokio::test]
async fn stops_at_max_steps() {
    // The model keeps asking for tools; the cap must end the loop.
    let model = ScriptModel::new(vec![tool_call_events(), tool_call_events()]);
    let agent = Agent::new(model).tool(weather_tool()).max_steps(2);

    let outcome = agent.run("loop forever?").await.unwrap();

    assert_eq!(outcome.stopped, StopCause::MaxSteps);
    assert_eq!(outcome.steps, 2);
}

#[tokio::test]
async fn emits_trace_events() {
    let tracer = RecordingTracer::new();
    let model = ScriptModel::new(vec![tool_call_events(), final_events()]);
    let agent = Agent::new(model)
        .tool(weather_tool())
        .tracer(Arc::new(tracer.clone()));

    let _ = agent.run("weather?").await.unwrap();

    let events = tracer.events();
    assert!(events
        .iter()
        .any(|e| matches!(e, TraceEvent::ToolSelected { name, .. } if name == "get_weather")));
    assert!(events
        .iter()
        .any(|e| matches!(e, TraceEvent::ToolExecuted { ok: true, .. })));
    assert!(
        events
            .iter()
            .filter(|e| matches!(e, TraceEvent::AgentStep { .. }))
            .count()
            >= 2
    );
}

#[tokio::test]
async fn unknown_tool_is_reported_back_not_fatal() {
    // Model calls a tool that isn't registered; the loop should feed the error
    // back as a tool result and continue to a final answer.
    let model = ScriptModel::new(vec![tool_call_events(), final_events()]);
    let agent = Agent::new(model).max_steps(5); // no tools registered

    let outcome = agent.run("weather?").await.unwrap();

    assert_eq!(outcome.stopped, StopCause::Final);
    assert_eq!(outcome.steps, 2);
    // The tool-result message should carry the error flag.
    let had_error_result = outcome.messages.iter().any(|m| {
        m.content
            .iter()
            .any(|b| matches!(b, ai_core::ContentBlock::ToolResult { is_error: true, .. }))
    });
    assert!(had_error_result);
}

#[tokio::test]
async fn without_tools_is_one_shot() {
    let model = ScriptModel::new(vec![final_events()]);
    let agent = Agent::new(model);

    let outcome = agent.run("hello").await.unwrap();

    assert_eq!(outcome.steps, 1);
    assert_eq!(outcome.stopped, StopCause::Final);
}
