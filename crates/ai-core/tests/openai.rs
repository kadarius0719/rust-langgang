//! Integration tests for the OpenAI-compatible adapter, against a mock server.
//! Built only with `--features openai` (see `required-features` in Cargo.toml).

use ai_core::providers::openai::OpenAiClient;
use ai_core::{ApiErrorKind, ChatModel, ChatRequest, Error, StopReason, ToolDef};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn req() -> ChatRequest {
    ChatRequest::builder("local-model")
        .system("be brief")
        .user("hi")
        .build()
        .unwrap()
}

#[tokio::test]
async fn non_streaming_chat_parses_response_and_shapes_request() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "model": "local-model",
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        // Verifies the domain -> wire mapping: system becomes a system message.
        .and(body_partial_json(serde_json::json!({
            "model": "local-model",
            "stream": false,
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let model = OpenAiClient::local(format!("{}/v1", server.uri())).chat_model("local-model");
    let response = model.chat(req()).await.unwrap();

    assert_eq!(response.text(), "Hello!");
    assert_eq!(response.usage.input_tokens, 5);
    assert_eq!(response.usage.output_tokens, 2);
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.model.as_deref(), Some("local-model"));
}

#[tokio::test]
async fn streaming_chat_accumulates() {
    let server = MockServer::start().await;
    let c1 = serde_json::json!({"choices": [{"delta": {"content": "Hel"}}]}).to_string();
    let c2 =
        serde_json::json!({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]})
            .to_string();
    let c3 =
        serde_json::json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
            .to_string();
    let sse = format!("data: {c1}\n\ndata: {c2}\n\ndata: {c3}\n\ndata: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(serde_json::json!({ "stream": true })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let model = OpenAiClient::local(format!("{}/v1", server.uri())).chat_model("m");
    let response = model
        .stream(req())
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();

    assert_eq!(response.text(), "Hello");
    assert_eq!(response.usage.input_tokens, 3);
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
}

#[tokio::test]
async fn tool_call_round_trips() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;

    let model = OpenAiClient::local(format!("{}/v1", server.uri())).chat_model("m");
    let request = ChatRequest::builder("m")
        .user("weather in Paris?")
        .tool(ToolDef::new(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        ))
        .build()
        .unwrap();

    let response = model.chat(request).await.unwrap();
    let tools = response.tool_uses();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].id, "call_1");
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(tools[0].args["city"], "Paris");
    assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
}

#[tokio::test]
async fn streaming_tool_call_assembles_arguments() {
    let server = MockServer::start().await;
    // OpenAI streams a tool call as: id+name first, then argument fragments.
    let c1 = serde_json::json!({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}
    ]}}]})
    .to_string();
    let c2 = serde_json::json!({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "{\"city\":"}}
    ]}}]})
    .to_string();
    let c3 = serde_json::json!({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "\"Paris\"}"}}
    ]}, "finish_reason": "tool_calls"}]})
    .to_string();
    let sse = format!("data: {c1}\n\ndata: {c2}\n\ndata: {c3}\n\ndata: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let model = OpenAiClient::local(format!("{}/v1", server.uri())).chat_model("m");
    let response = model
        .stream(req())
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();

    let tools = response.tool_uses();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].id, "call_1");
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(tools[0].args["city"], "Paris");
    assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
}

#[tokio::test]
async fn error_status_maps_to_normalized_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_json(serde_json::json!({"error": {"message": "slow down"}})),
        )
        .mount(&server)
        .await;

    let model = OpenAiClient::local(format!("{}/v1", server.uri())).chat_model("m");
    let err = model.chat(req()).await.unwrap_err();

    match err {
        Error::Provider {
            status,
            kind,
            message,
            ..
        } => {
            assert_eq!(status, Some(429));
            assert_eq!(kind, ApiErrorKind::RateLimited);
            assert_eq!(message, "slow down");
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
}
