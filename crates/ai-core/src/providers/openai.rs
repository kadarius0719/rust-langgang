//! OpenAI-compatible Chat Completions adapter.
//!
//! Covers OpenAI and any OpenAI-compatible endpoint via a configurable
//! `base_url` — including offline/local runners (llama.cpp `llama-server`, LM
//! Studio, Ollama's `/v1`). Construct with [`OpenAiClient::new`] for the hosted
//! API or [`OpenAiClient::local`] for a local runner, then build a
//! [`OpenAiModel`] with [`OpenAiClient::chat_model`].
//!
//! Wire DTOs are private and kept strictly separate from the domain types; the
//! adapter converts in both directions and merges [`ChatRequest::extra`] into the
//! request body so unmodeled provider params still work.

use std::sync::Arc;

use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    auth::{Auth, BearerAuth, NoAuth},
    error::{Error, Result},
    message::{ContentBlock, ImageSource, Message, Role},
    model::{ChatClient, ChatModel},
    request::ChatRequest,
    response::{ChatResponse, StopReason, Usage},
    stream::{ChatStream, StreamEvent},
    structured::ResponseFormat,
    tool::{ToolChoice, ToolDef},
};

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// A client for an OpenAI-compatible endpoint.
///
/// Cloning is cheap (it shares the underlying `reqwest::Client`).
#[derive(Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    auth: Arc<dyn Auth>,
}

impl OpenAiClient {
    /// A client for the hosted OpenAI API, authenticated with a bearer key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: OPENAI_BASE_URL.to_string(),
            auth: Arc::new(BearerAuth(api_key.into())),
        }
    }

    /// A client for a local/offline runner — no authentication, custom URL.
    ///
    /// `base_url` should include the version path, e.g.
    /// `http://localhost:11434/v1` (Ollama) or `http://localhost:8080/v1`
    /// (llama.cpp `llama-server`).
    pub fn local(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            auth: Arc::new(NoAuth),
        }
    }

    /// Override the base URL (covers OpenRouter, Together, Groq, Azure, …).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Override the authentication scheme (bring your own [`Auth`]).
    pub fn with_auth(mut self, auth: Arc<dyn Auth>) -> Self {
        self.auth = auth;
        self
    }

    /// Inject a custom `reqwest::Client` (proxy, timeouts, mTLS, middleware).
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Build a model handle for the given model id.
    pub fn chat_model(&self, model: impl Into<String>) -> OpenAiModel {
        OpenAiModel {
            client: self.clone(),
            model: model.into(),
        }
    }

    async fn post_chat(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let builder = self.http.post(url).json(body);
        let builder = self.auth.apply(builder);
        Ok(builder.send().await?)
    }
}

impl ChatClient for OpenAiClient {
    type Model = OpenAiModel;

    fn chat_model(&self, model: impl Into<String>) -> Self::Model {
        OpenAiClient::chat_model(self, model)
    }
}

/// A configured model handle produced by [`OpenAiClient::chat_model`].
#[derive(Clone)]
pub struct OpenAiModel {
    client: OpenAiClient,
    model: String,
}

impl ChatModel for OpenAiModel {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let body = build_body(&self.model, &request, false)?;
        let response = self.client.post_chat(&body).await?;
        let status = response.status();
        let bytes = response.bytes().await?;
        if !status.is_success() {
            return Err(provider_error(status, &bytes));
        }
        let raw: Value = serde_json::from_slice(&bytes)?;
        let parsed: ChatCompletionResponse = serde_json::from_slice(&bytes)?;
        Ok(to_chat_response(parsed, raw))
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let body = build_body(&self.model, &request, true)?;
        let response = self.client.post_chat(&body).await?;
        let status = response.status();
        if !status.is_success() {
            let bytes = response.bytes().await?;
            return Err(provider_error(status, &bytes));
        }
        Ok(sse_to_chat_stream(response))
    }
}

// ---------------------------------------------------------------------------
// Request: domain -> wire
// ---------------------------------------------------------------------------

fn build_body(model: &str, request: &ChatRequest, stream: bool) -> Result<Value> {
    let wire = WireRequest {
        model,
        messages: to_wire_messages(request),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stop: if request.stop.is_empty() {
            None
        } else {
            Some(&request.stop)
        },
        tools: request.tools.iter().map(to_wire_tool).collect(),
        tool_choice: request.tool_choice.as_ref().map(to_wire_tool_choice),
        response_format: request
            .response_format
            .as_ref()
            .and_then(to_wire_response_format),
        stream,
        stream_options: stream.then_some(StreamOptions {
            include_usage: true,
        }),
    };

    let mut value = serde_json::to_value(&wire)?;
    if !request.extra.is_empty() {
        if let Value::Object(map) = &mut value {
            // Typed fields win; `extra` only fills params we don't model.
            for (key, val) in &request.extra {
                map.entry(key.clone()).or_insert_with(|| val.clone());
            }
        }
    }
    Ok(value)
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<WireToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<WireResponseFormat>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<WireContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireContent {
    Text(String),
    Parts(Vec<WirePart>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum WirePart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: WireImageUrl },
}

#[derive(Serialize)]
struct WireImageUrl {
    url: String,
}

#[derive(Serialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireFnCall,
}

#[derive(Serialize)]
struct WireFnCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireToolFn,
}

#[derive(Serialize)]
struct WireToolFn {
    name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    description: String,
    parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireToolChoice {
    Mode(&'static str),
    Named(NamedToolChoice),
}

#[derive(Serialize)]
struct NamedToolChoice {
    #[serde(rename = "type")]
    kind: &'static str,
    function: NamedFn,
}

#[derive(Serialize)]
struct NamedFn {
    name: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireResponseFormat {
    JsonObject,
    JsonSchema { json_schema: WireJsonSchema },
}

#[derive(Serialize)]
struct WireJsonSchema {
    name: String,
    schema: Value,
    strict: bool,
}

fn to_wire_messages(request: &ChatRequest) -> Vec<WireMessage> {
    let mut out = Vec::new();
    if let Some(system) = &request.system {
        out.push(WireMessage {
            role: "system",
            content: Some(WireContent::Text(system.clone())),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    for message in &request.messages {
        match message.role {
            Role::User => out.push(user_message(message)),
            Role::Assistant => out.push(assistant_message(message)),
            Role::Tool => {
                for block in &message.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = block
                    {
                        out.push(WireMessage {
                            role: "tool",
                            content: Some(WireContent::Text(content.clone())),
                            tool_calls: Vec::new(),
                            tool_call_id: Some(tool_use_id.clone()),
                        });
                    }
                }
            }
        }
    }
    out
}

fn user_message(message: &Message) -> WireMessage {
    let has_image = message
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    let content = if has_image {
        let parts = message
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(WirePart::Text { text: text.clone() }),
                ContentBlock::Image { source } => Some(WirePart::ImageUrl {
                    image_url: WireImageUrl {
                        url: image_url(source),
                    },
                }),
                _ => None,
            })
            .collect();
        WireContent::Parts(parts)
    } else {
        WireContent::Text(message.text())
    };
    WireMessage {
        role: "user",
        content: Some(content),
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

fn assistant_message(message: &Message) -> WireMessage {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text: t } => text.push_str(t),
            ContentBlock::ToolUse { id, name, args } => tool_calls.push(WireToolCall {
                id: id.clone(),
                kind: "function",
                function: WireFnCall {
                    name: name.clone(),
                    arguments: serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string()),
                },
            }),
            _ => {}
        }
    }
    WireMessage {
        role: "assistant",
        content: (!text.is_empty()).then_some(WireContent::Text(text)),
        tool_calls,
        tool_call_id: None,
    }
}

fn image_url(source: &ImageSource) -> String {
    match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
    }
}

fn to_wire_tool(tool: &ToolDef) -> WireTool {
    WireTool {
        kind: "function",
        function: WireToolFn {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict: tool.strict,
        },
    }
}

fn to_wire_tool_choice(choice: &ToolChoice) -> WireToolChoice {
    match choice {
        ToolChoice::Auto => WireToolChoice::Mode("auto"),
        ToolChoice::None => WireToolChoice::Mode("none"),
        ToolChoice::Required => WireToolChoice::Mode("required"),
        ToolChoice::Tool { name } => WireToolChoice::Named(NamedToolChoice {
            kind: "function",
            function: NamedFn { name: name.clone() },
        }),
    }
}

fn to_wire_response_format(format: &ResponseFormat) -> Option<WireResponseFormat> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(WireResponseFormat::JsonObject),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => Some(WireResponseFormat::JsonSchema {
            json_schema: WireJsonSchema {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

// ---------------------------------------------------------------------------
// Response: wire -> domain (non-streaming)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<RespChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct RespChoice {
    #[serde(default)]
    message: RespMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct RespMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<RespToolCall>,
}

#[derive(Deserialize)]
struct RespToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    function: RespFn,
}

#[derive(Default, Deserialize)]
struct RespFn {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

fn to_chat_response(parsed: ChatCompletionResponse, raw: Value) -> ChatResponse {
    let (message, finish_reason) = parsed
        .choices
        .into_iter()
        .next()
        .map(|c| (c.message, c.finish_reason))
        .unwrap_or_default();

    let mut content = Vec::new();
    if let Some(text) = message.content {
        if !text.is_empty() {
            content.push(ContentBlock::text(text));
        }
    }
    for call in message.tool_calls {
        let args = serde_json::from_str(&call.function.arguments)
            .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
        content.push(ContentBlock::ToolUse {
            id: call.id,
            name: call.function.name,
            args,
        });
    }

    ChatResponse {
        message: Message {
            role: Role::Assistant,
            content,
        },
        usage: parsed.usage.map(Usage::from).unwrap_or_default(),
        stop_reason: finish_reason.map(|r| map_finish_reason(&r)),
        model: parsed.model,
        raw,
    }
}

// ---------------------------------------------------------------------------
// Response: wire -> domain (streaming, SSE)
// ---------------------------------------------------------------------------

fn sse_to_chat_stream(response: reqwest::Response) -> ChatStream {
    let stream = async_stream::try_stream! {
        yield StreamEvent::MessageStart;
        let mut events = response.bytes_stream().eventsource();
        while let Some(event) = events.next().await {
            let event = event.map_err(|e| Error::stream(e.to_string()))?;
            if event.data == "[DONE]" {
                break;
            }
            let chunk: StreamChunk = serde_json::from_str(&event.data)?;
            for stream_event in chunk_to_events(chunk) {
                yield stream_event;
            }
        }
        yield StreamEvent::MessageStop;
    };
    ChatStream::new(stream)
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<DeltaFn>,
}

#[derive(Deserialize)]
struct DeltaFn {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

fn chunk_to_events(chunk: StreamChunk) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    for choice in chunk.choices {
        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                out.push(StreamEvent::TextDelta(text));
            }
        }
        for call in choice.delta.tool_calls {
            if let (Some(id), Some(function)) = (&call.id, &call.function) {
                if let Some(name) = &function.name {
                    out.push(StreamEvent::ToolCallStart {
                        index: call.index,
                        id: id.clone(),
                        name: name.clone(),
                    });
                }
            }
            if let Some(function) = &call.function {
                if let Some(arguments) = &function.arguments {
                    if !arguments.is_empty() {
                        out.push(StreamEvent::ToolCallArgsDelta {
                            index: call.index,
                            delta: arguments.clone(),
                        });
                    }
                }
            }
        }
        if let Some(reason) = choice.finish_reason {
            out.push(StreamEvent::Stop(map_finish_reason(&reason)));
        }
    }
    if let Some(usage) = chunk.usage {
        out.push(StreamEvent::Usage(usage.into()));
    }
    out
}

// ---------------------------------------------------------------------------
// Shared wire types / helpers
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionDetails>,
}

#[derive(Deserialize)]
struct PromptDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct CompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

impl From<WireUsage> for Usage {
    fn from(usage: WireUsage) -> Self {
        Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            cache_read_input_tokens: usage.prompt_tokens_details.and_then(|d| d.cached_tokens),
            cache_creation_input_tokens: None,
            reasoning_tokens: usage
                .completion_tokens_details
                .and_then(|d| d.reasoning_tokens),
        }
    }
}

fn map_finish_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "content_filter" => StopReason::Refusal,
        other => StopReason::Other(other.to_string()),
    }
}

#[derive(Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    error: Option<ErrorBody>,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[serde(default)]
    message: String,
}

fn provider_error(status: reqwest::StatusCode, body: &[u8]) -> Error {
    let message = serde_json::from_slice::<ErrorEnvelope>(body)
        .ok()
        .and_then(|e| e.error)
        .map(|e| e.message)
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| String::from_utf8_lossy(body).into_owned());
    Error::provider_status("openai", status.as_u16(), message)
}
