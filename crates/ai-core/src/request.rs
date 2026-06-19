//! The provider-neutral chat request and its builder.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::{Error, Result},
    message::Message,
    structured::ResponseFormat,
    tool::{ToolChoice, ToolDef},
};

/// A provider-neutral chat request.
///
/// Adapters translate this into each provider's wire format. The `extra` map is
/// a passthrough for provider-specific knobs we don't model first-class, so
/// callers are never blocked by missing fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model identifier (provider-specific).
    pub model: String,
    /// Conversation so far (must be non-empty).
    pub messages: Vec<Message>,
    /// System prompt — a first-class field, not a message role.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Tools the model may call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    /// Constraint on tool usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Desired output format (structured output).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Provider-specific passthrough fields, merged into the wire request.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, Value>,
}

impl ChatRequest {
    /// Start building a request for `model`.
    pub fn builder(model: impl Into<String>) -> ChatRequestBuilder {
        ChatRequestBuilder::new(model)
    }
}

/// Builder for [`ChatRequest`].
///
/// The two true invariants — a non-empty model and at least one message — are
/// validated in [`ChatRequestBuilder::build`]; everything else is optional. We
/// keep a plain builder rather than a typestate explosion so docs and error
/// messages stay readable.
#[derive(Debug, Clone)]
pub struct ChatRequestBuilder {
    model: String,
    messages: Vec<Message>,
    system: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Vec<String>,
    tools: Vec<ToolDef>,
    tool_choice: Option<ToolChoice>,
    response_format: Option<ResponseFormat>,
    extra: serde_json::Map<String, Value>,
}

impl ChatRequestBuilder {
    /// Create a builder for `model`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            extra: serde_json::Map::new(),
        }
    }

    /// Append a message.
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Append several messages.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Append a user text message.
    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(Message::user(text))
    }

    /// Append an assistant text message.
    pub fn assistant(self, text: impl Into<String>) -> Self {
        self.message(Message::assistant(text))
    }

    /// Set the system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the maximum number of output tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the nucleus sampling probability mass.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Add a stop sequence.
    pub fn stop(mut self, stop: impl Into<String>) -> Self {
        self.stop.push(stop.into());
        self
    }

    /// Add a tool the model may call.
    pub fn tool(mut self, tool: ToolDef) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add several tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDef>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Constrain how tools may be used.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Request a specific output format.
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Set a provider-specific passthrough field.
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Validate and build the request.
    ///
    /// # Errors
    /// Returns [`Error::InvalidRequest`] if the model is empty or there are no
    /// messages.
    pub fn build(self) -> Result<ChatRequest> {
        if self.model.trim().is_empty() {
            return Err(Error::invalid_request("model must not be empty"));
        }
        if self.messages.is_empty() {
            return Err(Error::invalid_request("at least one message is required"));
        }
        Ok(ChatRequest {
            model: self.model,
            messages: self.messages,
            system: self.system,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop,
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: self.response_format,
            extra: self.extra,
        })
    }
}
