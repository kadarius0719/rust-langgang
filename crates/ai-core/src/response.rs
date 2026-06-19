//! Response and usage types.

use serde::{Deserialize, Serialize};

use crate::message::{ContentBlock, Message};

/// Token accounting for a request.
///
/// `input_tokens`/`output_tokens` are the canonical fields every provider maps
/// onto; the optional fields are populated only when a provider reports them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub input_tokens: u32,
    /// Completion tokens.
    pub output_tokens: u32,
    /// Tokens served from the provider's prompt cache, if reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    /// Tokens written to the provider's prompt cache, if reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Reasoning/thinking tokens, if reported separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl Usage {
    /// `input_tokens + output_tokens`.
    pub fn total(&self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

/// Why the model stopped generating.
///
/// Unknown values from a provider are preserved in [`StopReason::Other`] so wire
/// drift never loses information.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StopReason {
    /// The model finished naturally.
    EndTurn,
    /// Hit the output token cap.
    MaxTokens,
    /// Hit a stop sequence.
    StopSequence,
    /// The model wants to call one or more tools.
    ToolUse,
    /// The model refused on safety grounds.
    Refusal,
    /// Any other provider-specific reason, preserved verbatim.
    Other(String),
}

impl StopReason {
    /// The canonical snake_case string for this reason.
    pub fn as_str(&self) -> &str {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::ToolUse => "tool_use",
            Self::Refusal => "refusal",
            Self::Other(s) => s,
        }
    }

    /// Parse from a wire string, mapping unknown values to [`StopReason::Other`].
    pub fn from_wire(s: &str) -> Self {
        match s {
            "end_turn" => Self::EndTurn,
            "max_tokens" => Self::MaxTokens,
            "stop_sequence" => Self::StopSequence,
            "tool_use" => Self::ToolUse,
            "refusal" => Self::Refusal,
            other => Self::Other(other.to_string()),
        }
    }
}

impl Serialize for StopReason {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self::from_wire(&s))
    }
}

/// A borrowed view of a tool call within a [`ChatResponse`].
#[derive(Debug, Clone, Copy)]
pub struct ToolUseRef<'a> {
    /// Provider-assigned call id.
    pub id: &'a str,
    /// Tool name.
    pub name: &'a str,
    /// Parsed JSON arguments.
    pub args: &'a serde_json::Value,
}

/// A completed (non-streaming) model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The assistant message (role is always [`Role::Assistant`]).
    pub message: Message,
    /// Token accounting.
    #[serde(default)]
    pub usage: Usage,
    /// Why generation stopped, if reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// The model id that produced this response, if reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The provider-native payload, for escape-hatch access. `Null` unless an
    /// adapter chooses to populate it.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl ChatResponse {
    /// Build a response from an assistant message with empty usage/metadata.
    pub fn from_message(message: Message) -> Self {
        Self {
            message,
            usage: Usage::default(),
            stop_reason: None,
            model: None,
            raw: serde_json::Value::Null,
        }
    }

    /// Concatenated text of all text blocks in the message.
    pub fn text(&self) -> String {
        self.message.text()
    }

    /// All tool calls the model requested, in order.
    pub fn tool_uses(&self) -> Vec<ToolUseRef<'_>> {
        self.message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, args } => Some(ToolUseRef { id, name, args }),
                _ => None,
            })
            .collect()
    }

    /// Whether the model requested any tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.message
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }
}
