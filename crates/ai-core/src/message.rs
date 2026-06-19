//! Provider-neutral conversation types.
//!
//! These are *our* canonical representation, deliberately decoupled from any
//! provider's wire format — adapters translate to/from these. The system prompt
//! is **not** a [`Role`]; it is a first-class field on the request
//! ([`crate::ChatRequest::system`]) because providers disagree on whether it is
//! a top-level field or a role, and the adapter is the right place to decide.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who authored a message.
///
/// Note there is no `System` — the system prompt lives on the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// End-user input.
    User,
    /// Model output.
    Assistant,
    /// A tool result fed back to the model.
    Tool,
}

/// A single message in a conversation: a role plus ordered content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored this message.
    pub role: Role,
    /// Ordered content blocks (text, images, tool calls, tool results, …).
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Construct a message from a role and content blocks.
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// A user message containing a single text block.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![ContentBlock::text(text)])
    }

    /// An assistant message containing a single text block.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![ContentBlock::text(text)])
    }

    /// A tool-result message answering a prior tool call.
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::new(
            Role::Tool,
            vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error,
            }],
        )
    }

    /// Concatenate all [`ContentBlock::Text`] blocks into one string.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for block in &self.content {
            if let ContentBlock::Text { text } = block {
                out.push_str(text);
            }
        }
        out
    }
}

/// A unit of message content.
///
/// `#[non_exhaustive]` so providers can grow new block kinds without a breaking
/// change to downstream code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// The text.
        text: String,
    },
    /// An image input.
    Image {
        /// Where the image comes from.
        source: ImageSource,
    },
    /// A tool call requested by the model.
    ToolUse {
        /// Provider-assigned call id (correlates with the matching result).
        id: String,
        /// Tool name.
        name: String,
        /// Parsed JSON arguments.
        #[serde(default)]
        args: Value,
    },
    /// A tool result fed back to the model.
    ToolResult {
        /// The [`ContentBlock::ToolUse::id`] this answers.
        tool_use_id: String,
        /// Result payload (stringified).
        content: String,
        /// Whether the tool errored.
        #[serde(default)]
        is_error: bool,
    },
    /// Model reasoning / thinking content.
    Thinking {
        /// The reasoning text (may be a summary or empty depending on provider).
        text: String,
        /// Opaque provider signature, preserved verbatim for round-tripping.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

impl ContentBlock {
    /// A [`ContentBlock::Text`].
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// A [`ContentBlock::Image`] from a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::Url { url: url.into() },
        }
    }

    /// A [`ContentBlock::Image`] from base64-encoded bytes.
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }
}

/// The source of an image content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageSource {
    /// A remote URL.
    Url {
        /// The image URL.
        url: String,
    },
    /// Inline base64-encoded bytes.
    Base64 {
        /// MIME type, e.g. `image/png`.
        media_type: String,
        /// Base64-encoded image data.
        data: String,
    },
}
