//! The crate-wide error type.
//!
//! We use [`thiserror`] for a structured, `#[non_exhaustive]` error enum and
//! never expose `anyhow` in the public API. Provider errors are normalized into
//! a single [`Error::Provider`] variant carrying an [`ApiErrorKind`] so callers
//! can branch on *kind* (rate limited, invalid auth, …) uniformly across every
//! provider.

use std::fmt;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// The crate-wide error type.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The request was rejected before it left the crate (missing model, no
    /// messages, an impossible combination of options, …).
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// A provider returned (or the transport surfaced) a normalized error.
    #[error("provider `{provider}` request failed [{kind}]: {message}")]
    Provider {
        /// Provider identifier, e.g. `"openai"`, `"anthropic"`, `"ollama"`.
        provider: String,
        /// HTTP status, when the transport is HTTP.
        status: Option<u16>,
        /// Normalized, provider-agnostic error classification.
        kind: ApiErrorKind,
        /// Human-readable detail (often the provider's own message).
        message: String,
    },

    /// An error occurred while reading or decoding a streamed response.
    #[error("stream error: {0}")]
    Stream(String),

    /// A tool invocation failed.
    #[error("tool error: {0}")]
    Tool(String),

    /// (De)serialization failure.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Transport-level HTTP error (only present with an HTTP provider feature).
    #[cfg(feature = "http")]
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// Any other boxed error, for escape hatches and `?` ergonomics.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Construct an [`Error::InvalidRequest`].
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::InvalidRequest(msg.into())
    }

    /// Construct an [`Error::Provider`] with no HTTP status.
    pub fn provider(
        provider: impl Into<String>,
        kind: ApiErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self::Provider {
            provider: provider.into(),
            status: None,
            kind,
            message: message.into(),
        }
    }

    /// Construct an [`Error::Provider`] from an HTTP status, classifying the
    /// [`ApiErrorKind`] from that status.
    pub fn provider_status(
        provider: impl Into<String>,
        status: u16,
        message: impl Into<String>,
    ) -> Self {
        Self::Provider {
            provider: provider.into(),
            status: Some(status),
            kind: ApiErrorKind::from_status(status),
            message: message.into(),
        }
    }

    /// Construct an [`Error::Stream`].
    pub fn stream(msg: impl Into<String>) -> Self {
        Self::Stream(msg.into())
    }

    /// Construct an [`Error::Tool`].
    pub fn tool(msg: impl Into<String>) -> Self {
        Self::Tool(msg.into())
    }
}

/// Provider-agnostic classification of an API error.
///
/// `#[non_exhaustive]` so adding kinds is not a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApiErrorKind {
    /// Too many requests / quota exceeded.
    RateLimited,
    /// Missing or invalid credentials.
    InvalidAuth,
    /// Input exceeded the model's context window.
    ContextLengthExceeded,
    /// Request or response blocked by a content filter / safety system.
    ContentFilter,
    /// Resource (model, endpoint) not found.
    NotFound,
    /// Malformed request the provider rejected.
    BadRequest,
    /// Provider-side failure (5xx).
    ServerError,
    /// Request timed out.
    Timeout,
    /// Anything not otherwise classified.
    Other,
}

impl ApiErrorKind {
    /// Best-effort classification from an HTTP status code.
    pub fn from_status(status: u16) -> Self {
        match status {
            400 => Self::BadRequest,
            401 | 403 => Self::InvalidAuth,
            404 => Self::NotFound,
            408 => Self::Timeout,
            429 => Self::RateLimited,
            500..=599 => Self::ServerError,
            _ => Self::Other,
        }
    }
}

impl fmt::Display for ApiErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::RateLimited => "rate limited",
            Self::InvalidAuth => "invalid auth",
            Self::ContextLengthExceeded => "context length exceeded",
            Self::ContentFilter => "content filtered",
            Self::NotFound => "not found",
            Self::BadRequest => "bad request",
            Self::ServerError => "server error",
            Self::Timeout => "timeout",
            Self::Other => "other",
        };
        f.write_str(s)
    }
}
