//! Pluggable authentication for HTTP providers.
//!
//! Providers don't hardcode an auth scheme — they hold an `Arc<dyn Auth>` and
//! call [`Auth::apply`] on each outgoing request. Built-in schemes cover the
//! common cases; implement [`Auth`] yourself for anything else (SigV4 for
//! Bedrock, OAuth refresh, signed headers) without forking the crate.

/// Applies credentials to an outgoing HTTP request.
pub trait Auth: Send + Sync {
    /// Add whatever headers/query the scheme requires and return the builder.
    fn apply(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder;
}

/// No authentication — for local runners (llama.cpp server, LM Studio, Ollama).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoAuth;

impl Auth for NoAuth {
    fn apply(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder
    }
}

/// `Authorization: Bearer <token>` — the OpenAI/hosted-gateway default.
#[derive(Debug, Clone)]
pub struct BearerAuth(pub String);

impl Auth for BearerAuth {
    fn apply(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.bearer_auth(&self.0)
    }
}

/// An arbitrary API-key header, e.g. `x-api-key: <value>`.
#[derive(Debug, Clone)]
pub struct ApiKeyAuth {
    /// Header name.
    pub header: String,
    /// Header value.
    pub value: String,
}

impl ApiKeyAuth {
    /// Construct an API-key header scheme.
    pub fn new(header: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            header: header.into(),
            value: value.into(),
        }
    }
}

impl Auth for ApiKeyAuth {
    fn apply(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.header(&self.header, &self.value)
    }
}
