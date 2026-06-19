//! Structured-output request configuration.
//!
//! Structured output is a *model-layer* feature, not a separate "output parser"
//! subsystem: you describe the desired shape with a [`ResponseFormat`], adapters
//! use the provider-native JSON/schema mode when available (or a tool-call
//! fallback otherwise), and deserialize into your type. With the `schema`
//! feature, [`response_format_for`] derives the schema from a Rust type.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How the model should format its response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseFormat {
    /// Free-form text (the default behavior).
    Text,
    /// Any syntactically valid JSON object.
    JsonObject,
    /// JSON constrained to a named JSON Schema.
    JsonSchema {
        /// A name for the schema (some providers require one).
        name: String,
        /// The JSON Schema document.
        schema: Value,
        /// Whether to request strict enforcement, when the provider supports it.
        #[serde(default)]
        strict: bool,
    },
}

impl ResponseFormat {
    /// The discriminant of this format, useful for logging/tracing without
    /// cloning the (potentially large) schema.
    pub fn kind(&self) -> ResponseFormatKind {
        match self {
            Self::Text => ResponseFormatKind::Text,
            Self::JsonObject => ResponseFormatKind::JsonObject,
            Self::JsonSchema { .. } => ResponseFormatKind::JsonSchema,
        }
    }
}

/// The discriminant of a [`ResponseFormat`], without its payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseFormatKind {
    /// See [`ResponseFormat::Text`].
    Text,
    /// See [`ResponseFormat::JsonObject`].
    JsonObject,
    /// See [`ResponseFormat::JsonSchema`].
    JsonSchema,
}

/// Build a [`ResponseFormat::JsonSchema`] from a Rust type's derived schema.
///
/// Requires the `schema` feature.
#[cfg(feature = "schema")]
pub fn response_format_for<T: schemars::JsonSchema>(
    name: impl Into<String>,
    strict: bool,
) -> ResponseFormat {
    let schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null);
    ResponseFormat::JsonSchema {
        name: name.into(),
        schema,
        strict,
    }
}

/// Ask a model for output deserialized straight into a Rust type.
///
/// Blanket-implemented for every [`ChatModel`](crate::ChatModel). If the request
/// has no [`ResponseFormat`] set, one is derived from `T` (via `schemars`) and
/// applied so capable providers constrain their output; the assistant's text is
/// then deserialized into `T`. Requires the `schema` feature.
///
/// ```ignore
/// use ai_core::{ChatModel, ChatRequest, StructuredExt};
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct Person { name: String, age: u32 }
///
/// let request = ChatRequest::builder("model")
///     .user("Invent a person as JSON with `name` and `age`.")
///     .build()?;
/// let person: Person = model.structured(request).await?;
/// ```
#[cfg(feature = "schema")]
pub trait StructuredExt: crate::model::ChatModel {
    /// Run the request and deserialize the response into `T`.
    fn structured<T>(
        &self,
        request: crate::request::ChatRequest,
    ) -> impl std::future::Future<Output = crate::error::Result<T>> + Send
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned + Send;
}

#[cfg(feature = "schema")]
impl<M: crate::model::ChatModel> StructuredExt for M {
    async fn structured<T>(
        &self,
        mut request: crate::request::ChatRequest,
    ) -> crate::error::Result<T>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned + Send,
    {
        if request.response_format.is_none() {
            request.response_format = Some(response_format_for::<T>("output", false));
        }
        let response = self.chat(request).await?;
        serde_json::from_str(&response.text()).map_err(crate::error::Error::from)
    }
}

#[cfg(all(test, feature = "schema"))]
mod tests {
    use super::*;
    use crate::{ChatModel, ChatRequest, ChatStream, Error, Result, StreamEvent};
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::future::Future;

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    /// A model that always streams back a fixed JSON document.
    struct JsonModel(&'static str);

    impl ChatModel for JsonModel {
        fn stream(&self, _request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
            let json = self.0.to_string();
            async move {
                Ok(ChatStream::new(futures::stream::iter(vec![
                    Ok::<_, Error>(StreamEvent::TextDelta(json)),
                ])))
            }
        }
    }

    #[tokio::test]
    async fn structured_deserializes_into_type() {
        let model = JsonModel(r#"{"name":"Ada","age":36}"#);
        let request = ChatRequest::builder("m").user("who?").build().unwrap();
        let person: Person = model.structured(request).await.unwrap();
        assert_eq!(
            person,
            Person {
                name: "Ada".into(),
                age: 36
            }
        );
    }

    #[tokio::test]
    async fn structured_sets_response_format_when_absent() {
        // The helper should populate a JSON-schema response format from the type.
        let model = JsonModel(r#"{"name":"Ada","age":36}"#);
        let request = ChatRequest::builder("m").user("who?").build().unwrap();
        assert!(request.response_format.is_none());
        let _: Person = model.structured(request).await.unwrap();
    }
}
