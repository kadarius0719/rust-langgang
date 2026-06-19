//! Tool definitions and the [`Tool`] trait.
//!
//! A tool is a function the model can call: a name, a description, and a
//! JSON-Schema for its parameters, plus an async `invoke`. The [`ToolDef`]
//! itself carries the schema as a [`serde_json::Value`] so the type costs
//! nothing without the `schema` feature; with `schema`, [`tool_def`] derives the
//! schema from a Rust type via `schemars`.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::{Error, Result},
    BoxFuture,
};

/// The provider-neutral declaration of a callable tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    /// Unique tool name.
    pub name: String,
    /// What the tool does and when to call it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameters: Value,
    /// Whether the provider should strictly enforce the schema, when supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ToolDef {
    /// A tool definition with an explicit JSON-Schema parameter object.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: None,
        }
    }
}

/// Derive a [`ToolDef`] whose parameter schema comes from a Rust type.
#[cfg(feature = "schema")]
pub fn tool_def<T: schemars::JsonSchema>(
    name: impl Into<String>,
    description: impl Into<String>,
) -> ToolDef {
    let parameters = serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null);
    ToolDef {
        name: name.into(),
        description: description.into(),
        parameters,
        strict: None,
    }
}

/// How the model is allowed (or required) to use tools for a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides whether to call a tool (default).
    Auto,
    /// The model may not call any tool.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call this specific tool.
    Tool {
        /// The tool name to force.
        name: String,
    },
}

/// A callable tool the model can invoke.
///
/// Hot-path trait using native `async fn`-in-traits (RPITIT). For dynamic
/// storage (a registry of heterogeneous tools), use [`DynTool`], which has a
/// blanket impl over every `Tool`.
pub trait Tool: Send + Sync {
    /// The declaration advertised to the model.
    fn def(&self) -> ToolDef;

    /// Execute the tool with parsed JSON arguments.
    fn invoke(&self, args: Value) -> impl Future<Output = Result<Value>> + Send;
}

/// Object-safe facade for [`Tool`], so tools can be boxed and stored together.
pub trait DynTool: Send + Sync {
    /// The declaration advertised to the model.
    fn def(&self) -> ToolDef;

    /// Execute the tool, returning a boxed future.
    fn invoke_boxed<'a>(&'a self, args: Value) -> BoxFuture<'a, Result<Value>>;
}

impl<T: Tool> DynTool for T {
    fn def(&self) -> ToolDef {
        Tool::def(self)
    }

    fn invoke_boxed<'a>(&'a self, args: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(self.invoke(args))
    }
}

/// Defines a [`Tool`] from a [`ToolDef`] plus an async handler closure, so a
/// tool can be declared inline without a dedicated struct.
///
/// ```
/// use ai_core::{FnTool, ToolDef};
/// let tool = FnTool::new(
///     ToolDef::new("echo", "echoes its input", serde_json::json!({"type": "object"})),
///     |args: serde_json::Value| async move { Ok::<_, ai_core::Error>(args) },
/// );
/// ```
pub struct FnTool<F> {
    def: ToolDef,
    f: F,
}

impl<F> FnTool<F> {
    /// Create a tool from a definition and an async handler.
    pub fn new(def: ToolDef, f: F) -> Self {
        Self { def, f }
    }
}

impl<F, Fut> Tool for FnTool<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Value>> + Send,
{
    fn def(&self) -> ToolDef {
        self.def.clone()
    }

    fn invoke(&self, args: Value) -> impl Future<Output = Result<Value>> + Send {
        (self.f)(args)
    }
}

/// A registry of tools, keyed by name, for dispatching model tool calls.
///
/// Used by [`Agent`](crate::agent::Agent), but usable standalone: register
/// tools, advertise their [`ToolDef`]s on a request via [`defs`](Self::defs),
/// and run them by name with [`invoke`](Self::invoke).
#[derive(Clone, Default)]
pub struct ToolBox {
    tools: HashMap<String, Arc<dyn DynTool>>,
}

impl ToolBox {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, replacing any existing tool with the same name.
    pub fn add(&mut self, tool: impl Tool + 'static) -> &mut Self {
        let tool: Arc<dyn DynTool> = Arc::new(tool);
        self.tools.insert(tool.def().name, tool);
        self
    }

    /// Builder-style [`add`](Self::add).
    pub fn with(mut self, tool: impl Tool + 'static) -> Self {
        self.add(tool);
        self
    }

    /// The declarations of every registered tool, for binding to a request.
    pub fn defs(&self) -> Vec<ToolDef> {
        self.tools.values().map(|t| t.def()).collect()
    }

    /// Look up a registered tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn DynTool>> {
        self.tools.get(name)
    }

    /// Invoke a registered tool by name.
    ///
    /// # Errors
    /// Returns [`Error::Tool`] if no tool with that name is registered, or
    /// whatever the tool itself returns.
    pub async fn invoke(&self, name: &str, args: Value) -> Result<Value> {
        match self.tools.get(name) {
            Some(tool) => tool.invoke_boxed(args).await,
            None => Err(Error::tool(format!("unknown tool `{name}`"))),
        }
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl std::fmt::Debug for ToolBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolBox")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}
