//! An explicit, inspectable tool-calling loop.
//!
//! This is the whole "agent" — no `AgentExecutor` ceremony. The loop is:
//! call the model → if it requested tools, run them via the [`ToolBox`] and
//! append the results → repeat until the model answers without tools or a
//! `max_steps` cap is hit. Each step emits [`TraceEvent`]s (when a [`Tracer`] is
//! attached) so the whole decision trail is auditable.
//!
//! ```ignore
//! use ai_core::{Agent, FnTool, ToolDef};
//!
//! let agent = Agent::new(model)
//!     .system("You are a helpful assistant.")
//!     .tool(FnTool::new(
//!         ToolDef::new("get_weather", "Get the weather for a city.",
//!             serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}})),
//!         |args| async move { Ok::<_, ai_core::Error>(serde_json::json!({"temp": "72F"})) },
//!     ))
//!     .max_steps(5);
//!
//! let outcome = agent.run("What's the weather in Paris?").await?;
//! println!("{}", outcome.text());
//! ```

use std::sync::Arc;

use crate::{
    error::Result,
    message::Message,
    model::ChatModel,
    request::ChatRequest,
    response::ChatResponse,
    tool::{Tool, ToolBox, ToolDef},
    trace::{NoopTracer, TraceEvent, TraceId, Tracer},
};

/// Why an agent run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopCause {
    /// The model produced a final answer with no tool calls.
    Final,
    /// The `max_steps` cap was reached before a final answer.
    MaxSteps,
}

/// The result of an [`Agent`] run.
#[derive(Debug)]
pub struct AgentOutcome {
    /// The last model response (the final answer on a [`StopCause::Final`] run).
    pub response: ChatResponse,
    /// The full transcript, including tool calls and tool results — ready to
    /// continue the conversation from.
    pub messages: Vec<Message>,
    /// How many model calls were made.
    pub steps: u32,
    /// Why the run stopped.
    pub stopped: StopCause,
}

impl AgentOutcome {
    /// The text of the final response.
    pub fn text(&self) -> String {
        self.response.text()
    }
}

/// A tool-calling loop over a [`ChatModel`].
///
/// Build one with [`Agent::new`] and the chained setters, then [`run`](Agent::run).
/// Per-request knobs beyond `system`/`max_tokens` are best applied by wrapping
/// the model with [`map_request`](crate::ChatModelExt::map_request) before
/// handing it to the agent.
pub struct Agent<M> {
    model: M,
    model_id: String,
    tools: ToolBox,
    system: Option<String>,
    max_tokens: Option<u32>,
    max_steps: u32,
    tracer: Arc<dyn Tracer>,
}

impl<M: ChatModel> Agent<M> {
    /// Create an agent over `model`, with no tools and a default step cap of 8.
    pub fn new(model: M) -> Self {
        Self {
            model,
            model_id: "model".to_string(),
            tools: ToolBox::new(),
            system: None,
            max_tokens: None,
            max_steps: 8,
            tracer: Arc::new(NoopTracer),
        }
    }

    /// Set the model id placed in each [`ChatRequest`].
    ///
    /// Handle-style adapters (e.g. the OpenAI one) already know their model and
    /// ignore this; set it for adapters that read `ChatRequest::model`.
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Register a tool the model may call.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.add(tool);
        self
    }

    /// Replace the whole tool registry.
    pub fn tools(mut self, tools: ToolBox) -> Self {
        self.tools = tools;
        self
    }

    /// Set the system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the output token cap per model call.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the maximum number of model calls before stopping (minimum 1).
    pub fn max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    /// Attach a tracer for the agent's own decision events.
    pub fn tracer(mut self, tracer: Arc<dyn Tracer>) -> Self {
        self.tracer = tracer;
        self
    }

    /// Run the loop starting from a single user message.
    pub async fn run(&self, user: impl Into<String>) -> Result<AgentOutcome> {
        self.run_messages(vec![Message::user(user)]).await
    }

    /// Run the loop starting from an existing message list.
    pub async fn run_messages(&self, initial: Vec<Message>) -> Result<AgentOutcome> {
        let trace_id = TraceId::next();
        let tool_defs = self.tools.defs();
        let mut messages = initial;
        let mut steps = 0u32;

        loop {
            steps += 1;
            self.tracer.record(TraceEvent::AgentStep {
                trace_id,
                step: steps,
                action: "call_model".to_string(),
            });

            let request = self.build_request(messages.clone(), tool_defs.clone())?;
            let response = self.model.chat(request).await?;
            messages.push(response.message.clone());

            let calls: Vec<(String, String, serde_json::Value)> = response
                .tool_uses()
                .iter()
                .map(|t| (t.id.to_string(), t.name.to_string(), t.args.clone()))
                .collect();

            if calls.is_empty() {
                return Ok(AgentOutcome {
                    response,
                    messages,
                    steps,
                    stopped: StopCause::Final,
                });
            }

            for (id, name, args) in calls {
                self.tracer.record(TraceEvent::ToolSelected {
                    trace_id,
                    id: id.clone(),
                    name: name.clone(),
                    args: args.clone(),
                });
                let (output, ok) = match self.tools.invoke(&name, args).await {
                    Ok(value) => (value_to_string(&value), true),
                    Err(e) => (e.to_string(), false),
                };
                self.tracer.record(TraceEvent::ToolExecuted {
                    trace_id,
                    id: id.clone(),
                    name,
                    ok,
                    output_len: output.len(),
                });
                messages.push(Message::tool_result(id, output, !ok));
            }

            if steps >= self.max_steps {
                return Ok(AgentOutcome {
                    response,
                    messages,
                    steps,
                    stopped: StopCause::MaxSteps,
                });
            }
        }
    }

    fn build_request(
        &self,
        messages: Vec<Message>,
        tool_defs: Vec<ToolDef>,
    ) -> Result<ChatRequest> {
        let mut builder = ChatRequest::builder(self.model_id.as_str())
            .messages(messages)
            .tools(tool_defs);
        if let Some(system) = &self.system {
            builder = builder.system(system.clone());
        }
        if let Some(max_tokens) = self.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }
        builder.build()
    }
}

/// Render a tool's JSON result as the string carried in a tool-result message.
fn value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
