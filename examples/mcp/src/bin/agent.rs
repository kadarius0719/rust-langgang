//! Connects an ai-core `Agent` to an MCP server (streamable HTTP) and lets the
//! model use the server's tools.
//!
//! The only glue is `McpTool`, which implements ai-core's `Tool` by forwarding
//! `invoke` to an MCP `tools/call` — so an MCP server's tools drop straight into
//! an `Agent` with no changes to the crate.
//!
//! Run `mcp-server` first, then: `cargo run --bin mcp-agent`
//! (set MODEL to a tool-capable model for the agent leg, e.g. MODEL=qwen2.5:3b)

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ai_core::{Agent, ContentBlock, Error, OpenAiClient, Result, Tool, ToolBox, ToolDef};
use serde::Deserialize;
use serde_json::{json, Value};

const MCP_URL: &str = "http://127.0.0.1:9000/mcp";
const OLLAMA_V1: &str = "http://localhost:11434/v1";

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

// ---------------------------------------------------------------------------
// A minimal MCP client (Streamable HTTP, JSON response mode).
// ---------------------------------------------------------------------------
struct McpClient {
    http: reqwest::Client,
    url: String,
    next_id: AtomicU64,
}

#[derive(Deserialize)]
struct McpToolInfo {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(rename = "inputSchema", default)]
    input_schema: Value,
}

impl McpClient {
    async fn connect(url: impl Into<String>) -> std::result::Result<Self, BoxErr> {
        let client = Self {
            http: reqwest::Client::new(),
            url: url.into(),
            next_id: AtomicU64::new(1),
        };
        client
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": { "name": "ai-core-mcp-demo", "version": "0.1.0" }
                }),
            )
            .await?;
        client.notify("notifications/initialized").await?;
        Ok(client)
    }

    async fn list_tools(&self) -> std::result::Result<Vec<McpToolInfo>, BoxErr> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result.get("tools").cloned().unwrap_or(Value::Null);
        Ok(serde_json::from_value(tools)?)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> std::result::Result<Value, BoxErr> {
        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;
        // MCP returns { content: [{ type: "text", text: "..." }], isError }.
        let text = result
            .get("content")
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(Value::as_str))
                    .collect::<String>()
            })
            .unwrap_or_default();
        // Hand back the tool's JSON if it parses, else the raw text.
        Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    async fn request(&self, method: &str, params: Value) -> std::result::Result<Value, BoxErr> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let resp = self
            .http
            .post(&self.url)
            .header("accept", "application/json, text/event-stream")
            .json(&body)
            .send()
            .await?;
        let value: Value = resp.json().await?;
        if let Some(err) = value.get("error") {
            return Err(format!("mcp error: {err}").into());
        }
        Ok(value.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn notify(&self, method: &str) -> std::result::Result<(), BoxErr> {
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": {} });
        self.http
            .post(&self.url)
            .header("accept", "application/json")
            .json(&body)
            .send()
            .await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The bridge: an MCP tool exposed as an ai-core `Tool`.
// ---------------------------------------------------------------------------
struct McpTool {
    client: Arc<McpClient>,
    def: ToolDef,
}

impl Tool for McpTool {
    fn def(&self) -> ToolDef {
        self.def.clone()
    }

    fn invoke(&self, args: Value) -> impl Future<Output = Result<Value>> + Send {
        let client = self.client.clone();
        let name = self.def.name.clone();
        async move {
            client
                .call_tool(&name, args)
                .await
                .map_err(|e| Error::tool(e.to_string()))
        }
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), BoxErr> {
    // 1. Connect to the MCP server and discover its tools.
    let client = Arc::new(McpClient::connect(MCP_URL).await?);
    let infos = client.list_tools().await?;
    let names: Vec<&str> = infos.iter().map(|t| t.name.as_str()).collect();
    println!("Discovered {} MCP tool(s): {names:?}", infos.len());

    // 2. Call one DIRECTLY (no model) — proves the transport works regardless of
    //    how good the model is at choosing tools.
    let direct = client
        .call_tool("get_weather", json!({ "city": "Paris" }))
        .await?;
    println!("Direct  tools/call  get_weather(Paris) -> {direct}");
    let sum = client
        .call_tool("calculate", json!({ "a": 128, "b": 47 }))
        .await?;
    println!("Direct  tools/call  calculate(128,47)  -> {sum}\n");

    // 3. Bridge every MCP tool into an ai-core ToolBox.
    let mut tools = ToolBox::new();
    for info in infos {
        tools.add(McpTool {
            client: client.clone(),
            def: ToolDef::new(info.name, info.description, info.input_schema),
        });
    }

    // 4. Run an Agent whose tools come entirely from the MCP server.
    let model_id = std::env::var("MODEL").unwrap_or_else(|_| "llama3.2:1b".to_string());
    println!("Running Agent (model: {model_id}) — tools sourced from MCP …");
    let model = OpenAiClient::local(OLLAMA_V1).chat_model(&model_id);
    let agent = Agent::new(model)
        .system("Use the available tools when relevant, then answer in one short sentence.")
        .tools(tools)
        .max_steps(5);

    let outcome = agent.run("What's the weather in Paris?").await?;
    let used: Vec<String> = outcome
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolUse { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();

    println!("Agent answer : {}", outcome.text());
    println!(
        "MCP tools the model invoked: {used:?} (steps={})",
        outcome.steps
    );
    Ok(())
}
