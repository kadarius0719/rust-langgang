//! A minimal MCP client (Streamable HTTP, JSON response mode) plus a bridge that
//! exposes an MCP tool as an ai-core `Tool`. The same code as `examples/mcp`,
//! reused here so the web UI's agent can pull its tools from a running MCP
//! server. See `examples/mcp/README.md` for the protocol notes.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ai_core::{Error, Result, Tool, ToolDef};
use serde::Deserialize;
use serde_json::{json, Value};

pub type BoxErr = Box<dyn std::error::Error + Send + Sync>;

pub struct McpClient {
    http: reqwest::Client,
    url: String,
    next_id: AtomicU64,
}

#[derive(Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
}

impl McpClient {
    pub async fn connect(url: &str) -> std::result::Result<Self, BoxErr> {
        let client = Self {
            http: reqwest::Client::new(),
            url: url.to_string(),
            next_id: AtomicU64::new(1),
        };
        client
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": { "name": "ai-core-web-ui", "version": "0.1.0" }
                }),
            )
            .await?;
        client.notify("notifications/initialized").await?;
        Ok(client)
    }

    pub async fn list_tools(&self) -> std::result::Result<Vec<McpToolInfo>, BoxErr> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result.get("tools").cloned().unwrap_or(Value::Null);
        Ok(serde_json::from_value(tools)?)
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> std::result::Result<Value, BoxErr> {
        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;
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

/// An MCP tool exposed as an ai-core `Tool`: `invoke` forwards to `tools/call`.
pub struct McpTool {
    client: Arc<McpClient>,
    def: ToolDef,
}

impl McpTool {
    pub fn new(client: Arc<McpClient>, def: ToolDef) -> Self {
        Self { client, def }
    }
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
