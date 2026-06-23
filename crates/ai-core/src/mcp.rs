//! Minimal MCP client support over Streamable HTTP plus a bridge into [`ToolBox`].

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{Error, Result, Tool, ToolBox, ToolDef};

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const MCP_CLIENT_NAME: &str = "ai-core";
const MCP_CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// An MCP client speaking JSON-RPC 2.0 over Streamable HTTP.
pub struct McpClient {
    http: reqwest::Client,
    url: String,
    next_id: AtomicU64,
}

/// Tool metadata advertised by an MCP server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolInfo {
    /// Remote tool name.
    pub name: String,
    /// What the tool does and when to call it.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the tool's arguments.
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
}

/// Per-server load report for [`load_mcp_tools`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpLoadReport {
    /// Sanitized label used to namespace registered tool names.
    pub label: String,
    /// Original server URL.
    pub url: String,
    /// Registered, namespaced tool names.
    pub tool_names: Vec<String>,
    /// Failure detail, if this server could not be loaded.
    pub error: Option<String>,
}

/// An MCP tool exposed as an `ai-core` [`Tool`].
pub struct McpTool {
    client: Arc<McpClient>,
    def: ToolDef,
    remote_name: String,
}

impl McpClient {
    /// Connect to an MCP server, initialize the session, and mark the client ready.
    pub async fn connect(url: impl Into<String>) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(5))
            .build()?;
        let client = Self {
            http,
            url: url.into(),
            next_id: AtomicU64::new(1),
        };
        client
            .request(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": MCP_CLIENT_NAME, "version": MCP_CLIENT_VERSION }
                }),
            )
            .await?;
        client.notify("notifications/initialized").await?;
        Ok(client)
    }

    /// The server URL this client is connected to.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Discover tools advertised by the server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result.get("tools").cloned().unwrap_or(Value::Null);
        Ok(serde_json::from_value(tools)?)
    }

    /// Invoke a remote tool by its original MCP name.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
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

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let value: Value = self
            .http
            .post(&self.url)
            .header("accept", "application/json, text/event-stream")
            .json(&body)
            .send()
            .await?
            .json()
            .await?;
        if let Some(err) = value.get("error") {
            return Err(Error::tool(format!("mcp error: {err}")));
        }
        Ok(value.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn notify(&self, method: &str) -> Result<()> {
        let body = json!({ "jsonrpc": "2.0", "method": method, "params": {} });
        self.http
            .post(&self.url)
            .header("accept", "application/json, text/event-stream")
            .json(&body)
            .send()
            .await?;
        Ok(())
    }
}

impl McpTool {
    /// Create a namespaced local tool that forwards to a remote MCP tool.
    pub fn new(
        client: Arc<McpClient>,
        local_name: impl Into<String>,
        remote_name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            client,
            def: ToolDef::new(local_name, description, parameters),
            remote_name: remote_name.into(),
        }
    }

    /// Build a namespaced local tool from discovered MCP metadata.
    pub fn from_info(client: Arc<McpClient>, server_label: &str, info: McpToolInfo) -> Self {
        let remote_name = info.name;
        let local_name = namespaced_tool_name(server_label, &remote_name);
        let description = describe_tool(server_label, &remote_name, &info.description);
        Self::new(
            client,
            local_name,
            remote_name,
            description,
            info.input_schema,
        )
    }
}

impl Tool for McpTool {
    fn def(&self) -> ToolDef {
        self.def.clone()
    }

    fn invoke(&self, args: Value) -> impl Future<Output = Result<Value>> + Send {
        let client = self.client.clone();
        let remote_name = self.remote_name.clone();
        async move { client.call_tool(&remote_name, args).await }
    }
}

/// Load tools from one or more MCP servers into a [`ToolBox`].
///
/// Each tool is namespaced as `<server_label>_<tool_name>`. Servers are loaded
/// independently; failures are reported and do not prevent successful servers
/// from contributing tools.
pub async fn load_mcp_tools<I, L, U>(tools: &mut ToolBox, servers: I) -> Vec<McpLoadReport>
where
    I: IntoIterator<Item = (L, U)>,
    L: AsRef<str>,
    U: AsRef<str>,
{
    let mut reports = Vec::new();
    for (label, url) in servers {
        let label = sanitize_server_label(label.as_ref());
        let url = url.as_ref().trim().to_string();
        match McpClient::connect(url.clone()).await {
            Ok(client) => {
                let client = Arc::new(client);
                match client.list_tools().await {
                    Ok(infos) => {
                        let mut tool_names = Vec::with_capacity(infos.len());
                        for info in infos {
                            let tool = McpTool::from_info(client.clone(), &label, info);
                            tool_names.push(tool.def().name.clone());
                            tools.add(tool);
                        }
                        reports.push(McpLoadReport {
                            label,
                            url,
                            tool_names,
                            error: None,
                        });
                    }
                    Err(err) => reports.push(McpLoadReport {
                        label,
                        url,
                        tool_names: Vec::new(),
                        error: Some(err.to_string()),
                    }),
                }
            }
            Err(err) => reports.push(McpLoadReport {
                label,
                url,
                tool_names: Vec::new(),
                error: Some(err.to_string()),
            }),
        }
    }
    reports
}

/// Sanitize a server label for use in namespaced tool names.
pub fn sanitize_server_label(label: &str) -> String {
    let mut out = String::with_capacity(label.len().max(1));
    for ch in label.trim().chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches('_');
    if out.is_empty() {
        "mcp".to_string()
    } else {
        out.to_string()
    }
}

/// Build the default server label for the Nth configured MCP URL.
pub fn default_server_label(index: usize) -> String {
    format!("mcp{}", index + 1)
}

fn namespaced_tool_name(server_label: &str, tool_name: &str) -> String {
    format!("{}_{}", sanitize_server_label(server_label), tool_name)
}

fn describe_tool(server_label: &str, tool_name: &str, description: &str) -> String {
    let prefix = format!(
        "Source server `{}`; original MCP tool `{}`.",
        sanitize_server_label(server_label),
        tool_name
    );
    let description = description.trim();
    if description.is_empty() {
        prefix
    } else {
        format!("{prefix} {description}")
    }
}
