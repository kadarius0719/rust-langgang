//! Connects an ai-core `Agent` to an MCP server (streamable HTTP) and lets the
//! model use the server's tools.
//!
//! The only glue is `McpTool`, which implements ai-core's `Tool` by forwarding
//! `invoke` to an MCP `tools/call` — so an MCP server's tools drop straight into
//! an `Agent` with no changes to the crate.
//!
//! Run `mcp-server` first, then: `cargo run --bin mcp-agent`
//! (set MODEL to a tool-capable model for the agent leg, e.g. MODEL=qwen2.5:3b)

use std::sync::Arc;

use ai_core::{load_mcp_tools, Agent, ContentBlock, McpClient, OpenAiClient, ToolBox};
use serde_json::json;

const MCP_URL: &str = "http://127.0.0.1:9000/mcp";
const OLLAMA_V1: &str = "http://localhost:11434/v1";

#[tokio::main]
async fn main() -> ai_core::Result<()> {
    let _ = dotenvy::dotenv();

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
    let reports = load_mcp_tools(&mut tools, [("mcp", MCP_URL)]).await;
    assert!(reports.iter().all(|report| report.error.is_none()));

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
