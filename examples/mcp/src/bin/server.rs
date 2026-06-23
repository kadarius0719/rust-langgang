//! A minimal MCP server over the Streamable HTTP transport (JSON response mode).
//!
//! Implements just enough of the protocol — JSON-RPC 2.0 over a single
//! `POST /mcp` endpoint — to handle `initialize`, `tools/list`, and
//! `tools/call`. It exposes two demo tools. Run it, then run `mcp-agent`.
//!
//! (A full server would also do session ids, SSE-streamed responses, and the
//! GET server→client channel; those are intentionally omitted for clarity.)

use axum::{http::StatusCode, routing::post, Json, Router};
use serde_json::{json, Value};

const ADDR: &str = "127.0.0.1:9000";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let app = Router::new().route("/mcp", post(handle));
    let listener = tokio::net::TcpListener::bind(ADDR).await?;
    println!("MCP server (streamable HTTP) listening on http://{ADDR}/mcp");
    println!("tools: get_weather, calculate");
    axum::serve(listener, app).await?;
    Ok(())
}

/// One JSON-RPC message in, one JSON-RPC message out.
async fn handle(Json(req): Json<Value>) -> (StatusCode, Json<Value>) {
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    let id = req.get("id").cloned();

    // Notifications carry no id and expect no response body.
    let Some(id) = id else {
        return (StatusCode::ACCEPTED, Json(json!({})));
    };

    let result = match method {
        "initialize" => json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "ai-core-demo-mcp", "version": "0.1.0" }
        }),
        "tools/list" => json!({ "tools": tool_defs() }),
        "tools/call" => {
            let params = req.get("params").cloned().unwrap_or(Value::Null);
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Value::Null);
            match call_tool(name, &args) {
                Ok(value) => json!({
                    "content": [{ "type": "text", "text": value.to_string() }],
                    "isError": false
                }),
                Err(message) => json!({
                    "content": [{ "type": "text", "text": message }],
                    "isError": true
                }),
            }
        }
        _ => {
            return rpc_error(id, -32601, "method not found");
        }
    };

    (
        StatusCode::OK,
        Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })),
    )
}

fn rpc_error(id: Value, code: i64, message: &str) -> (StatusCode, Json<Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message }
        })),
    )
}

fn tool_defs() -> Value {
    json!([
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "inputSchema": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        },
        {
            "name": "calculate",
            "description": "Add two numbers a and b.",
            "inputSchema": {
                "type": "object",
                "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                "required": ["a", "b"]
            }
        }
    ])
}

fn call_tool(name: &str, args: &Value) -> Result<Value, String> {
    match name {
        "get_weather" => {
            let city = args.get("city").and_then(Value::as_str).unwrap_or("?");
            Ok(json!({ "city": city, "temp_f": 72, "conditions": "sunny" }))
        }
        "calculate" => {
            let a = args.get("a").and_then(Value::as_f64).unwrap_or(0.0);
            let b = args.get("b").and_then(Value::as_f64).unwrap_or(0.0);
            Ok(json!({ "sum": a + b }))
        }
        other => Err(format!("unknown tool `{other}`")),
    }
}
