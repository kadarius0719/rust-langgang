# mcp — an `ai-core` Agent using tools from an MCP server

Shows that an [MCP](https://modelcontextprotocol.io) server's tools drop into an
`ai-core` `Agent` with **no changes to the crate** — the only glue is a ~15-line
`McpTool` that implements ai-core's `Tool` trait by forwarding `invoke` to an MCP
`tools/call`.

```
ai-core Agent ──▶ McpTool (impl Tool) ──HTTP JSON-RPC──▶ MCP server (:9000)
      │                                                   tools: get_weather, calculate
      └──▶ Ollama /v1 (:11434)  (the model that decides which tool to call)
```

Two binaries:

- **`mcp-server`** — a minimal MCP server over the **Streamable HTTP** transport
  (`POST /mcp`, JSON-RPC 2.0): handles `initialize`, `tools/list`, `tools/call`,
  and exposes two demo tools.
- **`mcp-agent`** — connects to it, discovers the tools, calls them directly
  (to prove the transport), then runs an ai-core `Agent` whose entire `ToolBox`
  is sourced from the MCP server.

## Run it

```sh
# 1. model (for the agent leg)
ollama serve && ollama pull llama3.2:1b

# 2. MCP server  (terminal A)
cargo run --bin mcp-server

# 3. agent       (terminal B)
cargo run --bin mcp-agent
```

Expected output:

```
Discovered 2 MCP tool(s): ["get_weather", "calculate"]
Direct  tools/call  get_weather(Paris) -> {"city":"Paris","conditions":"sunny","temp_f":72}
Direct  tools/call  calculate(128,47)  -> {"sum":175.0}

Running Agent (model: llama3.2:1b) — tools sourced from MCP …
Agent answer : The current weather in Paris is sunny with a temperature of 72°F.
MCP tools the model invoked: ["get_weather"] (steps=2)
```

## The integration, in full

```rust
struct McpTool { client: Arc<McpClient>, def: ToolDef }

impl Tool for McpTool {
    fn def(&self) -> ToolDef { self.def.clone() }
    fn invoke(&self, args: Value) -> impl Future<Output = Result<Value>> + Send {
        let (client, name) = (self.client.clone(), self.def.name.clone());
        async move { client.call_tool(&name, args).await.map_err(|e| Error::tool(e.to_string())) }
    }
}

// discovery → ToolBox → Agent (all existing ai-core API):
for info in client.list_tools().await? {
    tools.add(McpTool { client: client.clone(),
        def: ToolDef::new(info.name, info.description, info.input_schema) });
}
let agent = Agent::new(model).tools(tools);
```

MCP's tool model (`name` + `description` + JSON-Schema `inputSchema` + a
JSON-in/JSON-out call) maps 1:1 onto ai-core's `ToolDef` + `Tool::invoke`, which
is why the bridge is so small.

## Notes & scope

- **The transport is intentionally minimal** — JSON response mode only. A
  production client/server would also do session ids (`Mcp-Session-Id`),
  SSE-streamed responses, the GET server→client channel, and resumability. For a
  real integration, use the official Rust SDK [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk).
- **Tool *selection* depends on the model.** The direct `tools/call` always
  works; whether the *agent* chooses the right tool is up to the LLM. `llama3.2:1b`
  is unreliable at this — set a tool-tuned model for the agent leg:
  ```sh
  ollama pull qwen2.5:3b && MODEL=qwen2.5:3b cargo run --bin mcp-agent
  ```
- Standalone crate (excluded from the workspace); depends on the crate via a
  relative path, so it builds from a clone.
