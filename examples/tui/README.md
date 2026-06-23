# TUI Agent Chat Example

A terminal user interface for chatting with an `ai-core` agent.

The example loads a local `.env` file automatically if present.

## Requirements

- Rust 1.82+
- For local models: Ollama, LM Studio, llama.cpp, or any other OpenAI-compatible server
- For OpenAI: a valid API key

## Usage

### Local model

```bash
# Start your local OpenAI-compatible server first
ollama serve

# Then run the TUI
cargo run
```

The app defaults to `http://localhost:11434/v1` when no API key is set.

Override the local endpoint with:

```bash
AI_BASE_URL=http://127.0.0.1:1234/v1 cargo run
```

You can also point to another OpenAI-compatible provider with `OPENAI_BASE_URL`.

### OpenAI API

```bash
OPENAI_API_KEY=sk-... MODEL=gpt-4o-mini cargo run
```

The default model is `gemma4:e4b`. Override it with `MODEL=...`.

You can also place those variables in a `.env` file:

```dotenv
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini
MCP_URLS=http://127.0.0.1:8811/mcp,http://127.0.0.1:9000/mcp
```

### MCP tools

The TUI can load tools from one or more MCP servers.

```bash
# Single server (backward compatible)
MCP_URL=http://127.0.0.1:8811/mcp cargo run

# Multiple servers
MCP_URLS=http://127.0.0.1:8811/mcp,http://127.0.0.1:9000/mcp cargo run
```

When `MCP_URLS` is set, the app assigns server labels like `mcp1`, `mcp2`, and
registers tool names as `<label>_<tool_name>` to avoid collisions.

## Controls

- Type a message and press Enter to send
- `Ctrl+C` exits
- `Ctrl+L` clears the chat history
- `Ctrl+T` toggles thinking visibility
- `Up` / `Down` scroll the chat
- `PageUp` / `PageDown` scroll faster
- `Home` jumps to the top
- `End` jumps to the latest message
