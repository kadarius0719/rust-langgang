# Dev container

A single container with everything the examples need — **Rust**, **Node**, and
**Ollama** — so they run anywhere with no local toolchain setup. Because every
example talks to `localhost` (Ollama `:11434`, the backend `:8080`, the MCP
server `:9000`), one container Just Works: no example changes, no compose wiring.

## Run it

**VS Code / Cursor:** install the *Dev Containers* extension, open this repo, and
choose **"Reopen in Container"**.

**CLI** (the [`@devcontainers/cli`](https://github.com/devcontainers/cli), needs Docker):

```sh
npm install -g @devcontainers/cli
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
```

The first build pulls the Rust image + Node feature and runs
[`post-create.sh`](./post-create.sh) (installs Ollama, warms the cargo/npm
caches). When it finishes it prints the run commands.

## Run the examples (inside the container)

```sh
ollama serve &
ollama pull llama3.2:1b                              # one time, ~1.3 GB

(cd examples/mcp && cargo run --bin mcp-server) &    # :9000  (for the MCP Tools page)
(cd examples/web-ui/backend && cargo run) &          # :8080
(cd examples/web-ui/frontend && npm run dev)         # :5173
```

Open the forwarded **port 5173** in your browser. Forwarded ports: `5173`
(frontend), `8080` (backend), `9000` (MCP server), `11434` (Ollama).

## Notes

- **CPU-only.** Ollama runs on CPU in the container, so the model is slower than
  on a GPU host — `llama3.2:1b` keeps it snappy enough for a demo. To use the
  host's GPU/Ollama instead, point the examples at `host.docker.internal:11434`
  (the URLs are constants in the example `main.rs` files).
- Give Docker a few GB of RAM; a 1B model needs ~2 GB to run.
- This only sets up the examples — the core crate (`crates/ai-core`) builds with
  `cargo build` anywhere; it has no system dependencies.
