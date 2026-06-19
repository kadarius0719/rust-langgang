#!/usr/bin/env bash
# Runs once after the dev container is created: installs Ollama, warms the Rust
# and npm caches so the first `cargo run` / `npm run dev` is quick.
set -uo pipefail

echo "▶ Installing Ollama …"
curl -fsSL https://ollama.com/install.sh | sh \
  || echo "  (the install script warned — the ollama binary should still be on PATH)"

echo "▶ Pre-fetching Rust dependencies …"
cargo fetch || true
(cd examples/mcp && cargo fetch) || true
(cd examples/web-ui/backend && cargo fetch) || true

echo "▶ Installing frontend dependencies …"
(cd examples/web-ui/frontend && npm install) || true

cat <<'EOF'

✅ Dev container ready. Run the examples (each in its own terminal):

  ollama serve
  ollama pull llama3.2:1b                                  # ~1.3 GB, one time

  (cd examples/mcp && cargo run --bin mcp-server)          # :9000  (for the MCP page)
  (cd examples/web-ui/backend && cargo run)                # :8080  (STORE=sqlite to persist)
  (cd examples/web-ui/frontend && npm run dev)             # :5173

Then open the forwarded port 5173 in your browser.

Quick standalone MCP demo (no web UI):
  (cd examples/mcp && cargo run --bin mcp-server) &        # terminal A
  (cd examples/mcp && cargo run --bin mcp-agent)           # terminal B
EOF
