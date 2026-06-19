import { useEffect, useState } from "react";
import { getJson, postJson } from "../api";
import { Feature } from "../components/Feature";
import type { ContentBlock, Message } from "../types";

type AgentOut = { answer: string; steps: number; stopped: string; transcript: Message[] };
type McpTool = { name: string; description: string };

function Block({ block }: { block: ContentBlock }) {
  if (block.type === "text" && block.text) {
    return <div className="block text">{block.text}</div>;
  }
  if (block.type === "tool_use") {
    return (
      <div className="block tool-use">
        🔧 calls <code>{String(block.name)}</code>(<code>{JSON.stringify(block.args)}</code>)
      </div>
    );
  }
  if (block.type === "tool_result") {
    return (
      <div className="block tool-result">
        ↩ tool result: <code>{String(block.content)}</code>
      </div>
    );
  }
  return null;
}

export default function Mcp() {
  const [tools, setTools] = useState<McpTool[] | null>(null);
  const [connErr, setConnErr] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("What's the weather in Paris?");
  const [out, setOut] = useState<AgentOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getJson<McpTool[]>("/api/mcp/tools")
      .then(setTools)
      .catch((e) => setConnErr(String(e)));
  }, []);

  async function run() {
    setBusy(true);
    setError(null);
    setOut(null);
    try {
      setOut(await postJson<AgentOut>("/api/mcp/agent", { prompt }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Feature
      title="MCP Tools"
      blurb="The agent's tools come from a separate MCP server — discovered and called over the wire."
      how={
        <>
          <p>
            The backend is also an <strong>MCP client</strong>: it connects to a
            running MCP server (<code>examples/mcp</code>), discovers its tools via{" "}
            <code>tools/list</code>, and wraps each one as an ai-core{" "}
            <code>Tool</code> whose <code>invoke</code> forwards to{" "}
            <code>tools/call</code>. The <code>Agent</code> then uses them exactly
            like local tools — no core changes; the bridge is ~15 lines.
          </p>
          <pre>{`impl Tool for McpTool {
    fn def(&self) -> ToolDef { self.def.clone() }
    fn invoke(&self, args: Value) -> impl Future<Output = Result<Value>> + Send {
        let (c, name) = (self.client.clone(), self.def.name.clone());
        async move { c.call_tool(&name, args).await }   // MCP tools/call
    }
}`}</pre>
          <p>
            Requires the MCP server: <code>cd examples/mcp &amp;&amp; cargo run --bin mcp-server</code>.
          </p>
        </>
      }
    >
      {connErr ? (
        <div className="error">
          ⚠ Can't reach the MCP server. Start it with{" "}
          <code>cd examples/mcp &amp;&amp; cargo run --bin mcp-server</code>.
          <div style={{ fontSize: "0.8rem", marginTop: "0.3rem" }}>{connErr}</div>
        </div>
      ) : (
        <p className="meta">
          Connected to MCP server · {tools ? `${tools.length} tool(s):` : "loading…"}
          {tools?.map((t) => (
            <span key={t.name}>
              {" "}
              <code>{t.name}</code>
            </span>
          ))}
        </p>
      )}

      <div className="row">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          disabled={busy || !!connErr}
        />
        <button onClick={run} disabled={busy || !!connErr}>
          {busy ? "Running…" : "Run via MCP"}
        </button>
      </div>
      {error && <div className="error">⚠ {error}</div>}
      {out && (
        <div className="result">
          <p>
            <strong>Answer:</strong> {out.answer}
          </p>
          <p className="meta">
            steps: {out.steps} · stopped: {out.stopped} · tools sourced from MCP
          </p>
          <h4>Transcript</h4>
          <ol className="transcript">
            {out.transcript.map((m, i) => (
              <li key={i} className={`tmsg ${m.role}`}>
                <span className="role">{m.role}</span>
                <div className="blocks">
                  {m.content.map((b, j) => (
                    <Block key={j} block={b} />
                  ))}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </Feature>
  );
}
