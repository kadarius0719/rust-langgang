import { useState } from "react";
import { postJson } from "../api";
import { Feature } from "../components/Feature";
import type { ContentBlock, Message } from "../types";

type AgentOut = {
  answer: string;
  steps: number;
  stopped: string;
  transcript: Message[];
};

function Block({ block }: { block: ContentBlock }) {
  if (block.type === "text" && block.text) {
    return <div className="block text">{block.text}</div>;
  }
  if (block.type === "tool_use") {
    return (
      <div className="block tool-use">
        🔧 calls <code>{String(block.name)}</code>(
        <code>{JSON.stringify(block.args)}</code>)
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

export default function Agent() {
  const [prompt, setPrompt] = useState("What's the weather in Paris?");
  const [out, setOut] = useState<AgentOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setOut(null);
    try {
      setOut(await postJson<AgentOut>("/api/agent", { prompt }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Feature
      title="Agent & Tools"
      blurb="The model can call tools; their results feed back until it produces a final answer."
      how={
        <>
          <p>
            An <code>Agent</code> wraps the model with a <code>ToolBox</code> of{" "}
            <code>FnTool</code>s (here <code>get_weather</code> and{" "}
            <code>calculate</code>). The loop is: call the model → if it requested
            tools, run them and append the results → repeat until it answers or{" "}
            <code>max_steps</code> is hit. The transcript below shows every step.
          </p>
          <pre>{`let agent = Agent::new(model)
    .tool(FnTool::new(get_weather_def, |args| async move { /* ... */ }))
    .max_steps(5);
let outcome = agent.run(prompt).await?;`}</pre>
        </>
      }
    >
      <div className="row">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          disabled={busy}
        />
        <button onClick={run} disabled={busy}>
          {busy ? "Running…" : "Run agent"}
        </button>
      </div>
      {error && <div className="error">⚠ {error}</div>}
      {out && (
        <div className="result">
          <p>
            <strong>Answer:</strong> {out.answer}
          </p>
          <p className="meta">
            steps: {out.steps} · stopped: {out.stopped} · {out.transcript.length} messages
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
