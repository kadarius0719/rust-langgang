// Talks to the Rust backend (see webui-backend). Permissive CORS lets this
// dev-server origin call it directly.
export const API = "http://localhost:8080";

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as T;
}

/** POST a prompt to /api/chat/stream and stream tokens out of the SSE body. */
export async function streamChat(
  body: unknown,
  onToken: (t: string) => void,
  onError: (m: string) => void,
) {
  const res = await fetch(`${API}/api/chat/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) return onError(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      const payload = data.join("\n");
      if (event === "error") return onError(payload);
      if (event === "done") continue;
      try {
        onToken((JSON.parse(payload) as { text: string }).text);
      } catch {
        /* ignore keep-alive / non-JSON frames */
      }
    }
  }
}
