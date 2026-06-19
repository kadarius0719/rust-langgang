import { useState } from "react";
import { postJson } from "../api";
import { Feature } from "../components/Feature";

type Person = { name: string; age: number; occupation: string };

export default function Structured() {
  const [prompt, setPrompt] = useState("Ada Lovelace, age 36, mathematician and writer.");
  const [person, setPerson] = useState<Person | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setPerson(null);
    try {
      setPerson(await postJson<Person>("/api/structured", { prompt }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Feature
      title="Structured Output"
      blurb="Get the model's answer back as a typed Rust value, not free text."
      how={
        <>
          <p>
            The backend derives a JSON Schema from the Rust type, asks the model to
            answer in that shape (<code>response_format</code>), and deserializes the
            reply straight into the struct via <code>model.structured::&lt;T&gt;()</code>.
          </p>
          <pre>{`#[derive(Deserialize, JsonSchema)]
struct Person { name: String, age: u32, occupation: String }

let person: Person = model.structured::<Person>(request).await?;`}</pre>
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
          {busy ? "Extracting…" : "Extract"}
        </button>
      </div>
      {error && <div className="error">⚠ {error}</div>}
      {person && (
        <div className="result">
          <div className="fields">
            <div className="field">
              <span className="key">name</span>
              <span className="val">{person.name}</span>
            </div>
            <div className="field">
              <span className="key">age</span>
              <span className="val">{person.age}</span>
            </div>
            <div className="field">
              <span className="key">occupation</span>
              <span className="val">{person.occupation}</span>
            </div>
          </div>
          <pre className="json">{JSON.stringify(person, null, 2)}</pre>
        </div>
      )}
    </Feature>
  );
}
