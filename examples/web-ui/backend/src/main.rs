//! A tiny HTTP backend that puts `ai-core` behind a REST + SSE API for a web UI,
//! with conversation **memory** (per-session history), **logging** (structured
//! trace events), and a pluggable memory backend: in-process by default, or
//! SQLite when run with `STORE=sqlite`. Fully offline via Ollama.
//!
//!   GET  /api/health            -> liveness
//!   POST /api/chat              -> non-streaming chat {prompt, session?} -> {text, usage}
//!   POST /api/chat/stream       -> SSE token stream    {prompt, session?} -> data: {"text": "..."}
//!   POST /api/agent             -> Agent tool-loop (local FnTools)
//!   POST /api/structured        -> structured output into a typed struct
//!   GET  /api/mcp/tools         -> tools advertised by the configured MCP server
//!   POST /api/mcp/agent         -> Agent whose tools are sourced from the MCP server
//!   GET  /api/history/{session} -> the stored conversation for a session
//!   GET  /api/logs              -> the recorded trace events

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;

use ai_core::{
    Agent, ChatHistory, ChatModel, ChatModelExt, ChatRequest, ChatStore, DynChatStore, FnTool,
    InMemoryChatStore, McpClient, McpTool, Message, OpenAiClient, OpenAiModel, RecordingTracer,
    StreamEvent, StructuredExt, ToolBox, ToolDef, TraceEvent, Traced, Tracer,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tower_http::cors::CorsLayer;

const OLLAMA_V1: &str = "http://localhost:11434/v1";
const MODEL: &str = "llama3.2:1b";
const SYSTEM: &str = "You are a concise, helpful assistant.";

// --- logging: a tracer that prints each event AND records it for /api/logs ---
#[derive(Clone)]
struct LogTracer {
    rec: RecordingTracer,
}

impl Tracer for LogTracer {
    fn record(&self, event: TraceEvent) {
        eprintln!(
            "[trace] {}",
            serde_json::to_string(&event).unwrap_or_default()
        );
        self.rec.record(event);
    }
}

// --- memory backend #2: a ChatStore backed by SQLite (one row per message) ---
#[derive(Clone)]
struct SqliteChatStore {
    pool: SqlitePool,
}

impl SqliteChatStore {
    async fn connect(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts).await?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS messages (
                 id      INTEGER PRIMARY KEY AUTOINCREMENT,
                 session TEXT NOT NULL,
                 payload TEXT NOT NULL)",
        )
        .execute(&pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session, id)")
            .execute(&pool)
            .await?;
        Ok(Self { pool })
    }
}

/// sqlx error -> ai-core error (via the `Box<dyn Error>` escape hatch).
fn db_err(err: sqlx::Error) -> ai_core::Error {
    ai_core::Error::from(Box::new(err) as Box<dyn std::error::Error + Send + Sync>)
}

// Implementing ai-core's `ChatStore` is all it takes to plug in a new backend;
// `serde` (de)serializes each provider-neutral `Message` to a JSON payload.
impl ChatStore for SqliteChatStore {
    fn load(&self, session: &str) -> impl Future<Output = ai_core::Result<Vec<Message>>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            let rows: Vec<String> =
                sqlx::query_scalar("SELECT payload FROM messages WHERE session = ? ORDER BY id")
                    .bind(&session)
                    .fetch_all(&pool)
                    .await
                    .map_err(db_err)?;
            let mut messages = Vec::with_capacity(rows.len());
            for row in rows {
                messages.push(serde_json::from_str(&row)?);
            }
            Ok(messages)
        }
    }

    fn save(
        &self,
        session: &str,
        messages: Vec<Message>,
    ) -> impl Future<Output = ai_core::Result<()>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            let mut tx = pool.begin().await.map_err(db_err)?;
            sqlx::query("DELETE FROM messages WHERE session = ?")
                .bind(&session)
                .execute(&mut *tx)
                .await
                .map_err(db_err)?;
            for message in &messages {
                let payload = serde_json::to_string(message)?;
                sqlx::query("INSERT INTO messages (session, payload) VALUES (?, ?)")
                    .bind(&session)
                    .bind(payload)
                    .execute(&mut *tx)
                    .await
                    .map_err(db_err)?;
            }
            tx.commit().await.map_err(db_err)?;
            Ok(())
        }
    }

    fn append(
        &self,
        session: &str,
        message: Message,
    ) -> impl Future<Output = ai_core::Result<()>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            let payload = serde_json::to_string(&message)?;
            sqlx::query("INSERT INTO messages (session, payload) VALUES (?, ?)")
                .bind(&session)
                .bind(payload)
                .execute(&pool)
                .await
                .map_err(db_err)?;
            Ok(())
        }
    }

    fn clear(&self, session: &str) -> impl Future<Output = ai_core::Result<()>> + Send {
        let pool = self.pool.clone();
        let session = session.to_string();
        async move {
            sqlx::query("DELETE FROM messages WHERE session = ?")
                .bind(&session)
                .execute(&pool)
                .await
                .map_err(db_err)?;
            Ok(())
        }
    }
}

#[derive(Clone)]
struct AppState {
    /// The model, wrapped so every call emits trace events.
    model: Arc<Traced<OpenAiModel>>,
    /// A read handle onto recorded trace events (shares the buffer).
    logs: RecordingTracer,
    /// Conversation memory — runtime-selected backend behind ai-core's facade.
    store: Arc<dyn DynChatStore>,
    model_id: String,
    /// URL of the MCP server whose tools the `/api/mcp/*` endpoints use.
    mcp_url: String,
}

#[derive(Deserialize)]
struct ChatIn {
    prompt: String,
    #[serde(default = "default_session")]
    session: String,
}

fn default_session() -> String {
    "default".to_string()
}

#[derive(Serialize)]
struct ChatOut {
    text: String,
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Serialize)]
struct Token {
    text: String,
}

/// Maps an ai-core error into a 500 with its message.
struct AppError(ai_core::Error);

impl From<ai_core::Error> for AppError {
    fn from(err: ai_core::Error) -> Self {
        Self(err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let base = OpenAiClient::local(OLLAMA_V1).chat_model(MODEL);
    let logs = RecordingTracer::new();
    // Attach the tracer non-invasively: the model now emits trace events.
    let model = Arc::new(base.traced(Arc::new(LogTracer { rec: logs.clone() })));

    // Pick the conversation-memory backend at runtime. Both implement ai-core's
    // `ChatStore`; `DynChatStore` erases them to one type behind an `Arc`, so the
    // handlers below are identical regardless of which one is chosen.
    let store: Arc<dyn DynChatStore> = match std::env::var("STORE").as_deref() {
        Ok("sqlite") => {
            println!("memory: SQLite (./webui.db — survives restarts)");
            Arc::new(SqliteChatStore::connect("webui.db").await?)
        }
        _ => {
            println!("memory: in-process InMemoryChatStore (cleared on restart)");
            Arc::new(InMemoryChatStore::new())
        }
    };

    let mcp_url =
        std::env::var("MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:9000/mcp".to_string());
    println!("mcp tools from: {mcp_url} (run the examples/mcp server to enable /api/mcp/*)");

    let state = AppState {
        model,
        logs,
        store,
        model_id: MODEL.to_string(),
        mcp_url,
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(chat_stream))
        .route("/api/agent", post(run_agent))
        .route("/api/structured", post(structured))
        .route("/api/mcp/tools", get(mcp_tools))
        .route("/api/mcp/agent", post(mcp_agent))
        .route("/api/history/{session}", get(history))
        .route("/api/logs", get(get_logs))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    println!("backend listening on http://127.0.0.1:8080 (model: {MODEL})");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Serialize)]
struct AgentOut {
    answer: String,
    steps: u32,
    stopped: String,
    transcript: Vec<Message>,
}

/// Run an Agent loop: the model may call tools, whose results feed back until it
/// answers. Returns the final answer plus the full transcript (tool calls and
/// results included) so the UI can show the decision trail.
async fn run_agent(
    State(state): State<AppState>,
    Json(body): Json<ChatIn>,
) -> Result<Json<AgentOut>, AppError> {
    let weather = FnTool::new(
        ToolDef::new(
            "get_weather",
            "Get the current weather for a city.",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        ),
        |args: serde_json::Value| async move {
            let city = args["city"].as_str().unwrap_or("?");
            Ok::<_, ai_core::Error>(json!({ "city": city, "temp_f": 72, "conditions": "sunny" }))
        },
    );
    let calculate = FnTool::new(
        ToolDef::new(
            "calculate",
            "Add two numbers a and b.",
            json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
        ),
        |args: serde_json::Value| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok::<_, ai_core::Error>(json!({ "sum": a + b }))
        },
    );

    let agent = Agent::new(state.model.clone())
        .system("You have tools. Use get_weather for weather and calculate for arithmetic, then answer in one short sentence.")
        .tool(weather)
        .tool(calculate)
        .max_steps(5);

    let outcome = agent.run(body.prompt).await?;
    Ok(Json(AgentOut {
        answer: outcome.text(),
        steps: outcome.steps,
        stopped: format!("{:?}", outcome.stopped),
        transcript: outcome.messages,
    }))
}

/// A small derived type the model fills in for the structured-output demo.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Person {
    /// Full name.
    name: String,
    /// Age in years.
    age: u32,
    /// Their occupation or role.
    occupation: String,
}

/// Structured output: ask the model for JSON constrained to `Person`'s schema
/// and deserialize straight into the Rust type.
async fn structured(
    State(state): State<AppState>,
    Json(body): Json<ChatIn>,
) -> Result<Json<Person>, AppError> {
    let request = ChatRequest::builder(&state.model_id)
        .system("Extract or invent a person from the user's text.")
        .user(body.prompt)
        .build()?;
    let person: Person = state.model.structured::<Person>(request).await?;
    Ok(Json(person))
}

/// Non-streaming chat that remembers the conversation per session.
async fn chat(
    State(state): State<AppState>,
    Json(body): Json<ChatIn>,
) -> Result<Json<ChatOut>, AppError> {
    let mut hist = ChatHistory::from_messages(state.store.load_boxed(&body.session).await?);
    hist.user(body.prompt);
    let request = hist.to_request(&state.model_id).system(SYSTEM).build()?;
    let response = state.model.chat(request).await?;
    hist.record_response(&response);
    state
        .store
        .save_boxed(&body.session, hist.into_messages())
        .await?;

    Ok(Json(ChatOut {
        text: response.text(),
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
    }))
}

/// Streaming chat with the same per-session memory.
async fn chat_stream(State(state): State<AppState>, Json(body): Json<ChatIn>) -> impl IntoResponse {
    let model = state.model.clone();
    let store = state.store.clone();
    let model_id = state.model_id.clone();

    let stream = async_stream::stream! {
        let prior = match store.load_boxed(&body.session).await {
            Ok(prior) => prior,
            Err(err) => { yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string())); return; }
        };
        let mut hist = ChatHistory::from_messages(prior);
        hist.user(body.prompt);

        let request = match hist.to_request(&model_id).system(SYSTEM).build() {
            Ok(request) => request,
            Err(err) => { yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string())); return; }
        };

        match model.stream(request).await {
            Ok(mut events) => {
                let mut full = String::new();
                while let Some(event) = events.next().await {
                    match event {
                        Ok(StreamEvent::TextDelta(text)) => {
                            full.push_str(&text);
                            if let Ok(ev) = Event::default().json_data(Token { text }) {
                                yield Ok::<_, Infallible>(ev);
                            }
                        }
                        Err(err) => {
                            yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string()));
                            return;
                        }
                        _ => {}
                    }
                }
                hist.assistant(full);
                let _ = store.save_boxed(&body.session, hist.into_messages()).await;
                yield Ok::<_, Infallible>(Event::default().event("done").data("[DONE]"));
            }
            Err(err) => {
                yield Ok::<_, Infallible>(Event::default().event("error").data(err.to_string()));
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Inspect a session's stored conversation (memory).
async fn history(
    State(state): State<AppState>,
    Path(session): Path<String>,
) -> Result<Json<Vec<Message>>, AppError> {
    Ok(Json(state.store.load_boxed(&session).await?))
}

/// Inspect the recorded trace events (logging).
async fn get_logs(State(state): State<AppState>) -> Json<Vec<TraceEvent>> {
    Json(state.logs.events())
}

#[derive(Serialize)]
struct McpToolView {
    name: String,
    description: String,
}

/// Map an MCP/transport error into a 500 (e.g. the MCP server isn't running).
fn mcp_err(err: ai_core::Error) -> AppError {
    AppError(err)
}

/// List the tools advertised by the configured MCP server.
async fn mcp_tools(State(state): State<AppState>) -> Result<Json<Vec<McpToolView>>, AppError> {
    let client = McpClient::connect(&state.mcp_url).await.map_err(mcp_err)?;
    let tools = client.list_tools().await.map_err(mcp_err)?;
    Ok(Json(
        tools
            .into_iter()
            .map(|t| McpToolView {
                name: t.name,
                description: t.description,
            })
            .collect(),
    ))
}

/// Run an Agent whose entire ToolBox is sourced from the MCP server.
async fn mcp_agent(
    State(state): State<AppState>,
    Json(body): Json<ChatIn>,
) -> Result<Json<AgentOut>, AppError> {
    let client = Arc::new(McpClient::connect(&state.mcp_url).await.map_err(mcp_err)?);
    let mut tools = ToolBox::new();
    for info in client.list_tools().await.map_err(mcp_err)? {
        let remote_name = info.name.clone();
        tools.add(McpTool::new(
            client.clone(),
            info.name,
            remote_name,
            info.description,
            info.input_schema,
        ));
    }

    let agent = Agent::new(state.model.clone())
        .system("Use the available tools when relevant, then answer in one short sentence.")
        .tools(tools)
        .max_steps(5);
    let outcome = agent.run(body.prompt).await?;
    Ok(Json(AgentOut {
        answer: outcome.text(),
        steps: outcome.steps,
        stopped: format!("{:?}", outcome.stopped),
        transcript: outcome.messages,
    }))
}
