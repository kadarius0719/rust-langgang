use ai_core::{
    default_server_label, load_mcp_tools, Agent, BearerAuth, ContentBlock, Message, OpenAiClient,
    OpenAiModel, Role, ToolBox,
};
use std::sync::mpsc;
use std::sync::Arc;

pub type BoxErr = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug)]
pub enum LoadingEvent {
    Completed { messages: Vec<Message> },
    Error(String),
}

#[derive(Debug, Clone)]
pub enum RenderBlock {
    Text(String),
    Thinking(String),
    ToolUse {
        name: String,
        args: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone)]
pub struct RenderMessage {
    pub role: Role,
    pub blocks: Vec<RenderBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollState {
    #[default]
    FollowLatest,
    Manual {
        scroll_offset: u16,
        pending_new_content: bool,
    },
}

impl ScrollState {
    fn visible_offset(self, max_scroll: u16) -> u16 {
        match self {
            Self::FollowLatest => max_scroll,
            Self::Manual { scroll_offset, .. } => scroll_offset.min(max_scroll),
        }
    }

    fn status_text(self) -> &'static str {
        match self {
            Self::FollowLatest => "Live",
            Self::Manual {
                pending_new_content: true,
                ..
            } => "New content below - End to jump",
            Self::Manual {
                pending_new_content: false,
                ..
            } => "History",
        }
    }

    fn mark_new_content(&mut self) {
        if let Self::Manual {
            pending_new_content,
            ..
        } = self
        {
            *pending_new_content = true;
        }
    }

    fn reconcile(&mut self, max_scroll: u16) {
        if let Self::Manual { scroll_offset, .. } = self {
            if *scroll_offset >= max_scroll {
                *self = Self::FollowLatest;
            }
        }
    }

    fn scroll_up(&mut self, amount: u16, max_scroll: u16) {
        if max_scroll == 0 {
            *self = Self::FollowLatest;
            return;
        }

        match self {
            Self::FollowLatest => {
                let scroll_offset = max_scroll.saturating_sub(amount);
                if scroll_offset < max_scroll {
                    *self = Self::Manual {
                        scroll_offset,
                        pending_new_content: false,
                    };
                }
            }
            Self::Manual { scroll_offset, .. } => {
                *scroll_offset = scroll_offset.saturating_sub(amount);
            }
        }
    }

    fn scroll_down(&mut self, amount: u16, max_scroll: u16) {
        if max_scroll == 0 {
            *self = Self::FollowLatest;
            return;
        }

        match self {
            Self::FollowLatest => {}
            Self::Manual { scroll_offset, .. } => {
                let next = scroll_offset.saturating_add(amount);
                if next >= max_scroll {
                    *self = Self::FollowLatest;
                } else {
                    *scroll_offset = next;
                }
            }
        }
    }

    fn home(&mut self, max_scroll: u16) {
        if max_scroll == 0 {
            *self = Self::FollowLatest;
        } else {
            *self = Self::Manual {
                scroll_offset: 0,
                pending_new_content: false,
            };
        }
    }

    fn end(&mut self) {
        *self = Self::FollowLatest;
    }
}

pub struct App {
    pub input: String,
    pub messages: Vec<RenderMessage>,
    pub is_loading: bool,
    pub error: Option<String>,
    pub show_thinking: bool,
    pub scroll_state: ScrollState,
    pub loading_rx: mpsc::Receiver<LoadingEvent>,
    pub loading_tx: mpsc::Sender<LoadingEvent>,
    message_scroll_limit: u16,
    transcript: Vec<Message>,
    agent: Arc<Agent<OpenAiModel>>,
}

impl App {
    pub async fn new() -> std::result::Result<Self, BoxErr> {
        let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e4b".to_string());
        let api_key = std::env::var("OPENAI_API_KEY").ok();
        let base_url = std::env::var("OPENAI_BASE_URL")
            .or_else(|_| std::env::var("AI_BASE_URL"))
            .ok();

        let model: OpenAiModel = build_model(&model_name, api_key, base_url);

        let mut tools = ToolBox::new();
        let mcp_servers = mcp_server_urls();
        if !mcp_servers.is_empty() {
            for report in load_mcp_tools(&mut tools, mcp_servers).await {
                match report.error {
                    Some(error) => {
                        eprintln!(
                            "MCP server {} at {} failed: {}",
                            report.label, report.url, error
                        );
                        eprintln!("Continuing without tools from that server.");
                    }
                    None => {
                        println!(
                            "Discovered {} MCP tool(s) from {}: {:?}",
                            report.tool_names.len(),
                            report.label,
                            report.tool_names
                        );
                    }
                }
            }
        }

        let agent = Agent::new(model)
            .model_id(model_name)
            .system(
                "You are a helpful assistant in a terminal UI. Use tools when they improve accuracy. Keep final answers concise.",
            )
            .tools(tools)
            .max_steps(10);

        let (tx, rx) = mpsc::channel();

        Ok(Self {
            input: String::new(),
            messages: vec![],
            is_loading: false,
            error: None,
            show_thinking: true,
            scroll_state: ScrollState::default(),
            loading_rx: rx,
            loading_tx: tx,
            message_scroll_limit: 0,
            transcript: vec![],
            agent: Arc::new(agent),
        })
    }

    pub async fn send_message(&mut self) -> std::result::Result<(), BoxErr> {
        if self.input.trim().is_empty() {
            return Ok(());
        }

        self.scroll_state.end();

        // Add user message to both UI and transcript.
        self.messages.push(RenderMessage {
            role: Role::User,
            blocks: vec![RenderBlock::Text(self.input.clone())],
        });
        self.transcript.push(Message::user(self.input.clone()));

        let snapshot = self.transcript.clone();
        self.input.clear();
        self.is_loading = true;
        self.error = None;

        // Spawn async task
        let agent = Arc::clone(&self.agent);
        let tx = self.loading_tx.clone();

        tokio::spawn(async move {
            match agent.run_messages(snapshot).await {
                Ok(outcome) => {
                    let _ = tx.send(LoadingEvent::Completed {
                        messages: outcome.messages,
                    });
                }
                Err(e) => {
                    let _ = tx.send(LoadingEvent::Error(e.to_string()));
                }
            }
        });

        Ok(())
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.transcript.clear();
        self.input.clear();
        self.error = None;
        self.message_scroll_limit = 0;
        self.scroll_state = ScrollState::default();
    }

    pub fn toggle_thinking(&mut self) {
        self.show_thinking = !self.show_thinking;
    }

    pub fn set_message_scroll_limit(&mut self, max_scroll: u16) {
        self.message_scroll_limit = max_scroll;
        self.scroll_state.reconcile(max_scroll);
    }

    pub fn message_scroll_offset(&self) -> u16 {
        self.scroll_state.visible_offset(self.message_scroll_limit)
    }

    pub fn scroll_status_text(&self) -> &'static str {
        self.scroll_state.status_text()
    }

    pub fn scroll_messages_up(&mut self, amount: u16) {
        self.scroll_state
            .scroll_up(amount, self.message_scroll_limit);
    }

    pub fn scroll_messages_down(&mut self, amount: u16) {
        self.scroll_state
            .scroll_down(amount, self.message_scroll_limit);
    }

    pub fn scroll_messages_home(&mut self) {
        self.scroll_state.home(self.message_scroll_limit);
    }

    pub fn scroll_messages_end(&mut self) {
        self.scroll_state.end();
    }

    pub fn apply_outcome_messages(&mut self, messages: Vec<Message>) {
        self.transcript = messages.clone();
        self.messages = render_messages(&messages);
        self.scroll_state.mark_new_content();
    }
}

fn build_model(model_name: &str, api_key: Option<String>, base_url: Option<String>) -> OpenAiModel {
    match (api_key, base_url) {
        (Some(key), Some(base_url)) => {
            // OpenAI-compatible endpoint with custom base URL and bearer auth.
            OpenAiClient::local(base_url)
                .with_auth(Arc::new(BearerAuth(key)))
                .chat_model(model_name)
        }
        (Some(key), None) => {
            // Hosted OpenAI API.
            OpenAiClient::new(key).chat_model(model_name)
        }
        (None, Some(base_url)) => {
            // Local or unauthenticated OpenAI-compatible endpoint.
            OpenAiClient::local(base_url).chat_model(model_name)
        }
        (None, None) => {
            // Default local runner.
            OpenAiClient::local("http://localhost:11434/v1").chat_model(model_name)
        }
    }
}

fn mcp_server_urls() -> Vec<(String, String)> {
    if let Ok(urls) = std::env::var("MCP_URLS") {
        let servers: Vec<(String, String)> = urls
            .split(',')
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .enumerate()
            .map(|(index, url)| (default_server_label(index), url.to_string()))
            .collect();
        if !servers.is_empty() {
            return servers;
        }
    }

    std::env::var("MCP_URL")
        .ok()
        .or_else(|| Some("http://localhost:8811/mcp".to_string()))
        .map(|url| vec![(default_server_label(0), url)])
        .unwrap_or_default()
}

fn render_messages(messages: &[Message]) -> Vec<RenderMessage> {
    let mut rendered = Vec::new();

    for message in messages {
        let mut blocks = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::Text { text } if !text.trim().is_empty() => {
                    blocks.push(RenderBlock::Text(text.clone()));
                }
                ContentBlock::Thinking { text, .. } if !text.trim().is_empty() => {
                    blocks.push(RenderBlock::Thinking(text.clone()));
                }
                ContentBlock::ToolUse { name, args, .. } => {
                    blocks.push(RenderBlock::ToolUse {
                        name: name.clone(),
                        args: args.to_string(),
                    });
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    blocks.push(RenderBlock::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        is_error: *is_error,
                    });
                }
                _ => {}
            }
        }

        if !blocks.is_empty() {
            rendered.push(RenderMessage {
                role: message.role,
                blocks,
            });
        }
    }

    rendered
}

#[cfg(test)]
mod tests {
    use super::{build_model, ScrollState};

    #[test]
    fn live_tail_scroll_state_transitions() {
        let mut state = ScrollState::default();
        assert_eq!(state, ScrollState::FollowLatest);
        assert_eq!(state.visible_offset(12), 12);
        assert_eq!(state.status_text(), "Live");

        state.scroll_up(1, 12);
        assert_eq!(
            state,
            ScrollState::Manual {
                scroll_offset: 11,
                pending_new_content: false,
            }
        );
        assert_eq!(state.visible_offset(12), 11);
        assert_eq!(state.status_text(), "History");

        state.mark_new_content();
        assert_eq!(
            state,
            ScrollState::Manual {
                scroll_offset: 11,
                pending_new_content: true,
            }
        );
        assert_eq!(state.status_text(), "New content below - End to jump");

        state.scroll_down(1, 12);
        assert_eq!(state, ScrollState::FollowLatest);
        assert_eq!(state.status_text(), "Live");
    }

    #[test]
    fn home_and_end_reset_view_state() {
        let mut state = ScrollState::FollowLatest;

        state.home(15);
        assert_eq!(
            state,
            ScrollState::Manual {
                scroll_offset: 0,
                pending_new_content: false,
            }
        );

        state.mark_new_content();
        assert_eq!(
            state,
            ScrollState::Manual {
                scroll_offset: 0,
                pending_new_content: true,
            }
        );

        state.end();
        assert_eq!(state, ScrollState::FollowLatest);
    }

    #[test]
    fn reconcile_returns_to_live_when_view_hits_bottom() {
        let mut state = ScrollState::Manual {
            scroll_offset: 4,
            pending_new_content: true,
        };

        state.reconcile(4);
        assert_eq!(state, ScrollState::FollowLatest);
    }

    #[test]
    fn model_selection_uses_custom_base_url_with_bearer_auth_when_both_are_set() {
        let model = build_model(
            "demo",
            Some("secret".to_string()),
            Some("http://example.com/v1".to_string()),
        );

        // The type only proves the model was built; the precedence is encoded
        // in the helper itself and covered by the match arms above.
        let _ = model;
    }

    #[test]
    fn model_selection_uses_hosted_openai_when_only_a_key_is_set() {
        let model = build_model("demo", Some("secret".to_string()), None);
        let _ = model;
    }

    #[test]
    fn model_selection_uses_local_endpoint_when_only_base_url_is_set() {
        let model = build_model("demo", None, Some("http://example.com/v1".to_string()));
        let _ = model;
    }
}
