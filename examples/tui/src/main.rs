//! A TUI agent chat interface using ai-core.
//!
//! Supports local or hosted OpenAI-compatible APIs and built-in tool calling.
//!
//! Usage:
//!   - Local Ollama: `cargo run`
//!   - Hosted OpenAI: `MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... cargo run`
//!   - Custom OpenAI-compatible endpoint: `OPENAI_BASE_URL=http://host/v1 OPENAI_API_KEY=... cargo run`
//!
//! Keys:
//!   - Type your message and press Enter to send
//!   - Ctrl+C to exit
//!   - Ctrl+L to clear history
//!   - Ctrl+T to toggle thinking visibility

mod app;
mod ui;

use app::{App, BoxErr, LoadingEvent};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io;
use std::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), BoxErr> {
    let _ = dotenvy::dotenv();

    // Create app first so startup errors are visible in the normal terminal.
    let mut app = App::new().await?;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Main loop
    let res = loop {
        // Draw UI
        terminal.draw(|f| ui::ui(f, &mut app))?;

        // Handle input
        if crossterm::event::poll(std::time::Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        break Ok(());
                    }
                    KeyCode::Char('l') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        app.clear_messages();
                    }
                    KeyCode::Char('t') if key.modifiers.contains(event::KeyModifiers::CONTROL) => {
                        app.toggle_thinking();
                    }
                    KeyCode::Up => {
                        app.scroll_messages_up(1);
                    }
                    KeyCode::Down => {
                        app.scroll_messages_down(1);
                    }
                    KeyCode::PageUp => {
                        app.scroll_messages_up(8);
                    }
                    KeyCode::PageDown => {
                        app.scroll_messages_down(8);
                    }
                    KeyCode::Home => {
                        app.scroll_messages_home();
                    }
                    KeyCode::End => {
                        app.scroll_messages_end();
                    }
                    KeyCode::Enter => {
                        if !app.input.is_empty() && !app.is_loading {
                            app.send_message().await?;
                        }
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Char(c) => {
                        app.input.push(c);
                    }
                    _ => {}
                }
            }
        }

        // Check if loading is complete
        if app.is_loading {
            match app.loading_rx.try_recv() {
                Ok(LoadingEvent::Completed { messages }) => {
                    app.apply_outcome_messages(messages);
                    app.is_loading = false;
                }
                Ok(LoadingEvent::Error(error)) => {
                    app.error = Some(error);
                    app.is_loading = false;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    app.error = Some("Connection lost".to_string());
                    app.is_loading = false;
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    };

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    res
}
