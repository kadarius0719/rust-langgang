//! Stream tokens from a local (offline) model to stdout as they arrive.
//!
//! ```sh
//! cargo run --example streaming --features openai
//! ```

use std::io::Write;

use ai_core::{ChatModel, ChatRequest, OpenAiClient, StreamEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> ai_core::Result<()> {
    let model = OpenAiClient::local("http://localhost:11434/v1").chat_model("llama3.1");

    let request = ChatRequest::builder("llama3.1")
        .user("Write a short haiku about the Rust programming language.")
        .build()?;

    let mut stream = model.stream(request).await?;
    while let Some(event) = stream.next().await {
        if let StreamEvent::TextDelta(text) = event? {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
    }
    println!();
    Ok(())
}
