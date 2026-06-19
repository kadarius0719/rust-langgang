//! Minimal chat against a local (offline) model.
//!
//! Run a local OpenAI-compatible server first, e.g. `ollama serve` (then
//! `ollama pull llama3.1`), and run:
//!
//! ```sh
//! cargo run --example chat --features openai
//! ```
//!
//! For the hosted OpenAI API instead, use
//! `OpenAiClient::new(std::env::var("OPENAI_API_KEY")?)`.

use ai_core::{ChatModel, ChatRequest, OpenAiClient};

#[tokio::main]
async fn main() -> ai_core::Result<()> {
    let model = OpenAiClient::local("http://localhost:11434/v1").chat_model("llama3.1");

    let request = ChatRequest::builder("llama3.1")
        .system("You are concise.")
        .user("Name three things Rust is good for.")
        .max_tokens(200)
        .build()?;

    let response = model.chat(request).await?;

    println!("{}", response.text());
    println!(
        "\n[tokens: {} in / {} out]",
        response.usage.input_tokens, response.usage.output_tokens
    );
    Ok(())
}
