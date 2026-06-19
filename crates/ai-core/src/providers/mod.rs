//! Provider adapters, each behind its own feature flag.
//!
//! The OpenAI-compatible adapter ([`openai`]) covers OpenAI itself plus any
//! OpenAI-compatible endpoint — including local/offline runners (llama.cpp
//! `llama-server`, LM Studio, Ollama's `/v1`) via a configurable `base_url`.

#[cfg(feature = "openai")]
pub mod openai;
