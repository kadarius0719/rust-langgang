//! A typed Runnable pipeline — no model or network required.
//!
//! ```sh
//! cargo run --example pipeline
//! ```

use ai_core::runnable::{from_fn, parallel};
use ai_core::{Error, Result, Runnable, RunnableExt};

/// A reusable `i32 -> i32` step. Returns `impl Runnable` so several of these
/// share one type (needed for the homogeneous `parallel`).
fn add(delta: i32) -> impl Runnable<In = i32, Out = i32> {
    from_fn(move |n: i32| async move { Ok::<_, Error>(n + delta) })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Sequential pipeline: parse -> double -> label.
    let pipeline = from_fn(|s: String| async move {
        s.trim()
            .parse::<i32>()
            .map_err(|e| Error::invalid_request(e.to_string()))
    })
    .then(add(0).map_out(|n| n * 2))
    .map_out(|n| format!("result = {n}"));

    println!("{}", pipeline.invoke("  21  ".to_string()).await?);

    // Fan-out: one input to several steps concurrently.
    let fan = parallel(vec![add(1), add(10), add(100)]);
    println!("fan-out: {:?}", fan.invoke(5).await?);

    // Batch: many inputs through one runnable.
    let doubled = add(0).map_out(|n| n * 2);
    let batched = doubled
        .batch(vec![1, 2, 3])
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    println!("batch: {batched:?}");

    Ok(())
}
