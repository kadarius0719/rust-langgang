//! Integration tests for the Runnable composition layer.

use std::future::Future;

use ai_core::runnable::{from_fn, model_runnable, parallel, parallel_map, Branch, DynRunnable};
use ai_core::{
    ChatModel, ChatRequest, ChatStream, Error, Result, Runnable, RunnableExt, StreamEvent,
};

/// A runnable `i32 -> i32` that adds `delta`. Returns `impl Runnable` so several
/// of these share one type (needed for the homogeneous `parallel`/`Branch`).
fn adder(delta: i32) -> impl Runnable<In = i32, Out = i32> {
    from_fn(move |n: i32| async move { Ok::<_, Error>(n + delta) })
}

#[tokio::test]
async fn then_chains_typed_steps() {
    let parse = from_fn(|s: String| async move {
        s.trim()
            .parse::<i32>()
            .map_err(|e| Error::invalid_request(e.to_string()))
    });
    let double = from_fn(|n: i32| async move { Ok::<_, Error>(n * 2) });

    let pipeline = parse.then(double);
    assert_eq!(pipeline.invoke("  21  ".to_string()).await.unwrap(), 42);
}

#[tokio::test]
async fn map_out_transforms_output() {
    let pipeline = adder(1).map_out(|n| format!("={n}"));
    assert_eq!(pipeline.invoke(6).await.unwrap(), "=7");
}

#[tokio::test]
async fn parallel_fans_out_to_a_vec() {
    let fan = parallel(vec![adder(1), adder(2), adder(3)]);
    assert_eq!(fan.invoke(10).await.unwrap(), vec![11, 12, 13]);
}

#[tokio::test]
async fn branch_routes_by_predicate() {
    let router = Branch::new(adder(0)).when(|n: &i32| *n < 0, adder(100));
    assert_eq!(router.invoke(-5).await.unwrap(), 95); // negative arm
    assert_eq!(router.invoke(5).await.unwrap(), 5); // default
}

#[tokio::test]
async fn with_fallback_recovers() {
    let failing = from_fn(|_: i32| async move { Err::<i32, Error>(Error::tool("boom")) });
    let ok = from_fn(|n: i32| async move { Ok::<_, Error>(n) });

    let pipeline = failing.with_fallback(ok);
    assert_eq!(pipeline.invoke(7).await.unwrap(), 7);
}

#[tokio::test]
async fn batch_runs_many_inputs() {
    let results = adder(1).batch(vec![1, 2, 3]).await;
    let values: Vec<i32> = results.into_iter().map(|r| r.unwrap()).collect();
    assert_eq!(values, vec![2, 3, 4]);
}

#[tokio::test]
async fn erased_parallel_map_collects_a_json_object() {
    // Differently-typed branches, fanned out into a keyed JSON object.
    let length = from_fn(|s: String| async move { Ok::<_, Error>(s.len()) }).erase();
    let upper = from_fn(|s: String| async move { Ok::<_, Error>(s.to_uppercase()) }).erase();

    let fan = parallel_map(vec![
        (
            "length".to_string(),
            Box::new(length) as Box<dyn DynRunnable>,
        ),
        ("upper".to_string(), Box::new(upper) as Box<dyn DynRunnable>),
    ]);

    let out = fan.invoke(serde_json::json!("hello")).await.unwrap();
    assert_eq!(out["length"], 5);
    assert_eq!(out["upper"], "HELLO");
}

#[derive(Clone)]
struct TextModel(&'static str);

impl ChatModel for TextModel {
    fn stream(&self, _request: ChatRequest) -> impl Future<Output = Result<ChatStream>> + Send {
        let text = self.0.to_string();
        async move {
            Ok(ChatStream::new(futures::stream::iter(vec![
                Ok::<_, Error>(StreamEvent::TextDelta(text)),
            ])))
        }
    }
}

#[tokio::test]
async fn model_runnable_composes_in_a_pipeline() {
    // build request -> model -> extract text
    let pipeline = from_fn(|q: String| async move { ChatRequest::builder("m").user(q).build() })
        .then(model_runnable(TextModel("pong")))
        .map_out(|response| response.text());

    let answer = pipeline.invoke("ping".to_string()).await.unwrap();
    assert_eq!(answer, "pong");
}

/// A `i32 -> i32` step that errors when `fail` is set (same type regardless of
/// the value, so several can live in one `parallel`).
fn maybe_fail(fail: bool) -> impl Runnable<In = i32, Out = i32> {
    from_fn(move |n: i32| async move {
        if fail {
            Err(Error::tool("boom"))
        } else {
            Ok(n)
        }
    })
}

#[tokio::test]
async fn parallel_propagates_first_error() {
    let fan = parallel(vec![maybe_fail(false), maybe_fail(true), maybe_fail(false)]);
    assert!(fan.invoke(1).await.is_err());
}

#[tokio::test]
async fn batch_isolates_per_input_failures() {
    let runnable = from_fn(|n: i32| async move {
        if n == 2 {
            Err(Error::tool("two"))
        } else {
            Ok(n * 10)
        }
    });
    let results = runnable.batch(vec![1, 2, 3]).await;
    assert_eq!(*results[0].as_ref().unwrap(), 10);
    assert!(results[1].is_err());
    assert_eq!(*results[2].as_ref().unwrap(), 30);
}

#[tokio::test]
async fn erased_step_composes_in_a_typed_combinator() {
    // An erased Box<dyn DynRunnable> flows through the typed `map_out`.
    let erased: Box<dyn DynRunnable> = Box::new(
        from_fn(|v: serde_json::Value| async move {
            Ok::<_, Error>(serde_json::json!(v.as_i64().unwrap_or(0) + 1))
        })
        .erase(),
    );
    let pipeline = erased.map_out(|v: serde_json::Value| v.as_i64().unwrap_or(0) * 2);
    assert_eq!(pipeline.invoke(serde_json::json!(10)).await.unwrap(), 22);
}

#[tokio::test]
async fn pipeline_is_send_across_spawn() {
    // Regression: an erased step composed into a combinator must stay `Send`,
    // i.e. survive `tokio::spawn`.
    let erased: Box<dyn DynRunnable> = Box::new(
        from_fn(|v: serde_json::Value| async move {
            Ok::<_, Error>(serde_json::json!({ "echoed": v }))
        })
        .erase(),
    );
    let pipeline = erased.map_out(|v: serde_json::Value| v.to_string());

    let handle = tokio::spawn(async move { pipeline.invoke(serde_json::json!("hi")).await });
    let out = handle.await.unwrap().unwrap();
    assert!(out.contains("echoed"));
}
