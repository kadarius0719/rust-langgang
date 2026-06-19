//! A small, composable pipeline layer.
//!
//! [`Runnable`] is the unit of composition: one required method, [`invoke`], plus
//! a default [`batch`]. It is **not** an LCEL DSL — you compose with ordinary
//! method chaining and the combinators here, and the typed path is fully
//! compile-time checked.
//!
//! Two composition paths, by design:
//! - **Typed** ([`RunnableExt::then`], [`parallel`], [`Branch`], …) — zero-cost,
//!   the input/output types line up at compile time. Use this by default.
//! - **Erased** ([`DynRunnable`], [`RunnableExt::erase`], [`parallel_map`]) — over
//!   [`serde_json::Value`], for *dynamic* or *heterogeneous* graphs (e.g. fanning
//!   one input out to differently-typed branches collected into a JSON object).
//!
//! A [`ChatModel`] joins a pipeline via [`model_runnable`].
//!
//! The layer is request/response oriented: per-step token streaming is
//! deliberately deferred. To stream tokens, call [`ChatModel::stream`] on the
//! model directly rather than through a pipeline.
//!
//! ```ignore
//! use ai_core::runnable::{from_fn, model_runnable, RunnableExt};
//!
//! // prompt-builder -> model -> extract text
//! let pipeline = from_fn(build_request)
//!     .then(model_runnable(model))
//!     .map_out(|resp| resp.text());
//! let answer = pipeline.invoke("summarize this".to_string()).await?;
//! ```
//!
//! [`invoke`]: Runnable::invoke
//! [`batch`]: Runnable::batch

use std::future::Future;
use std::marker::PhantomData;

use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

use crate::{
    error::{Error, Result},
    model::ChatModel,
    request::ChatRequest,
    response::ChatResponse,
    BoxFuture,
};

/// A composable step from `In` to `Out`.
///
/// Implement [`invoke`](Runnable::invoke); [`batch`](Runnable::batch) runs inputs
/// concurrently by default.
pub trait Runnable: Send + Sync {
    /// Input type.
    type In: Send;
    /// Output type.
    type Out: Send;

    /// Run the step on a single input.
    fn invoke(&self, input: Self::In) -> impl Future<Output = Result<Self::Out>> + Send;

    /// Run the step on many inputs concurrently, one result per input.
    fn batch(&self, inputs: Vec<Self::In>) -> impl Future<Output = Vec<Result<Self::Out>>> + Send {
        async move {
            futures::future::join_all(inputs.into_iter().map(|input| self.invoke(input))).await
        }
    }
}

/// Fluent combinators on every [`Runnable`].
pub trait RunnableExt: Runnable + Sized {
    /// Pipe this step's output into `next`'s input (compile-time type-checked).
    fn then<R>(self, next: R) -> Pipe<Self, R>
    where
        R: Runnable<In = Self::Out>,
    {
        Pipe {
            first: self,
            second: next,
        }
    }

    /// Transform the output with a plain function.
    fn map_out<F, O>(self, f: F) -> MapOut<Self, F>
    where
        F: Fn(Self::Out) -> O + Send + Sync,
        O: Send,
    {
        MapOut { inner: self, f }
    }

    /// Fall back to `fallback` if this step errors.
    fn with_fallback<B>(self, fallback: B) -> WithFallback<Self, B>
    where
        B: Runnable<In = Self::In, Out = Self::Out>,
        Self::In: Clone,
    {
        WithFallback {
            primary: self,
            fallback,
        }
    }

    /// Erase this step to a [`DynRunnable`] over [`serde_json::Value`], for use in
    /// dynamic/heterogeneous graphs.
    fn erase(self) -> Erased<Self>
    where
        Self::In: DeserializeOwned,
        Self::Out: Serialize,
    {
        Erased { inner: self }
    }
}

impl<R: Runnable> RunnableExt for R {}

/// Wraps an async function as a [`Runnable`]. Build with [`from_fn`].
pub struct RunnableFn<F, In, Out> {
    f: F,
    _marker: PhantomData<fn(In) -> Out>,
}

/// Wrap an async function/closure as a [`Runnable`].
pub fn from_fn<F, Fut, In, Out>(f: F) -> RunnableFn<F, In, Out>
where
    F: Fn(In) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Out>> + Send,
    In: Send,
    Out: Send,
{
    RunnableFn {
        f,
        _marker: PhantomData,
    }
}

impl<F, Fut, In, Out> Runnable for RunnableFn<F, In, Out>
where
    F: Fn(In) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Out>> + Send,
    In: Send,
    Out: Send,
{
    type In = In;
    type Out = Out;

    fn invoke(&self, input: In) -> impl Future<Output = Result<Out>> + Send {
        (self.f)(input)
    }
}

/// Sequential composition. See [`RunnableExt::then`].
pub struct Pipe<A, B> {
    first: A,
    second: B,
}

impl<A, B> Runnable for Pipe<A, B>
where
    A: Runnable,
    B: Runnable<In = A::Out>,
{
    type In = A::In;
    type Out = B::Out;

    async fn invoke(&self, input: A::In) -> Result<B::Out> {
        let mid = self.first.invoke(input).await?;
        self.second.invoke(mid).await
    }
}

/// Output transformation. See [`RunnableExt::map_out`].
pub struct MapOut<R, F> {
    inner: R,
    f: F,
}

impl<R, F, O> Runnable for MapOut<R, F>
where
    R: Runnable,
    F: Fn(R::Out) -> O + Send + Sync,
    O: Send,
{
    type In = R::In;
    type Out = O;

    async fn invoke(&self, input: R::In) -> Result<O> {
        let out = self.inner.invoke(input).await?;
        Ok((self.f)(out))
    }
}

/// Error-recovery wrapper. See [`RunnableExt::with_fallback`].
pub struct WithFallback<A, B> {
    primary: A,
    fallback: B,
}

impl<A, B> Runnable for WithFallback<A, B>
where
    A: Runnable,
    B: Runnable<In = A::In, Out = A::Out>,
    A::In: Clone,
{
    type In = A::In;
    type Out = A::Out;

    async fn invoke(&self, input: A::In) -> Result<A::Out> {
        match self.primary.invoke(input.clone()).await {
            Ok(out) => Ok(out),
            Err(_) => self.fallback.invoke(input).await,
        }
    }
}

/// Homogeneous fan-out: run the same input through every runnable concurrently
/// and collect their outputs. Fail-fast — the first error is returned and any
/// still-in-flight branches are dropped. Build with [`parallel`].
pub struct Parallel<R> {
    runnables: Vec<R>,
}

/// Fan one input out to each runnable concurrently, collecting `Vec<Out>`.
///
/// All branches share a type — for *differently-typed* branches, [`erase`] them
/// to `Box<dyn DynRunnable>` (which is itself a `Runnable`), or use
/// [`parallel_map`].
///
/// Note that `from_fn(a)` and `from_fn(b)` with *different* closures are
/// *different* types, so they cannot share a `Vec`. A function returning a
/// single `impl Runnable` gives every call the same type:
///
/// ```
/// use ai_core::{from_fn, parallel, Error, Runnable};
///
/// fn scaler(factor: i64) -> impl Runnable<In = i64, Out = i64> {
///     from_fn(move |n: i64| async move { Ok::<_, Error>(n * factor) })
/// }
/// let fan = parallel(vec![scaler(2), scaler(3), scaler(10)]);
/// // fan.invoke(7).await == Ok(vec![14, 21, 70])
/// ```
///
/// [`erase`]: RunnableExt::erase
pub fn parallel<R: Runnable>(runnables: Vec<R>) -> Parallel<R> {
    Parallel { runnables }
}

impl<R> Runnable for Parallel<R>
where
    R: Runnable,
    R::In: Clone,
{
    type In = R::In;
    type Out = Vec<R::Out>;

    async fn invoke(&self, input: R::In) -> Result<Vec<R::Out>> {
        let futures = self.runnables.iter().map(|r| r.invoke(input.clone()));
        futures::future::try_join_all(futures).await
    }
}

/// First-match routing. Build with [`Branch::new`] + [`Branch::when`].
pub struct Branch<R: Runnable> {
    #[allow(clippy::type_complexity)]
    arms: Vec<(Box<dyn Fn(&R::In) -> bool + Send + Sync>, R)>,
    default: R,
}

impl<R: Runnable> Branch<R> {
    /// A branch that runs `default` when no arm matches.
    pub fn new(default: R) -> Self {
        Self {
            arms: Vec::new(),
            default,
        }
    }

    /// Route to `runnable` when `predicate` is the first to match the input.
    pub fn when<P>(mut self, predicate: P, runnable: R) -> Self
    where
        P: Fn(&R::In) -> bool + Send + Sync + 'static,
    {
        self.arms.push((Box::new(predicate), runnable));
        self
    }
}

impl<R: Runnable> Runnable for Branch<R> {
    type In = R::In;
    type Out = R::Out;

    async fn invoke(&self, input: R::In) -> Result<R::Out> {
        for (predicate, runnable) in &self.arms {
            if predicate(&input) {
                return runnable.invoke(input).await;
            }
        }
        self.default.invoke(input).await
    }
}

/// A [`ChatModel`] as a [`Runnable`] from [`ChatRequest`] to [`ChatResponse`].
/// Build with [`model_runnable`].
pub struct ModelRunnable<M> {
    model: M,
}

/// Adapt a [`ChatModel`] into a [`Runnable`] so it composes in a pipeline.
pub fn model_runnable<M: ChatModel>(model: M) -> ModelRunnable<M> {
    ModelRunnable { model }
}

impl<M: ChatModel> Runnable for ModelRunnable<M> {
    type In = ChatRequest;
    type Out = ChatResponse;

    fn invoke(&self, input: ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send {
        self.model.chat(input)
    }
}

// ---------------------------------------------------------------------------
// Erased path (over serde_json::Value) for dynamic / heterogeneous graphs
// ---------------------------------------------------------------------------

/// Object-safe, value-typed runnable for dynamic graphs.
///
/// Get one by [`erase`](RunnableExt::erase)-ing a typed runnable whose input is
/// `DeserializeOwned` and output is `Serialize`. `Box<dyn DynRunnable>` is itself
/// a [`Runnable`] over [`serde_json::Value`], so erased steps compose with the
/// typed combinators too.
pub trait DynRunnable: Send + Sync {
    /// Run on a JSON value, returning a JSON value.
    fn invoke_boxed(&self, input: Value) -> BoxFuture<'_, Result<Value>>;
}

/// A typed runnable erased to [`DynRunnable`]. Build with [`RunnableExt::erase`].
pub struct Erased<R> {
    inner: R,
}

impl<R> DynRunnable for Erased<R>
where
    R: Runnable,
    R::In: DeserializeOwned,
    R::Out: Serialize,
{
    fn invoke_boxed(&self, input: Value) -> BoxFuture<'_, Result<Value>> {
        Box::pin(async move {
            let typed: R::In = serde_json::from_value(input)?;
            let out = self.inner.invoke(typed).await?;
            Ok(serde_json::to_value(out)?)
        })
    }
}

// Generalized over the trait object's lifetime so that an erased step composed
// into a combinator stays `Send` (and thus `tokio::spawn`-able), not just the
// `'static` case.
impl<'r> Runnable for Box<dyn DynRunnable + 'r> {
    type In = Value;
    type Out = Value;

    fn invoke(&self, input: Value) -> impl Future<Output = Result<Value>> + Send {
        self.as_ref().invoke_boxed(input)
    }
}

/// Heterogeneous keyed fan-out: send one [`Value`] to every named branch
/// concurrently and collect the results into a JSON object. Build with
/// [`parallel_map`].
pub struct ParallelMap {
    branches: Vec<(String, Box<dyn DynRunnable>)>,
}

/// Fan one [`Value`] out to each named (erased) branch, collecting a JSON object
/// keyed by branch name. Fail-fast — the first error is returned and any
/// still-in-flight branches are dropped; duplicate branch keys are an error.
pub fn parallel_map(branches: Vec<(String, Box<dyn DynRunnable>)>) -> ParallelMap {
    ParallelMap { branches }
}

impl Runnable for ParallelMap {
    type In = Value;
    type Out = Value;

    async fn invoke(&self, input: Value) -> Result<Value> {
        let futures = self.branches.iter().map(|(key, runnable)| {
            let input = input.clone();
            async move {
                runnable
                    .invoke_boxed(input)
                    .await
                    .map(|value| (key.clone(), value))
            }
        });
        let mut object = serde_json::Map::new();
        for (key, value) in futures::future::try_join_all(futures).await? {
            if object.contains_key(&key) {
                return Err(Error::invalid_request(format!(
                    "duplicate parallel_map branch key `{key}`"
                )));
            }
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}
