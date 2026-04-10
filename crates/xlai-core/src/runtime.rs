use std::future::Future;
use std::pin::Pin;

use futures_core::Stream;
use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[cfg(not(target_arch = "wasm32"))]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + 'a>>;

#[cfg(not(target_arch = "wasm32"))]
pub trait RuntimeBound: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T> RuntimeBound for T where T: Send + Sync + ?Sized {}

#[cfg(target_arch = "wasm32")]
pub trait RuntimeBound {}
#[cfg(target_arch = "wasm32")]
impl<T> RuntimeBound for T where T: ?Sized {}

#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T> MaybeSend for T where T: Send + ?Sized {}

#[cfg(target_arch = "wasm32")]
pub trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSend for T where T: ?Sized {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RuntimeCapability {
    Chat,
    Embeddings,
    ImageGeneration,
    Transcription,
    Tts,
    ToolCalling,
    SkillResolution,
    KnowledgeSearch,
    VectorSearch,
    FileSystem,
}
