use serde::{Deserialize, Serialize};

use crate::chat::{ToolCall, ToolResult};
use crate::error::XlaiError;
use crate::filesystem::SkillFileSystem;
use crate::metadata::{ChunkId, DocumentId, Metadata, SkillId};
use crate::runtime::{BoxFuture, RuntimeBound};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Skill {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub prompt_fragment: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeDocument {
    pub id: DocumentId,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeChunk {
    pub id: ChunkId,
    pub document_id: DocumentId,
    pub content: String,
    #[serde(default)]
    pub metadata: Metadata,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeQuery {
    pub text: String,
    pub limit: usize,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeHit {
    pub chunk: KnowledgeChunk,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRecord {
    pub id: ChunkId,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchQuery {
    pub namespace: String,
    pub vector: Vec<f32>,
    pub limit: usize,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchHit {
    pub id: ChunkId,
    pub score: f32,
    #[serde(default)]
    pub metadata: Metadata,
}

pub trait ToolExecutor: RuntimeBound {
    fn call_tool(&self, call: ToolCall) -> BoxFuture<'_, Result<ToolResult, XlaiError>>;
}

pub trait SkillStore: SkillFileSystem {
    fn resolve_skills<'a>(
        &'a self,
        ids: &'a [SkillId],
    ) -> BoxFuture<'a, Result<Vec<Skill>, XlaiError>>;
}

pub trait KnowledgeStore: RuntimeBound {
    fn upsert_documents(
        &self,
        documents: Vec<KnowledgeDocument>,
    ) -> BoxFuture<'_, Result<(), XlaiError>>;

    fn search(&self, query: KnowledgeQuery) -> BoxFuture<'_, Result<Vec<KnowledgeHit>, XlaiError>>;
}

pub trait VectorStore: RuntimeBound {
    fn upsert<'a>(
        &'a self,
        namespace: &'a str,
        records: Vec<EmbeddingRecord>,
    ) -> BoxFuture<'a, Result<(), XlaiError>>;

    fn search(
        &self,
        query: VectorSearchQuery,
    ) -> BoxFuture<'_, Result<Vec<VectorSearchHit>, XlaiError>>;
}
