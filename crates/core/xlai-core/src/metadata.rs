use std::collections::BTreeMap;

use serde_json::Value;

/// Structured metadata attached to runtime entities.
///
/// Message metadata is especially useful for local-only annotations in persisted
/// chat histories, such as marking entries as editable reminders or tracking
/// bookkeeping needed when replaying a session later.
pub type Metadata = BTreeMap<String, Value>;
pub type SkillId = String;
pub type DocumentId = String;
pub type ChunkId = String;
