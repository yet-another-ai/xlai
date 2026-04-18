//! Composes and inserts **ephemeral** per-request system reminders for [`super::Agent`].
//!
//! Reminder rows are sent only on the wire to the model; they are not appended to the agent’s
//! in-memory transcript and must be stripped from any inbound history so callers never treat them
//! as part of user-visible chat history.

use std::collections::BTreeSet;

use serde_json::{Value, json};
use tera::Context;
use xlai_core::{ChatMessage, ErrorKind, MessageRole, Skill, SkillId, SkillLoadPolicy, XlaiError};

use crate::{EmbeddedPromptStore, XlaiRuntime};

/// Metadata key marking synthetic reminder rows (internal only).
pub(crate) const XLAI_SYSTEM_REMINDER_METADATA_KEY: &str = "xlai_system_reminder";

#[must_use]
pub(crate) fn strip_internal_reminders(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    messages
        .into_iter()
        .filter(|m| {
            !m.metadata
                .get(XLAI_SYSTEM_REMINDER_METADATA_KEY)
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .collect()
}

const AVAILABLE_SKILLS_TEMPLATE: &str = "system/system-reminder-available-skills.md";
const INVOKED_SKILLS_TEMPLATE: &str = "system/system-reminder-invoked-skills.md";

/// Inserts a `system` reminder near the tail without splitting assistant→tool groups.
#[must_use]
pub(crate) fn insert_system_reminder_near_tail(
    mut messages: Vec<ChatMessage>,
    body: String,
) -> Vec<ChatMessage> {
    let reminder = reminder_message(body);
    if messages.is_empty() {
        messages.push(reminder);
        return messages;
    }
    let mut insert_at = messages.len() - 1;
    while insert_at > 0 && messages[insert_at].role == MessageRole::Tool {
        insert_at -= 1;
    }
    if messages[insert_at].role != MessageRole::Assistant
        && insert_at + 1 < messages.len()
        && messages[insert_at + 1].role == MessageRole::Tool
    {
        insert_at += 1;
    }
    messages.insert(insert_at, reminder);
    messages
}

fn reminder_message(body: String) -> ChatMessage {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        XLAI_SYSTEM_REMINDER_METADATA_KEY.to_owned(),
        Value::Bool(true),
    );
    ChatMessage {
        role: MessageRole::System,
        content: xlai_core::ChatContent::text(body),
        tool_name: None,
        tool_call_id: None,
        metadata,
    }
}

/// Ordered unique skill ids from successful `skill` tool results in `messages`.
fn collect_invoked_skill_ids(messages: &[ChatMessage]) -> Vec<SkillId> {
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::new();
    for m in messages {
        if m.role != MessageRole::Tool {
            continue;
        }
        if m.tool_name.as_deref() != Some("skill") {
            continue;
        }
        let Some(Value::String(id)) = m.metadata.get("skill_id") else {
            continue;
        };
        if seen.insert(id.clone()) {
            ordered.push(id.clone());
        }
    }
    ordered
}

fn skills_for_tera(skills: &[Skill]) -> Vec<Value> {
    skills
        .iter()
        .map(|s| {
            json!({
                "name": s.id,
                "description": s.description,
            })
        })
        .collect()
}

/// Builds optional reminder text from embedded prompts, runtime state, and an optional user fragment.
pub(crate) async fn compose_system_reminder_body(
    runtime: &XlaiRuntime,
    messages: &[ChatMessage],
    user_fragment: Option<&str>,
) -> Result<Option<String>, XlaiError> {
    let mut parts: Vec<String> = Vec::new();

    match runtime.list_skills().await {
        Ok(skills) if !skills.is_empty() => {
            let mut ctx = Context::new();
            ctx.insert("skills", &skills_for_tera(&skills));
            let rendered = EmbeddedPromptStore::render(AVAILABLE_SKILLS_TEMPLATE, &ctx)?;
            let trimmed = rendered.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_owned());
            }
        }
        Ok(_) => {}
        Err(e) if e.kind == ErrorKind::Unsupported => {}
        Err(e) => return Err(e),
    }

    let invoked_ids = collect_invoked_skill_ids(messages);
    if !invoked_ids.is_empty() {
        let mut skills: Vec<Skill> = Vec::new();
        match runtime.resolve_skills(&invoked_ids).await {
            Ok(resolved) => {
                for id in &invoked_ids {
                    if let Some(s) = resolved.iter().find(|s| &s.id == id) {
                        skills.push(s.clone());
                    } else {
                        skills.push(Skill {
                            id: id.clone(),
                            name: id.clone(),
                            description: String::new(),
                            prompt_fragment: String::new(),
                            resources: Vec::new(),
                            entrypoints: Vec::new(),
                            load_policy: Some(SkillLoadPolicy::default()),
                            tags: Vec::new(),
                            metadata: Default::default(),
                        });
                    }
                }
            }
            Err(_) => {
                for id in &invoked_ids {
                    skills.push(Skill {
                        id: id.clone(),
                        name: id.clone(),
                        description: String::new(),
                        prompt_fragment: String::new(),
                        resources: Vec::new(),
                        entrypoints: Vec::new(),
                        load_policy: Some(SkillLoadPolicy::default()),
                        tags: Vec::new(),
                        metadata: Default::default(),
                    });
                }
            }
        }

        let mut ctx = Context::new();
        ctx.insert("skills", &skills_for_tera(&skills));
        let rendered = EmbeddedPromptStore::render(INVOKED_SKILLS_TEMPLATE, &ctx)?;
        let trimmed = rendered.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_owned());
        }
    }

    if let Some(u) = user_fragment {
        let u = u.trim();
        if !u.is_empty() {
            parts.push(u.to_owned());
        }
    }

    if parts.is_empty() {
        return Ok(None);
    }

    Ok(Some(parts.join("\n\n---\n\n")))
}
