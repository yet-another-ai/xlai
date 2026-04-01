use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;
use tera::Context;
use xlai_core::{ToolDefinition, ToolParameter, ToolParameterType, ToolResult, XlaiError};

use crate::chat::ToolOrigin;
use crate::{Chat, EmbeddedPromptStore, XlaiRuntime};

const TOOL_NAME: &str = "skill";
const TOOL_DESCRIPTION_TEMPLATE: &str = "system/tool-description-skill.md";

pub(super) fn register(chat: &mut Chat, runtime: Arc<XlaiRuntime>) -> Result<(), XlaiError> {
    let definition = definition()?;

    chat.register_tool_with_origin(definition, ToolOrigin::Builtin, move |arguments| {
        let runtime = runtime.clone();
        async move { resolve(runtime.as_ref(), arguments).await }
    });

    Ok(())
}

fn definition() -> Result<ToolDefinition, XlaiError> {
    let mut context = Context::new();
    context.insert("skill_tag_name", TOOL_NAME);

    Ok(ToolDefinition {
        name: TOOL_NAME.to_owned(),
        description: EmbeddedPromptStore::render(TOOL_DESCRIPTION_TEMPLATE, &context)?,
        execution_mode: Default::default(),
        parameters: vec![
            ToolParameter {
                name: "skill".to_owned(),
                description: "The skill identifier to resolve and apply".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            },
            ToolParameter {
                name: "args".to_owned(),
                description: "Optional task-specific input to pair with the resolved skill"
                    .to_owned(),
                kind: ToolParameterType::String,
                required: false,
            },
        ],
    })
}

async fn resolve(runtime: &XlaiRuntime, arguments: Value) -> Result<ToolResult, XlaiError> {
    let skill_id = match required_string_argument(&arguments, "skill") {
        Ok(skill_id) => skill_id,
        Err(message) => return Ok(tool_error_result(message)),
    };
    let args = optional_string_argument(&arguments, "args").unwrap_or_default();

    let resolved_skills = runtime
        .resolve_skills(std::slice::from_ref(&skill_id))
        .await?;
    let Some(skill) = resolved_skills.into_iter().next() else {
        return Ok(tool_error_result(format!(
            "skill `{skill_id}` could not be resolved from the configured skill store"
        )));
    };

    let mut metadata = BTreeMap::new();
    metadata.insert("skill_id".to_owned(), skill.id.clone());
    metadata.insert("skill_name".to_owned(), skill.name.clone());

    let mut content = format!(
        "Resolved skill `{}` ({})\n\nDescription:\n{}\n\nPrompt fragment:\n{}",
        skill.id, skill.name, skill.description, skill.prompt_fragment
    );
    if !args.is_empty() {
        metadata.insert("args".to_owned(), args.clone());
        content.push_str("\n\nArguments:\n");
        content.push_str(&args);
    }

    Ok(ToolResult {
        tool_name: TOOL_NAME.to_owned(),
        content,
        is_error: false,
        metadata,
    })
}

fn required_string_argument(arguments: &Value, name: &str) -> Result<String, String> {
    optional_string_argument(arguments, name)
        .ok_or_else(|| format!("tool `{TOOL_NAME}` requires a string argument named `{name}`"))
}

fn optional_string_argument(arguments: &Value, name: &str) -> Option<String> {
    arguments
        .as_object()
        .and_then(|map| map.get(name))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn tool_error_result(message: impl Into<String>) -> ToolResult {
    ToolResult {
        tool_name: TOOL_NAME.to_owned(),
        content: message.into(),
        is_error: true,
        metadata: BTreeMap::new(),
    }
}
