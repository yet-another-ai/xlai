use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;
use tera::Context;
use xlai_core::{Skill, ToolDefinition, ToolResult, ToolSchema, XlaiError};

use crate::chat::ToolOrigin;
use crate::skill_store::resolve_skill_resource_path;
use crate::{Chat, EmbeddedPromptStore, XlaiRuntime};

const SKILL_TOOL_NAME: &str = "skill";
const SKILL_RESOURCE_TOOL_NAME: &str = "skill_resource";
const SKILL_TOOL_DESCRIPTION_TEMPLATE: &str = "system/tool-description-skill.md";
const SKILL_RESOURCE_TOOL_DESCRIPTION_TEMPLATE: &str = "system/tool-description-skill-resource.md";

pub(super) fn register(chat: &mut Chat, runtime: Arc<XlaiRuntime>) -> Result<(), XlaiError> {
    let skill_definition = skill_definition()?;
    let skill_resource_definition = skill_resource_definition()?;
    let skill_runtime = runtime.clone();

    chat.register_tool_with_origin(skill_definition, ToolOrigin::Builtin, move |arguments| {
        let runtime = skill_runtime.clone();
        async move { resolve_skill(runtime.as_ref(), arguments).await }
    });

    chat.register_tool_with_origin(
        skill_resource_definition,
        ToolOrigin::Builtin,
        move |arguments| {
            let runtime = runtime.clone();
            async move { resolve_skill_resource(runtime.as_ref(), arguments).await }
        },
    );

    Ok(())
}

fn skill_definition() -> Result<ToolDefinition, XlaiError> {
    let mut context = Context::new();
    context.insert("skill_tag_name", SKILL_TOOL_NAME);
    let schema = ToolSchema::object(
        BTreeMap::from([
            (
                "skill".to_owned(),
                ToolSchema::string().with_description("The skill identifier to resolve and apply"),
            ),
            (
                "args".to_owned(),
                ToolSchema::string().with_description(
                    "Optional task-specific input to pair with the resolved skill",
                ),
            ),
        ]),
        vec!["skill".to_owned()],
    );

    Ok(ToolDefinition::new(
        SKILL_TOOL_NAME,
        EmbeddedPromptStore::render(SKILL_TOOL_DESCRIPTION_TEMPLATE, &context)?,
        schema,
    )
    .with_execution_mode(Default::default()))
}

fn skill_resource_definition() -> Result<ToolDefinition, XlaiError> {
    let schema = ToolSchema::object(
        BTreeMap::from([
            (
                "skill".to_owned(),
                ToolSchema::string().with_description("The skill identifier to read from"),
            ),
            (
                "path".to_owned(),
                ToolSchema::string().with_description(
                    "The declared relative resource path to load from the skill package",
                ),
            ),
        ]),
        vec!["skill".to_owned(), "path".to_owned()],
    );

    Ok(ToolDefinition::new(
        SKILL_RESOURCE_TOOL_NAME,
        EmbeddedPromptStore::load(SKILL_RESOURCE_TOOL_DESCRIPTION_TEMPLATE)?,
        schema,
    )
    .with_execution_mode(Default::default()))
}

async fn resolve_skill(runtime: &XlaiRuntime, arguments: Value) -> Result<ToolResult, XlaiError> {
    let skill_id = match required_string_argument(&arguments, "skill", SKILL_TOOL_NAME) {
        Ok(skill_id) => skill_id,
        Err(message) => return Ok(tool_error_result(SKILL_TOOL_NAME, message)),
    };
    let args = optional_string_argument(&arguments, "args").unwrap_or_default();

    let resolved_skills = runtime
        .resolve_skills(std::slice::from_ref(&skill_id))
        .await?;
    let Some(skill) = resolved_skills.into_iter().next() else {
        return Ok(tool_error_result(
            SKILL_TOOL_NAME,
            format!("skill `{skill_id}` could not be resolved from the configured skill store"),
        ));
    };

    let eager_paths = skill
        .load_policy
        .as_ref()
        .map(|policy| policy.eager_paths.clone())
        .unwrap_or_default();
    let eager_resources = read_eager_resources(runtime, &skill, &eager_paths).await?;
    let optional_resources = skill
        .resources
        .iter()
        .filter(|resource| !eager_paths.iter().any(|path| path == &resource.path))
        .map(|resource| {
            let mut descriptor = resource.path.clone();
            if let Some(kind) = &resource.kind {
                descriptor.push_str(&format!(" [{kind}]"));
            }
            if let Some(purpose) = &resource.purpose {
                descriptor.push_str(&format!(" - {purpose}"));
            }
            descriptor
        })
        .collect::<Vec<_>>();

    let mut metadata = BTreeMap::new();
    metadata.insert("skill_id".to_owned(), Value::String(skill.id.clone()));
    metadata.insert("skill_name".to_owned(), Value::String(skill.name.clone()));

    let mut content = format!(
        "Resolved skill `{}` ({})\n\nDescription:\n{}\n\nEntrypoints:\n{}\n\nPrompt fragment:\n{}",
        skill.id,
        skill.name,
        skill.description,
        skill.entrypoints.join("\n"),
        skill.prompt_fragment
    );

    if !eager_resources.is_empty() {
        content.push_str("\n\nLoaded resources:");
        for (path, body) in eager_resources {
            content.push_str(&format!("\n\n--- {path} ---\n{body}"));
        }
    }

    if !optional_resources.is_empty() {
        content.push_str("\n\nOptional resources available via `skill_resource`:");
        for descriptor in optional_resources {
            content.push_str(&format!("\n- {descriptor}"));
        }
    }

    if !args.is_empty() {
        metadata.insert("args".to_owned(), Value::String(args.clone()));
        content.push_str("\n\nArguments:\n");
        content.push_str(&args);
    }

    Ok(ToolResult {
        tool_name: SKILL_TOOL_NAME.to_owned(),
        content,
        is_error: false,
        metadata,
    })
}

async fn resolve_skill_resource(
    runtime: &XlaiRuntime,
    arguments: Value,
) -> Result<ToolResult, XlaiError> {
    let skill_id = match required_string_argument(&arguments, "skill", SKILL_RESOURCE_TOOL_NAME) {
        Ok(skill_id) => skill_id,
        Err(message) => return Ok(tool_error_result(SKILL_RESOURCE_TOOL_NAME, message)),
    };
    let resource_path = match required_string_argument(&arguments, "path", SKILL_RESOURCE_TOOL_NAME)
    {
        Ok(path) => path,
        Err(message) => return Ok(tool_error_result(SKILL_RESOURCE_TOOL_NAME, message)),
    };

    let resolved_skills = runtime
        .resolve_skills(std::slice::from_ref(&skill_id))
        .await?;
    let Some(skill) = resolved_skills.into_iter().next() else {
        return Ok(tool_error_result(
            SKILL_RESOURCE_TOOL_NAME,
            format!("skill `{skill_id}` could not be resolved from the configured skill store"),
        ));
    };

    let absolute_path = match resolve_skill_resource_path(&skill, &resource_path) {
        Ok(path) => path,
        Err(error) => {
            return Ok(tool_error_result(
                SKILL_RESOURCE_TOOL_NAME,
                error.to_string(),
            ));
        }
    };
    let bytes = runtime.read_skill_file(&absolute_path).await?;
    let content = match String::from_utf8(bytes) {
        Ok(content) => content,
        Err(error) => {
            return Ok(tool_error_result(
                SKILL_RESOURCE_TOOL_NAME,
                format!(
                    "skill resource `{}` is not valid UTF-8: {error}",
                    resource_path
                ),
            ));
        }
    };

    let mut metadata = BTreeMap::new();
    metadata.insert("skill_id".to_owned(), Value::String(skill.id.clone()));
    metadata.insert("skill_name".to_owned(), Value::String(skill.name.clone()));
    metadata.insert(
        "resource_path".to_owned(),
        Value::String(resource_path.clone()),
    );

    Ok(ToolResult {
        tool_name: SKILL_RESOURCE_TOOL_NAME.to_owned(),
        content: format!(
            "Resolved skill resource `{}` from `{}`\n\n{}",
            resource_path, skill.id, content
        ),
        is_error: false,
        metadata,
    })
}

async fn read_eager_resources(
    runtime: &XlaiRuntime,
    skill: &Skill,
    eager_paths: &[String],
) -> Result<Vec<(String, String)>, XlaiError> {
    let mut resources = Vec::new();

    for path in eager_paths {
        let absolute_path = resolve_skill_resource_path(skill, path)?;
        let bytes = runtime.read_skill_file(&absolute_path).await?;
        let content = String::from_utf8(bytes).map_err(|error| {
            XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("skill resource `{path}` is not valid UTF-8: {error}"),
            )
        })?;
        resources.push((path.clone(), content));
    }

    Ok(resources)
}

fn required_string_argument(
    arguments: &Value,
    name: &str,
    tool_name: &str,
) -> Result<String, String> {
    optional_string_argument(arguments, name)
        .ok_or_else(|| format!("tool `{tool_name}` requires a string argument named `{name}`"))
}

fn optional_string_argument(arguments: &Value, name: &str) -> Option<String> {
    arguments
        .as_object()
        .and_then(|map| map.get(name))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn tool_error_result(tool_name: &str, message: impl Into<String>) -> ToolResult {
    ToolResult {
        tool_name: tool_name.to_owned(),
        content: message.into(),
        is_error: true,
        metadata: BTreeMap::new(),
    }
}
