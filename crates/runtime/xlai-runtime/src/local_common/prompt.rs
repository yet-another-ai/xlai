use serde_json::json;
use tera::Context;
use xlai_core::{ErrorKind, StructuredOutput, StructuredOutputFormat, XlaiError};

use super::prompt_store::EmbeddedPromptStore;
use super::request::{PreparedLocalChatRequest, PromptMessage, PromptRole};
use super::tool_calling::tool_call_instruction;

const MANUAL_PROMPT_TEMPLATE: &str = "chat/manual.txt";
const COMBINED_INSTRUCTION_TEMPLATE: &str = "system/combined-instruction.md";
const MERGED_SYSTEM_MESSAGE_TEMPLATE: &str = "system/merged-system-message.md";

pub fn render_manual_prompt(messages: &[PromptMessage]) -> Result<String, XlaiError> {
    let messages = messages
        .iter()
        .map(|message| {
            json!({
                "label": message.role.as_manual_label(),
                "content": message.content.trim(),
            })
        })
        .collect::<Vec<_>>();

    let mut context = Context::new();
    context.insert("messages", &messages);
    EmbeddedPromptStore::render(MANUAL_PROMPT_TEMPLATE, &context)
}

pub fn prompt_messages_with_constraints(
    prepared: &PreparedLocalChatRequest,
) -> Result<Vec<PromptMessage>, XlaiError> {
    let instruction = combined_instruction(prepared)?;
    let Some(instruction) = instruction else {
        return Ok(prepared.messages.clone());
    };

    let mut messages = prepared.messages.clone();
    if let Some(system_message) = messages
        .iter_mut()
        .find(|message| message.role == PromptRole::System)
    {
        system_message.content = merge_system_message(&system_message.content, &instruction)?;
    } else {
        messages.insert(
            0,
            PromptMessage {
                role: PromptRole::System,
                content: instruction,
            },
        );
    }

    Ok(messages)
}

fn combined_instruction(prepared: &PreparedLocalChatRequest) -> Result<Option<String>, XlaiError> {
    let mut sections = Vec::new();

    if !prepared.available_tools.is_empty() {
        sections.push(tool_call_instruction(&prepared.available_tools)?);
    }

    if let Some(structured_output) = &prepared.structured_output {
        sections.push(structured_output_instruction(structured_output)?);
    }

    if sections.is_empty() {
        return Ok(None);
    }

    let mut context = Context::new();
    context.insert("sections", &sections);
    EmbeddedPromptStore::render(COMBINED_INSTRUCTION_TEMPLATE, &context).map(Some)
}

fn merge_system_message(existing: &str, instruction: &str) -> Result<String, XlaiError> {
    let mut context = Context::new();
    context.insert("existing", existing);
    context.insert("instruction", instruction);
    context.insert("has_existing_content", &!existing.trim().is_empty());
    EmbeddedPromptStore::render(MERGED_SYSTEM_MESSAGE_TEMPLATE, &context)
}

pub fn structured_output_instruction(
    structured_output: &StructuredOutput,
) -> Result<String, XlaiError> {
    let mut context = Context::new();
    if let Some(name) = structured_output.name.as_deref() {
        context.insert("name", name);
    }
    if let Some(description) = structured_output.description.as_deref() {
        context.insert("description", description);
    }

    match &structured_output.format {
        StructuredOutputFormat::JsonSchema { schema } => {
            let schema = serde_json::to_string_pretty(schema).map_err(|error| {
                XlaiError::new(
                    ErrorKind::Validation,
                    format!("structured output schema could not be serialized: {error}"),
                )
            })?;
            context.insert("schema", &schema);
            EmbeddedPromptStore::render("system/structured-output-json-schema.md", &context)
        }
        StructuredOutputFormat::LarkGrammar { grammar } => {
            context.insert("grammar", grammar);
            EmbeddedPromptStore::render("system/structured-output-lark-grammar.md", &context)
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::{prompt_messages_with_constraints, render_manual_prompt};
    use crate::local_common::request::{PreparedLocalChatRequest, PromptMessage, PromptRole};
    use xlai_core::{
        StructuredOutput, ToolCallExecutionMode, ToolDefinition, ToolParameter, ToolParameterType,
    };

    fn prepared_request(messages: Vec<PromptMessage>) -> PreparedLocalChatRequest {
        PreparedLocalChatRequest {
            messages,
            available_tools: Vec::new(),
            structured_output: None,
            temperature: 0.8,
            max_output_tokens: 64,
            execution: None,
            cancellation: None,
        }
    }

    fn normalize_newlines(text: &str) -> String {
        text.replace("\r\n", "\n").replace('\r', "\n")
    }

    #[test]
    fn manual_prompt_renderer_uses_embedded_template() {
        let prompt = match render_manual_prompt(&[
            PromptMessage {
                role: PromptRole::System,
                content: "Be concise.".to_owned(),
            },
            PromptMessage {
                role: PromptRole::User,
                content: "Say hello".to_owned(),
            },
        ]) {
            Ok(prompt) => prompt,
            Err(error) => panic!("render manual prompt failed: {error}"),
        };

        assert_eq!(
            normalize_newlines(&prompt),
            "System: Be concise.\n\nUser: Say hello\n\nAssistant:"
        );
    }

    #[test]
    fn prompt_messages_merge_instruction_into_existing_system_message() {
        let mut prepared = prepared_request(vec![
            PromptMessage {
                role: PromptRole::System,
                content: "Be concise.".to_owned(),
            },
            PromptMessage {
                role: PromptRole::User,
                content: "What is the weather?".to_owned(),
            },
        ]);
        prepared.available_tools = vec![ToolDefinition {
            name: "lookup_weather".to_owned(),
            description: "Lookup weather by city".to_owned(),
            input_schema: None,
            parameters: vec![ToolParameter {
                name: "city".to_owned(),
                description: "City name".to_owned(),
                kind: ToolParameterType::String,
                required: true,
            }],
            execution_mode: ToolCallExecutionMode::Concurrent,
        }];

        let messages = match prompt_messages_with_constraints(&prepared) {
            Ok(messages) => messages,
            Err(error) => panic!("build prompt messages failed: {error}"),
        };

        assert_eq!(messages[0].role, PromptRole::System);
        let system_message = normalize_newlines(&messages[0].content);
        assert!(system_message.starts_with("Be concise.\n\n"));
        assert!(system_message.contains("lookup_weather"));
        assert_eq!(messages[1].role, PromptRole::User);
    }

    #[test]
    fn prompt_messages_insert_system_message_when_missing() {
        let mut prepared = prepared_request(vec![PromptMessage {
            role: PromptRole::User,
            content: "Return a person".to_owned(),
        }]);
        prepared.structured_output = Some(StructuredOutput::json_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false
        })));

        let messages = match prompt_messages_with_constraints(&prepared) {
            Ok(messages) => messages,
            Err(error) => panic!("build prompt messages failed: {error}"),
        };

        assert_eq!(messages[0].role, PromptRole::System);
        assert!(messages[0].content.contains("\"required\": ["));
        assert_eq!(messages[1].role, PromptRole::User);
    }
}
