use serde_json::Value;
use tera::Context;
use xlai_core::{ErrorKind, StructuredOutput, StructuredOutputFormat, XlaiError};
use xlai_llama_cpp_sys as sys;

use crate::prompt_store::EmbeddedPromptStore;
use crate::request::{PreparedRequest, PromptMessage, PromptRole};
use crate::tool_calling::tool_call_instruction;
use crate::{LlamaCppConfig, LoadedModel, map_provider_error};

pub(crate) fn render_prompt(
    config: &LlamaCppConfig,
    loaded: &LoadedModel,
    prepared: &PreparedRequest,
) -> Result<String, XlaiError> {
    let prompt_messages = prompt_messages_with_constraints(prepared)?;

    if let Some(template) = config
        .chat_template
        .as_deref()
        .or(loaded.default_chat_template.as_deref())
    {
        let template_messages = prompt_messages
            .iter()
            .map(|message| sys::ChatMessage {
                role: message.role.as_template_role(),
                content: message.content.as_str(),
            })
            .collect::<Vec<_>>();
        return sys::apply_chat_template(template, &template_messages, true)
            .map_err(map_provider_error);
    }

    Ok(render_manual_prompt(&prompt_messages))
}

pub(crate) fn render_manual_prompt(messages: &[PromptMessage]) -> String {
    let mut prompt = String::new();

    for message in messages {
        prompt.push_str(message.role.as_manual_label());
        prompt.push_str(": ");
        prompt.push_str(message.content.trim());
        prompt.push_str("\n\n");
    }

    prompt.push_str("Assistant:");
    prompt
}

pub(crate) fn prompt_messages_with_constraints(
    prepared: &PreparedRequest,
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
        if !system_message.content.trim().is_empty() {
            system_message.content.push_str("\n\n");
        }
        system_message.content.push_str(&instruction);
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

fn combined_instruction(prepared: &PreparedRequest) -> Result<Option<String>, XlaiError> {
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

    Ok(Some(sections.join("\n\n")))
}

pub(crate) fn structured_output_instruction(
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

pub(crate) fn validate_structured_output_schema(
    structured_output: &StructuredOutput,
) -> Result<(), XlaiError> {
    match &structured_output.format {
        StructuredOutputFormat::JsonSchema { schema } => jsonschema::validator_for(schema)
            .map(|_| ())
            .map_err(|error| {
                XlaiError::new(
                    ErrorKind::Validation,
                    format!("structured output schema is invalid: {error}"),
                )
            }),
        StructuredOutputFormat::LarkGrammar { grammar } => {
            if grammar.trim().is_empty() {
                return Err(XlaiError::new(
                    ErrorKind::Validation,
                    "structured output Lark grammar must not be empty",
                ));
            }
            Ok(())
        }
    }
}

pub(crate) fn validate_structured_output(
    structured_output: &StructuredOutput,
    generated: &str,
) -> Result<(), XlaiError> {
    if let StructuredOutputFormat::JsonSchema { schema } = &structured_output.format {
        let generated = generated.trim();
        let value: Value = serde_json::from_str(generated).map_err(|error| {
            XlaiError::new(
                ErrorKind::Provider,
                format!("llama.cpp structured output was not valid JSON: {error}"),
            )
        })?;
        let validator = jsonschema::validator_for(schema).map_err(|error| {
            XlaiError::new(
                ErrorKind::Validation,
                format!("structured output schema is invalid: {error}"),
            )
        })?;
        if let Err(error) = validator.validate(&value) {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                format!("llama.cpp structured output did not match the requested schema: {error}"),
            ));
        }
    }
    Ok(())
}
