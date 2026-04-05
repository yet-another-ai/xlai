use xlai_core::XlaiError;
use xlai_sys as sys;

use crate::request::PreparedRequest;
use crate::{LlamaCppConfig, LoadedModel, map_provider_error};

#[cfg(test)]
pub(crate) use xlai_runtime::local_common::validate_structured_output_schema;
pub(crate) use xlai_runtime::local_common::{
    prompt_messages_with_constraints, render_manual_prompt, validate_structured_output,
};

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

    render_manual_prompt(&prompt_messages)
}
