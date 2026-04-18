use xlai_core::{ChatRequest, XlaiError};

use crate::LlamaCppConfig;
use xlai_runtime::local_common::{LocalChatPrepareOptions, PreparedLocalChatRequest};

pub(crate) type PreparedRequest = PreparedLocalChatRequest;

#[cfg(test)]
pub(crate) use xlai_runtime::local_common::{PromptMessage, PromptRole, extract_text_content};

pub(crate) fn prepared_from_core_request(
    config: &LlamaCppConfig,
    request: ChatRequest,
) -> Result<PreparedRequest, XlaiError> {
    PreparedLocalChatRequest::from_chat_request(
        request,
        &LocalChatPrepareOptions {
            default_temperature: config.temperature,
            default_max_output_tokens: config.max_output_tokens,
            expected_model_name: Some(config.resolved_model_name()),
        },
    )
}

pub(crate) fn validate_prepared_for_llama(
    prepared: &PreparedRequest,
    config: &LlamaCppConfig,
) -> Result<(), XlaiError> {
    prepared.validate_common()?;

    if config.n_gpu_layers > 0 && !xlai_sys_llama::supports_gpu_offload() {
        return Err(XlaiError::new(
            xlai_core::ErrorKind::Unsupported,
            "this xlai llama.cpp build was compiled without GPU offload support",
        ));
    }

    Ok(())
}
