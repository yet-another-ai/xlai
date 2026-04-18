//! Local (non-HTTP) chat preparation shared by **llama.cpp** and **transformers.js** backends.
//!
//! # Supported integration surface
//!
//! These items are intended for backend implementations and advanced callers:
//!
//! - Request shaping: [`PreparedLocalChatRequest`], [`LocalChatPrepareOptions`], [`PromptMessage`],
//!   [`PromptRole`], [`extract_text_content`].
//! - Prompt rendering: [`render_manual_prompt`], [`prompt_messages_with_constraints`],
//!   [`structured_output_instruction`].
//! - Tool JSON: [`tool_call_instruction`], [`tool_response_schema`], [`parse_tool_response`],
//!   [`ToolResponse`].
//! - Schema validation: [`validate_structured_output_schema`], [`validate_structured_output`].
//!
//! Submodules [`request`] and [`tool_calling`] remain public so paths like
//! `xlai_runtime::local_common::request::PreparedLocalChatRequest` stay stable.
//!
//! This module was previously the `xlai-local-common` crate; it lives inside `xlai-runtime`
//! so the published runtime does not depend on a separate unpublished helper crate.

mod prompt;
mod prompt_store;
pub mod request;
mod schema;
pub mod tool_calling;

pub use prompt::{
    prompt_messages_with_constraints, render_manual_prompt, structured_output_instruction,
};
pub use request::{
    LocalChatPrepareOptions, PreparedLocalChatRequest, PromptMessage, PromptRole,
    extract_text_content,
};
pub use schema::{validate_structured_output, validate_structured_output_schema};
pub use tool_calling::{
    ToolResponse, parse_tool_response, tool_call_instruction, tool_response_schema,
};
