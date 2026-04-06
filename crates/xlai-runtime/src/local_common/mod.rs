//! Shared preparation, prompts, and tool JSON envelope for local (non-API) chat backends.

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
