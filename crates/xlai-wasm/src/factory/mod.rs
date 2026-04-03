//! Build [`Chat`] / [`Agent`] runtimes for OpenAI and (wasm) transformers.js.

mod openai;

#[cfg(target_arch = "wasm32")]
mod js_reflect;
#[cfg(target_arch = "wasm32")]
mod transformers;

pub(crate) use openai::{create_agent_session_inner, create_chat_session_inner};

#[cfg(target_arch = "wasm32")]
pub(crate) use openai::{
    create_agent_session_with_dyn_file_system, create_chat_session_with_dyn_file_system,
};

#[cfg(target_arch = "wasm32")]
pub(crate) use js_reflect::{js_file_system_arc, parse_transformers_session_options};
#[cfg(target_arch = "wasm32")]
pub(crate) use transformers::{
    create_transformers_agent_session_with_dyn_file_system,
    create_transformers_chat_session_with_dyn_file_system,
};
