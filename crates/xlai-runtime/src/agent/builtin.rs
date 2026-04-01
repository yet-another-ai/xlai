use std::sync::Arc;

use xlai_core::{RuntimeCapability, XlaiError};

use crate::{Chat, XlaiRuntime};

mod skill;

type BuiltinToolInstaller = fn(&mut Chat, Arc<XlaiRuntime>) -> Result<(), XlaiError>;

struct BuiltinToolRegistration {
    capability: RuntimeCapability,
    install: BuiltinToolInstaller,
}

const BUILTIN_TOOLS: &[BuiltinToolRegistration] = &[BuiltinToolRegistration {
    capability: RuntimeCapability::SkillResolution,
    install: skill::register,
}];

pub(super) fn register_builtin_tools(
    chat: &mut Chat,
    runtime: Arc<XlaiRuntime>,
) -> Result<(), XlaiError> {
    for builtin in BUILTIN_TOOLS {
        if runtime.has_capability(builtin.capability) {
            (builtin.install)(chat, runtime.clone())?;
        }
    }

    Ok(())
}
