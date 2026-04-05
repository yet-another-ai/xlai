use std::sync::LazyLock;

use rust_embed::RustEmbed;
use tera::{Context, Tera};
use xlai_core::{ErrorKind, XlaiError};

#[derive(RustEmbed)]
#[folder = "prompts/local_common/"]
struct LocalChatPromptAssets;

static LOCAL_CHAT_PROMPT_ENGINE: LazyLock<Result<Tera, XlaiError>> =
    LazyLock::new(build_prompt_engine);

pub(crate) struct EmbeddedPromptStore;

impl EmbeddedPromptStore {
    pub(crate) fn render(name: &str, context: &Context) -> Result<String, XlaiError> {
        prompt_engine()?.render(name, context).map_err(|error| {
            XlaiError::new(
                ErrorKind::Configuration,
                format!("failed to render prompt template {name}: {error}"),
            )
        })
    }
}

fn prompt_engine() -> Result<&'static Tera, XlaiError> {
    match &*LOCAL_CHAT_PROMPT_ENGINE {
        Ok(engine) => Ok(engine),
        Err(error) => Err(error.clone()),
    }
}

fn build_prompt_engine() -> Result<Tera, XlaiError> {
    let mut tera = Tera::default();
    tera.autoescape_on(Vec::new());

    for name in LocalChatPromptAssets::iter() {
        let name = name.as_ref();
        let source = embedded_prompt_source(name)?;
        tera.add_raw_template(name, &source).map_err(|error| {
            XlaiError::new(
                ErrorKind::Configuration,
                format!("failed to register embedded prompt template {name}: {error}"),
            )
        })?;
    }

    Ok(tera)
}

fn embedded_prompt_source(name: &str) -> Result<String, XlaiError> {
    let file = LocalChatPromptAssets::get(name).ok_or_else(|| {
        XlaiError::new(
            ErrorKind::Configuration,
            format!("embedded prompt template not found: {name}"),
        )
    })?;

    String::from_utf8(file.data.into_owned()).map_err(|error| {
        XlaiError::new(
            ErrorKind::Configuration,
            format!("embedded prompt template {name} is not valid UTF-8: {error}"),
        )
    })
}
