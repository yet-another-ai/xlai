use std::sync::LazyLock;

use rust_embed::RustEmbed;
use tera::{Context, Tera};
use xlai_core::{ErrorKind, XlaiError};

#[derive(RustEmbed)]
#[folder = "prompts/"]
struct PromptAssets;

static PROMPT_ENGINE: LazyLock<Result<Tera, XlaiError>> = LazyLock::new(build_prompt_engine);

/// Loads embedded prompt assets and renders them with `tera`.
pub struct EmbeddedPromptStore;

impl EmbeddedPromptStore {
    #[must_use]
    pub fn names() -> Vec<String> {
        let mut names = PromptAssets::iter()
            .map(|name| name.as_ref().to_owned())
            .collect::<Vec<_>>();
        names.sort_unstable();
        names
    }

    /// Loads the raw embedded prompt source.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt does not exist or is not valid UTF-8.
    pub fn load(name: &str) -> Result<String, XlaiError> {
        embedded_prompt_source(name)
    }

    /// Renders an embedded prompt template with the provided `tera` context.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt does not exist, cannot be parsed as a
    /// template, or fails to render with the provided context.
    pub fn render(name: &str, context: &Context) -> Result<String, XlaiError> {
        prompt_engine()?.render(name, context).map_err(|error| {
            XlaiError::new(
                ErrorKind::Configuration,
                format!("failed to render prompt template {name}: {error}"),
            )
        })
    }
}

fn prompt_engine() -> Result<&'static Tera, XlaiError> {
    match &*PROMPT_ENGINE {
        Ok(engine) => Ok(engine),
        Err(error) => Err(error.clone()),
    }
}

fn build_prompt_engine() -> Result<Tera, XlaiError> {
    let mut tera = Tera::default();
    tera.autoescape_on(Vec::new());

    for name in PromptAssets::iter() {
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
    let file = PromptAssets::get(name).ok_or_else(|| {
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

#[cfg(test)]
mod tests {
    use tera::Context;
    use xlai_core::XlaiError;

    use super::EmbeddedPromptStore;

    #[test]
    fn embedded_prompt_store_lists_prompt_assets() {
        let names = EmbeddedPromptStore::names();

        assert!(
            names.contains(&"system/tool-description-skill.md".to_owned()),
            "expected embedded prompt catalog to include the checked-in prompt asset"
        );
    }

    #[test]
    fn embedded_prompt_store_loads_raw_prompt_assets() -> Result<(), XlaiError> {
        let prompt = EmbeddedPromptStore::load("system/tool-description-skill.md")?;

        if !prompt.contains("Execute a skill within the main conversation") {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Configuration,
                "expected raw embedded prompt content to contain the prompt body",
            ));
        }

        Ok(())
    }

    #[test]
    fn embedded_prompt_store_renders_prompt_assets() -> Result<(), XlaiError> {
        let mut context = Context::new();
        context.insert("skill_tag_name", "tool_skill");

        let rendered = EmbeddedPromptStore::render("system/tool-description-skill.md", &context)?;

        if !rendered.contains("tool_skill") {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Configuration,
                "expected rendered prompt content to include the injected template variable",
            ));
        }

        Ok(())
    }
}
