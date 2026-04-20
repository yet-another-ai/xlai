use crate::OpenRouterConfig;

pub(super) fn test_config() -> OpenRouterConfig {
    OpenRouterConfig::new("https://openrouter.ai/api/v1", "k", "openai/gpt-4.1")
}
