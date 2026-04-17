use xlai_backend_openai::OpenAiConfig;
use xlai_core::{
    ErrorKind, ImageGenerationOutputFormat, ImageGenerationRequest, MediaSource, RuntimeCapability,
    XlaiError,
};
use xlai_runtime::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_IMAGE_MODEL in the protected e2e environment"]
async fn openai_image_generation_smoke_test() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_image_generation_runtime()?;

    let response = runtime
        .generate_image(ImageGenerationRequest {
            model: None,
            prompt: "A simple pixel-art robot waving hello".to_owned(),
            size: Some("1024x1024".to_owned()),
            quality: None,
            background: None,
            output_format: Some(ImageGenerationOutputFormat::Png),
            count: Some(1),
            metadata: Default::default(),
        })
        .await?;

    assert!(
        !response.images.is_empty(),
        "image generation should return at least one image",
    );
    assert_generated_image_payload(&response.images[0])?;

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_IMAGE_MODEL in the protected e2e environment"]
async fn openai_image_generation_runtime_reports_capability() -> Result<(), XlaiError> {
    let _ = dotenvy::dotenv();

    let runtime = build_image_generation_runtime()?;

    assert!(
        runtime.has_capability(RuntimeCapability::ImageGeneration),
        "runtime should expose image generation capability when backend is configured",
    );

    let response = runtime
        .generate_image(ImageGenerationRequest {
            model: None,
            prompt: "A simple flat icon of a mountain landscape".to_owned(),
            size: Some("1024x1024".to_owned()),
            quality: None,
            background: None,
            output_format: Some(ImageGenerationOutputFormat::Png),
            count: Some(1),
            metadata: Default::default(),
        })
        .await?;

    assert!(
        !response.images.is_empty(),
        "capability check should still return at least one image",
    );
    assert_generated_image_payload(&response.images[0])?;

    Ok(())
}

fn assert_generated_image_payload(image: &xlai_core::GeneratedImage) -> Result<(), XlaiError> {
    match &image.image {
        MediaSource::InlineData { data, .. } => {
            if data.len() <= 16 {
                return Err(XlaiError::new(
                    ErrorKind::Provider,
                    "generated inline image should contain non-trivial bytes",
                ));
            }
        }
        MediaSource::Url { url } => {
            if url.trim().is_empty() {
                return Err(XlaiError::new(
                    ErrorKind::Provider,
                    "generated image URL should not be empty",
                ));
            }
        }
    }

    Ok(())
}

fn build_image_generation_runtime() -> Result<xlai_runtime::XlaiRuntime, XlaiError> {
    let api_key = require_env("OPENAI_API_KEY")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let chat_model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned());
    let image_model = require_env("OPENAI_IMAGE_MODEL")?;

    RuntimeBuilder::new()
        .with_image_generation_backend(
            OpenAiConfig::new(base_url, api_key, chat_model).with_image_model(image_model),
        )
        .build()
}

fn require_env(name: &str) -> Result<String, XlaiError> {
    std::env::var(name).map_err(|error| {
        XlaiError::new(
            ErrorKind::Configuration,
            format!("{name} must be set for e2e tests: {error}"),
        )
    })
}
