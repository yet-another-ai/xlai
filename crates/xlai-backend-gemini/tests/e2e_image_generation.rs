use xlai_backend_gemini::GeminiConfig;
use xlai_core::{ImageGenerationBackend, ImageGenerationModel, ImageGenerationRequest, XlaiError};

#[tokio::test]
async fn gemini_image_generation_smoke_test() -> Result<(), XlaiError> {
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            println!(
                "Skipping gemini_image_generation_smoke_test because GEMINI_API_KEY is not set"
            );
            return Ok(());
        }
    };

    let base_url = std::env::var("GEMINI_BASE_URL")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta".to_owned());
    let model_name = std::env::var("GEMINI_IMAGE_MODEL")
        .unwrap_or_else(|_| "gemini-3.1-flash-image-preview".to_owned());

    let config = GeminiConfig::new(base_url, api_key, model_name);
    let model = config.into_image_generation_model();

    let request = ImageGenerationRequest {
        model: None,
        prompt: "A cute nano banana wearing sunglasses".to_owned(),
        size: Some("1:1".to_owned()),
        quality: None,
        background: None,
        output_format: None,
        count: Some(1),
        metadata: Default::default(),
    };

    let response = model.generate_image(request).await?;
    assert!(
        !response.images.is_empty(),
        "Response should contain at least one image"
    );
    assert!(
        matches!(
            response.images[0].image,
            xlai_core::MediaSource::InlineData { .. }
        ),
        "Image should be inline data"
    );

    Ok(())
}
