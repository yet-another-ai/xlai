//! Runtime tests: image generation capability wiring.
use std::sync::{Arc, Mutex};

use xlai_core::{ErrorKind, ImageGenerationRequest, RuntimeCapability, XlaiError};

use super::common::{
    RecordingImageGenerationModel, lock_unpoisoned, sample_generated_image_response,
};
use crate::RuntimeBuilder;

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_generate_image_uses_configured_model() -> Result<(), XlaiError> {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(RecordingImageGenerationModel::new(
        requests.clone(),
        vec![sample_generated_image_response()],
    ));

    let runtime = RuntimeBuilder::new()
        .with_image_generation_model(model)
        .build()?;

    let request = ImageGenerationRequest {
        model: Some("gpt-image-1".to_owned()),
        prompt: "A cat sitting on a windowsill".to_owned(),
        size: Some("1024x1024".to_owned()),
        quality: None,
        background: None,
        output_format: None,
        count: Some(1),
        metadata: Default::default(),
    };

    let response = runtime.generate_image(request.clone()).await?;
    assert_eq!(response.images.len(), 1);
    assert!(runtime.has_capability(RuntimeCapability::ImageGeneration));

    let requests = lock_unpoisoned(&requests);
    assert_eq!(requests.as_slice(), &[request]);

    Ok(())
}

#[allow(clippy::panic_in_result_fn)]
#[tokio::test]
async fn runtime_generate_image_errors_without_configured_model() -> Result<(), XlaiError> {
    let runtime = RuntimeBuilder::new()
        .with_chat_model(Arc::new(super::common::RecordingChatModel::new(
            Arc::new(Mutex::new(Vec::new())),
            Vec::new(),
        )))
        .build()?;

    let result = runtime
        .generate_image(ImageGenerationRequest {
            model: None,
            prompt: "A cat sitting on a windowsill".to_owned(),
            size: None,
            quality: None,
            background: None,
            output_format: None,
            count: None,
            metadata: Default::default(),
        })
        .await;
    let error = match result {
        Err(error) => error,
        Ok(_) => {
            return Err(XlaiError::new(
                ErrorKind::Provider,
                "missing image generation model should fail",
            ));
        }
    };

    assert_eq!(error.kind, ErrorKind::Unsupported);
    assert!(error.message.contains("image generation model"));

    Ok(())
}
