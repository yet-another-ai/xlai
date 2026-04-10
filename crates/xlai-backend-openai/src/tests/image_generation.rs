use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::json;
use xlai_core::{
    ErrorKind, ImageGenerationBackground, ImageGenerationOutputFormat, ImageGenerationQuality,
    ImageGenerationRequest, MediaSource,
};

use crate::image_generation::{OpenAiImageGenerationRequest, OpenAiImageGenerationResponse};

use super::common::test_config;

#[test]
fn image_generation_request_uses_explicit_or_configured_model() {
    let config = test_config().with_image_model("gpt-image-config");
    let request = ImageGenerationRequest {
        model: None,
        prompt: "A lantern floating above a lake".to_owned(),
        size: Some("1024x1024".to_owned()),
        quality: Some(ImageGenerationQuality::High),
        background: Some(ImageGenerationBackground::Transparent),
        output_format: Some(ImageGenerationOutputFormat::Png),
        count: Some(2),
        metadata: BTreeMap::new(),
    };

    let payload = OpenAiImageGenerationRequest::from_core_request(&config, request.clone());
    assert!(payload.is_ok(), "build image request");
    let payload_json = serde_json::to_value(payload.unwrap_or_else(|_| unreachable!()));
    assert!(payload_json.is_ok(), "serialize payload");
    let payload_json = payload_json.unwrap_or_else(|_| json!({}));
    assert_eq!(payload_json["model"], json!("gpt-image-config"));
    assert_eq!(payload_json["n"], json!(2));
    assert_eq!(payload_json["quality"], json!("high"));
    assert_eq!(payload_json["background"], json!("transparent"));
    assert_eq!(payload_json["output_format"], json!("png"));

    let mut explicit_model_request = request;
    explicit_model_request.model = Some("gpt-image-explicit".to_owned());
    let explicit = OpenAiImageGenerationRequest::from_core_request(&config, explicit_model_request);
    assert!(explicit.is_ok(), "build explicit-model image request");
    let explicit_json = serde_json::to_value(explicit.unwrap_or_else(|_| unreachable!()));
    assert!(explicit_json.is_ok(), "serialize explicit payload");
    assert_eq!(
        explicit_json.unwrap_or_else(|_| json!({}))["model"],
        json!("gpt-image-explicit")
    );
}

#[test]
fn image_generation_request_rejects_zero_count() {
    let config = test_config();
    let result = OpenAiImageGenerationRequest::from_core_request(
        &config,
        ImageGenerationRequest {
            model: None,
            prompt: "A lantern floating above a lake".to_owned(),
            size: None,
            quality: None,
            background: None,
            output_format: None,
            count: Some(0),
            metadata: BTreeMap::new(),
        },
    );
    assert!(result.is_err(), "zero count should be rejected");
    let Err(error) = result else {
        return;
    };
    assert_eq!(error.kind, ErrorKind::Validation);
}

#[test]
fn image_generation_response_decodes_inline_payloads_and_metadata() {
    let encoded = STANDARD.encode([137, 80, 78, 71]);
    let response: Result<OpenAiImageGenerationResponse, _> = serde_json::from_value(json!({
        "created": 1710000000,
        "data": [
            {
                "b64_json": encoded,
                "revised_prompt": "A lantern floating above a moonlit lake",
                "seed": 1234
            }
        ],
        "usage": {
            "total_tokens": 88
        }
    }));
    assert!(response.is_ok(), "deserialize image response");
    let Ok(response) = response else {
        return;
    };

    let response = response.into_core_response(Some(ImageGenerationOutputFormat::Png));
    assert!(response.is_ok(), "map image response");
    let Ok(response) = response else {
        return;
    };

    assert_eq!(response.metadata.get("created"), Some(&json!(1710000000)));
    assert_eq!(
        response.metadata.get("usage"),
        Some(&json!({ "total_tokens": 88 }))
    );
    assert_eq!(response.images.len(), 1);
    assert_eq!(
        response.images[0].revised_prompt.as_deref(),
        Some("A lantern floating above a moonlit lake")
    );
    assert_eq!(response.images[0].mime_type.as_deref(), Some("image/png"));
    assert_eq!(response.images[0].metadata.get("seed"), Some(&json!(1234)));
    assert!(matches!(
        response.images[0].image,
        MediaSource::InlineData { .. }
    ));
}

#[test]
fn image_generation_response_preserves_urls_when_provider_returns_them() {
    let response: Result<OpenAiImageGenerationResponse, _> = serde_json::from_value(json!({
        "data": [
            {
                "url": "https://example.com/generated.webp"
            }
        ]
    }));
    assert!(response.is_ok(), "deserialize url image response");
    let Ok(response) = response else {
        return;
    };

    let mapped = response.into_core_response(Some(ImageGenerationOutputFormat::Webp));
    assert!(mapped.is_ok(), "map url image response");
    let Ok(mapped) = mapped else {
        return;
    };

    assert!(matches!(
        mapped.images[0].image,
        MediaSource::Url { ref url } if url == "https://example.com/generated.webp"
    ));
    assert_eq!(mapped.images[0].mime_type.as_deref(), Some("image/webp"));
}
