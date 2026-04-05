//! Chat backend for browser inference via a JavaScript adapter that runs
//! `@huggingface/transformers` with optional llguidance constrained decoding.
//!
//! On non-`wasm32` targets, [`ChatModel::generate`] returns [`ErrorKind::Unsupported`].

#[cfg(not(target_arch = "wasm32"))]
use xlai_core::ErrorKind;
use xlai_core::{BoxFuture, ChatBackend, ChatModel, ChatRequest, ChatResponse, XlaiError};

/// Configuration for the transformers.js-backed chat model (HF model id and defaults).
#[derive(Clone, Debug)]
pub struct TransformersJsConfig {
    pub model_id: String,
    pub temperature: f32,
    pub max_output_tokens: u32,
}

impl TransformersJsConfig {
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            temperature: 0.8,
            max_output_tokens: 256,
        }
    }

    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    #[must_use]
    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::fmt;

    use serde::Deserialize;
    use serde_json::{Value, json};

    use js_sys::{Function, Promise, Reflect};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use xlai_core::{
        BoxFuture, ChatContent, ChatMessage, ChatModel, ChatRequest, ChatResponse, ErrorKind,
        FinishReason, MessageRole, StructuredOutputFormat, TokenUsage, XlaiError,
    };
    use xlai_runtime::local_common::{
        LocalChatPrepareOptions, PreparedLocalChatRequest, ToolResponse, parse_tool_response,
        prompt_messages_with_constraints, render_manual_prompt, tool_response_schema,
        validate_structured_output,
    };

    use super::TransformersJsConfig;

    fn prepared_from_request(
        config: &TransformersJsConfig,
        request: ChatRequest,
    ) -> Result<PreparedLocalChatRequest, XlaiError> {
        PreparedLocalChatRequest::from_chat_request(
            request,
            &LocalChatPrepareOptions {
                default_temperature: config.temperature,
                default_max_output_tokens: config.max_output_tokens,
                expected_model_name: Some(config.model_id.clone()),
            },
        )
    }

    fn build_prompt(prepared: &PreparedLocalChatRequest) -> Result<String, XlaiError> {
        let messages = prompt_messages_with_constraints(prepared)?;
        render_manual_prompt(&messages)
    }

    fn grammar_and_tool_schema(
        prepared: &PreparedLocalChatRequest,
    ) -> Result<(Option<Value>, Option<Value>), XlaiError> {
        if !prepared.available_tools.is_empty() {
            return Ok((None, Some(tool_response_schema(&prepared.available_tools))));
        }

        let Some(structured) = &prepared.structured_output else {
            return Ok((None, None));
        };

        let grammar = match &structured.format {
            StructuredOutputFormat::JsonSchema { schema } => {
                json!({ "type": "json_schema", "schema": schema })
            }
            StructuredOutputFormat::LarkGrammar { grammar } => {
                json!({
                    "type": "lark",
                    "grammar": grammar,
                    "startSymbol": "start",
                })
            }
        };

        Ok((Some(grammar), None))
    }

    fn finish_reason_from_js(s: Option<&str>) -> FinishReason {
        let Some(s) = s.map(str::trim) else {
            return FinishReason::Completed;
        };
        match s.to_ascii_lowercase().as_str() {
            "length" => FinishReason::Length,
            "stopped" | "eos" => FinishReason::Stopped,
            _ => FinishReason::Completed,
        }
    }

    #[derive(serde::Serialize)]
    #[serde(rename_all = "camelCase")]
    struct JsGenerateRequest {
        prompt: String,
        model: String,
        temperature: f32,
        max_new_tokens: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        grammar: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_schema: Option<Value>,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct JsGenerateResponse {
        text: String,
        #[serde(default)]
        finish_reason: Option<String>,
        #[serde(default)]
        usage: Option<JsUsageFields>,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct JsUsageFields {
        #[serde(default)]
        input_tokens: Option<u32>,
        #[serde(default)]
        output_tokens: Option<u32>,
        #[serde(default)]
        total_tokens: Option<u32>,
    }

    fn js_provider_error(message: impl Into<String>) -> XlaiError {
        XlaiError::new(ErrorKind::Provider, message.into())
    }

    fn ensure_adapter(adapter: &wasm_bindgen::JsValue) -> Result<(), XlaiError> {
        if adapter.is_null() || adapter.is_undefined() {
            return Err(js_provider_error(
                "transformers.js adapter must be a non-null object with a generate() method",
            ));
        }
        let generate = Reflect::get(adapter, &wasm_bindgen::JsValue::from_str("generate"))
            .map_err(|_| js_provider_error("failed to read adapter.generate"))?;
        if generate.is_undefined() || generate.is_null() {
            return Err(js_provider_error(
                "transformers.js adapter must expose generate(request) -> Promise",
            ));
        }
        let _ = generate
            .dyn_into::<Function>()
            .map_err(|_| js_provider_error("adapter.generate must be a function"))?;
        Ok(())
    }

    async fn call_adapter_generate(
        adapter: &wasm_bindgen::JsValue,
        body: JsGenerateRequest,
    ) -> Result<JsGenerateResponse, XlaiError> {
        ensure_adapter(adapter)?;
        let generate = Reflect::get(adapter, &wasm_bindgen::JsValue::from_str("generate"))
            .map_err(|_| js_provider_error("failed to read adapter.generate"))?;
        let generate: Function = generate
            .dyn_into()
            .map_err(|_| js_provider_error("adapter.generate must be a function"))?;

        let arg = serde_wasm_bindgen::to_value(&body)
            .map_err(|e| js_provider_error(format!("failed to serialize generate request: {e}")))?;

        let promise_val = generate
            .call1(adapter, &arg)
            .map_err(|e| js_provider_error(format!("adapter.generate call failed: {e:?}")))?;
        let promise = Promise::resolve(&promise_val);
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| js_provider_error(format!("adapter.generate promise failed: {e:?}")))?;

        serde_wasm_bindgen::from_value(result)
            .map_err(|e| js_provider_error(format!("invalid generate response: {e}")))
    }

    fn usage_from_js(u: Option<JsUsageFields>) -> Option<TokenUsage> {
        let u = u?;
        let input_tokens = u.input_tokens.unwrap_or(0);
        let output_tokens = u.output_tokens.unwrap_or(0);
        let total_tokens = u
            .total_tokens
            .unwrap_or_else(|| input_tokens.saturating_add(output_tokens));
        Some(TokenUsage {
            input_tokens,
            output_tokens,
            total_tokens,
        })
    }

    pub(super) struct Bundle {
        config: TransformersJsConfig,
        adapter: wasm_bindgen::JsValue,
    }

    impl Bundle {
        pub(super) fn new(config: TransformersJsConfig, adapter: wasm_bindgen::JsValue) -> Self {
            Self { config, adapter }
        }

        pub(super) fn config(&self) -> &TransformersJsConfig {
            &self.config
        }

        pub(super) fn adapter(&self) -> wasm_bindgen::JsValue {
            self.adapter.clone()
        }

        pub(super) fn into_model(self) -> ChatModelImpl {
            ChatModelImpl {
                config: self.config,
                adapter: self.adapter,
            }
        }
    }

    impl fmt::Debug for Bundle {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("TransformersJsBundle")
                .field("config", &self.config)
                .finish_non_exhaustive()
        }
    }

    impl Clone for Bundle {
        fn clone(&self) -> Self {
            Self {
                config: self.config.clone(),
                adapter: self.adapter.clone(),
            }
        }
    }

    pub(super) struct ChatModelImpl {
        config: TransformersJsConfig,
        adapter: wasm_bindgen::JsValue,
    }

    impl ChatModelImpl {
        pub(super) fn new(config: TransformersJsConfig, adapter: wasm_bindgen::JsValue) -> Self {
            Self { config, adapter }
        }
    }

    impl fmt::Debug for ChatModelImpl {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("TransformersJsChatModel")
                .field("config", &self.config)
                .finish_non_exhaustive()
        }
    }

    impl Clone for ChatModelImpl {
        fn clone(&self) -> Self {
            Self {
                config: self.config.clone(),
                adapter: self.adapter.clone(),
            }
        }
    }

    impl ChatModel for ChatModelImpl {
        fn provider_name(&self) -> &'static str {
            "transformers.js"
        }

        fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
            let config = self.config.clone();
            let adapter = self.adapter.clone();
            Box::pin(async move {
                let prepared = prepared_from_request(&config, request)?;
                prepared.validate_common()?;

                let prompt = build_prompt(&prepared)?;
                let (grammar, tool_schema) = grammar_and_tool_schema(&prepared)?;
                let tool_mode = !prepared.available_tools.is_empty();

                let js_req = JsGenerateRequest {
                    prompt,
                    model: config.model_id.clone(),
                    temperature: prepared.temperature,
                    max_new_tokens: prepared.max_output_tokens,
                    grammar,
                    tool_schema,
                };

                let js_resp = call_adapter_generate(&adapter, js_req).await?;
                let mut generated = js_resp.text;
                let usage = usage_from_js(js_resp.usage);

                let (message_text, tool_calls, finish_reason) = if tool_mode {
                    match parse_tool_response(&generated, &prepared.available_tools)? {
                        ToolResponse::AssistantMessage(message_text) => (
                            message_text,
                            Vec::new(),
                            finish_reason_from_js(js_resp.finish_reason.as_deref()),
                        ),
                        ToolResponse::ToolCalls(tool_calls) => {
                            generated.clear();
                            (String::new(), tool_calls, FinishReason::ToolCalls)
                        }
                    }
                } else {
                    if let Some(structured_output) = &prepared.structured_output {
                        validate_structured_output(structured_output, &generated)?;
                    }
                    (
                        generated,
                        Vec::new(),
                        finish_reason_from_js(js_resp.finish_reason.as_deref()),
                    )
                };

                Ok(ChatResponse {
                    message: ChatMessage {
                        role: MessageRole::Assistant,
                        content: ChatContent::text(message_text),
                        tool_name: None,
                        tool_call_id: None,
                        metadata: Default::default(),
                    },
                    tool_calls,
                    usage,
                    finish_reason,
                    metadata: Default::default(),
                })
            })
        }
    }
}

#[cfg(target_arch = "wasm32")]
use wasm::{Bundle as WasmBundle, ChatModelImpl as WasmChatModelImpl};

/// Bundle of config plus a JavaScript adapter (wasm only). The adapter must expose
/// `async generate(request)` returning `{ text, finishReason?, usage? }` (camelCase).
#[cfg(target_arch = "wasm32")]
pub struct TransformersJsBundle {
    inner: WasmBundle,
}

#[cfg(target_arch = "wasm32")]
impl TransformersJsBundle {
    #[must_use]
    pub fn new(config: TransformersJsConfig, adapter: wasm_bindgen::JsValue) -> Self {
        Self {
            inner: WasmBundle::new(config, adapter),
        }
    }

    #[must_use]
    pub fn adapter(&self) -> wasm_bindgen::JsValue {
        self.inner.adapter()
    }

    #[must_use]
    pub fn config(&self) -> &TransformersJsConfig {
        self.inner.config()
    }
}

#[cfg(target_arch = "wasm32")]
impl std::fmt::Debug for TransformersJsBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(target_arch = "wasm32")]
impl Clone for TransformersJsBundle {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub struct TransformersJsChatModel {
    inner: WasmChatModelImpl,
}

#[cfg(target_arch = "wasm32")]
impl TransformersJsChatModel {
    #[must_use]
    pub fn new(config: TransformersJsConfig, adapter: wasm_bindgen::JsValue) -> Self {
        Self {
            inner: WasmChatModelImpl::new(config, adapter),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl std::fmt::Debug for TransformersJsChatModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

#[cfg(target_arch = "wasm32")]
impl Clone for TransformersJsChatModel {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl ChatBackend for TransformersJsBundle {
    type Model = TransformersJsChatModel;

    fn into_chat_model(self) -> Self::Model {
        TransformersJsChatModel {
            inner: self.inner.into_model(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl ChatModel for TransformersJsChatModel {
    fn provider_name(&self) -> &'static str {
        self.inner.provider_name()
    }

    fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
        self.inner.generate(request)
    }
}

/// Stub bundle for non-wasm builds (CI); [`ChatModel::generate`] is unsupported.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
pub struct TransformersJsBundle {
    pub config: TransformersJsConfig,
}

#[cfg(not(target_arch = "wasm32"))]
impl TransformersJsBundle {
    #[must_use]
    pub fn new(config: TransformersJsConfig) -> Self {
        Self { config }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
pub struct TransformersJsChatModel {
    _config: TransformersJsConfig,
}

#[cfg(not(target_arch = "wasm32"))]
impl TransformersJsChatModel {
    #[must_use]
    pub fn new(config: TransformersJsConfig) -> Self {
        Self { _config: config }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ChatBackend for TransformersJsBundle {
    type Model = TransformersJsChatModel;

    fn into_chat_model(self) -> Self::Model {
        TransformersJsChatModel::new(self.config)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ChatModel for TransformersJsChatModel {
    fn provider_name(&self) -> &'static str {
        "transformers.js"
    }

    fn generate(&self, _request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
        Box::pin(async move {
            Err(XlaiError::new(
                ErrorKind::Unsupported,
                "xlai-backend-transformersjs only runs in wasm32 builds; use wasm-pack for the browser",
            ))
        })
    }
}

#[cfg(test)]
mod tests;
