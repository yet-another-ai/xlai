use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use async_stream::try_stream;
use tokio::sync::mpsc;
use xlai_core::{
    BoxFuture, BoxStream, ChatBackend, ChatChunk, ChatContent, ChatMessage, ChatModel, ChatRequest,
    ChatResponse, ErrorKind, FinishReason, MessageRole, StructuredOutputFormat, XlaiError,
};
use xlai_llama_cpp_sys as sys;

mod prompt;
mod request;

#[cfg(test)]
mod tests;

use prompt::{render_prompt, validate_structured_output};
use request::PreparedRequest;

#[derive(Clone, Debug)]
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub model_name: Option<String>,
    pub context_size: Option<u32>,
    pub threads: Option<i32>,
    pub n_gpu_layers: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_output_tokens: u32,
    pub seed: u32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub chat_template: Option<String>,
}

impl LlamaCppConfig {
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            model_name: None,
            context_size: None,
            threads: None,
            n_gpu_layers: 0,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            max_output_tokens: 256,
            seed: sys::raw::LLAMA_DEFAULT_SEED,
            use_mmap: true,
            use_mlock: false,
            chat_template: None,
        }
    }

    #[must_use]
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    #[must_use]
    pub fn with_context_size(mut self, context_size: u32) -> Self {
        self.context_size = Some(context_size);
        self
    }

    #[must_use]
    pub fn with_threads(mut self, threads: i32) -> Self {
        self.threads = Some(threads);
        self
    }

    #[must_use]
    pub fn with_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    #[must_use]
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    #[must_use]
    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    #[must_use]
    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    #[must_use]
    pub fn with_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    #[must_use]
    pub fn with_chat_template(mut self, chat_template: impl Into<String>) -> Self {
        self.chat_template = Some(chat_template.into());
        self
    }

    #[must_use]
    pub fn resolved_model_name(&self) -> String {
        if let Some(model_name) = &self.model_name {
            return model_name.clone();
        }

        self.model_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.model_path.display().to_string())
    }
}

#[derive(Clone, Debug)]
pub struct LlamaCppChatModel {
    config: LlamaCppConfig,
    runtime: Arc<RuntimeState>,
}

#[derive(Debug, Default)]
struct RuntimeState {
    loaded: OnceLock<Result<Arc<Mutex<LoadedModel>>, String>>,
}

#[derive(Debug)]
pub(crate) struct LoadedModel {
    model: sys::Model,
    default_chat_template: Option<String>,
}

impl LlamaCppChatModel {
    #[must_use]
    pub fn new(config: LlamaCppConfig) -> Self {
        Self {
            config,
            runtime: Arc::new(RuntimeState::default()),
        }
    }

    fn load_model(&self) -> Result<Arc<Mutex<LoadedModel>>, XlaiError> {
        let loaded = self.runtime.loaded.get_or_init(|| {
            load_model(&self.config)
                .map(|loaded| Arc::new(Mutex::new(loaded)))
                .map_err(|error| error.message)
        });

        loaded
            .clone()
            .map_err(|message| XlaiError::new(ErrorKind::Provider, message))
    }

    fn run_generation<F>(
        &self,
        prepared: PreparedRequest,
        mut emit: F,
    ) -> Result<ChatResponse, XlaiError>
    where
        F: FnMut(ChatChunk) -> Result<(), XlaiError>,
    {
        prepared.validate_against(&self.config)?;

        let loaded = self.load_model()?;
        let loaded = loaded.lock().map_err(|_| {
            XlaiError::new(
                ErrorKind::Provider,
                "llama.cpp model lock was poisoned by a previous panic",
            )
        })?;

        let prompt = render_prompt(&self.config, &loaded, &prepared)?;
        let vocab = loaded.model.vocab().map_err(map_provider_error)?;
        let mut prompt_tokens = vocab
            .tokenize(&prompt, true, true)
            .map_err(map_provider_error)?;
        if prompt_tokens.is_empty() {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                "llama.cpp prompt tokenization produced no input tokens",
            ));
        }

        let context_size = resolve_context_size(&self.config, &loaded.model);
        let required_context = prompt_tokens
            .len()
            .saturating_add(usize::try_from(prepared.max_output_tokens).unwrap_or(usize::MAX));
        if required_context >= usize::try_from(context_size).unwrap_or(usize::MAX) {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                format!(
                    "prompt and requested output exceed the configured llama.cpp context ({required_context} tokens required, {context_size} available)"
                ),
            ));
        }

        let mut context = loaded
            .model
            .new_context(&sys::ContextParams {
                n_ctx: context_size,
                n_batch: prompt_tokens
                    .len()
                    .try_into()
                    .unwrap_or(u32::MAX)
                    .min(context_size),
                n_threads: self
                    .config
                    .threads
                    .unwrap_or(sys::ContextParams::default().n_threads),
                n_threads_batch: self
                    .config
                    .threads
                    .unwrap_or(sys::ContextParams::default().n_threads_batch),
            })
            .map_err(map_provider_error)?;

        if loaded.model.has_encoder() {
            context
                .encode(&mut prompt_tokens)
                .map_err(map_provider_error)?;
            let start_token = loaded
                .model
                .decoder_start_token()
                .unwrap_or_else(|| vocab.bos_token());
            let mut start = vec![start_token];
            context.decode(&mut start).map_err(map_provider_error)?;
        } else {
            context
                .decode(&mut prompt_tokens)
                .map_err(map_provider_error)?;
        }

        let mut sampler = if let Some(structured_output) = &prepared.structured_output {
            match &structured_output.format {
                StructuredOutputFormat::JsonSchema { schema } => {
                    let schema = serde_json::to_string(schema).map_err(|error| {
                        XlaiError::new(
                            ErrorKind::Validation,
                            format!("structured output schema could not be serialized: {error}"),
                        )
                    })?;
                    sys::Sampler::new_with_json_schema(
                        &sys::SamplerParams {
                            seed: self.config.seed,
                            temperature: prepared.temperature,
                            top_k: self.config.top_k,
                            top_p: self.config.top_p,
                        },
                        &vocab,
                        &schema,
                    )
                    .map_err(map_provider_error)?
                }
                StructuredOutputFormat::LarkGrammar { grammar } => {
                    sys::Sampler::new_with_llguidance(
                        &sys::SamplerParams {
                            seed: self.config.seed,
                            temperature: prepared.temperature,
                            top_k: self.config.top_k,
                            top_p: self.config.top_p,
                        },
                        &vocab,
                        "lark",
                        grammar,
                    )
                    .map_err(map_provider_error)?
                }
            }
        } else {
            sys::Sampler::new(&sys::SamplerParams {
                seed: self.config.seed,
                temperature: prepared.temperature,
                top_k: self.config.top_k,
                top_p: self.config.top_p,
            })
            .map_err(map_provider_error)?
        };

        emit(ChatChunk::MessageStart {
            role: MessageRole::Assistant,
        })?;

        let mut generated = String::new();
        let mut output_tokens = 0_u32;
        let mut finish_reason = FinishReason::Length;

        for _ in 0..prepared.max_output_tokens {
            let token = context.sample(&mut sampler).map_err(map_provider_error)?;
            if vocab.is_eog(token) {
                finish_reason = FinishReason::Stopped;
                break;
            }

            sampler.accept(token);
            let piece = vocab
                .token_to_piece(token, true)
                .map_err(map_provider_error)?;
            if !piece.is_empty() {
                generated.push_str(&piece);
                emit(ChatChunk::ContentDelta(xlai_core::StreamTextDelta {
                    part_index: 0,
                    delta: piece,
                }))?;
            }

            let mut decode_token = vec![token];
            context
                .decode(&mut decode_token)
                .map_err(map_provider_error)?;
            output_tokens = output_tokens.saturating_add(1);
        }

        if let Some(structured_output) = &prepared.structured_output {
            validate_structured_output(structured_output, &generated)?;
        }

        Ok(ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: ChatContent::text(generated),
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            tool_calls: Vec::new(),
            usage: Some(xlai_core::TokenUsage {
                input_tokens: prompt_tokens.len().try_into().unwrap_or(u32::MAX),
                output_tokens,
                total_tokens: prompt_tokens
                    .len()
                    .try_into()
                    .unwrap_or(u32::MAX)
                    .saturating_add(output_tokens),
            }),
            finish_reason,
            metadata: Default::default(),
        })
    }
}

impl ChatBackend for LlamaCppConfig {
    type Model = LlamaCppChatModel;

    fn into_chat_model(self) -> Self::Model {
        LlamaCppChatModel::new(self)
    }
}

impl ChatModel for LlamaCppChatModel {
    fn provider_name(&self) -> &'static str {
        "llama.cpp"
    }

    fn generate(&self, request: ChatRequest) -> BoxFuture<'_, Result<ChatResponse, XlaiError>> {
        let model = self.clone();
        Box::pin(async move {
            let prepared = PreparedRequest::from_core_request(&model.config, request)?;
            tokio::task::spawn_blocking(move || model.run_generation(prepared, |_| Ok(())))
                .await
                .map_err(map_join_error)?
        })
    }

    fn generate_stream(&self, request: ChatRequest) -> BoxStream<'_, Result<ChatChunk, XlaiError>> {
        let model = self.clone();
        Box::pin(try_stream! {
            let prepared = PreparedRequest::from_core_request(&model.config, request)?;
            let (sender, mut receiver) = mpsc::unbounded_channel();

            tokio::spawn(async move {
                let sender_for_generation = sender.clone();
                let outcome = tokio::task::spawn_blocking(move || {
                    model.run_generation(prepared, |chunk| {
                        sender_for_generation.send(Ok(chunk)).map_err(|_| {
                            XlaiError::new(
                                ErrorKind::Provider,
                                "llama.cpp stream receiver closed before generation finished",
                            )
                        })
                    })
                })
                .await
                .map_err(map_join_error);

                match outcome {
                    Ok(Ok(response)) => {
                        let _ = sender.send(Ok(ChatChunk::Finished(response)));
                    }
                    Ok(Err(error)) | Err(error) => {
                        let _ = sender.send(Err(error));
                    }
                }
            });

            while let Some(item) = receiver.recv().await {
                yield item?;
            }
        })
    }
}

fn load_model(config: &LlamaCppConfig) -> Result<LoadedModel, XlaiError> {
    let model = sys::Model::load_from_file(
        &config.model_path,
        &sys::ModelParams {
            n_gpu_layers: config.n_gpu_layers,
            use_mmap: config.use_mmap,
            use_mlock: config.use_mlock,
        },
    )
    .map_err(map_provider_error)?;

    Ok(LoadedModel {
        default_chat_template: model.default_chat_template(),
        model,
    })
}

fn resolve_context_size(config: &LlamaCppConfig, model: &sys::Model) -> u32 {
    config
        .context_size
        .unwrap_or_else(|| model.train_context_size().max(2048))
}

pub(crate) fn map_provider_error(error: sys::LlamaError) -> XlaiError {
    XlaiError::new(ErrorKind::Provider, error.to_string())
}

fn map_join_error(error: tokio::task::JoinError) -> XlaiError {
    XlaiError::new(
        ErrorKind::Provider,
        format!("llama.cpp background generation task failed: {error}"),
    )
}
