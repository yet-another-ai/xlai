use std::ffi::{CStr, CString, c_char};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::Once;

pub mod raw {
    #![allow(
        clippy::all,
        clippy::pedantic,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        rustdoc::bare_urls
    )]

    include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));
}

unsafe extern "C" {
    fn xlai_llama_sampler_init_llguidance(
        vocab: *const raw::llama_vocab,
        grammar_kind: *const c_char,
        grammar_data: *const c_char,
    ) -> *mut raw::llama_sampler;
    fn xlai_llama_sampler_init_json_schema(
        vocab: *const raw::llama_vocab,
        json_schema: *const c_char,
    ) -> *mut raw::llama_sampler;
    fn xlai_llama_last_error_message() -> *const c_char;
}

pub type Token = raw::llama_token;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlamaError {
    message: String,
}

impl LlamaError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    #[must_use]
    pub fn into_inner(self) -> String {
        self.message
    }
}

impl Display for LlamaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LlamaError {}

#[derive(Clone, Debug)]
pub struct ModelParams {
    pub n_gpu_layers: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
}

impl Default for ContextParams {
    fn default() -> Self {
        let default_threads = std::thread::available_parallelism()
            .ok()
            .and_then(|count| i32::try_from(count.get()).ok())
            .unwrap_or(1);
        Self {
            n_ctx: 0,
            n_batch: 512,
            n_threads: default_threads,
            n_threads_batch: default_threads,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SamplerParams {
    pub seed: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            seed: raw::LLAMA_DEFAULT_SEED,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

#[derive(Debug)]
pub struct Model {
    inner: NonNull<raw::llama_model>,
}

// SAFETY: the model handle is only used through the safe wrapper API, and the
// backend crate serializes access to the loaded model through a mutex.
unsafe impl Send for Model {}

impl Model {
    /// # Errors
    ///
    /// Returns an error if the model path cannot be converted to a C string or
    /// the upstream library fails to load the GGUF model.
    pub fn load_from_file(path: &Path, params: &ModelParams) -> Result<Self, LlamaError> {
        ensure_backend_initialized();

        let path = path
            .to_str()
            .ok_or_else(|| LlamaError::new("model path must be valid UTF-8"))?;
        let path = CString::new(path)
            .map_err(|_| LlamaError::new("model path may not contain interior NUL bytes"))?;

        // SAFETY: `llama_model_default_params` is a pure upstream initializer.
        let mut raw_params = unsafe { raw::llama_model_default_params() };
        raw_params.n_gpu_layers = params.n_gpu_layers;
        raw_params.use_mmap = params.use_mmap;
        raw_params.use_mlock = params.use_mlock;

        // SAFETY: `path` is a live NUL-terminated C string, and `raw_params`
        // comes from llama.cpp with a few field overrides.
        let model = unsafe { raw::llama_model_load_from_file(path.as_ptr(), raw_params) };
        let inner = NonNull::new(model)
            .ok_or_else(|| LlamaError::new("llama.cpp could not load the requested model"))?;

        Ok(Self { inner })
    }

    #[must_use]
    pub fn description(&self) -> String {
        let mut buffer = vec![0_u8; 256];

        loop {
            // SAFETY: the model pointer is valid for the lifetime of `self`,
            // and the buffer points to writable storage of the declared size.
            let written = unsafe {
                raw::llama_model_desc(
                    self.inner.as_ptr(),
                    buffer.as_mut_ptr().cast::<c_char>(),
                    buffer.len(),
                )
            };

            if written < 0 {
                return "unknown model".to_owned();
            }

            if let Ok(required) = usize::try_from(written) {
                if required < buffer.len() {
                    return String::from_utf8_lossy(&buffer[..required]).into_owned();
                }
                buffer.resize(required.saturating_add(1), 0);
                continue;
            }

            return "unknown model".to_owned();
        }
    }

    #[must_use]
    pub fn default_chat_template(&self) -> Option<String> {
        // SAFETY: the model pointer is valid; a null name asks for the default
        // template according to the llama.cpp API.
        let template = unsafe { raw::llama_model_chat_template(self.inner.as_ptr(), ptr::null()) };
        if template.is_null() {
            return None;
        }

        // SAFETY: llama.cpp returns a stable NUL-terminated string for the
        // lifetime of the model.
        Some(
            unsafe { CStr::from_ptr(template) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    #[must_use]
    pub fn train_context_size(&self) -> u32 {
        let train_ctx = unsafe { raw::llama_model_n_ctx_train(self.inner.as_ptr()) };
        u32::try_from(train_ctx).unwrap_or(0)
    }

    #[must_use]
    pub fn has_encoder(&self) -> bool {
        // SAFETY: the model pointer is valid for the lifetime of `self`.
        unsafe { raw::llama_model_has_encoder(self.inner.as_ptr()) }
    }

    #[must_use]
    pub fn decoder_start_token(&self) -> Option<Token> {
        // SAFETY: the model pointer is valid for the lifetime of `self`.
        let token = unsafe { raw::llama_model_decoder_start_token(self.inner.as_ptr()) };
        if token == raw::LLAMA_TOKEN_NULL {
            None
        } else {
            Some(token)
        }
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp reports a null vocab pointer for a loaded
    /// model.
    pub fn vocab(&self) -> Result<VocabRef<'_>, LlamaError> {
        // SAFETY: the model pointer is valid and llama.cpp returns a vocab that
        // is owned by the model.
        let vocab = unsafe { raw::llama_model_get_vocab(self.inner.as_ptr()) };
        let inner = NonNull::new(vocab.cast_mut())
            .ok_or_else(|| LlamaError::new("llama.cpp returned a null vocab pointer"))?;
        Ok(VocabRef {
            inner,
            marker: PhantomData,
        })
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp fails to create a context from the model.
    pub fn new_context(&self, params: &ContextParams) -> Result<Context, LlamaError> {
        // SAFETY: `llama_context_default_params` is a pure upstream initializer.
        let mut raw_params = unsafe { raw::llama_context_default_params() };
        raw_params.n_ctx = params.n_ctx;
        raw_params.n_batch = params.n_batch;
        raw_params.n_ubatch = params.n_batch;
        raw_params.n_threads = params.n_threads;
        raw_params.n_threads_batch = params.n_threads_batch;

        // SAFETY: the model pointer is valid and the params struct came from the
        // upstream defaults with a small set of field overrides.
        let context = unsafe { raw::llama_init_from_model(self.inner.as_ptr(), raw_params) };
        let inner = NonNull::new(context)
            .ok_or_else(|| LlamaError::new("llama.cpp could not create a context"))?;
        Ok(Context { inner })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: the pointer was returned by llama.cpp and has not been freed.
        unsafe { raw::llama_model_free(self.inner.as_ptr()) };
    }
}

pub struct VocabRef<'a> {
    inner: NonNull<raw::llama_vocab>,
    marker: PhantomData<&'a Model>,
}

impl<'a> VocabRef<'a> {
    #[must_use]
    pub fn bos_token(&self) -> Token {
        // SAFETY: the vocab pointer is borrowed from a live model.
        unsafe { raw::llama_vocab_bos(self.inner.as_ptr()) }
    }

    #[must_use]
    pub fn is_eog(&self, token: Token) -> bool {
        // SAFETY: the vocab pointer is borrowed from a live model.
        unsafe { raw::llama_vocab_is_eog(self.inner.as_ptr(), token) }
    }

    /// # Errors
    ///
    /// Returns an error when the provided text cannot be tokenized.
    pub fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>, LlamaError> {
        let text = CString::new(text)
            .map_err(|_| LlamaError::new("prompt text may not contain interior NUL bytes"))?;
        let text_len = i32::try_from(text.as_bytes().len())
            .map_err(|_| LlamaError::new("prompt text is too large for llama.cpp tokenization"))?;

        // SAFETY: the vocab pointer is valid; a null output buffer with length 0
        // asks llama.cpp to report the required token count.
        let required = unsafe {
            raw::llama_tokenize(
                self.inner.as_ptr(),
                text.as_ptr(),
                text_len,
                ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };

        if required == i32::MIN {
            return Err(LlamaError::new(
                "tokenization overflowed llama.cpp's 32-bit token count",
            ));
        }

        let required = required
            .checked_abs()
            .ok_or_else(|| LlamaError::new("tokenization failed unexpectedly"))?;
        let required = usize::try_from(required)
            .map_err(|_| LlamaError::new("tokenization requested too many tokens"))?;

        let mut tokens = vec![0; required];
        // SAFETY: the token buffer is writable and large enough based on the
        // sizing call above.
        let written = unsafe {
            raw::llama_tokenize(
                self.inner.as_ptr(),
                text.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                i32::try_from(tokens.len())
                    .map_err(|_| LlamaError::new("token buffer exceeds llama.cpp limits"))?,
                add_special,
                parse_special,
            )
        };

        if written < 0 {
            return Err(LlamaError::new("llama.cpp failed to tokenize the prompt"));
        }

        let written = usize::try_from(written)
            .map_err(|_| LlamaError::new("tokenization returned an invalid length"))?;
        tokens.truncate(written);
        Ok(tokens)
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp cannot render the token piece.
    pub fn token_to_piece(&self, token: Token, special: bool) -> Result<String, LlamaError> {
        let mut buffer = vec![0_u8; 32];

        loop {
            // SAFETY: the vocab pointer is valid and the buffer points to
            // writable storage.
            let written = unsafe {
                raw::llama_token_to_piece(
                    self.inner.as_ptr(),
                    token,
                    buffer.as_mut_ptr().cast::<c_char>(),
                    i32::try_from(buffer.len())
                        .map_err(|_| LlamaError::new("piece buffer exceeds llama.cpp limits"))?,
                    0,
                    special,
                )
            };

            if written >= 0 {
                let written = usize::try_from(written)
                    .map_err(|_| LlamaError::new("llama.cpp returned an invalid piece length"))?;
                return Ok(String::from_utf8_lossy(&buffer[..written]).into_owned());
            }

            let required = usize::try_from(written.checked_abs().unwrap_or(i32::MAX))
                .map_err(|_| LlamaError::new("llama.cpp requested an invalid piece buffer size"))?;
            let next_size = required.max(buffer.len().saturating_mul(2));
            buffer.resize(next_size, 0);
        }
    }
}

pub struct Context {
    inner: NonNull<raw::llama_context>,
}

impl Context {
    /// # Errors
    ///
    /// Returns an error if llama.cpp fails while encoding the provided tokens.
    pub fn encode(&mut self, tokens: &mut [Token]) -> Result<(), LlamaError> {
        if tokens.is_empty() {
            return Ok(());
        }

        // SAFETY: the batch borrows the mutable token slice for the duration of
        // the call that immediately follows.
        let batch = unsafe {
            raw::llama_batch_get_one(
                tokens.as_mut_ptr(),
                i32::try_from(tokens.len())
                    .map_err(|_| LlamaError::new("token batch exceeds llama.cpp limits"))?,
            )
        };

        // SAFETY: the context is valid and the batch points to live token
        // storage for the duration of the call.
        let status = unsafe { raw::llama_encode(self.inner.as_ptr(), batch) };
        if status != 0 {
            return Err(LlamaError::new(format!(
                "llama.cpp failed to encode the prompt batch with status {status}"
            )));
        }

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp fails while decoding the provided tokens.
    pub fn decode(&mut self, tokens: &mut [Token]) -> Result<(), LlamaError> {
        if tokens.is_empty() {
            return Ok(());
        }

        // SAFETY: the batch borrows the mutable token slice for the duration of
        // the call that immediately follows.
        let batch = unsafe {
            raw::llama_batch_get_one(
                tokens.as_mut_ptr(),
                i32::try_from(tokens.len())
                    .map_err(|_| LlamaError::new("token batch exceeds llama.cpp limits"))?,
            )
        };

        // SAFETY: the context is valid and the batch points to live token
        // storage for the duration of the call.
        let status = unsafe { raw::llama_decode(self.inner.as_ptr(), batch) };
        if status != 0 {
            return Err(LlamaError::new(format!(
                "llama.cpp failed to decode the prompt batch with status {status}"
            )));
        }

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the sampler could not select a token.
    pub fn sample(&mut self, sampler: &mut Sampler) -> Result<Token, LlamaError> {
        // SAFETY: both the sampler and context are valid live llama.cpp objects.
        let token =
            unsafe { raw::llama_sampler_sample(sampler.inner.as_ptr(), self.inner.as_ptr(), -1) };
        if token == raw::LLAMA_TOKEN_NULL {
            return Err(LlamaError::new("llama.cpp returned a null sampled token"));
        }
        Ok(token)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the pointer was returned by llama.cpp and has not been freed.
        unsafe { raw::llama_free(self.inner.as_ptr()) };
    }
}

pub struct Sampler {
    inner: NonNull<raw::llama_sampler>,
}

impl Sampler {
    /// # Errors
    ///
    /// Returns an error if llama.cpp fails to construct the sampler chain.
    pub fn new(params: &SamplerParams) -> Result<Self, LlamaError> {
        Self::new_with_grammar(params, None)
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp fails to construct the sampler chain or
    /// the optional grammar sampler.
    pub fn new_with_grammar(
        params: &SamplerParams,
        grammar: Option<(&VocabRef<'_>, &str, &str)>,
    ) -> Result<Self, LlamaError> {
        // SAFETY: `llama_sampler_chain_default_params` is a pure upstream
        // initializer that returns a by-value configuration struct.
        let chain_params = unsafe { raw::llama_sampler_chain_default_params() };
        // SAFETY: the params were created by llama.cpp itself.
        let sampler = unsafe { raw::llama_sampler_chain_init(chain_params) };
        let inner = NonNull::new(sampler)
            .ok_or_else(|| LlamaError::new("llama.cpp could not create a sampler chain"))?;

        let mut sampler = Self { inner };

        if let Some((vocab, grammar_str, grammar_root)) = grammar {
            sampler.add_grammar_sampler(vocab, grammar_str, grammar_root)?;
        }

        if params.temperature <= 0.0 {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_greedy() },
                "greedy sampler",
            )?;
            return Ok(sampler);
        }

        if params.top_k > 0 {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_top_k(params.top_k) },
                "top-k sampler",
            )?;
        }

        if (0.0..1.0).contains(&params.top_p) {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_top_p(params.top_p, 1) },
                "top-p sampler",
            )?;
        }

        sampler.add_owned_sampler(
            // SAFETY: this constructor allocates a standalone sampler.
            unsafe { raw::llama_sampler_init_temp(params.temperature) },
            "temperature sampler",
        )?;
        sampler.add_owned_sampler(
            // SAFETY: this constructor allocates a standalone sampler.
            unsafe { raw::llama_sampler_init_dist(params.seed) },
            "distribution sampler",
        )?;

        Ok(sampler)
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp fails to construct the sampler chain or
    /// initialize a JSON-schema-constrained sampler.
    pub fn new_with_json_schema(
        params: &SamplerParams,
        vocab: &VocabRef<'_>,
        json_schema: &str,
    ) -> Result<Self, LlamaError> {
        // SAFETY: `llama_sampler_chain_default_params` is a pure upstream
        // initializer that returns a by-value configuration struct.
        let chain_params = unsafe { raw::llama_sampler_chain_default_params() };
        // SAFETY: the params were created by llama.cpp itself.
        let sampler = unsafe { raw::llama_sampler_chain_init(chain_params) };
        let inner = NonNull::new(sampler)
            .ok_or_else(|| LlamaError::new("llama.cpp could not create a sampler chain"))?;

        let mut sampler = Self { inner };
        sampler.add_json_schema_sampler(vocab, json_schema)?;

        if params.temperature <= 0.0 {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_greedy() },
                "greedy sampler",
            )?;
            return Ok(sampler);
        }

        if params.top_k > 0 {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_top_k(params.top_k) },
                "top-k sampler",
            )?;
        }

        if (0.0..1.0).contains(&params.top_p) {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_top_p(params.top_p, 1) },
                "top-p sampler",
            )?;
        }

        sampler.add_owned_sampler(
            // SAFETY: this constructor allocates a standalone sampler.
            unsafe { raw::llama_sampler_init_temp(params.temperature) },
            "temperature sampler",
        )?;
        sampler.add_owned_sampler(
            // SAFETY: this constructor allocates a standalone sampler.
            unsafe { raw::llama_sampler_init_dist(params.seed) },
            "distribution sampler",
        )?;

        Ok(sampler)
    }

    /// # Errors
    ///
    /// Returns an error if llama.cpp fails to construct the sampler chain or
    /// initialize an LLGuidance grammar sampler.
    pub fn new_with_llguidance(
        params: &SamplerParams,
        vocab: &VocabRef<'_>,
        grammar_kind: &str,
        grammar_data: &str,
    ) -> Result<Self, LlamaError> {
        // SAFETY: `llama_sampler_chain_default_params` is a pure upstream
        // initializer that returns a by-value configuration struct.
        let chain_params = unsafe { raw::llama_sampler_chain_default_params() };
        // SAFETY: the params were created by llama.cpp itself.
        let sampler = unsafe { raw::llama_sampler_chain_init(chain_params) };
        let inner = NonNull::new(sampler)
            .ok_or_else(|| LlamaError::new("llama.cpp could not create a sampler chain"))?;

        let mut sampler = Self { inner };
        sampler.add_llguidance_sampler(vocab, grammar_kind, grammar_data)?;

        if params.temperature <= 0.0 {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_greedy() },
                "greedy sampler",
            )?;
            return Ok(sampler);
        }

        if params.top_k > 0 {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_top_k(params.top_k) },
                "top-k sampler",
            )?;
        }

        if (0.0..1.0).contains(&params.top_p) {
            sampler.add_owned_sampler(
                // SAFETY: this constructor allocates a standalone sampler.
                unsafe { raw::llama_sampler_init_top_p(params.top_p, 1) },
                "top-p sampler",
            )?;
        }

        sampler.add_owned_sampler(
            // SAFETY: this constructor allocates a standalone sampler.
            unsafe { raw::llama_sampler_init_temp(params.temperature) },
            "temperature sampler",
        )?;
        sampler.add_owned_sampler(
            // SAFETY: this constructor allocates a standalone sampler.
            unsafe { raw::llama_sampler_init_dist(params.seed) },
            "distribution sampler",
        )?;

        Ok(sampler)
    }

    pub fn accept(&mut self, token: Token) {
        // SAFETY: the sampler is valid and llama.cpp accepts any token returned
        // by the same model.
        unsafe { raw::llama_sampler_accept(self.inner.as_ptr(), token) };
    }

    fn add_owned_sampler(
        &mut self,
        sampler: *mut raw::llama_sampler,
        name: &str,
    ) -> Result<(), LlamaError> {
        let sampler = NonNull::new(sampler)
            .ok_or_else(|| LlamaError::new(format!("llama.cpp could not create the {name}")))?;
        // SAFETY: the chain and owned sampler are valid. Ownership of the child
        // sampler moves into the chain.
        unsafe { raw::llama_sampler_chain_add(self.inner.as_ptr(), sampler.as_ptr()) };
        Ok(())
    }

    fn add_grammar_sampler(
        &mut self,
        vocab: &VocabRef<'_>,
        grammar_str: &str,
        grammar_root: &str,
    ) -> Result<(), LlamaError> {
        let grammar_str = CString::new(grammar_str)
            .map_err(|_| LlamaError::new("grammar may not contain interior NUL bytes"))?;
        let grammar_root = CString::new(grammar_root)
            .map_err(|_| LlamaError::new("grammar root may not contain interior NUL bytes"))?;
        self.add_owned_sampler(
            // SAFETY: the vocab pointer is borrowed from a live model, both C
            // strings are valid for the duration of the call, and ownership of
            // the returned sampler transfers into the sampler chain.
            unsafe {
                raw::llama_sampler_init_grammar(
                    vocab.inner.as_ptr(),
                    grammar_str.as_ptr(),
                    grammar_root.as_ptr(),
                )
            },
            "grammar sampler",
        )
    }

    fn add_json_schema_sampler(
        &mut self,
        vocab: &VocabRef<'_>,
        json_schema: &str,
    ) -> Result<(), LlamaError> {
        let json_schema = CString::new(json_schema)
            .map_err(|_| LlamaError::new("JSON schema may not contain interior NUL bytes"))?;
        let sampler =
            NonNull::new(
                // SAFETY: the vocab pointer is borrowed from a live model, the JSON
                // schema C string remains alive for the duration of the call, and
                // ownership of the returned sampler transfers into the sampler
                // chain when initialization succeeds.
                unsafe {
                    xlai_llama_sampler_init_json_schema(vocab.inner.as_ptr(), json_schema.as_ptr())
                },
            )
            .ok_or_else(|| {
                LlamaError::new(llguidance_error_message().unwrap_or_else(|| {
                    "llama.cpp could not create the JSON schema sampler".to_owned()
                }))
            })?;
        // SAFETY: the chain and owned sampler are valid. Ownership of the child
        // sampler moves into the chain.
        unsafe { raw::llama_sampler_chain_add(self.inner.as_ptr(), sampler.as_ptr()) };
        Ok(())
    }

    fn add_llguidance_sampler(
        &mut self,
        vocab: &VocabRef<'_>,
        grammar_kind: &str,
        grammar_data: &str,
    ) -> Result<(), LlamaError> {
        let grammar_kind = CString::new(grammar_kind)
            .map_err(|_| LlamaError::new("grammar kind may not contain interior NUL bytes"))?;
        let grammar_data = CString::new(grammar_data)
            .map_err(|_| LlamaError::new("grammar data may not contain interior NUL bytes"))?;
        let sampler =
            NonNull::new(
                // SAFETY: the vocab pointer is borrowed from a live model, both C
                // strings remain alive for the duration of the call, and ownership
                // of the returned sampler transfers into the sampler chain when
                // initialization succeeds.
                unsafe {
                    xlai_llama_sampler_init_llguidance(
                        vocab.inner.as_ptr(),
                        grammar_kind.as_ptr(),
                        grammar_data.as_ptr(),
                    )
                },
            )
            .ok_or_else(|| {
                LlamaError::new(llguidance_error_message().unwrap_or_else(|| {
                    "llama.cpp could not create the LLGuidance sampler".to_owned()
                }))
            })?;
        // SAFETY: the chain and owned sampler are valid. Ownership of the child
        // sampler moves into the chain.
        unsafe { raw::llama_sampler_chain_add(self.inner.as_ptr(), sampler.as_ptr()) };
        Ok(())
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // SAFETY: the pointer was returned by llama.cpp and has not been freed.
        unsafe { raw::llama_sampler_free(self.inner.as_ptr()) };
    }
}

/// # Errors
///
/// Returns an error if the template or any message strings contain interior
/// NUL bytes, or if llama.cpp cannot render the final prompt.
pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage<'_>],
    add_assistant: bool,
) -> Result<String, LlamaError> {
    let template = CString::new(template)
        .map_err(|_| LlamaError::new("chat template may not contain interior NUL bytes"))?;

    let role_strings = messages
        .iter()
        .map(|message| {
            CString::new(message.role)
                .map_err(|_| LlamaError::new("message role may not contain interior NUL bytes"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let content_strings = messages
        .iter()
        .map(|message| {
            CString::new(message.content)
                .map_err(|_| LlamaError::new("message content may not contain interior NUL bytes"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let raw_messages = role_strings
        .iter()
        .zip(content_strings.iter())
        .map(|(role, content)| raw::llama_chat_message {
            role: role.as_ptr(),
            content: content.as_ptr(),
        })
        .collect::<Vec<_>>();

    let capacity = messages
        .iter()
        .map(|message| message.role.len().saturating_add(message.content.len()))
        .sum::<usize>()
        .saturating_mul(2)
        .saturating_add(128)
        .max(256);
    let mut buffer = vec![0_u8; capacity];

    loop {
        // SAFETY: all C strings in `raw_messages` remain alive for the duration
        // of the call, and the output buffer is writable and sized correctly.
        let written = unsafe {
            raw::llama_chat_apply_template(
                template.as_ptr(),
                raw_messages.as_ptr(),
                raw_messages.len(),
                add_assistant,
                buffer.as_mut_ptr().cast::<c_char>(),
                i32::try_from(buffer.len())
                    .map_err(|_| LlamaError::new("template buffer exceeds llama.cpp limits"))?,
            )
        };

        if written < 0 {
            return Err(LlamaError::new(
                "llama.cpp failed to apply the chat template",
            ));
        }

        let written = usize::try_from(written)
            .map_err(|_| LlamaError::new("llama.cpp returned an invalid template length"))?;
        if written < buffer.len() {
            return Ok(String::from_utf8_lossy(&buffer[..written]).into_owned());
        }

        buffer.resize(written.saturating_add(1), 0);
    }
}

#[must_use]
pub fn supports_gpu_offload() -> bool {
    ensure_backend_initialized();
    // SAFETY: this is a pure capability query in llama.cpp.
    unsafe { raw::llama_supports_gpu_offload() }
}

fn ensure_backend_initialized() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: llama.cpp documents this as a one-time process-level init.
        unsafe { raw::llama_backend_init() };
    });
}

fn llguidance_error_message() -> Option<String> {
    // SAFETY: the wrapper returns either null or a stable NUL-terminated
    // thread-local string pointer valid until the next wrapper call.
    let message = unsafe { xlai_llama_last_error_message() };
    if message.is_null() {
        return None;
    }

    // SAFETY: non-null pointers from the wrapper always reference a
    // NUL-terminated string.
    Some(
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned(),
    )
}
