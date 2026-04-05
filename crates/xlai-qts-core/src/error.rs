use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Qwen3TtsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid string: {0}")]
    Nul(#[from] std::ffi::NulError),

    #[error("invalid UTF-8 in path")]
    InvalidPath,

    #[error("missing or unreadable model file: {0}")]
    ModelFile(PathBuf),

    #[error("file is not a valid GGUF container: {0}")]
    InvalidGguf(PathBuf),

    #[error("file is not a valid ONNX model: {0}")]
    InvalidOnnx(PathBuf),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("missing tokenizer token: {0}")]
    MissingTokenizerToken(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("missing GGUF tensor: {0}")]
    MissingTensor(String),

    #[error("invalid GGUF tensor: {0}")]
    InvalidTensor(String),

    #[error("unsupported GGUF tensor type for {0}")]
    UnsupportedTensorType(String),

    #[error("stage not implemented yet: {0}")]
    NotImplemented(&'static str),

    #[error("onnx runtime error: {0}")]
    Ort(String),
}
