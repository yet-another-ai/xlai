pub use xlai_backend_openai::{OpenAiChatModel, OpenAiConfig};
pub use xlai_core as core;
pub use xlai_runtime::{
    Chat, ChatExecutionEvent, DirectoryFileSystem, FileSystem, FsEntry, FsEntryKind, FsPath,
    MemoryFileSystem, ReadableFileSystem, RuntimeBuilder, ToolCallExecutionMode,
    WritableFileSystem, XlaiRuntime,
};

use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use js_sys::{Function, Promise, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::{JsValue, wasm_bindgen};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
use xlai_core::{ChatResponse, FinishReason, MessageRole, TokenUsage};
#[cfg(target_arch = "wasm32")]
use xlai_core::{ErrorKind, ToolDefinition, ToolResult, XlaiError};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmChatRequest {
    prompt: String,
    api_key: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmChatSessionOptions {
    api_key: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
}

impl From<WasmChatRequest> for WasmChatSessionOptions {
    fn from(value: WasmChatRequest) -> Self {
        Self {
            api_key: value.api_key,
            base_url: value.base_url,
            model: value.model,
            system_prompt: value.system_prompt,
            temperature: value.temperature,
            max_output_tokens: value.max_output_tokens,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WasmChatResponse {
    message: WasmChatMessage,
    finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<WasmChatUsage>,
}

#[derive(Serialize)]
struct WasmChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WasmChatUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WasmFsEntry {
    path: String,
    kind: &'static str,
}

#[cfg(target_arch = "wasm32")]
#[derive(Deserialize)]
struct WasmFsEntryInput {
    path: String,
    kind: String,
}

impl From<ChatResponse> for WasmChatResponse {
    fn from(response: ChatResponse) -> Self {
        Self {
            message: WasmChatMessage {
                role: message_role_label(response.message.role),
                content: response.message.content,
            },
            finish_reason: finish_reason_label(response.finish_reason),
            usage: response.usage.map(Into::into),
        }
    }
}

impl From<TokenUsage> for WasmChatUsage {
    fn from(usage: TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl From<FsEntry> for WasmFsEntry {
    fn from(entry: FsEntry) -> Self {
        Self {
            path: entry.path.as_str().to_owned(),
            kind: fs_entry_kind_label(entry.kind),
        }
    }
}

#[wasm_bindgen(js_name = MemoryFileSystem)]
pub struct WasmMemoryFileSystem {
    inner: Arc<MemoryFileSystem>,
}

#[wasm_bindgen(js_class = MemoryFileSystem)]
impl WasmMemoryFileSystem {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MemoryFileSystem::new()),
        }
    }

    pub async fn read(&self, path: String) -> Result<Vec<u8>, JsValue> {
        self.inner.read(&FsPath::from(path)).await.map_err(js_error)
    }

    pub async fn exists(&self, path: String) -> Result<bool, JsValue> {
        self.inner.exists(&FsPath::from(path)).await.map_err(js_error)
    }

    pub async fn write(&self, path: String, data: Vec<u8>) -> Result<(), JsValue> {
        self.inner
            .write(&FsPath::from(path), data)
            .await
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = createDirAll)]
    pub async fn create_dir_all(&self, path: String) -> Result<(), JsValue> {
        self.inner
            .create_dir_all(&FsPath::from(path))
            .await
            .map_err(js_error)
    }

    pub async fn list(&self, path: String) -> Result<JsValue, JsValue> {
        let entries = self
            .inner
            .list(&FsPath::from(path))
            .await
            .map_err(js_error)?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<WasmFsEntry>>();
        serde_wasm_bindgen::to_value(&entries).map_err(js_error)
    }

    #[wasm_bindgen(js_name = deletePath)]
    pub async fn delete_path(&self, path: String) -> Result<(), JsValue> {
        self.inner
            .delete(&FsPath::from(path))
            .await
            .map_err(js_error)
    }
}

impl Default for WasmMemoryFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_arch = "wasm32")]
struct JsFileSystem {
    callbacks: JsValue,
}

#[cfg(target_arch = "wasm32")]
impl JsFileSystem {
    fn new(callbacks: JsValue) -> Self {
        Self { callbacks }
    }

    async fn call0(&self, name: &str, path: &FsPath) -> Result<JsValue, XlaiError> {
        let callback = js_callback(&self.callbacks, name)?;
        let result = callback
            .call1(&self.callbacks, &JsValue::from_str(path.as_str()))
            .map_err(file_system_js_value_error)?;
        JsFuture::from(Promise::resolve(&result))
            .await
            .map_err(file_system_js_value_error)
    }

    async fn call1(&self, name: &str, path: &FsPath, value: JsValue) -> Result<JsValue, XlaiError> {
        let callback = js_callback(&self.callbacks, name)?;
        let result = callback
            .call2(&self.callbacks, &JsValue::from_str(path.as_str()), &value)
            .map_err(file_system_js_value_error)?;
        JsFuture::from(Promise::resolve(&result))
            .await
            .map_err(file_system_js_value_error)
    }
}

#[cfg(target_arch = "wasm32")]
impl ReadableFileSystem for JsFileSystem {
    fn read<'a>(&'a self, path: &'a FsPath) -> xlai_core::BoxFuture<'a, Result<Vec<u8>, XlaiError>> {
        Box::pin(async move {
            let value = self.call0("read", path).await?;
            Ok(Uint8Array::new(&value).to_vec())
        })
    }

    fn exists<'a>(&'a self, path: &'a FsPath) -> xlai_core::BoxFuture<'a, Result<bool, XlaiError>> {
        Box::pin(async move {
            let value = self.call0("exists", path).await?;
            value.as_bool().ok_or_else(|| {
                XlaiError::new(
                    xlai_core::ErrorKind::FileSystem,
                    "filesystem exists() callback must return a boolean",
                )
            })
        })
    }
}

#[cfg(target_arch = "wasm32")]
impl WritableFileSystem for JsFileSystem {
    fn write<'a>(
        &'a self,
        path: &'a FsPath,
        data: Vec<u8>,
    ) -> xlai_core::BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            let bytes = Uint8Array::from(data.as_slice());
            self.call1("write", path, JsValue::from(bytes)).await?;
            Ok(())
        })
    }

    fn delete<'a>(&'a self, path: &'a FsPath) -> xlai_core::BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            self.call0("deletePath", path).await?;
            Ok(())
        })
    }

    fn create_dir_all<'a>(
        &'a self,
        path: &'a FsPath,
    ) -> xlai_core::BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            self.call0("createDirAll", path).await?;
            Ok(())
        })
    }
}

#[cfg(target_arch = "wasm32")]
impl DirectoryFileSystem for JsFileSystem {
    fn list<'a>(&'a self, path: &'a FsPath) -> xlai_core::BoxFuture<'a, Result<Vec<FsEntry>, XlaiError>> {
        Box::pin(async move {
            let value = self.call0("list", path).await?;
            let entries: Vec<WasmFsEntryInput> =
                serde_wasm_bindgen::from_value(value).map_err(file_system_js_error)?;
            entries
                .into_iter()
                .map(|entry| {
                    let kind = match entry.kind.as_str() {
                        "file" => FsEntryKind::File,
                        "directory" => FsEntryKind::Directory,
                        other => {
                            return Err(XlaiError::new(
                                xlai_core::ErrorKind::FileSystem,
                                format!("unsupported filesystem entry kind: {other}"),
                            ));
                        }
                    };

                    Ok(FsEntry {
                        path: FsPath::from(entry.path),
                        kind,
                    })
                })
                .collect()
        })
    }
}

#[wasm_bindgen(js_name = ChatSession)]
pub struct WasmChatSession {
    inner: Chat,
}

#[wasm_bindgen(js_class = ChatSession)]
impl WasmChatSession {
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen(js_name = registerTool)]
    pub fn register_tool(&mut self, definition: JsValue, callback: Function) -> Result<(), JsValue> {
        let definition: ToolDefinition =
            serde_wasm_bindgen::from_value(definition).map_err(js_error)?;

        self.inner.register_tool(definition, move |arguments| {
            let callback = callback.clone();
            async move {
                let arguments = serde_wasm_bindgen::to_value(&arguments).map_err(tool_js_error)?;
                let result = callback
                    .call1(&JsValue::NULL, &arguments)
                    .map_err(tool_js_value_error)?;
                let result = JsFuture::from(Promise::resolve(&result))
                    .await
                    .map_err(tool_js_value_error)?;
                serde_wasm_bindgen::from_value::<ToolResult>(result).map_err(tool_js_error)
            }
        });

        Ok(())
    }

    pub async fn prompt(&self, content: String) -> Result<JsValue, JsValue> {
        serialize_chat_response(self.inner.prompt(content).await.map_err(js_error)?)
    }
}

#[must_use]
pub fn builder() -> RuntimeBuilder {
    RuntimeBuilder::new()
}

#[wasm_bindgen]
#[must_use]
pub fn package_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[wasm_bindgen]
pub async fn chat(options: JsValue) -> Result<JsValue, JsValue> {
    let options: WasmChatRequest = serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    let prompt = options.prompt.clone();
    let chat = create_chat_session_inner(options.into(), None)?;
    chat.prompt(prompt).await
}

#[wasm_bindgen(js_name = createChatSession)]
pub fn create_chat_session(options: JsValue) -> Result<WasmChatSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_chat_session_inner(options, None)
}

#[wasm_bindgen(js_name = createChatSessionWithMemoryFileSystem)]
pub fn create_chat_session_with_memory_file_system(
    options: JsValue,
    file_system: &WasmMemoryFileSystem,
) -> Result<WasmChatSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    create_chat_session_inner(options, Some(file_system.inner.clone()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createChatSessionWithFileSystem)]
pub fn create_chat_session_with_file_system(
    options: JsValue,
    file_system: JsValue,
) -> Result<WasmChatSession, JsValue> {
    let options: WasmChatSessionOptions =
        serde_wasm_bindgen::from_value(options).map_err(js_error)?;
    let file_system: Arc<dyn FileSystem> = Arc::new(JsFileSystem::new(file_system));
    create_chat_session_with_dyn_file_system(options, Some(file_system))
}

fn create_chat_session_inner(
    options: WasmChatSessionOptions,
    file_system: Option<Arc<MemoryFileSystem>>,
) -> Result<WasmChatSession, JsValue> {
    let file_system = file_system.map(|file_system| -> Arc<dyn FileSystem> { file_system });
    create_chat_session_with_dyn_file_system(options, file_system)
}

fn create_chat_session_with_dyn_file_system(
    options: WasmChatSessionOptions,
    file_system: Option<Arc<dyn FileSystem>>,
) -> Result<WasmChatSession, JsValue> {
    let mut runtime_builder = RuntimeBuilder::new().with_chat_backend(OpenAiConfig::new(
        options
            .base_url
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
        options.api_key,
        options
            .model
            .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
    ));

    if let Some(file_system) = file_system {
        runtime_builder = runtime_builder.with_file_system(file_system);
    }

    let runtime = runtime_builder
        .build()
        .map_err(js_error)?;

    let mut chat = runtime.chat_session();

    if let Some(system_prompt) = options.system_prompt {
        chat = chat.with_system_prompt(system_prompt);
    }

    if let Some(temperature) = options.temperature {
        chat = chat.with_temperature(temperature);
    }

    if let Some(max_output_tokens) = options.max_output_tokens {
        chat = chat.with_max_output_tokens(max_output_tokens);
    }

    Ok(WasmChatSession { inner: chat })
}

fn serialize_chat_response(response: ChatResponse) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&WasmChatResponse::from(response)).map_err(js_error)
}

fn js_error(error: impl ToString) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(target_arch = "wasm32")]
fn tool_js_error(error: impl ToString) -> XlaiError {
    XlaiError::new(ErrorKind::Tool, error.to_string())
}

#[cfg(target_arch = "wasm32")]
fn tool_js_value_error(error: JsValue) -> XlaiError {
    XlaiError::new(
        ErrorKind::Tool,
        error
            .as_string()
            .unwrap_or_else(|| format!("javascript callback failed: {error:?}")),
    )
}

#[cfg(target_arch = "wasm32")]
fn file_system_js_error(error: impl ToString) -> XlaiError {
    XlaiError::new(xlai_core::ErrorKind::FileSystem, error.to_string())
}

#[cfg(target_arch = "wasm32")]
fn file_system_js_value_error(error: JsValue) -> XlaiError {
    XlaiError::new(
        xlai_core::ErrorKind::FileSystem,
        error
            .as_string()
            .unwrap_or_else(|| format!("filesystem callback failed: {error:?}")),
    )
}

#[cfg(target_arch = "wasm32")]
fn js_callback(target: &JsValue, name: &str) -> Result<Function, XlaiError> {
    let value = Reflect::get(target, &JsValue::from_str(name)).map_err(file_system_js_value_error)?;
    value.dyn_into::<Function>().map_err(|_| {
        XlaiError::new(
            xlai_core::ErrorKind::FileSystem,
            format!("filesystem callback `{name}` must be a function"),
        )
    })
}

const fn finish_reason_label(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Completed => "completed",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Length => "length",
        FinishReason::Stopped => "stopped",
    }
}

const fn message_role_label(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

const fn fs_entry_kind_label(kind: FsEntryKind) -> &'static str {
    match kind {
        FsEntryKind::File => "file",
        FsEntryKind::Directory => "directory",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use xlai_core::{ChatMessage, FinishReason, FsEntry, FsEntryKind, FsPath, MessageRole, TokenUsage};

    use super::{WasmChatResponse, WasmFsEntry};

    #[test]
    fn wasm_chat_response_uses_js_friendly_field_values() {
        let response = WasmChatResponse::from(xlai_core::ChatResponse {
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: "hello from wasm".to_owned(),
                tool_name: None,
                tool_call_id: None,
                metadata: BTreeMap::new(),
            },
            tool_calls: Vec::new(),
            usage: Some(TokenUsage {
                input_tokens: 5,
                output_tokens: 7,
                total_tokens: 12,
            }),
            finish_reason: FinishReason::Stopped,
            metadata: BTreeMap::new(),
        });

        assert_eq!(response.message.role, "assistant");
        assert_eq!(response.message.content, "hello from wasm");
        assert_eq!(response.finish_reason, "stopped");

        assert!(response.usage.is_some());
        let Some(usage) = response.usage else {
            return;
        };

        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn wasm_fs_entry_uses_js_friendly_field_values() {
        let entry = WasmFsEntry::from(FsEntry {
            path: FsPath::from("/docs/readme.md"),
            kind: FsEntryKind::File,
        });

        assert_eq!(entry.path, "/docs/readme.md");
        assert_eq!(entry.kind, "file");
    }
}
