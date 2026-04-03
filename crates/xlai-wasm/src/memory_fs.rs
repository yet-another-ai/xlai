//! In-memory [`MemoryFileSystem`] exposed to JavaScript.

use std::sync::Arc;

use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;
use xlai_runtime::{
    DirectoryFileSystem, FsPath, MemoryFileSystem, ReadableFileSystem, WritableFileSystem,
};

use crate::types::WasmFsEntry;
use crate::wasm_helpers::js_error;

#[wasm_bindgen(js_name = MemoryFileSystem)]
pub struct WasmMemoryFileSystem {
    pub(crate) inner: Arc<MemoryFileSystem>,
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
        self.inner
            .exists(&FsPath::from(path))
            .await
            .map_err(js_error)
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
