//! JavaScript-backed file system (browser callbacks).

use js_sys::{Promise, Uint8Array};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;
use xlai_core::XlaiError;
use xlai_runtime::{
    DirectoryFileSystem, FsEntry, FsEntryKind, FsPath, ReadableFileSystem, WritableFileSystem,
};

use crate::types::WasmFsEntryInput;
use crate::wasm_helpers::{file_system_js_error, file_system_js_value_error, js_callback};

pub(crate) struct JsFileSystem {
    callbacks: JsValue,
}

impl JsFileSystem {
    pub(crate) fn new(callbacks: JsValue) -> Self {
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

impl ReadableFileSystem for JsFileSystem {
    fn read<'a>(
        &'a self,
        path: &'a FsPath,
    ) -> xlai_core::BoxFuture<'a, Result<Vec<u8>, XlaiError>> {
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

impl DirectoryFileSystem for JsFileSystem {
    fn list<'a>(
        &'a self,
        path: &'a FsPath,
    ) -> xlai_core::BoxFuture<'a, Result<Vec<FsEntry>, XlaiError>> {
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
