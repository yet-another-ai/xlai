use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard};

use xlai_core::{
    BoxFuture, DirectoryFileSystem, ErrorKind, FileSystem, FsEntry, FsEntryKind, FsPath,
    ReadableFileSystem, WritableFileSystem, XlaiError,
};

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct MemoryFileSystem {
    state: std::sync::Arc<Mutex<MemoryFileSystemState>>,
}

#[derive(Default)]
struct MemoryFileSystemState {
    files: BTreeMap<FsPath, Vec<u8>>,
    directories: BTreeSet<FsPath>,
}

impl MemoryFileSystem {
    #[must_use]
    pub fn new() -> Self {
        let mut state = MemoryFileSystemState::default();
        state.directories.insert(FsPath::from("/"));
        Self {
            state: std::sync::Arc::new(Mutex::new(state)),
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, MemoryFileSystemState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl Default for MemoryFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadableFileSystem for MemoryFileSystem {
    fn read<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<u8>, XlaiError>> {
        Box::pin(async move {
            let path = normalize_virtual_path(path)?;
            let state = self.lock_state();
            state.files.get(&path).cloned().ok_or_else(|| {
                XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("file not found: {}", path.as_str()),
                )
            })
        })
    }

    fn exists<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<bool, XlaiError>> {
        Box::pin(async move {
            let path = normalize_virtual_path(path)?;
            let state = self.lock_state();
            Ok(state.files.contains_key(&path) || state.directories.contains(&path))
        })
    }
}

impl WritableFileSystem for MemoryFileSystem {
    fn write<'a>(
        &'a self,
        path: &'a FsPath,
        data: Vec<u8>,
    ) -> BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            let path = normalize_virtual_path(path)?;
            let parent = parent_path(&path).ok_or_else(|| {
                XlaiError::new(
                    ErrorKind::FileSystem,
                    "cannot write data to the filesystem root",
                )
            })?;
            let mut state = self.lock_state();
            if !state.directories.contains(&parent) {
                return Err(XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("parent directory does not exist: {}", parent.as_str()),
                ));
            }
            state.files.insert(path, data);
            Ok(())
        })
    }

    fn delete<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            let path = normalize_virtual_path(path)?;
            let mut state = self.lock_state();

            if state.files.remove(&path).is_some() {
                return Ok(());
            }

            if path.as_str() == "/" {
                state.files.clear();
                state.directories.clear();
                state.directories.insert(FsPath::from("/"));
                return Ok(());
            }

            if !state.directories.contains(&path) {
                return Err(XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("path not found: {}", path.as_str()),
                ));
            }

            let prefix = path_prefix(&path);
            state
                .files
                .retain(|entry_path, _| !entry_path.as_str().starts_with(&prefix));
            state.directories.retain(|entry_path| {
                entry_path.as_str() == "/" || !entry_path.as_str().starts_with(&prefix)
            });
            state.directories.remove(&path);
            Ok(())
        })
    }

    fn create_dir_all<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            let path = normalize_virtual_path(path)?;
            let mut state = self.lock_state();
            for ancestor in ancestor_paths(&path) {
                state.directories.insert(ancestor);
            }
            Ok(())
        })
    }
}

impl DirectoryFileSystem for MemoryFileSystem {
    fn list<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<FsEntry>, XlaiError>> {
        Box::pin(async move {
            let path = normalize_virtual_path(path)?;
            let state = self.lock_state();
            if !state.directories.contains(&path) {
                return Err(XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("directory not found: {}", path.as_str()),
                ));
            }

            let mut entries = BTreeMap::new();
            for directory in &state.directories {
                if let Some(name) = direct_child_name(&path, directory) {
                    entries.insert(
                        name,
                        FsEntry {
                            path: directory.clone(),
                            kind: FsEntryKind::Directory,
                        },
                    );
                }
            }
            for file_path in state.files.keys() {
                if let Some(name) = direct_child_name(&path, file_path) {
                    entries.insert(
                        name,
                        FsEntry {
                            path: file_path.clone(),
                            kind: FsEntryKind::File,
                        },
                    );
                }
            }

            Ok(entries.into_values().collect())
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
pub struct LocalFileSystem {
    root: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl LocalFileSystem {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve_path(&self, path: &FsPath) -> Result<PathBuf, XlaiError> {
        let path = normalize_virtual_path(path)?;
        if path.as_str() == "/" {
            return Ok(self.root.clone());
        }

        let mut resolved = self.root.clone();
        for segment in path.as_str().trim_start_matches('/').split('/') {
            resolved.push(segment);
        }
        Ok(resolved)
    }

    fn to_virtual_path(&self, path: &Path) -> Result<FsPath, XlaiError> {
        let relative = path.strip_prefix(&self.root).map_err(|error| {
            XlaiError::new(
                ErrorKind::FileSystem,
                format!("failed to resolve relative path: {error}"),
            )
        })?;
        let mut virtual_path = String::from("/");
        let mut components = relative.components().peekable();
        while let Some(component) = components.next() {
            virtual_path.push_str(&component.as_os_str().to_string_lossy());
            if components.peek().is_some() {
                virtual_path.push('/');
            }
        }
        Ok(FsPath::from(virtual_path))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ReadableFileSystem for LocalFileSystem {
    fn read<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<u8>, XlaiError>> {
        Box::pin(async move {
            let resolved = self.resolve_path(path)?;
            std::fs::read(&resolved).map_err(|error| {
                XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("failed to read {}: {error}", path.as_str()),
                )
            })
        })
    }

    fn exists<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<bool, XlaiError>> {
        Box::pin(async move {
            let resolved = self.resolve_path(path)?;
            Ok(resolved.exists())
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl WritableFileSystem for LocalFileSystem {
    fn write<'a>(
        &'a self,
        path: &'a FsPath,
        data: Vec<u8>,
    ) -> BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            let resolved = self.resolve_path(path)?;
            let parent = resolved.parent().ok_or_else(|| {
                XlaiError::new(
                    ErrorKind::FileSystem,
                    "cannot write data to the filesystem root",
                )
            })?;
            if !parent.exists() {
                return Err(XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("parent directory does not exist: {}", path.as_str()),
                ));
            }
            std::fs::write(&resolved, data).map_err(|error| {
                XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("failed to write {}: {error}", path.as_str()),
                )
            })
        })
    }

    fn delete<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            let resolved = self.resolve_path(path)?;
            if !resolved.exists() {
                return Err(XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("path not found: {}", path.as_str()),
                ));
            }
            if resolved.is_dir() {
                std::fs::remove_dir_all(&resolved).map_err(|error| {
                    XlaiError::new(
                        ErrorKind::FileSystem,
                        format!("failed to delete directory {}: {error}", path.as_str()),
                    )
                })
            } else {
                std::fs::remove_file(&resolved).map_err(|error| {
                    XlaiError::new(
                        ErrorKind::FileSystem,
                        format!("failed to delete file {}: {error}", path.as_str()),
                    )
                })
            }
        })
    }

    fn create_dir_all<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>> {
        Box::pin(async move {
            let resolved = self.resolve_path(path)?;
            std::fs::create_dir_all(&resolved).map_err(|error| {
                XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("failed to create directory {}: {error}", path.as_str()),
                )
            })
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl DirectoryFileSystem for LocalFileSystem {
    fn list<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<FsEntry>, XlaiError>> {
        Box::pin(async move {
            let resolved = self.resolve_path(path)?;
            let read_dir = std::fs::read_dir(&resolved).map_err(|error| {
                XlaiError::new(
                    ErrorKind::FileSystem,
                    format!("failed to list {}: {error}", path.as_str()),
                )
            })?;

            let mut entries = Vec::new();
            for entry in read_dir {
                let entry = entry.map_err(|error| {
                    XlaiError::new(
                        ErrorKind::FileSystem,
                        format!("failed to read entry in {}: {error}", path.as_str()),
                    )
                })?;
                let file_type = entry.file_type().map_err(|error| {
                    XlaiError::new(
                        ErrorKind::FileSystem,
                        format!("failed to inspect entry in {}: {error}", path.as_str()),
                    )
                })?;
                let kind = if file_type.is_dir() {
                    FsEntryKind::Directory
                } else {
                    FsEntryKind::File
                };
                entries.push(FsEntry {
                    path: self.to_virtual_path(&entry.path())?,
                    kind,
                });
            }
            entries.sort_by(|left, right| left.path.cmp(&right.path));
            Ok(entries)
        })
    }
}

pub fn boxed_file_system(file_system: impl FileSystem + 'static) -> std::sync::Arc<dyn FileSystem> {
    std::sync::Arc::new(file_system)
}

fn normalize_virtual_path(path: &FsPath) -> Result<FsPath, XlaiError> {
    let raw = path.as_str();
    if !raw.starts_with('/') {
        return Err(XlaiError::new(
            ErrorKind::Validation,
            format!("filesystem paths must start with '/': {raw}"),
        ));
    }

    let mut segments = Vec::new();
    for segment in raw.split('/') {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." {
            return Err(XlaiError::new(
                ErrorKind::Validation,
                format!("filesystem paths must not contain '..': {raw}"),
            ));
        }
        segments.push(segment);
    }

    if segments.is_empty() {
        return Ok(FsPath::from("/"));
    }

    Ok(FsPath::from(format!("/{}", segments.join("/"))))
}

fn ancestor_paths(path: &FsPath) -> Vec<FsPath> {
    let mut ancestors = vec![FsPath::from("/")];
    if path.as_str() == "/" {
        return ancestors;
    }

    let mut current = String::new();
    for segment in path.as_str().trim_start_matches('/').split('/') {
        current.push('/');
        current.push_str(segment);
        ancestors.push(FsPath::from(current.clone()));
    }
    ancestors
}

fn parent_path(path: &FsPath) -> Option<FsPath> {
    if path.as_str() == "/" {
        return None;
    }

    let trimmed = path.as_str().trim_start_matches('/');
    match trimmed.rsplit_once('/') {
        Some((parent, _)) if !parent.is_empty() => Some(FsPath::from(format!("/{parent}"))),
        _ => Some(FsPath::from("/")),
    }
}

fn path_prefix(path: &FsPath) -> String {
    if path.as_str() == "/" {
        "/".to_owned()
    } else {
        format!("{}/", path.as_str())
    }
}

fn direct_child_name(parent: &FsPath, candidate: &FsPath) -> Option<String> {
    if parent == candidate {
        return None;
    }

    let remainder = if parent.as_str() == "/" {
        candidate.as_str().strip_prefix('/')?
    } else {
        candidate.as_str().strip_prefix(&path_prefix(parent))?
    };
    if remainder.is_empty() || remainder.contains('/') {
        return None;
    }
    Some(remainder.to_owned())
}

#[cfg(test)]
mod tests {
    use super::{MemoryFileSystem, ReadableFileSystem, WritableFileSystem};
    use xlai_core::{DirectoryFileSystem, FsEntryKind, FsPath};

    #[cfg(not(target_arch = "wasm32"))]
    use super::LocalFileSystem;

    #[tokio::test]
    async fn memory_file_system_round_trips_data() {
        let fs = MemoryFileSystem::new();
        let path = FsPath::from("/docs/readme.txt");

        let create_result = fs.create_dir_all(&FsPath::from("/docs")).await;
        assert!(
            create_result.is_ok(),
            "expected directory creation to succeed"
        );

        let write_result = fs.write(&path, b"hello".to_vec()).await;
        assert!(write_result.is_ok(), "expected file write to succeed");

        let bytes = fs.read(&path).await;
        assert_eq!(bytes.ok(), Some(b"hello".to_vec()));
    }

    #[tokio::test]
    async fn memory_file_system_lists_direct_children() {
        let fs = MemoryFileSystem::new();

        assert!(
            fs.create_dir_all(&FsPath::from("/docs/guides"))
                .await
                .is_ok()
        );
        assert!(
            fs.write(&FsPath::from("/docs/readme.md"), b"readme".to_vec())
                .await
                .is_ok()
        );

        let entries = fs.list(&FsPath::from("/docs")).await;
        assert!(entries.is_ok(), "expected list to succeed");
        let entries = entries.unwrap_or_default();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].path, FsPath::from("/docs/guides"));
        assert_eq!(entries[0].kind, FsEntryKind::Directory);
        assert_eq!(entries[1].path, FsPath::from("/docs/readme.md"));
        assert_eq!(entries[1].kind, FsEntryKind::File);
    }

    #[tokio::test]
    async fn memory_file_system_normalizes_virtual_paths() {
        let fs = MemoryFileSystem::new();

        assert!(
            fs.create_dir_all(&FsPath::from("//docs/./guides//"))
                .await
                .is_ok()
        );
        assert!(
            fs.write(&FsPath::from("//docs//guide.txt"), b"normalized".to_vec())
                .await
                .is_ok()
        );

        let bytes = fs.read(&FsPath::from("/docs/./guide.txt")).await;
        assert_eq!(bytes.ok(), Some(b"normalized".to_vec()));

        let exists = fs.exists(&FsPath::from("/docs/guides")).await;
        assert_eq!(exists.ok(), Some(true));
    }

    #[tokio::test]
    async fn memory_file_system_rejects_invalid_paths() {
        let fs = MemoryFileSystem::new();

        let relative = fs.read(&FsPath::from("docs/readme.md")).await;
        assert!(relative.is_err(), "expected relative paths to be rejected");

        let escaped = fs.create_dir_all(&FsPath::from("/docs/../secret")).await;
        assert!(
            escaped.is_err(),
            "expected parent traversal segments to be rejected"
        );
    }

    #[tokio::test]
    async fn memory_file_system_deletes_directory_trees() {
        let fs = MemoryFileSystem::new();

        assert!(
            fs.create_dir_all(&FsPath::from("/docs/guides"))
                .await
                .is_ok()
        );
        assert!(
            fs.write(&FsPath::from("/docs/guides/intro.md"), b"intro".to_vec())
                .await
                .is_ok()
        );
        assert!(
            fs.write(&FsPath::from("/docs/readme.md"), b"readme".to_vec())
                .await
                .is_ok()
        );

        assert!(fs.delete(&FsPath::from("/docs")).await.is_ok());

        let docs_exists = fs.exists(&FsPath::from("/docs")).await;
        assert_eq!(docs_exists.ok(), Some(false));

        let nested_exists = fs.exists(&FsPath::from("/docs/guides/intro.md")).await;
        assert_eq!(nested_exists.ok(), Some(false));

        let root_entries = fs.list(&FsPath::from("/")).await;
        assert!(root_entries.is_ok(), "expected root list to succeed");
        let root_entries = root_entries.unwrap_or_default();
        assert!(root_entries.is_empty(), "expected /docs tree to be removed");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn local_file_system_scopes_operations_to_root() {
        let temp_dir = std::env::temp_dir().join(format!(
            "xlai-local-fs-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let fs = LocalFileSystem::new(&temp_dir);
        let path = FsPath::from("/nested/file.txt");

        assert!(fs.create_dir_all(&FsPath::from("/nested")).await.is_ok());
        assert!(fs.write(&path, b"native".to_vec()).await.is_ok());
        let bytes = fs.read(&path).await;
        assert_eq!(bytes.ok(), Some(b"native".to_vec()));

        let escaped = fs.read(&FsPath::from("/../secret.txt")).await;
        assert!(escaped.is_err(), "expected parent traversal to be rejected");

        let cleanup_result = std::fs::remove_dir_all(temp_dir);
        assert!(
            cleanup_result.is_ok(),
            "expected temp directory cleanup to succeed"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn local_file_system_lists_and_deletes_paths() {
        let temp_dir = std::env::temp_dir().join(format!(
            "xlai-local-fs-list-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));
        let fs = LocalFileSystem::new(&temp_dir);

        assert!(
            fs.create_dir_all(&FsPath::from("/nested/child"))
                .await
                .is_ok()
        );
        assert!(
            fs.write(&FsPath::from("/nested/readme.txt"), b"native".to_vec())
                .await
                .is_ok()
        );

        let entries = fs.list(&FsPath::from("/nested")).await;
        assert!(entries.is_ok(), "expected list to succeed");
        let entries = entries.unwrap_or_default();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].path, FsPath::from("/nested/child"));
        assert_eq!(entries[0].kind, FsEntryKind::Directory);
        assert_eq!(entries[1].path, FsPath::from("/nested/readme.txt"));
        assert_eq!(entries[1].kind, FsEntryKind::File);

        assert!(fs.delete(&FsPath::from("/nested/readme.txt")).await.is_ok());
        let exists = fs.exists(&FsPath::from("/nested/readme.txt")).await;
        assert_eq!(exists.ok(), Some(false));

        let cleanup_result = std::fs::remove_dir_all(temp_dir);
        assert!(
            cleanup_result.is_ok(),
            "expected temp directory cleanup to succeed"
        );
    }
}
