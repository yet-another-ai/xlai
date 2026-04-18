use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::error::XlaiError;
use crate::runtime::{BoxFuture, RuntimeBound};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FsPath(String);

impl FsPath {
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Display for FsPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<String> for FsPath {
    fn from(path: String) -> Self {
        Self(path)
    }
}

impl From<&str> for FsPath {
    fn from(path: &str) -> Self {
        Self(path.to_owned())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FsEntryKind {
    File,
    Directory,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FsEntry {
    pub path: FsPath,
    pub kind: FsEntryKind,
}

pub trait ReadableFileSystem: RuntimeBound {
    fn read<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<u8>, XlaiError>>;

    fn exists<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<bool, XlaiError>>;
}

pub trait WritableFileSystem: RuntimeBound {
    fn write<'a>(&'a self, path: &'a FsPath, data: Vec<u8>)
    -> BoxFuture<'a, Result<(), XlaiError>>;

    fn delete<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>>;

    fn create_dir_all<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<(), XlaiError>>;
}

pub trait DirectoryFileSystem: RuntimeBound {
    fn list<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<FsEntry>, XlaiError>>;
}

pub trait SkillFileSystem: ReadableFileSystem + DirectoryFileSystem {}

impl<T> SkillFileSystem for T where T: ReadableFileSystem + DirectoryFileSystem + ?Sized {}

pub trait FileSystem: ReadableFileSystem + WritableFileSystem + DirectoryFileSystem {}

impl<T> FileSystem for T where
    T: ReadableFileSystem + WritableFileSystem + DirectoryFileSystem + ?Sized
{
}
