use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use gray_matter::Matter;
use gray_matter::engine::YAML;
use serde::Deserialize;
use xlai_core::{
    BoxFuture, DirectoryFileSystem, ErrorKind, FsEntryKind, FsPath, ReadableFileSystem, Skill,
    SkillFileSystem, SkillStore, XlaiError,
};

const SKILL_FILE_NAME: &str = "SKILL.md";

#[derive(Clone)]
pub struct MarkdownSkillStore {
    file_system: Arc<dyn SkillFileSystem>,
    roots: Vec<FsPath>,
}

impl MarkdownSkillStore {
    #[must_use]
    pub fn new(file_system: Arc<dyn SkillFileSystem>) -> Self {
        Self::with_roots(file_system, vec![FsPath::from("/")])
    }

    #[must_use]
    pub fn with_roots(file_system: Arc<dyn SkillFileSystem>, roots: Vec<FsPath>) -> Self {
        Self { file_system, roots }
    }
}

impl SkillStore for MarkdownSkillStore {
    fn resolve_skills<'a>(
        &'a self,
        ids: &'a [String],
    ) -> BoxFuture<'a, Result<Vec<Skill>, XlaiError>> {
        Box::pin(async move {
            let mut resolved = discover_skills(self.file_system.as_ref(), &self.roots).await?;
            resolved.retain(|skill| ids.contains(&skill.id));
            Ok(resolved)
        })
    }
}

impl ReadableFileSystem for MarkdownSkillStore {
    fn read<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<Vec<u8>, XlaiError>> {
        self.file_system.read(path)
    }

    fn exists<'a>(&'a self, path: &'a FsPath) -> BoxFuture<'a, Result<bool, XlaiError>> {
        self.file_system.exists(path)
    }
}

impl DirectoryFileSystem for MarkdownSkillStore {
    fn list<'a>(
        &'a self,
        path: &'a FsPath,
    ) -> BoxFuture<'a, Result<Vec<xlai_core::FsEntry>, XlaiError>> {
        self.file_system.list(path)
    }
}

async fn discover_skills(
    file_system: &dyn SkillFileSystem,
    roots: &[FsPath],
) -> Result<Vec<Skill>, XlaiError> {
    let mut skill_files = BTreeSet::new();
    for root in roots {
        collect_skill_files(file_system, root, &mut skill_files).await?;
    }

    let mut resolved = Vec::with_capacity(skill_files.len());
    let mut seen_ids = BTreeMap::new();
    for skill_path in skill_files {
        let skill = load_skill(file_system, &skill_path).await?;
        if let Some(existing_path) = seen_ids.insert(skill.id.clone(), skill_path.clone()) {
            return Err(XlaiError::new(
                ErrorKind::Skill,
                format!(
                    "duplicate skill id `{}` found in {} and {}",
                    skill.id,
                    existing_path.as_str(),
                    skill_path.as_str()
                ),
            ));
        }
        resolved.push(skill);
    }

    Ok(resolved)
}

fn collect_skill_files<'a>(
    file_system: &'a dyn SkillFileSystem,
    path: &'a FsPath,
    skill_files: &'a mut BTreeSet<FsPath>,
) -> BoxFuture<'a, Result<(), XlaiError>> {
    Box::pin(async move {
        for entry in file_system.list(path).await? {
            match entry.kind {
                FsEntryKind::Directory => {
                    collect_skill_files(file_system, &entry.path, skill_files).await?;
                }
                FsEntryKind::File if file_name(&entry.path) == Some(SKILL_FILE_NAME) => {
                    skill_files.insert(entry.path);
                }
                FsEntryKind::File => {}
            }
        }

        Ok(())
    })
}

async fn load_skill(file_system: &dyn SkillFileSystem, path: &FsPath) -> Result<Skill, XlaiError> {
    let bytes = file_system.read(path).await?;
    let markdown = String::from_utf8(bytes).map_err(|error| {
        XlaiError::new(
            ErrorKind::Skill,
            format!("skill file {} is not valid UTF-8: {error}", path.as_str()),
        )
    })?;

    let matter = Matter::<YAML>::new();
    let parsed = matter
        .parse::<SkillFrontMatter>(&markdown)
        .map_err(|error| {
            XlaiError::new(
                ErrorKind::Skill,
                format!("failed to parse skill markdown {}: {error}", path.as_str()),
            )
        })?;
    let front_matter = parsed.data.ok_or_else(|| {
        XlaiError::new(
            ErrorKind::Skill,
            format!(
                "skill markdown {} is missing front matter metadata",
                path.as_str()
            ),
        )
    })?;

    let skill_dir = parent_directory(path);
    let mut metadata = BTreeMap::new();
    metadata.insert("skill_path".to_owned(), path.as_str().to_owned());
    metadata.insert("skill_dir".to_owned(), skill_dir.as_str().to_owned());

    Ok(Skill {
        id: front_matter.name.clone(),
        name: front_matter.name,
        description: front_matter.description,
        prompt_fragment: parsed.content.trim().to_owned(),
        tags: front_matter.tags,
        metadata,
    })
}

fn file_name(path: &FsPath) -> Option<&str> {
    path.as_str().rsplit('/').next()
}

fn parent_directory(path: &FsPath) -> FsPath {
    let value = path.as_str();
    match value.rsplit_once('/') {
        Some(("", _)) | None => FsPath::from("/"),
        Some((parent, _)) => FsPath::from(parent),
    }
}

#[derive(Clone, Debug, Deserialize)]
struct SkillFrontMatter {
    name: String,
    description: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xlai_core::{
        FsPath, ReadableFileSystem, SkillFileSystem, SkillStore, WritableFileSystem, XlaiError,
    };

    use super::MarkdownSkillStore;
    use crate::MemoryFileSystem;

    #[tokio::test]
    async fn markdown_skill_store_resolves_skill_metadata_from_skill_markdown()
    -> Result<(), XlaiError> {
        let file_system = Arc::new(MemoryFileSystem::new());
        file_system
            .create_dir_all(&FsPath::from("/skills/review"))
            .await?;
        file_system
            .write(
                &FsPath::from("/skills/review/SKILL.md"),
                br#"---
name: review.code
description: Reviews code with a bug-finding mindset.
tags:
  - review
  - quality
---
Prioritize bugs, regressions, and missing tests.
"#
                .to_vec(),
            )
            .await?;
        file_system
            .write(
                &FsPath::from("/skills/review/README.md"),
                b"extra file".to_vec(),
            )
            .await?;

        let file_system_trait: Arc<dyn SkillFileSystem> = file_system;
        let skill_store = MarkdownSkillStore::new(file_system_trait);
        let skills = skill_store
            .resolve_skills(&["review.code".to_owned()])
            .await?;

        if skills.len() != 1 {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("expected exactly one resolved skill, got {}", skills.len()),
            ));
        }
        if skills[0].id != "review.code" {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected skill id: {}", skills[0].id),
            ));
        }
        if skills[0].name != "review.code" {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected skill name: {}", skills[0].name),
            ));
        }
        if skills[0].description != "Reviews code with a bug-finding mindset." {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected skill description: {}", skills[0].description),
            ));
        }
        if skills[0].prompt_fragment != "Prioritize bugs, regressions, and missing tests." {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!(
                    "unexpected skill prompt fragment: {}",
                    skills[0].prompt_fragment
                ),
            ));
        }
        if skills[0].metadata.get("skill_dir").map(String::as_str) != Some("/skills/review") {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                "expected resolved skill metadata to include the skill directory",
            ));
        }
        let extra = skill_store
            .read(&FsPath::from("/skills/review/README.md"))
            .await?;
        if extra != b"extra file".to_vec() {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::FileSystem,
                "expected markdown skill store to delegate extra file reads",
            ));
        }

        Ok(())
    }
}
