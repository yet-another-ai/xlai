use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use gray_matter::Matter;
use gray_matter::engine::YAML;
use serde::Deserialize;
use serde_json::Value;
use xlai_core::{
    BoxFuture, DirectoryFileSystem, ErrorKind, FsEntryKind, FsPath, ReadableFileSystem, Skill,
    SkillFileSystem, SkillLoadPolicy, SkillResource, SkillStore, XlaiError,
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

    fn list_skills<'a>(&'a self) -> BoxFuture<'a, Result<Vec<Skill>, XlaiError>> {
        Box::pin(async move { discover_skills(self.file_system.as_ref(), &self.roots).await })
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
    let resources = collect_skill_resources(file_system, &skill_dir, &front_matter).await?;
    let eager_paths = eager_resource_paths(&front_matter, &resources)?;
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "skill_path".to_owned(),
        Value::String(path.as_str().to_owned()),
    );
    metadata.insert(
        "skill_dir".to_owned(),
        Value::String(skill_dir.as_str().to_owned()),
    );

    Ok(Skill {
        id: front_matter.name.clone(),
        name: front_matter.name,
        description: front_matter.description,
        prompt_fragment: parsed.content.trim().to_owned(),
        resources,
        entrypoints: vec![SKILL_FILE_NAME.to_owned()],
        load_policy: Some(SkillLoadPolicy { eager_paths }),
        tags: front_matter.tags,
        metadata,
    })
}

async fn collect_skill_resources(
    file_system: &dyn SkillFileSystem,
    skill_dir: &FsPath,
    front_matter: &SkillFrontMatter,
) -> Result<Vec<SkillResource>, XlaiError> {
    let mut resources = BTreeMap::new();

    for declared in &front_matter.resources {
        let path = normalize_skill_resource_path(skill_dir, &declared.path)?;
        ensure_skill_resource_exists(file_system, &path).await?;
        resources.insert(
            declared.path.clone(),
            SkillResource {
                path: declared.path.clone(),
                kind: declared.kind.clone(),
                purpose: declared.purpose.clone(),
                required: declared.required,
            },
        );
    }

    for path in &front_matter.load {
        let path = normalize_skill_resource_path(skill_dir, path)?;
        ensure_skill_resource_exists(file_system, &path).await?;
        let relative_path = path_relative_to_dir(skill_dir, &path)?;
        resources
            .entry(relative_path.clone())
            .or_insert_with(|| SkillResource {
                path: relative_path,
                kind: None,
                purpose: None,
                required: false,
            });
    }

    Ok(resources.into_values().collect())
}

fn eager_resource_paths(
    front_matter: &SkillFrontMatter,
    resources: &[SkillResource],
) -> Result<Vec<String>, XlaiError> {
    let declared_paths = resources
        .iter()
        .map(|resource| resource.path.as_str())
        .collect::<BTreeSet<_>>();
    let mut eager_paths = BTreeSet::new();

    for resource in resources {
        if resource.required {
            eager_paths.insert(resource.path.clone());
        }
    }

    for path in &front_matter.load {
        if !declared_paths.contains(path.as_str()) {
            return Err(XlaiError::new(
                ErrorKind::Skill,
                format!("skill load path `{path}` was not resolved into a declared resource"),
            ));
        }
        eager_paths.insert(path.clone());
    }

    Ok(eager_paths.into_iter().collect())
}

pub(crate) fn skill_directory(skill: &Skill) -> Result<FsPath, XlaiError> {
    skill
        .metadata
        .get("skill_dir")
        .and_then(Value::as_str)
        .map(FsPath::from)
        .ok_or_else(|| XlaiError::new(ErrorKind::Skill, "resolved skill is missing `skill_dir`"))
}

pub(crate) fn resolve_skill_resource_path(
    skill: &Skill,
    relative_path: &str,
) -> Result<FsPath, XlaiError> {
    if !skill
        .resources
        .iter()
        .any(|resource| resource.path == relative_path)
    {
        return Err(XlaiError::new(
            ErrorKind::Skill,
            format!(
                "resource `{relative_path}` is not declared by skill `{}`",
                skill.id
            ),
        ));
    }

    normalize_skill_resource_path(&skill_directory(skill)?, relative_path)
}

fn normalize_skill_resource_path(
    skill_dir: &FsPath,
    relative_path: &str,
) -> Result<FsPath, XlaiError> {
    if relative_path.is_empty() || relative_path.starts_with('/') {
        return Err(XlaiError::new(
            ErrorKind::Skill,
            format!("skill resource path `{relative_path}` must be a non-empty relative path"),
        ));
    }

    let mut normalized = Vec::new();
    for segment in relative_path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(XlaiError::new(
                ErrorKind::Skill,
                format!("skill resource path `{relative_path}` contains invalid segments"),
            ));
        }
        normalized.push(segment);
    }

    let joined = normalized.join("/");
    let base = skill_dir.as_str().trim_end_matches('/');
    if base.is_empty() || base == "/" {
        Ok(FsPath::from(format!("/{joined}")))
    } else {
        Ok(FsPath::from(format!("{base}/{joined}")))
    }
}

fn path_relative_to_dir(skill_dir: &FsPath, absolute_path: &FsPath) -> Result<String, XlaiError> {
    let skill_dir = skill_dir.as_str().trim_end_matches('/');
    let absolute = absolute_path.as_str();

    if skill_dir.is_empty() || skill_dir == "/" {
        return Ok(absolute.trim_start_matches('/').to_owned());
    }

    absolute
        .strip_prefix(&format!("{skill_dir}/"))
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            XlaiError::new(
                ErrorKind::Skill,
                format!(
                    "resource path {} escapes skill directory {}",
                    absolute_path.as_str(),
                    skill_dir
                ),
            )
        })
}

async fn ensure_skill_resource_exists(
    file_system: &dyn SkillFileSystem,
    path: &FsPath,
) -> Result<(), XlaiError> {
    if file_system.exists(path).await? {
        return Ok(());
    }

    Err(XlaiError::new(
        ErrorKind::Skill,
        format!("declared skill resource {} does not exist", path.as_str()),
    ))
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
    #[serde(default)]
    resources: Vec<SkillResourceFrontMatter>,
    #[serde(default)]
    load: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct SkillResourceFrontMatter {
    path: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    purpose: Option<String>,
    #[serde(default)]
    required: bool,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::Value;
    use xlai_core::{
        FsPath, ReadableFileSystem, SkillFileSystem, SkillStore, WritableFileSystem, XlaiError,
    };

    use super::{MarkdownSkillStore, resolve_skill_resource_path};
    use crate::MemoryFileSystem;

    #[allow(clippy::panic_in_result_fn)]
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
        if skills[0].metadata.get("skill_dir").and_then(Value::as_str) != Some("/skills/review") {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                "expected resolved skill metadata to include the skill directory",
            ));
        }
        if skills[0].entrypoints != vec!["SKILL.md".to_owned()] {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected entrypoints: {:?}", skills[0].entrypoints),
            ));
        }
        if !skills[0].resources.is_empty() {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("expected no resources, got {}", skills[0].resources.len()),
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

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn markdown_skill_store_resolves_declared_resources_and_load_policy()
    -> Result<(), XlaiError> {
        let file_system = Arc::new(MemoryFileSystem::new());
        file_system
            .create_dir_all(&FsPath::from("/skills/review/references"))
            .await?;
        file_system
            .create_dir_all(&FsPath::from("/skills/review/templates"))
            .await?;
        file_system
            .write(
                &FsPath::from("/skills/review/SKILL.md"),
                br#"---
name: review.code
description: Reviews code with a bug-finding mindset.
tags:
  - review
resources:
  - path: references/checklist.md
    kind: markdown
    purpose: review checklist
    required: true
  - path: templates/final.md
    purpose: response template
load:
  - templates/final.md
---
Base prompt.
"#
                .to_vec(),
            )
            .await?;
        file_system
            .write(
                &FsPath::from("/skills/review/references/checklist.md"),
                b"Checklist".to_vec(),
            )
            .await?;
        file_system
            .write(
                &FsPath::from("/skills/review/templates/final.md"),
                b"Template".to_vec(),
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
                format!("expected 1 skill, got {}", skills.len()),
            ));
        }
        if skills[0].resources.len() != 2 {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("expected 2 resources, got {}", skills[0].resources.len()),
            ));
        }
        if !skills[0]
            .resources
            .iter()
            .any(|resource| resource.path == "references/checklist.md" && resource.required)
        {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                "expected required checklist resource",
            ));
        }
        if !skills[0]
            .resources
            .iter()
            .any(|resource| resource.path == "templates/final.md" && !resource.required)
        {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                "expected optional template resource",
            ));
        }
        let eager_paths = skills[0]
            .load_policy
            .as_ref()
            .map(|policy| &policy.eager_paths);
        let expected_paths = vec![
            "references/checklist.md".to_owned(),
            "templates/final.md".to_owned(),
        ];
        if eager_paths != Some(&expected_paths) {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected eager paths: {eager_paths:?}"),
            ));
        }

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn markdown_skill_store_rejects_path_traversal_resources() -> Result<(), XlaiError> {
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
resources:
  - path: ../secret.md
---
Base prompt.
"#
                .to_vec(),
            )
            .await?;

        let file_system_trait: Arc<dyn SkillFileSystem> = file_system;
        let skill_store = MarkdownSkillStore::new(file_system_trait);
        let error = match skill_store
            .resolve_skills(&["review.code".to_owned()])
            .await
        {
            Ok(_) => {
                return Err(XlaiError::new(
                    xlai_core::ErrorKind::Skill,
                    "path traversal should fail",
                ));
            }
            Err(error) => error,
        };

        if error.kind != xlai_core::ErrorKind::Skill {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected error kind: {:?}", error.kind),
            ));
        }
        if !error.to_string().contains("invalid segments") {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected error: {error}"),
            ));
        }

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn markdown_skill_store_rejects_missing_declared_resources() -> Result<(), XlaiError> {
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
resources:
  - path: references/missing.md
---
Base prompt.
"#
                .to_vec(),
            )
            .await?;

        let file_system_trait: Arc<dyn SkillFileSystem> = file_system;
        let skill_store = MarkdownSkillStore::new(file_system_trait);
        let error = match skill_store
            .resolve_skills(&["review.code".to_owned()])
            .await
        {
            Ok(_) => {
                return Err(XlaiError::new(
                    xlai_core::ErrorKind::Skill,
                    "missing resource should fail",
                ));
            }
            Err(error) => error,
        };

        if error.kind != xlai_core::ErrorKind::Skill {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected error kind: {:?}", error.kind),
            ));
        }
        if !error.to_string().contains("does not exist") {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected error: {error}"),
            ));
        }

        Ok(())
    }

    #[allow(clippy::panic_in_result_fn)]
    #[tokio::test]
    async fn resolve_skill_resource_path_only_allows_declared_files() -> Result<(), XlaiError> {
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
resources:
  - path: README.md
---
Base prompt.
"#
                .to_vec(),
            )
            .await?;
        file_system
            .write(&FsPath::from("/skills/review/README.md"), b"extra".to_vec())
            .await?;

        let file_system_trait: Arc<dyn SkillFileSystem> = file_system;
        let skill_store = MarkdownSkillStore::new(file_system_trait);
        let skill = match skill_store
            .resolve_skills(&["review.code".to_owned()])
            .await?
            .into_iter()
            .next()
        {
            Some(skill) => skill,
            None => {
                return Err(XlaiError::new(
                    xlai_core::ErrorKind::Skill,
                    "skill should resolve",
                ));
            }
        };

        let resolved = resolve_skill_resource_path(&skill, "README.md")?;
        if resolved.as_str() != "/skills/review/README.md" {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected resource path: {}", resolved.as_str()),
            ));
        }
        let error = match resolve_skill_resource_path(&skill, "NOT_DECLARED.md") {
            Ok(path) => {
                return Err(XlaiError::new(
                    xlai_core::ErrorKind::Skill,
                    format!("undeclared path should fail, got {}", path.as_str()),
                ));
            }
            Err(error) => error,
        };
        if !error.to_string().contains("not declared") {
            return Err(XlaiError::new(
                xlai_core::ErrorKind::Skill,
                format!("unexpected error: {error}"),
            ));
        }

        Ok(())
    }
}
