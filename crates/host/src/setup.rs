//! Pinned downloads into private staging directories; activation exposes complete models.
use reqwest::{
    Client, StatusCode,
    header::{CONTENT_RANGE, RANGE},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io,
    path::{Component, Path, PathBuf},
    sync::Mutex,
    time::Duration,
};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct FilePin {
    pub size: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ModelArtifact {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub files: BTreeMap<String, FilePin>,
    #[serde(default)]
    pub prepared: BTreeMap<String, FilePin>,
}
impl ModelArtifact {
    pub fn download_bytes(&self) -> u64 {
        self.files.values().map(|f| f.size).sum()
    }
    pub fn stored_bytes(&self) -> u64 {
        self.download_bytes() + self.prepared.values().map(|f| f.size).sum::<u64>()
    }
}
pub(crate) fn artifacts() -> Vec<ModelArtifact> {
    serde_json::from_str(include_str!("model-artifacts.json")).expect("bundled model pins")
}
pub(crate) fn separation_artifact() -> ModelArtifact {
    serde_json::from_str(include_str!("separation-artifact.json")).expect("bundled separation pins")
}

#[derive(Clone, Default, Serialize)]
pub(crate) struct Progress {
    pub target: Option<String>,
    pub started_at: u64,
    pub activated: bool,
    pub models: Vec<String>,
    pub status: &'static str,
    pub phase: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub download_required: bool,
    pub error: Option<String>,
}
#[derive(Default)]
pub(crate) struct SetupService {
    progress: Mutex<Progress>,
}
impl SetupService {
    pub fn progress(&self) -> Progress {
        self.progress.lock().expect("setup progress").clone()
    }
    pub fn begin(&self, models: &[ModelArtifact], download: bool) {
        *self.progress.lock().expect("setup progress") = Progress {
            target: download.then(|| {
                if models.len() == 1 && models[0].id == "mossformer2-16k" {
                    "separation"
                } else {
                    "models"
                }
                .into()
            }),
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            models: models.iter().map(|m| m.id.clone()).collect(),
            status: if download { "downloading" } else { "loading" },
            phase: if download {
                "Checking installed models"
            } else {
                "Loading models…"
            }
            .into(),
            total_bytes: models.iter().map(ModelArtifact::download_bytes).sum(),
            download_required: download,
            ..Progress::default()
        };
    }
    pub fn loading(&self) {
        let mut progress = self.progress.lock().expect("setup progress");
        progress.status = "loading";
        progress.phase = "Loading models…".into();
    }
    pub fn check_downloads(&self, root: &Path, models: &[ModelArtifact]) {
        // Presence is only a setup-action hint. The initializer still verifies every model.
        // Errors and invalid contents do not establish that another download is necessary.
        let missing = models.iter().any(|model| {
            model
                .files
                .keys()
                .any(|name| missing_file(&root.join(&model.id).join(name)))
        });
        self.progress
            .lock()
            .expect("setup progress")
            .download_required = missing;
    }
    pub fn phase(&self, phase: impl Into<String>) {
        self.progress.lock().expect("setup progress").phase = phase.into();
    }
    pub fn activated(&self) {
        self.progress.lock().expect("setup progress").activated = true;
    }
    pub fn bytes(&self, bytes: u64) {
        self.progress
            .lock()
            .expect("setup progress")
            .downloaded_bytes = bytes;
    }
    pub fn missing(&self) {
        self.progress.lock().expect("setup progress").status = "missing";
    }
    pub fn finish(&self, result: Result<(), String>, missing: bool) {
        let mut progress = self.progress.lock().expect("setup progress");
        progress.status = if result.is_ok() {
            "ready"
        } else if missing {
            "missing"
        } else {
            "failed"
        };
        progress.error = result.err();
    }
}

pub(crate) fn initial_models_missing(
    root: &Path,
    models: &[ModelArtifact],
    reports: &[speakerdesk_core::ipc::SetupModelReport],
) -> bool {
    !std::fs::symlink_metadata(root).is_ok_and(|meta| meta.is_symlink() || !meta.is_dir())
        && reports.len() >= models.len()
        && reports.iter().take(models.len()).any(|report| !report.ok)
        && models.iter().zip(reports).all(|(model, report)| {
            report.ok
                || (!std::fs::symlink_metadata(root.join(&model.id))
                    .is_ok_and(|meta| meta.is_symlink() || !meta.is_dir())
                    && model
                        .files
                        .keys()
                        .all(|name| missing_file(&root.join(&model.id).join(name))))
        })
}

fn missing_file(path: &Path) -> bool {
    if !matches!(std::fs::symlink_metadata(path), Err(error) if error.kind() == io::ErrorKind::NotFound)
    {
        return false;
    }
    // Windows also reports NotFound when a parent is a regular file.
    path.ancestors()
        .skip(1)
        .filter(|p| !p.as_os_str().is_empty())
        .all(|parent| match std::fs::metadata(parent) {
            Ok(metadata) => metadata.is_dir(),
            Err(error) => error.kind() == io::ErrorKind::NotFound,
        })
}

pub fn directory_free_bytes(path: &Path) -> io::Result<u64> {
    fs4::available_space(path)
}
fn error(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}
fn checked(cancel: &CancellationToken) -> io::Result<()> {
    if cancel.is_cancelled() {
        Err(error(
            "Model setup cancelled. Downloaded files can be resumed.",
        ))
    } else {
        Ok(())
    }
}

// The directory is private and has one installer owner. Reject existing links at every component.
async fn directory(path: &Path) -> io::Result<()> {
    let mut current = PathBuf::new();
    for part in path.components() {
        if matches!(part, Component::ParentDir) {
            return Err(error("Invalid model directory"));
        }
        current.push(part);
        if let Component::Prefix(prefix) = part {
            if matches!(
                prefix.kind(),
                std::path::Prefix::Disk(_) | std::path::Prefix::VerbatimDisk(_)
            ) && path.components().nth(1) != Some(Component::RootDir)
            {
                return Err(error("Invalid model directory"));
            }
            // A Windows drive prefix is not an addressable directory until its root is appended.
            continue;
        }
        match fs::symlink_metadata(&current).await {
            Ok(meta) if meta.is_dir() && !meta.is_symlink() => {}
            Ok(_) => return Err(error("Model directory contains a link or non-directory")),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&current).await?;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
fn relative(name: &str) -> io::Result<&Path> {
    let path = Path::new(name);
    if name.is_empty()
        || path
            .components()
            .any(|p| !matches!(p, Component::Normal(_)))
    {
        return Err(error("Invalid model artifact path"));
    }
    Ok(path)
}
async fn regular(path: &Path) -> io::Result<Option<u64>> {
    match fs::symlink_metadata(path).await {
        Ok(meta) if meta.is_file() && !meta.is_symlink() => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if meta.nlink() != 1 {
                    return Err(error("Model artifact has multiple hard links"));
                }
            }
            Ok(Some(meta.len()))
        }
        Ok(_) => Err(error("Model artifact is not a regular file")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}
fn partial_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}
fn open_options() -> fs::OpenOptions {
    let mut options = fs::OpenOptions::new();
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    options
}
async fn matches(path: &Path, pin: &FilePin, cancel: &CancellationToken) -> io::Result<bool> {
    checked(cancel)?;
    if regular(path).await? != Some(pin.size) {
        return Ok(false);
    }
    let mut file = open_options().read(true).open(path).await?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        checked(cancel)?;
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()) == pin.sha256)
}
async fn installed(
    root: &Path,
    model: &ModelArtifact,
    cancel: &CancellationToken,
) -> io::Result<bool> {
    match fs::symlink_metadata(root).await {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
        Ok(meta) if !meta.is_dir() || meta.is_symlink() => {
            return Err(error("Model directory is not a regular directory"));
        }
        _ => {}
    }
    let inventory_root = root.to_owned();
    let inventory_model = model.clone();
    if !tokio::task::spawn_blocking(move || inventory(&inventory_root, &inventory_model, false))
        .await
        .map_err(|e| error(e.to_string()))??
    {
        return Ok(false);
    }
    for (name, pin) in model.files.iter().chain(&model.prepared) {
        let path = root.join(relative(name)?);
        let mut parent = root.to_owned();
        for component in relative(name)?
            .parent()
            .unwrap_or(Path::new(""))
            .components()
        {
            parent.push(component);
            match fs::symlink_metadata(&parent).await {
                Ok(meta) if meta.is_dir() && !meta.is_symlink() => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
                Err(e) => return Err(e),
                _ => return Err(error("Model artifact parent is not a regular directory")),
            }
        }
        if !matches(&path, pin, cancel).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) struct Installer {
    pub root: PathBuf,
    client: Client,
    // The lock lives through conversion, activation and initialization, including cancellation.
    _owner: std::fs::File,
}
impl Installer {
    pub async fn open(root: &Path) -> io::Result<Self> {
        if fs::symlink_metadata(root)
            .await
            .is_ok_and(|meta| meta.is_symlink())
        {
            return Err(error("Model root is a symlink"));
        }
        fs::create_dir_all(root).await?;
        let root = fs::canonicalize(root).await?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).await?;
        }
        let lock = root.join(".setup.lock");
        regular(&lock).await?;
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW);
        }
        let owner = options.open(lock)?;
        fs4::FileExt::try_lock(&owner)
            .map_err(|_| error("Another process is setting up these models"))?;
        Ok(Self {
            root,
            client: Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(30))
                .https_only(true)
                .build()
                .map_err(|e| error(e.to_string()))?,
            _owner: owner,
        })
    }
    pub async fn installed(
        &self,
        model: &ModelArtifact,
        cancel: &CancellationToken,
    ) -> io::Result<bool> {
        installed(&self.root.join(relative(&model.id)?), model, cancel).await
    }
    pub async fn needs_preparation(
        &self,
        model: &ModelArtifact,
        staging: &Path,
        cancel: &CancellationToken,
    ) -> io::Result<bool> {
        for (name, pin) in &model.prepared {
            if !matches(&staging.join(relative(name)?), pin, cancel).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub async fn stage(
        &self,
        model: &ModelArtifact,
        completed: u64,
        progress: &SetupService,
        cancel: &CancellationToken,
    ) -> io::Result<PathBuf> {
        checked(cancel)?;
        let staging = self.root.join(".staging").join(relative(&model.id)?);
        directory(&staging).await?;
        let inspect = staging.clone();
        let pinned = model.clone();
        if !tokio::task::spawn_blocking(move || inventory(&inspect, &pinned, true))
            .await
            .map_err(|e| error(e.to_string()))??
        {
            checked(cancel)?;
            // A crash after RENAME_SWAP may leave the previous model here; it is private staging.
            fs::remove_dir_all(&staging).await?;
            directory(&staging).await?;
        }
        for name in model.prepared.keys() {
            let path = staging.join(relative(name)?);
            directory(path.parent().expect("prepared parent")).await?;
            regular(&path).await?;
        }
        let mut missing = 0;
        for (name, pin) in model.files.iter().chain(&model.prepared) {
            let path = staging.join(relative(name)?);
            directory(path.parent().expect("artifact parent")).await?;
            if matches(&path, pin, cancel).await? {
                continue;
            }
            let partial = partial_path(&path);
            let mut stored = regular(&partial).await?.unwrap_or(0);
            if stored >= pin.size && !matches(&partial, pin, cancel).await? {
                stored = 0;
            }
            missing += pin.size.saturating_sub(stored);
        }
        if directory_free_bytes(&staging)? < missing {
            return Err(error(format!(
                "Model setup needs {missing} more bytes of free disk space."
            )));
        }
        let mut done = completed;
        for (name, pin) in &model.files {
            checked(cancel)?;
            progress.phase(format!("Downloading {} · {name}", model.name));
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{name}?download=true",
                model.repo, model.revision
            );
            self.download(&url, &staging.join(name), pin, done, progress, cancel)
                .await?;
            done += pin.size;
            progress.bytes(done);
        }
        Ok(staging)
    }
    async fn download(
        &self,
        url: &str,
        path: &Path,
        pin: &FilePin,
        done: u64,
        progress: &SetupService,
        cancel: &CancellationToken,
    ) -> io::Result<()> {
        if matches(path, pin, cancel).await? {
            let partial = partial_path(path);
            if regular(&partial).await?.is_some() {
                fs::remove_file(partial).await?;
            }
            return Ok(());
        }
        let partial = partial_path(path);
        let mut offset = regular(&partial).await?.unwrap_or(0);
        if offset >= pin.size {
            if offset == pin.size && matches(&partial, pin, cancel).await? {
                checked(cancel)?;
                fs::rename(&partial, path).await?;
                return Ok(());
            }
            open_options()
                .write(true)
                .truncate(true)
                .open(&partial)
                .await?;
            offset = 0;
        }
        let mut request = self.client.get(url);
        if offset != 0 {
            request = request.header(RANGE, format!("bytes={offset}-"));
        }
        let mut response = tokio::select! { biased; _ = cancel.cancelled() => return checked(cancel), response = request.send() => response.map_err(|e| error(e.to_string()))? };
        match response.status() {
            StatusCode::OK => offset = 0,
            StatusCode::PARTIAL_CONTENT => {
                let expected = format!("bytes {offset}-{}/{}", pin.size - 1, pin.size);
                if response
                    .headers()
                    .get(CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok())
                    != Some(expected.as_str())
                {
                    return Err(error("Invalid model download Content-Range"));
                }
            }
            status => return Err(error(format!("Model download returned HTTP {status}"))),
        }
        if response
            .content_length()
            .is_some_and(|len| len != pin.size - offset)
        {
            return Err(error("Model download length differs from its pin"));
        }
        let mut output = open_options()
            .write(true)
            .create(true)
            .truncate(offset == 0)
            .append(offset != 0)
            .open(&partial)
            .await?;
        loop {
            let chunk = tokio::select! { biased; _ = cancel.cancelled() => return checked(cancel), chunk = response.chunk() => chunk.map_err(|e| error(e.to_string()))? };
            let Some(chunk) = chunk else {
                break;
            };
            if chunk.len() as u64 > pin.size - offset {
                return Err(error("Model download exceeds its pinned size"));
            }
            output.write_all(&chunk).await?;
            output.flush().await?;
            offset += chunk.len() as u64;
            progress.bytes(done + offset);
        }
        output.sync_all().await?;
        drop(output);
        progress.phase(format!(
            "Verifying {}",
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        if !matches(&partial, pin, cancel).await? {
            return Err(error(
                "Model checksum differs. Retry setup to download this file again.",
            ));
        }
        checked(cancel)?;
        fs::rename(&partial, path).await?;
        Ok(())
    }
    pub async fn activate(
        &self,
        model: &ModelArtifact,
        staging: &Path,
        cancel: &CancellationToken,
    ) -> io::Result<()> {
        if !installed(staging, model, cancel).await? {
            return Err(error("Prepared model failed verification"));
        }
        // Pin verification is installation-only. No recursive receipt checks run during inference.
        checked(cancel)?;
        let source = staging.to_owned();
        let target = self.root.join(relative(&model.id)?);
        let parent = self.root.clone();
        let model = model.clone();
        let cancellation = cancel.clone();
        tokio::task::spawn_blocking(move || {
            sync_artifact(&source, &model)?;
            checked(&cancellation)?;
            match std::fs::symlink_metadata(&target) {
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    speakerdesk_core::filesystem::rename(&source, &target)?;
                }
                Err(e) => return Err(e),
                Ok(meta) if meta.is_symlink() || !meta.is_dir() => {
                    return Err(error("Model destination is not a regular directory"));
                }
                Ok(_) => {
                    exchange_directories(&source, &target)?;
                    sync_directory(&parent)?;
                    std::fs::remove_dir_all(&source)?;
                    sync_directory(&parent)?;
                }
            }
            Ok(())
        })
        .await
        .map_err(|e| error(e.to_string()))?
    }
}
fn sync_directory(path: &Path) -> io::Result<()> {
    speakerdesk_core::filesystem::sync_directory(path)
}
fn sync_artifact(root: &Path, model: &ModelArtifact) -> io::Result<()> {
    for name in model.files.keys().chain(model.prepared.keys()) {
        let path = root.join(name);
        speakerdesk_core::filesystem::sync_file(&path)?;
        // Windows publishes the complete prepared tree with a write-through directory rename.
        #[cfg(not(windows))]
        {
            let mut parent = path.parent().expect("artifact parent");
            while parent.starts_with(root) {
                sync_directory(parent)?;
                if parent == root {
                    break;
                }
                parent = parent.parent().expect("model parent");
            }
        }
    }
    Ok(())
}
fn inventory(root: &Path, model: &ModelArtifact, partials: bool) -> io::Result<bool> {
    let mut files = std::collections::HashSet::new();
    let mut parents = std::collections::HashSet::new();
    for name in model.files.keys().chain(model.prepared.keys()) {
        let path = relative(name)?.to_owned();
        files.insert(path.clone());
        if partials {
            files.insert(partial_path(&path));
        }
        for parent in path.ancestors().skip(1) {
            parents.insert(parent.to_owned());
        }
    }
    let mut directories = vec![root.to_owned()];
    let mut exact = true;
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            let path = entry.path();
            let relative = path.strip_prefix(root).map_err(|e| error(e.to_string()))?;
            if kind.is_dir() {
                exact &= parents.contains(relative);
                directories.push(path);
            } else if kind.is_file() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    if entry.metadata()?.nlink() != 1 {
                        return Err(error("Model artifact has multiple hard links"));
                    }
                }
                exact &= files.contains(relative);
            } else {
                return Err(error("Unexpected link or special file in model directory"));
            }
        }
    }
    Ok(exact)
}
fn exchange_directories(source: &Path, target: &Path) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let source =
            CString::new(source.as_os_str().as_bytes()).map_err(|e| error(e.to_string()))?;
        let target =
            CString::new(target.as_os_str().as_bytes()).map_err(|e| error(e.to_string()))?;
        // RENAME_SWAP atomically retains the previous directory at the staging path.
        if unsafe { libc::renamex_np(source.as_ptr(), target.as_ptr(), libc::RENAME_SWAP) } == 0 {
            return Ok(());
        }
        Err(io::Error::last_os_error())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (source, target);
        Err(error(
            "Atomic model replacement is not implemented on this platform",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{HeaderMap, Response},
        routing::get,
    };
    use std::sync::Arc;
    fn pin(bytes: &[u8]) -> FilePin {
        FilePin {
            size: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    }
    #[test]
    fn initializer_presence_distinguishes_missing_downloads_from_load_failure() {
        let temp = tempfile::tempdir().unwrap();
        let models = artifacts();
        let reports: Vec<_> = models
            .iter()
            .map(|model| speakerdesk_core::ipc::SetupModelReport {
                model: model.id.clone(),
                ok: false,
                timing_ms: 0,
            })
            .collect();
        let progress = SetupService::default();
        progress.begin(&models, false);
        progress.check_downloads(temp.path(), &models);
        progress.finish(
            Err("Replay initializer failure".into()),
            initial_models_missing(temp.path(), &models[..3], &reports),
        );
        assert!(
            progress.progress().download_required,
            "first setup needs model files"
        );
        assert_eq!(progress.progress().status, "missing");
        assert_eq!(
            progress.progress().error.as_deref(),
            Some("Replay initializer failure")
        );
        #[cfg(unix)]
        {
            let directory = temp.path().join(&models[0].id);
            std::os::unix::fs::symlink("missing-model-directory", &directory).unwrap();
            assert!(!initial_models_missing(temp.path(), &models, &reports));
            std::fs::remove_file(&directory).unwrap();
            std::fs::create_dir_all(&directory).unwrap();
            let link = directory.join(models[0].files.keys().next().unwrap());
            std::os::unix::fs::symlink("missing-model-file", &link).unwrap();
            assert!(!initial_models_missing(temp.path(), &models, &reports));
            std::fs::remove_file(link).unwrap();
        }
        progress.begin(&models, true);
        progress.finish(Err("HTTP 503 during a requested download".into()), false);
        assert_eq!(progress.progress().status, "failed");
        assert!(progress.progress().download_required);
        // A bad present model alongside absent models must stay a real failure.
        let present = temp
            .path()
            .join(&models[0].id)
            .join(models[0].files.keys().next().unwrap());
        std::fs::create_dir_all(present.parent().unwrap()).unwrap();
        std::fs::write(present, b"invalid model").unwrap();
        progress.begin(&models, false);
        progress.check_downloads(temp.path(), &models);
        progress.finish(
            Err("A present model could not load".into()),
            initial_models_missing(temp.path(), &models[..3], &reports),
        );
        assert_eq!(progress.progress().status, "failed");
        assert!(progress.progress().download_required);
        // Empty files deliberately establish presence, not model validity or readiness.
        // Prepared outputs may be missing without requiring a network transfer.
        for model in &models {
            for name in model.files.keys() {
                let path = temp.path().join(&model.id).join(name);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"").unwrap();
            }
        }
        progress.begin(&models, false);
        progress.check_downloads(temp.path(), &models);
        progress.finish(
            Err("Replay initializer failure".into()),
            initial_models_missing(temp.path(), &models[..3], &reports),
        );
        assert!(
            !progress.progress().download_required,
            "load failure is not evidence of a missing download"
        );
        assert_eq!(progress.progress().status, "failed");
        let unknown = temp.path().join("not-a-directory");
        std::fs::write(&unknown, b"invalid model root").unwrap();
        progress.check_downloads(&unknown, &models);
        assert!(
            !progress.progress().download_required,
            "filesystem errors leave download need unknown"
        );
        assert!(!initial_models_missing(&unknown, &models[..3], &reports));
        let blocked = temp.path().join("blocked-model-directory");
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join(&models[0].id), b"not a model directory").unwrap();
        progress.check_downloads(&blocked, &models[..1]);
        assert!(!progress.progress().download_required);
        assert!(!initial_models_missing(
            &blocked,
            &models[..1],
            &reports[..1]
        ));
    }
    async fn server(
        mode: &'static str,
        seen: Arc<Mutex<Vec<Option<String>>>>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = Router::new().route(
            "/file",
            get(move |headers: HeaderMap| {
                let seen = seen.clone();
                async move {
                    let range = headers.get(RANGE).map(|v| v.to_str().unwrap().to_owned());
                    seen.lock().unwrap().push(range.clone());
                    if mode == "hang" {
                        std::future::pending::<()>().await;
                    }
                    let mut response = Response::builder();
                    let bytes: &'static [u8] = match mode {
                        "bad-hash" => b"abcxef",
                        "short" => b"abc",
                        "oversized" => b"abcdefgh",
                        "resume" | "bad-range" if range.is_some() => {
                            response = response.status(206).header(
                                CONTENT_RANGE,
                                if mode == "resume" {
                                    "bytes 3-5/6"
                                } else {
                                    "bytes 2-5/6"
                                },
                            );
                            b"def"
                        }
                        _ => b"abcdef",
                    };
                    response.body(Body::from(bytes)).unwrap()
                }
            }),
        );
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}/file"), task)
    }
    #[tokio::test]
    async fn actual_http_resume_restart_and_checksum_matrix() {
        for mode in [
            "resume",
            "restart",
            "bad-range",
            "short",
            "oversized",
            "bad-hash",
        ] {
            let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
            let mut installer = Installer::open(temp.path()).await.unwrap();
            installer.client = Client::new();
            let path = temp.path().join("model.bin");
            fs::write(temp.path().join("model.bin.part"), b"abc")
                .await
                .unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (url, task) = server(mode, seen.clone()).await;
            let result = installer
                .download(
                    &url,
                    &path,
                    &pin(b"abcdef"),
                    0,
                    &SetupService::default(),
                    &CancellationToken::new(),
                )
                .await;
            assert_eq!(seen.lock().unwrap().as_slice(), &[Some("bytes=3-".into())]);
            if ["resume", "restart"].contains(&mode) {
                result.unwrap();
                assert_eq!(fs::read(&path).await.unwrap(), b"abcdef");
            } else {
                assert!(result.is_err(), "{mode}");
                assert!(!path.exists(), "{mode}");
            }
            task.abort();
        }
    }
    #[tokio::test]
    async fn complete_partial_verifies_without_network_and_corrupt_partial_retries() {
        for bytes in [
            b"abcdef".as_slice(),
            b"xxxxxx".as_slice(),
            b"toolong".as_slice(),
        ] {
            let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
            let mut installer = Installer::open(temp.path()).await.unwrap();
            installer.client = Client::new();
            let path = temp.path().join("weights");
            fs::write(temp.path().join("weights.part"), bytes)
                .await
                .unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let (url, task) = server("restart", seen.clone()).await;
            installer
                .download(
                    &url,
                    &path,
                    &pin(b"abcdef"),
                    0,
                    &SetupService::default(),
                    &CancellationToken::new(),
                )
                .await
                .unwrap();
            assert_eq!(fs::read(path).await.unwrap(), b"abcdef");
            assert_eq!(seen.lock().unwrap().len(), usize::from(bytes != b"abcdef"));
            task.abort();
        }
    }
    #[tokio::test]
    async fn cancellation_interrupts_headers_and_holds_the_storage_owner_until_cleanup() {
        let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let mut installer = Installer::open(temp.path()).await.unwrap();
        installer.client = Client::new();
        assert!(Installer::open(temp.path()).await.is_err());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (url, server) = server("hang", seen.clone()).await;
        let token = CancellationToken::new();
        let cancel = token.clone();
        let root = temp.path().to_owned();
        let download = tokio::spawn(async move {
            let result = installer
                .download(
                    &url,
                    &root.join("weights"),
                    &pin(b"abcdef"),
                    0,
                    &SetupService::default(),
                    &token,
                )
                .await;
            assert!(result.is_err());
        });
        while seen.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), download)
            .await
            .unwrap()
            .unwrap();
        Installer::open(temp.path()).await.unwrap();
        server.abort();
    }
    #[tokio::test]
    async fn publication_never_exposes_unverified_or_cancelled_models() {
        let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let installer = Installer::open(temp.path()).await.unwrap();
        let model = ModelArtifact {
            id: "fixture".into(),
            name: "Replay".into(),
            repo: "fixture/test".into(),
            revision: "pin".into(),
            files: BTreeMap::from([
                ("weights".into(), pin(b"abcdef")),
                ("config.json".into(), pin(b"config")),
            ]),
            prepared: BTreeMap::new(),
        };
        let staging = temp.path().join(".staging/fixture");
        directory(&staging).await.unwrap();
        fs::write(staging.join("weights"), b"abcdef").await.unwrap();
        fs::write(staging.join("config.json"), b"wrong!")
            .await
            .unwrap();
        assert!(
            installer
                .activate(&model, &staging, &CancellationToken::new())
                .await
                .is_err()
        );
        assert!(!temp.path().join("fixture").exists());
        fs::write(staging.join("config.json"), b"config")
            .await
            .unwrap();
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(
            installer
                .activate(&model, &staging, &cancelled)
                .await
                .is_err()
        );
        assert!(!temp.path().join("fixture").exists());
        installer
            .activate(&model, &staging, &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(
            fs::read(temp.path().join("fixture/weights")).await.unwrap(),
            b"abcdef"
        );
        assert!(
            installer
                .installed(&model, &CancellationToken::new())
                .await
                .unwrap()
        );
        #[cfg(target_os = "macos")]
        {
            directory(&staging).await.unwrap();
            fs::write(staging.join("weights"), b"abcdef").await.unwrap();
            fs::write(staging.join("config.json"), b"config")
                .await
                .unwrap();
            fs::write(temp.path().join("fixture/weights"), b"oldbad")
                .await
                .unwrap();
            installer
                .activate(&model, &staging, &CancellationToken::new())
                .await
                .unwrap();
            assert!(
                !staging.exists(),
                "swapped invalid model should be cleaned up"
            );
            assert_eq!(
                fs::read(temp.path().join("fixture/weights")).await.unwrap(),
                b"abcdef"
            );
        }
    }
    #[tokio::test]
    async fn unexpected_active_inventory_is_repairable_and_old_staging_does_not_poison_retries() {
        let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let installer = Installer::open(temp.path()).await.unwrap();
        let model = ModelArtifact {
            id: "fixture".into(),
            name: "Fixture".into(),
            repo: "fixture/test".into(),
            revision: "pin".into(),
            files: BTreeMap::from([("weights".into(), pin(b"abcdef"))]),
            prepared: BTreeMap::new(),
        };
        let active = temp.path().join("fixture");
        directory(&active).await.unwrap();
        fs::write(active.join("weights"), b"abcdef").await.unwrap();
        fs::create_dir(active.join("unexpected-empty-directory"))
            .await
            .unwrap();
        assert!(
            !installer
                .installed(&model, &CancellationToken::new())
                .await
                .unwrap()
        );
        let staging = temp.path().join(".staging/fixture");
        directory(&staging).await.unwrap();
        fs::write(staging.join("weights"), b"abcdef").await.unwrap();
        #[cfg(target_os = "macos")]
        {
            installer
                .activate(&model, &staging, &CancellationToken::new())
                .await
                .unwrap();
            assert!(
                installer
                    .installed(&model, &CancellationToken::new())
                    .await
                    .unwrap()
            );
            assert!(!staging.exists());
            // Model the private old directory retained by an interruption immediately after swap.
            directory(&staging).await.unwrap();
            fs::write(staging.join("unknown"), b"old").await.unwrap();
            assert!(!inventory(&staging, &model, true).unwrap());
            let prepared_model = ModelArtifact {
                files: BTreeMap::new(),
                prepared: model.files.clone(),
                ..model.clone()
            };
            installer
                .stage(
                    &prepared_model,
                    0,
                    &SetupService::default(),
                    &CancellationToken::new(),
                )
                .await
                .unwrap();
            assert!(!staging.join("unknown").exists());
            fs::write(staging.join("weights"), b"abcdef").await.unwrap();
            installer
                .activate(&prepared_model, &staging, &CancellationToken::new())
                .await
                .unwrap();
            assert!(
                installer
                    .installed(&prepared_model, &CancellationToken::new())
                    .await
                    .unwrap()
            );
        }
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn links_never_write_outside_model_storage() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let outside = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        fs::write(outside.path().join("sentinel"), b"private")
            .await
            .unwrap();
        let installer = Installer::open(temp.path()).await.unwrap();
        symlink(
            outside.path().join("sentinel"),
            temp.path().join("weights.part"),
        )
        .unwrap();
        assert!(
            installer
                .download(
                    "https://example.invalid",
                    &temp.path().join("weights"),
                    &pin(b"abcdef"),
                    0,
                    &SetupService::default(),
                    &CancellationToken::new()
                )
                .await
                .is_err()
        );
        symlink(outside.path(), temp.path().join(".staging")).unwrap();
        assert!(
            directory(&temp.path().join(".staging/model"))
                .await
                .is_err()
        );
        assert_eq!(
            fs::read(outside.path().join("sentinel")).await.unwrap(),
            b"private"
        );
        assert!(!outside.path().join("model").exists());
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn hard_linked_partials_do_not_truncate_or_append_to_outside_files() {
        let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let outside = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let sentinel = outside.path().join("private");
        fs::write(&sentinel, b"private").await.unwrap();
        let installer = Installer::open(temp.path()).await.unwrap();
        std::fs::hard_link(&sentinel, temp.path().join("weights.part")).unwrap();
        assert!(
            installer
                .download(
                    "https://example.invalid",
                    &temp.path().join("weights"),
                    &pin(b"abcdef"),
                    0,
                    &SetupService::default(),
                    &CancellationToken::new()
                )
                .await
                .is_err()
        );
        assert_eq!(fs::read(&sentinel).await.unwrap(), b"private");
    }
    #[tokio::test]
    async fn interrupted_body_keeps_a_resumable_prefix() {
        let temp = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        let mut installer = Installer::open(temp.path()).await.unwrap();
        installer.client = Client::new();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/file", listener.local_addr().unwrap());
        let sender = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0u8; 2048];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabc")
                .await
                .unwrap();
            socket.shutdown().await.unwrap();
        });
        let path = temp.path().join("weights");
        assert!(
            installer
                .download(
                    &url,
                    &path,
                    &pin(b"abcdef"),
                    0,
                    &SetupService::default(),
                    &CancellationToken::new()
                )
                .await
                .is_err()
        );
        sender.await.unwrap();
        assert_eq!(fs::read(partial_path(&path)).await.unwrap(), b"abc");
        assert!(!path.exists());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (url, server) = server("resume", seen.clone()).await;
        installer
            .download(
                &url,
                &path,
                &pin(b"abcdef"),
                0,
                &SetupService::default(),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(fs::read(path).await.unwrap(), b"abcdef");
        assert_eq!(seen.lock().unwrap()[0], Some("bytes=3-".into()));
        server.abort();
    }
    #[tokio::test]
    #[ignore = "downloads the pinned 1.25 MB Silero artifact to SPEAKERDESK_SETUP_TEST_ROOT"]
    async fn real_pinned_download_activates_into_disposable_storage() {
        let root = PathBuf::from(
            std::env::var_os("SPEAKERDESK_SETUP_TEST_ROOT").expect("explicit disposable directory"),
        );
        let installer = Installer::open(&root).await.unwrap();
        let model = artifacts().remove(0);
        let cancel = CancellationToken::new();
        let progress = SetupService::default();
        progress.begin(std::slice::from_ref(&model), true);
        let staging = installer
            .stage(&model, 0, &progress, &cancel)
            .await
            .unwrap();
        installer.activate(&model, &staging, &cancel).await.unwrap();
        assert!(installer.installed(&model, &cancel).await.unwrap());
        assert_eq!(progress.progress().downloaded_bytes, model.download_bytes());
    }
    #[test]
    fn catalog_is_pinned_and_paths_are_relative() {
        let catalog = artifacts();
        assert_eq!(catalog.len(), 6);
        for model in catalog {
            assert!(relative(&model.id).is_ok());
            assert_eq!(model.revision.len(), 40);
            for (path, pin) in model.files.iter().chain(&model.prepared) {
                assert!(relative(path).is_ok());
                assert!(pin.size > 0);
                assert_eq!(pin.sha256.len(), 64);
                assert!(pin.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            }
        }
        for path in ["/tmp/no", "../bad", "x/../../bad", ""] {
            assert!(relative(path).is_err());
        }
    }
}
