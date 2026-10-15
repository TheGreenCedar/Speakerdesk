//! The inference child owns one private Windows ML runtime and serial DirectML sessions.
use ort::{environment::Environment, session::Session};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use speakerdesk_core::error::ChildError;
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_SHARE_READ, FILE_SHARE_WRITE,
};
use windows_sys::Win32::System::LibraryLoader::{
    AddDllDirectory, GetModuleFileNameW, GetModuleHandleW, LOAD_LIBRARY_SEARCH_SYSTEM32,
    LOAD_LIBRARY_SEARCH_USER_DIRS, RemoveDllDirectory, SetDefaultDllDirectories,
};

#[path = "windows_placement.rs"]
mod placement;
use placement::{PlacementPolicy, admit_placement, placement_assignments};

static RUNTIME_READY: AtomicBool = AtomicBool::new(false);

#[derive(Clone)]
pub struct Backend {
    environment: Environment,
    directory: Rc<DllDirectory>,
}

impl Backend {
    pub fn new() -> Result<Self, ChildError> {
        let executable = std::env::current_exe().map_err(error)?;
        let directory = executable
            .parent()
            .ok_or_else(|| error("Missing executable directory"))?
            .join("runtime");
        Self::from_directory(&directory)
    }

    pub fn from_directory(directory: &Path) -> Result<Self, ChildError> {
        verify_runtime(directory)?;
        let directory = Rc::new(DllDirectory::open(directory)?);
        reject_foreign_runtime(&directory.path)?;
        let environment = ort::init_from(directory.path.join("onnxruntime.dll"))
            .map_err(error)?
            .with_telemetry(false)
            .build()
            .map_err(error)?;
        reject_foreign_runtime(&directory.path)?;
        RUNTIME_READY.store(true, Ordering::Release);
        Ok(Self {
            environment,
            directory,
        })
    }

    pub fn load(&self, path: &Path) -> Result<Session, ChildError> {
        self.load_with_options(
            path,
            ort::session::builder::GraphOptimizationLevel::Level3,
            true,
            true,
            false,
            None,
        )
    }

    pub fn load_unoptimized(&self, path: &Path) -> Result<Session, ChildError> {
        self.load_with_options(
            path,
            ort::session::builder::GraphOptimizationLevel::Disable,
            true,
            true,
            false,
            None,
        )
    }

    pub fn load_without_memory_arena(&self, path: &Path) -> Result<Session, ChildError> {
        self.load_with_options(
            path,
            ort::session::builder::GraphOptimizationLevel::Level3,
            false,
            true,
            false,
            None,
        )
    }

    pub(crate) fn load_cohere(
        &self,
        admission: &ModelAdmission,
        graph: &str,
    ) -> Result<Session, ChildError> {
        if !matches!(graph, "encoder" | "decoder-init" | "decoder-step") {
            return Err(error("Unknown Cohere graph"));
        }
        // Both reads use the admitted handles retained against writes/replacement.
        let graph_name = format!("{graph}.onnx");
        let policy: PlacementPolicy =
            serde_json::from_slice(&admission.read(&format!("{graph}-placement-policy.json"))?)
                .map_err(error)?;
        let model_sha256 = format!("{:x}", Sha256::digest(admission.read(&graph_name)?));
        if policy.model_sha256 != model_sha256 {
            return Err(error("placement policy belongs to a different model hash"));
        }
        // The encoder keeps ordered, unfused reconstruction; DML pooling reuses
        // transient buffers in all three graphs. Keep ORT optimization disabled
        // so it cannot expand the packed UINT4 constants.
        let decoder = graph != "encoder";
        self.load_with_options(
            &admission.directory.join(graph_name),
            ort::session::builder::GraphOptimizationLevel::Disable,
            true,
            decoder,
            true,
            Some((&policy, &model_sha256)),
        )
    }

    fn load_with_options(
        &self,
        path: &Path,
        optimization: ort::session::builder::GraphOptimizationLevel,
        memory_arena: bool,
        graph_fusion: bool,
        cpu_arena: bool,
        placement: Option<(&PlacementPolicy, &str)>,
    ) -> Result<Session, ChildError> {
        if !fs::symlink_metadata(path).map_err(error)?.is_file() || path.is_symlink() {
            return Err(error("Model graph must be a regular file"));
        }
        // DirectML requires serial execution and disabled memory-pattern optimization.
        let mut builder = Session::builder(&self.environment)
            .map_err(error)?
            .with_optimization_level(optimization)
            .map_err(error)?
            .with_parallel_execution(false)
            .map_err(error)?
            .with_memory_pattern(false)
            .map_err(error)?
            .with_intra_threads(4)
            .map_err(error)?
            // The DML factory reads this before registration; staged models must not retain every stage's scratch space.
            .with_config_entry(
                "ep.dml.disable_memory_arena",
                if memory_arena { "0" } else { "1" },
            )
            .map_err(error)?
            .with_config_entry(
                "ep.dml.disable_graph_fusion",
                if graph_fusion { "0" } else { "1" },
            )
            .map_err(error)?;
        if placement.is_some() {
            builder = builder
                .with_config_entry("session.record_ep_graph_assignment_info", "1")
                .map_err(error)?;
        }
        let session = builder
            .with_execution_providers([
                ort::ep::DirectML::default()
                    .with_performance_preference(
                        ort::ep::directml::PerformancePreference::HighPerformance,
                    )
                    .build()
                    .error_on_failure(),
                ort::ep::CPU::default()
                    .with_arena_allocator(cpu_arena)
                    .build(),
            ])
            .map_err(error)?
            .commit_from_file(path)
            .map_err(error)?;
        reject_foreign_runtime(&self.directory.path)?;
        if let Some((policy, model_sha256)) = placement {
            let assignments = placement_assignments(&session).map_err(error)?;
            admit_placement(policy, model_sha256, &assignments).map_err(error)?;
        }
        Ok(session)
    }
}

pub fn error(cause: impl std::fmt::Display) -> ChildError {
    ChildError::failed(format!("Windows inference: {cause}"), false)
}

pub(crate) fn run_options() -> Result<ort::session::RunOptions, ChildError> {
    // ORT otherwise implicitly loads a DLL from the environment or default search path.
    if !RUNTIME_READY.load(Ordering::Acquire) {
        return Err(error(
            "Private runtime must be initialized before native operations",
        ));
    }
    ort::session::RunOptions::new().map_err(error)
}

#[derive(Deserialize)]
struct RuntimePin {
    files: BTreeMap<String, FilePin>,
}
#[derive(Deserialize)]
struct ModelPin {
    id: String,
    files: BTreeMap<String, FilePin>,
    #[serde(default)]
    prepared: BTreeMap<String, FilePin>,
}
impl ModelPin {
    fn all_files(mut self) -> Result<BTreeMap<String, FilePin>, ChildError> {
        if self
            .files
            .keys()
            .any(|name| self.prepared.contains_key(name))
        {
            return Err(error("Downloaded and prepared artifact names overlap"));
        }
        self.files.extend(self.prepared);
        Ok(self.files)
    }
}
#[derive(Deserialize)]
struct FilePin {
    sha256: String,
    size: Option<u64>,
}

fn model_pin(id: &str) -> Result<ModelPin, ChildError> {
    let pins: Vec<ModelPin> =
        serde_json::from_str(include_str!("windows-model-artifacts.json")).map_err(error)?;
    let pin = pins
        .into_iter()
        .find(|pin| pin.id == id)
        .ok_or_else(|| error(format!("No admitted Windows artifact for {id}")))?;
    if pin.files.keys().any(|name| pin.prepared.contains_key(name)) {
        return Err(error("Downloaded and prepared artifact names overlap"));
    }
    Ok(pin)
}

pub fn verify_model(directory: &Path, id: &str) -> Result<(), ChildError> {
    verify_files(directory, &model_pin(id)?.all_files()?)
}

/// Hold verified file identities until their native model objects are dropped.
pub(crate) struct ModelAdmission {
    directory: PathBuf,
    files: BTreeMap<String, fs::File>,
    // Drop this last so the directory cannot move before its file guards close.
    _directory: fs::File,
}

impl ModelAdmission {
    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, ChildError> {
        let mut file = self.file(name)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(error)?;
        Ok(bytes)
    }

    pub(crate) fn file(&self, name: &str) -> Result<fs::File, ChildError> {
        use std::io::Seek;
        let mut file = self
            .files
            .get(name)
            .ok_or_else(|| error(format!("Artifact was not admitted: {name}")))?
            .try_clone()
            .map_err(error)?;
        file.rewind().map_err(error)?;
        Ok(file)
    }
}

pub(crate) fn admit_model(directory: &Path, id: &str) -> Result<ModelAdmission, ChildError> {
    admit_files(directory, &model_pin(id)?.all_files()?)
}

// Preparation needs pinned source handles before the prepared output exists.
pub(crate) fn admit_source_files(directory: &Path, id: &str) -> Result<ModelAdmission, ChildError> {
    admit_files(directory, &model_pin(id)?.files)
}

fn admit_files(
    directory: &Path,
    files: &BTreeMap<String, FilePin>,
) -> Result<ModelAdmission, ChildError> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    verify_directory(directory)?;
    let directory = directory.canonicalize().map_err(error)?;
    // Lock this artifact directory, not its unrelated ancestors. Readers and
    // ordinary directory access remain possible, but directory replacement does not.
    let directory_handle = fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(&directory)
        .map_err(error)?;
    let meta = directory_handle.metadata().map_err(error)?;
    if !meta.is_dir() || meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(error("Artifact directory is missing or a reparse point"));
    }
    let mut handles = BTreeMap::new();
    for (name, pin) in files {
        verify_filename(name)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(directory.join(name))
            .map_err(error)?;
        // Metadata and hash come from the very handle retained against writes/deletion.
        handles.insert(name.clone(), verify_open_file(file, name, pin)?);
    }
    Ok(ModelAdmission {
        directory,
        files: handles,
        _directory: directory_handle,
    })
}

fn verify_runtime(directory: &Path) -> Result<(), ChildError> {
    let mut pin: RuntimePin =
        serde_json::from_str(include_str!("windows-runtime.json")).map_err(error)?;
    pin.files.retain(|name, _| name.ends_with(".dll"));
    verify_runtime_files(directory, &pin.files)
}

fn verify_runtime_files(
    directory: &Path,
    files: &BTreeMap<String, FilePin>,
) -> Result<(), ChildError> {
    verify_files(directory, files)?;
    // USER_DIRS precedes System32: an extra DLL could shadow a pinned DLL's dependency.
    for entry in fs::read_dir(directory).map_err(error)? {
        let name = entry.map_err(error)?.file_name();
        if Path::new(&name)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
            && !files.keys().any(|pinned| name.eq_ignore_ascii_case(pinned))
        {
            return Err(error(format!(
                "Unadmitted DLL in private runtime: {}",
                name.to_string_lossy()
            )));
        }
    }
    Ok(())
}

fn verify_files(directory: &Path, files: &BTreeMap<String, FilePin>) -> Result<(), ChildError> {
    verify_directory(directory)?;
    for (name, pin) in files {
        verify_filename(name)?;
        let path = directory.join(name);
        verify_file_metadata(&fs::symlink_metadata(&path).map_err(error)?, name, pin)?;
        verify_open_file(fs::File::open(path).map_err(error)?, name, pin)?;
    }
    Ok(())
}

fn verify_directory(directory: &Path) -> Result<(), ChildError> {
    use std::os::windows::fs::MetadataExt;
    for parent in directory
        .ancestors()
        .filter(|path| !path.as_os_str().is_empty())
    {
        let meta = fs::symlink_metadata(parent).map_err(error)?;
        if !meta.is_dir() || meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(error("Artifact directory is missing or a reparse point"));
        }
    }
    Ok(())
}

fn verify_filename(name: &str) -> Result<(), ChildError> {
    if Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err(error("Invalid bundled artifact filename"));
    }
    Ok(())
}

fn verify_file_metadata(meta: &fs::Metadata, name: &str, pin: &FilePin) -> Result<(), ChildError> {
    use std::os::windows::fs::MetadataExt;
    if !meta.is_file()
        || meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
        || pin.size.is_some_and(|size| size != meta.len())
    {
        return Err(error(format!(
            "Artifact file is missing, linked or has the wrong size: {name}"
        )));
    }
    Ok(())
}

fn verify_open_file(mut file: fs::File, name: &str, pin: &FilePin) -> Result<fs::File, ChildError> {
    verify_file_metadata(&file.metadata().map_err(error)?, name, pin)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(error)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    if format!("{:x}", digest.finalize()) != pin.sha256 {
        return Err(error(format!("Artifact file failed verification: {name}")));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_files_are_required_verified_and_locked_without_overriding_sources() {
        let pin = |name: &str, content: &[u8]| {
            (
                name.to_string(),
                FilePin {
                    sha256: format!("{:x}", Sha256::digest(content)),
                    size: Some(content.len() as u64),
                },
            )
        };
        let duplicate = ModelPin {
            id: "fixture".into(),
            files: BTreeMap::from([pin("same", b"source")]),
            prepared: BTreeMap::from([pin("same", b"prepared")]),
        };
        assert!(duplicate.all_files().is_err());
        let pins = ModelPin {
            id: "fixture".into(),
            files: BTreeMap::from([pin("source", b"source")]),
            prepared: BTreeMap::from([pin("weights", b"prepared")]),
        }
        .all_files()
        .unwrap();
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("source"), b"source").unwrap();
        assert!(admit_files(root.path(), &pins).is_err());
        fs::write(root.path().join("weights"), b"tampered").unwrap();
        assert!(admit_files(root.path(), &pins).is_err());
        fs::write(root.path().join("weights"), b"prepared").unwrap();
        let admission = admit_files(root.path(), &pins).unwrap();
        assert!(
            fs::OpenOptions::new()
                .write(true)
                .open(root.path().join("weights"))
                .is_err()
        );
        drop(admission);
        assert!(
            fs::OpenOptions::new()
                .write(true)
                .open(root.path().join("weights"))
                .is_ok()
        );
    }

    #[test]
    fn lazy_model_admission_blocks_file_and_directory_replacement_until_drop() {
        use std::io::Write;

        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("model");
        fs::create_dir(&directory).unwrap();
        let originals = [
            ("encoder.onnx", b"original graph".as_slice()),
            ("mossformer.weights", b"original shared weights".as_slice()),
        ];
        let mut pins = BTreeMap::new();
        for (name, bytes) in originals {
            fs::write(directory.join(name), bytes).unwrap();
            pins.insert(
                name.to_owned(),
                FilePin {
                    sha256: format!("{:x}", Sha256::digest(bytes)),
                    size: Some(bytes.len() as u64),
                },
            );
        }
        let admission = Rc::new(admit_files(&directory, &pins).unwrap());
        // A live recording keeps its lease after the cached network owner is dropped.
        let recording = admission.clone();
        drop(admission);
        for (name, bytes) in originals {
            let file = directory.join(name);
            assert_eq!(
                fs::read(&file).unwrap(),
                bytes,
                "native readers must remain allowed"
            );
            assert!(
                fs::OpenOptions::new().write(true).open(&file).is_err(),
                "{name} must deny writes while admitted"
            );
            assert!(
                fs::OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&file)
                    .is_err(),
                "{name} must deny truncation while admitted"
            );
            let replacement = root.path().join(format!("replacement-{name}"));
            fs::write(&replacement, b"replacement").unwrap();
            assert!(
                fs::rename(&replacement, &file).is_err(),
                "{name} must deny replacement while admitted"
            );
            assert!(fs::remove_file(&file).is_err(), "{name} must deny deletion");
            assert_eq!(fs::read(&file).unwrap(), bytes);
        }
        let moved = root.path().join("previous-model");
        assert!(
            fs::rename(&directory, &moved).is_err(),
            "the complete admitted directory must not move"
        );
        // Unrelated artifacts under the same parent are not locked.
        let unrelated = root.path().join("other-model");
        fs::create_dir(&unrelated).unwrap();
        fs::rename(&unrelated, root.path().join("other-model-moved")).unwrap();
        drop(recording);
        for (name, bytes) in originals {
            let file = directory.join(name);
            fs::OpenOptions::new()
                .write(true)
                .open(&file)
                .unwrap()
                .write_all(b"changed")
                .unwrap();
            fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&file)
                .unwrap();
            assert_eq!(fs::metadata(&file).unwrap().len(), 0);
            fs::rename(root.path().join(format!("replacement-{name}")), &file).unwrap();
            assert_eq!(fs::read(&file).unwrap(), b"replacement");
            fs::write(file, bytes).unwrap();
        }
        // A failed hash check after the first file was locked must release all guards.
        pins.get_mut("mossformer.weights").unwrap().sha256 = "00".repeat(32);
        assert!(admit_files(&directory, &pins).is_err());
        for (name, _) in originals {
            fs::OpenOptions::new()
                .write(true)
                .open(directory.join(name))
                .unwrap();
        }
        fs::rename(&directory, &moved).unwrap();
        let replacement = root.path().join("replacement-model");
        fs::create_dir(&replacement).unwrap();
        fs::write(replacement.join("encoder.onnx"), b"replacement tree").unwrap();
        fs::rename(&replacement, &directory).unwrap();
        fs::remove_dir_all(moved).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn runtime_admission_rejects_unpinned_dependency_dlls() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("runtime.dll"), b"pinned").unwrap();
        let pins = BTreeMap::from([(
            "runtime.dll".to_owned(),
            FilePin {
                sha256: format!("{:x}", Sha256::digest(b"pinned")),
                size: Some(6),
            },
        )]);
        fs::write(directory.path().join("LICENSE.txt"), b"notice").unwrap();
        verify_runtime_files(directory.path(), &pins).unwrap();
        for name in ["d3d12.dll", "DBGHELP.DLL", "unexpected.DlL"] {
            let injected = directory.path().join(name);
            fs::write(&injected, b"unadmitted dependency").unwrap();
            assert!(
                verify_runtime_files(directory.path(), &pins).is_err(),
                "{name} could shadow a system dependency"
            );
            fs::remove_file(injected).unwrap();
        }
        fs::rename(
            directory.path().join("runtime.dll"),
            directory.path().join("RUNTIME.DLL"),
        )
        .unwrap();
        verify_runtime_files(directory.path(), &pins).unwrap();
    }

    #[test]
    fn artifact_admission_rejects_modified_missing_and_nonregular_files() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("model.onnx");
        fs::write(&file, b"original graph").unwrap();
        let pins = BTreeMap::from([(
            "model.onnx".to_owned(),
            FilePin {
                sha256: format!("{:x}", Sha256::digest(b"original graph")),
                size: Some(14),
            },
        )]);
        verify_files(directory.path(), &pins).unwrap();
        fs::write(&file, b"modified graph").unwrap();
        assert!(verify_files(directory.path(), &pins).is_err());
        fs::remove_file(&file).unwrap();
        assert!(verify_files(directory.path(), &pins).is_err());
        fs::create_dir(&file).unwrap();
        assert!(verify_files(directory.path(), &pins).is_err());
        assert!(verify_model(directory.path(), "unadmitted-model").is_err());
    }
}

struct DllDirectory {
    path: PathBuf,
    cookie: *mut std::ffi::c_void,
}
impl DllDirectory {
    fn open(path: &Path) -> Result<Self, ChildError> {
        use std::os::windows::ffi::OsStrExt;
        let path = path.canonicalize().map_err(error)?;
        let wide: Vec<_> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        // Exclude PATH and the working directory from transitive dependency lookup.
        unsafe {
            if SetDefaultDllDirectories(
                LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS,
            ) == 0
            {
                return Err(error(std::io::Error::last_os_error()));
            }
            let cookie = AddDllDirectory(wide.as_ptr());
            if cookie.is_null() {
                return Err(error(std::io::Error::last_os_error()));
            }
            Ok(Self { path, cookie })
        }
    }
}
impl Drop for DllDirectory {
    fn drop(&mut self) {
        unsafe {
            RemoveDllDirectory(self.cookie);
        }
    }
}

fn reject_foreign_runtime(directory: &Path) -> Result<(), ChildError> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    for name in [
        "onnxruntime.dll",
        "DirectML.dll",
        "Microsoft.Windows.AI.MachineLearning.dll",
    ] {
        let wide: Vec<_> = std::ffi::OsStr::new(name)
            .encode_wide()
            .chain(Some(0))
            .collect();
        let module = unsafe { GetModuleHandleW(wide.as_ptr()) };
        if module.is_null() {
            continue;
        }
        let mut buffer = vec![0_u16; 32768];
        let size = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) }
            as usize;
        if size == 0 || size == buffer.len() {
            return Err(error(std::io::Error::last_os_error()));
        }
        let loaded = PathBuf::from(std::ffi::OsString::from_wide(&buffer[..size]))
            .canonicalize()
            .map_err(error)?;
        if loaded != directory.join(name).canonicalize().map_err(error)? {
            return Err(error(format!("A different {name} is already loaded")));
        }
    }
    Ok(())
}
