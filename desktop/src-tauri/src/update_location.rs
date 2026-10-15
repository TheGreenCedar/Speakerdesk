//! Read-only installation preflight. No probe file is written into the app.
use std::path::{Component, Path, PathBuf};

pub fn bundle(executable: &Path) -> Result<PathBuf, &'static str> {
    let executable = executable
        .canonicalize()
        .map_err(|_| "Speakerdesk's installed app location could not be verified.")?;
    let macos = executable
        .parent()
        .filter(|path| path.file_name().is_some_and(|name| name == "MacOS"));
    let contents = macos
        .and_then(Path::parent)
        .filter(|path| path.file_name().is_some_and(|name| name == "Contents"));
    let app = contents.and_then(Path::parent).filter(|path| path.extension().is_some_and(|ext| ext == "app"))
        .ok_or("Install Speakerdesk's approved app bundle before updating. The development executable cannot self-update.")?;
    if app
        .components()
        .any(|component| matches!(component, Component::Normal(name) if name == "AppTranslocation"))
    {
        return Err("Move Speakerdesk to Applications, close this copy, and reopen the installed app before updating.");
    }
    if !app.is_dir() || !executable.is_file() {
        return Err("Speakerdesk's installed app bundle is unavailable.");
    }
    Ok(app.to_owned())
}

#[cfg(target_os = "macos")]
pub fn preflight(executable: &Path) -> Result<(), &'static str> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let app = bundle(executable)?;
    let parent = app
        .parent()
        .ok_or("Speakerdesk's install folder is unavailable.")?;
    for path in [&app, parent] {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| "Speakerdesk's app location is invalid.")?;
        let mut filesystem = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // statfs initializes the entire output on success. This only reads
        // mount flags; the official updater handles writable-location prompts.
        let result = unsafe { libc::statfs(path.as_ptr(), filesystem.as_mut_ptr()) };
        if result != 0 {
            return Err("Speakerdesk could not verify whether its install volume is writable. Reopen the installed app and try again.");
        }
        let filesystem = unsafe { filesystem.assume_init() };
        if filesystem.f_flags & libc::MNT_RDONLY as u32 != 0 {
            return Err("Speakerdesk is on a read-only volume. Move it to Applications, close this copy, and reopen the installed app before updating.");
        }
    }
    Ok(())
}

#[cfg(windows)]
pub fn preflight(executable: &Path) -> Result<(), &'static str> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ADD_FILE, FILE_DELETE_CHILD, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };
    if registered_value("DisplayName")? != "Speakerdesk"
        || !registered_value("MainBinaryName")?.eq_ignore_ascii_case("speakerdesk.exe")
    {
        return Err("Speakerdesk's current-user installer registration does not match this app. Reinstall the approved Windows installer.");
    }
    let directory = registry_path(&registered_value("InstallLocation")?)?;
    let uninstaller = registry_path(&registered_value("UninstallString")?)?;
    let directory = windows_installation(executable, &directory, &uninstaller)?;
    // Request the rights NSIS needs on the existing directory. Opening this
    // handle changes no bytes and creates no probe file inside the installation.
    std::fs::OpenOptions::new()
        .read(true)
        .access_mode(FILE_ADD_FILE | FILE_DELETE_CHILD)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(directory)
        .map_err(|_| "Speakerdesk cannot replace files in its install folder. Reinstall the approved Windows installer for this user and try again.")?;
    Ok(())
}

#[cfg(windows)]
fn registered_value(name: &str) -> Result<String, &'static str> {
    use windows_sys::Win32::{
        Foundation::ERROR_SUCCESS,
        System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_SZ},
    };
    let key: Vec<_> = "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Speakerdesk"
        .encode_utf16()
        .chain(Some(0))
        .collect();
    let name: Vec<_> = name.encode_utf16().chain(Some(0)).collect();
    let mut bytes = 0;
    let result = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            name.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut bytes,
        )
    };
    if result != ERROR_SUCCESS || !(2..=65536).contains(&bytes) || bytes % 2 != 0 {
        return Err("Install Speakerdesk's approved Windows installer for this user before updating. A development or portable copy cannot self-update.");
    }
    let mut value = vec![0_u16; bytes as usize / 2];
    let result = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            name.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            value.as_mut_ptr().cast(),
            &mut bytes,
        )
    };
    let length = bytes as usize / 2;
    if result != ERROR_SUCCESS
        || length == 0
        || length > value.len()
        || value[length - 1] != 0
        || value[..length - 1].contains(&0)
    {
        return Err("Speakerdesk's Windows installation registration is invalid. Reinstall the approved installer for this user.");
    }
    String::from_utf16(&value[..length - 1])
        .map_err(|_| "Speakerdesk's Windows installation path is invalid.")
}

#[cfg(any(windows, test))]
fn registry_path(value: &str) -> Result<PathBuf, &'static str> {
    let value = if let Some(value) = value.strip_prefix('"') {
        value
            .strip_suffix('"')
            .ok_or("Invalid Windows installation path")?
    } else {
        value
    };
    if value.is_empty() || value.contains(['"', '\0']) {
        return Err("Invalid Windows installation path");
    }
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("Windows installation path must be absolute");
    }
    Ok(path)
}

#[cfg(any(windows, test))]
fn windows_installation(
    executable: &Path,
    registered: &Path,
    uninstaller: &Path,
) -> Result<PathBuf, &'static str> {
    fn linked(metadata: &std::fs::Metadata) -> bool {
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            metadata.file_attributes()
                & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
                != 0
        }
        #[cfg(not(windows))]
        metadata.file_type().is_symlink()
    }
    let directory_metadata = std::fs::symlink_metadata(registered)
        .map_err(|_| "Speakerdesk's registered Windows install folder is unavailable.")?;
    if !directory_metadata.is_dir() || linked(&directory_metadata) {
        return Err("Speakerdesk's Windows install folder must be a regular directory.");
    }
    let directory = registered
        .canonicalize()
        .map_err(|_| "Speakerdesk's Windows install folder could not be verified.")?;
    for file in [executable, uninstaller] {
        let metadata = std::fs::symlink_metadata(file)
            .map_err(|_| "Speakerdesk's installed executable or uninstaller is unavailable.")?;
        if !metadata.is_file() || metadata.len() == 0 || linked(&metadata) {
            return Err(
                "Speakerdesk's installed executable and uninstaller must be regular files.",
            );
        }
    }
    let executable = executable
        .canonicalize()
        .map_err(|_| "Speakerdesk's installed executable could not be verified.")?;
    let uninstaller = uninstaller
        .canonicalize()
        .map_err(|_| "Speakerdesk's installed uninstaller could not be verified.")?;
    if executable != directory.join("speakerdesk.exe")
        || uninstaller != directory.join("uninstall.exe")
    {
        return Err("Open Speakerdesk from its registered Windows install folder before updating. Reinstall the approved installer if this copy was moved.");
    }
    Ok(directory)
}

#[cfg(not(any(target_os = "macos", windows)))]
pub fn preflight(_executable: &Path) -> Result<(), &'static str> {
    Err("Self-updating requires Speakerdesk's installed macOS or Windows app.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "speakerdesk-update-location-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn executable(&self, prefix: &str) -> PathBuf {
            let path = self
                .0
                .join(prefix)
                .join("Speakerdesk.app/Contents/MacOS/speakerdesk");
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, b"synthetic executable").unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn installed_bundle_preflight_reads_only_owned_fixture_bytes() {
        let fixture = Fixture::new();
        let executable = fixture.executable("");
        let original = fs::read(&executable).unwrap();
        assert_eq!(
            bundle(&executable).unwrap(),
            fixture.0.join("Speakerdesk.app").canonicalize().unwrap()
        );
        #[cfg(target_os = "macos")]
        assert_eq!(preflight(&executable), Ok(()));
        assert_eq!(fs::read(&executable).unwrap(), original);
        assert_eq!(
            fs::read_dir(executable.parent().unwrap()).unwrap().count(),
            1
        );
    }
    #[test]
    fn translocated_bundle_and_missing_or_development_executable_are_refused() {
        let fixture = Fixture::new();
        assert!(bundle(&fixture.executable("AppTranslocation/random/d")).is_err());
        let development = fixture.0.join("speakerdesk");
        fs::write(&development, b"development").unwrap();
        assert!(bundle(&development).is_err());
        assert!(bundle(&fixture.0.join("missing")).is_err());
        assert_eq!(fs::read(development).unwrap(), b"development");
    }
    #[cfg(unix)]
    #[test]
    fn an_alias_cannot_hide_a_translocated_bundle() {
        let fixture = Fixture::new();
        let actual = fixture.executable("AppTranslocation/random/d");
        let alias = fixture.0.join("normal-link");
        std::os::unix::fs::symlink(actual, &alias).unwrap();
        assert!(bundle(&alias).is_err());
    }

    #[test]
    fn windows_registration_binds_the_exact_installed_executable_and_uninstaller() {
        let fixture = Fixture::new();
        let install = fixture.0.join("Custom per-user installation");
        fs::create_dir(&install).unwrap();
        let executable = install.join("speakerdesk.exe");
        let uninstaller = install.join("uninstall.exe");
        fs::write(&executable, b"app").unwrap();
        fs::write(&uninstaller, b"uninstaller").unwrap();
        assert_eq!(
            windows_installation(&executable, &install, &uninstaller).unwrap(),
            install.canonicalize().unwrap()
        );
        let portable = fixture.0.join("speakerdesk.exe");
        fs::write(&portable, b"portable copy").unwrap();
        assert!(windows_installation(&portable, &install, &uninstaller).is_err());
        assert!(windows_installation(&executable, &install, &portable).is_err());
        fs::remove_file(&uninstaller).unwrap();
        assert!(windows_installation(&executable, &install, &uninstaller).is_err());
        assert_eq!(fs::read(&executable).unwrap(), b"app");
        assert_eq!(fs::read_dir(&install).unwrap().count(), 1);
        assert!(registry_path("relative/path").is_err());
        assert!(registry_path(&format!("\"{}\" /unexpected", install.display())).is_err());
        assert_eq!(
            registry_path(&format!("\"{}\"", install.display())).unwrap(),
            install
        );
    }

    #[cfg(unix)]
    #[test]
    fn windows_installation_refuses_linked_resources_and_install_roots() {
        let fixture = Fixture::new();
        let executable = fixture.0.join("speakerdesk.exe");
        let uninstaller = fixture.0.join("uninstall.exe");
        fs::write(&executable, b"app").unwrap();
        fs::write(&uninstaller, b"uninstaller").unwrap();
        let alias = fixture.0.join("linked installation");
        std::os::unix::fs::symlink(&fixture.0, &alias).unwrap();
        assert!(windows_installation(&executable, &alias, &uninstaller).is_err());
        fs::remove_file(&executable).unwrap();
        std::os::unix::fs::symlink(&uninstaller, &executable).unwrap();
        assert!(windows_installation(&executable, &fixture.0, &uninstaller).is_err());
    }
}
