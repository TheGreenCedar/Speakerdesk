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

#[cfg(not(target_os = "macos"))]
pub fn preflight(_executable: &Path) -> Result<(), &'static str> {
    Err("Self-updating is currently available for Speakerdesk's installed macOS app.")
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
}
