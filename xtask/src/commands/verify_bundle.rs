use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

/// Reject a missing path, and a directory that still carries Python artifacts.
pub fn run(path: Option<PathBuf>) -> anyhow::Result<()> {
    let Some(path) = path else {
        bail!("verify-bundle requires a path");
    };
    if !path.exists() {
        bail!("verify-bundle path is missing: {}", path.display());
    }
    let hits = python_artifacts(&path)?;
    if !hits.is_empty() {
        bail!(
            "verify-bundle failed: Python remains in {}:\n{}",
            path.display(),
            hits.join("\n")
        );
    }
    println!("verify-bundle ok: {}", path.display());
    Ok(())
}

fn python_artifacts(root: &Path) -> anyhow::Result<Vec<String>> {
    let root = fs::canonicalize(root).context("resolve bundle root")?;
    let mut hits = Vec::new();
    visit(&root, &root, &mut HashSet::new(), &mut hits)?;
    hits.sort();
    Ok(hits)
}

fn visit(
    root: &Path,
    path: &Path,
    visited: &mut HashSet<PathBuf>,
    hits: &mut Vec<String>,
) -> anyhow::Result<()> {
    let resolved = fs::canonicalize(path).with_context(|| format!("resolve {}", path.display()))?;
    if !resolved.starts_with(root) {
        bail!("bundle link escapes root: {}", path.display());
    }
    if is_python_artifact(path) || is_python_artifact(&resolved) {
        hits.push(relative(root, path));
        return Ok(());
    }
    if !visited.insert(resolved.clone()) {
        return Ok(());
    }
    let meta = fs::metadata(&resolved).with_context(|| format!("stat {}", resolved.display()))?;
    if meta.file_type().is_dir() {
        for entry in
            fs::read_dir(&resolved).with_context(|| format!("read {}", resolved.display()))?
        {
            let entry = entry.with_context(|| format!("read {}", path.display()))?;
            visit(root, &entry.path(), visited, hits)?;
        }
    }
    Ok(())
}

fn is_python_artifact(path: &Path) -> bool {
    let Some(name) = path.file_name() else {
        return false;
    };
    let name = name.to_string_lossy();
    name.ends_with(".py")
        || name.contains("libpython")
        || name.contains("Python.framework")
        || name.contains("speakerdesk-runtime")
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::run;

    #[test]
    fn temp_dir_flags_python_bundle_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("helper.py"), b"").unwrap();
        fs::write(root.join("libpython3.13.so"), b"").unwrap();
        fs::create_dir_all(root.join("Python.framework/Versions")).unwrap();
        fs::write(root.join("Python.framework/Versions/Python"), b"").unwrap();
        fs::write(root.join("speakerdesk-runtime"), b"").unwrap();

        let error = run(Some(root.to_path_buf())).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("helper.py"), "{text}");
        assert!(text.contains("libpython"), "{text}");
        assert!(text.contains("Python.framework"), "{text}");
        assert!(text.contains("speakerdesk-runtime"), "{text}");
    }

    #[test]
    fn clean_temp_dir_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("app.js"), b"ok").unwrap();
        run(Some(dir.path().to_path_buf())).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn follows_bundle_and_framework_links_without_skipping_python() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("actual.app");
        let version = root.join("Frameworks/Native.framework/Versions/A");
        fs::create_dir_all(&version).unwrap();
        fs::write(version.join("Native"), b"native fixture").unwrap();
        symlink("A", version.parent().unwrap().join("Current")).unwrap();
        symlink(
            "Versions/Current/Native",
            root.join("Frameworks/Native.framework/Native"),
        )
        .unwrap();
        let alias = dir.path().join("Speakerdesk.app");
        symlink(&root, &alias).unwrap();
        run(Some(alias.clone())).unwrap();
        fs::write(version.join("helper.py"), b"python fixture").unwrap();
        assert!(
            run(Some(alias))
                .unwrap_err()
                .to_string()
                .contains("Python remains")
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_escaping_broken_and_cyclic_links() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("app");
        fs::create_dir(&root).unwrap();
        fs::write(dir.path().join("outside"), b"outside").unwrap();
        let link = root.join("innocent");
        for target in ["../outside", "missing", "innocent"] {
            symlink(target, &link).unwrap();
            assert!(run(Some(root.clone())).is_err(), "{target}");
            fs::remove_file(&link).unwrap();
        }
    }

    #[test]
    fn missing_path_argument_is_an_error() {
        let error = run(None).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("path"), "expected a path error, got {text}");
    }

    #[test]
    fn missing_directory_is_an_error() {
        let missing = PathBuf::from("/tmp/speakerdesk-verify-bundle-missing-1c9e");
        let _ = fs::remove_dir_all(&missing);
        let error = run(Some(missing)).unwrap_err();
        let text = error.to_string();
        assert!(
            text.contains("missing")
                || text.contains("not found")
                || text.contains("does not exist"),
            "{text}"
        );
    }
}
