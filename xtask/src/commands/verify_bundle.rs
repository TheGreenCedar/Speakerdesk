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
    let mut hits = Vec::new();
    visit(root, root, &mut hits)?;
    hits.sort();
    Ok(hits)
}

fn visit(root: &Path, path: &Path, hits: &mut Vec<String>) -> anyhow::Result<()> {
    let meta = fs::symlink_metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if path != root && is_python_artifact(path) {
        hits.push(relative(root, path));
        return Ok(());
    }
    if meta.file_type().is_dir() {
        for entry in fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
            let entry = entry.with_context(|| format!("read {}", path.display()))?;
            visit(root, &entry.path(), hits)?;
        }
    } else if path == root && is_python_artifact(path) {
        hits.push(relative(root, path));
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
