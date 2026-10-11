//! The port ledger (`xtask/port-ledger.toml`, plan §7.1) tracks every Python
//! file slated for porting, simplification, dropping or delegation to an
//! xtask/Swift/allowlisted destination.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::Deserialize;

const LEDGER_PATH: &str = "xtask/port-ledger.toml";
/// Directories scanned recursively for `*.py`.
const RECURSIVE_ROOTS: [&str; 2] = ["tests", "speakerdesk"];
/// Directories scanned only at the top level for `*.py`.
const FLAT_ROOTS: [&str; 1] = ["scripts"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Disposition {
    Port,
    Simplify,
    Drop,
    Memo,
    Xtask,
    Swift,
    Allowlist,
}

impl fmt::Display for Disposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Port => "port",
            Self::Simplify => "simplify",
            Self::Drop => "drop",
            Self::Memo => "memo",
            Self::Xtask => "xtask",
            Self::Swift => "swift",
            Self::Allowlist => "allowlist",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pending,
    Done,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Pending => "pending",
            Self::Done => "done",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Deserialize)]
pub struct Row {
    pub path: PathBuf,
    pub disposition: Disposition,
    #[serde(default)]
    pub rust: Vec<PathBuf>,
    #[serde(default)]
    pub reason: Option<String>,
    pub status: Status,
    #[serde(default)]
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct Ledger {
    pub row: Vec<Row>,
}

#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub rows: usize,
    pub scanned: usize,
    pub by_disposition: BTreeMap<Disposition, usize>,
    pub by_status: BTreeMap<Status, usize>,
}

pub fn check(root: &Path) -> anyhow::Result<()> {
    let ledger = load(root)?;
    let report = validate(root, &ledger);
    println!(
        "ledger: {} rows covering {} tracked Python files",
        report.rows, report.scanned
    );
    for (disposition, count) in &report.by_disposition {
        println!("  {disposition}: {count}");
    }
    for (status, count) in &report.by_status {
        println!("  status {status}: {count}");
    }
    if report.errors.is_empty() {
        println!("ledger --check: ok");
        return Ok(());
    }
    for error in &report.errors {
        eprintln!("error: {error}");
    }
    bail!("{} ledger error(s)", report.errors.len())
}

fn load(root: &Path) -> anyhow::Result<Ledger> {
    let path = root.join(LEDGER_PATH);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

fn validate(root: &Path, ledger: &Ledger) -> Report {
    let mut report = Report {
        rows: ledger.row.len(),
        ..Report::default()
    };
    let mut listed = BTreeMap::new();
    for row in &ledger.row {
        *report.by_disposition.entry(row.disposition).or_default() += 1;
        *report.by_status.entry(row.status).or_default() += 1;
        let path = &row.path;
        if let Some(first) = listed.insert(path.clone(), row) {
            let _ = first;
            report
                .errors
                .push(format!("{}: listed more than once", path.display()));
            continue;
        }
        if !in_scope(path) {
            report.errors.push(format!(
                "{}: outside the tracked scope (tests/**, speakerdesk/**, scripts/*.py)",
                path.display()
            ));
        }
        let exists = root.join(path).is_file();
        if row.deleted && exists {
            report.errors.push(format!(
                "{}: marked deleted but still exists on disk",
                path.display()
            ));
        }
        if !exists && !row.deleted && row.status != Status::Done {
            report.errors.push(format!(
                "{}: missing on disk; mark it done or deleted = true",
                path.display()
            ));
        }
        if row.disposition == Disposition::Drop {
            match &row.reason {
                Some(reason) if has_decision_ref(reason) => {}
                Some(_) => report.errors.push(format!(
                    "{}: drop reason must cite a locked decision (e.g. \"D11\")",
                    path.display()
                )),
                None => report
                    .errors
                    .push(format!("{}: drop row requires a reason", path.display())),
            }
        }
        if row.status == Status::Done {
            for rust_path in &row.rust {
                if !root.join(rust_path).exists() {
                    report.errors.push(format!(
                        "{}: done row lists missing Rust path {}",
                        path.display(),
                        rust_path.display()
                    ));
                }
            }
        }
    }
    for file in scan_python_files(root) {
        report.scanned += 1;
        if !listed.contains_key(&file) {
            report.errors.push(format!(
                "{}: tracked Python file missing from the ledger",
                file.display()
            ));
        }
    }
    report
}

fn has_decision_ref(reason: &str) -> bool {
    let bytes = reason.as_bytes();
    bytes
        .windows(2)
        .any(|w| w[0] == b'D' && w[1].is_ascii_digit())
}

fn in_scope(path: &Path) -> bool {
    if path.extension() != Some(std::ffi::OsStr::new("py")) {
        return false;
    }
    let mut components = path.components();
    let Some(std::path::Component::Normal(top)) = components.next() else {
        return false;
    };
    if FLAT_ROOTS.contains(&top.to_string_lossy().as_ref()) {
        return path.parent() == Some(Path::new(top));
    }
    RECURSIVE_ROOTS.contains(&top.to_string_lossy().as_ref())
}

fn scan_python_files(root: &Path) -> BTreeSet<PathBuf> {
    let mut files = BTreeSet::new();
    for base in RECURSIVE_ROOTS {
        walk(root.join(base), true, root, &mut files);
    }
    for base in FLAT_ROOTS {
        walk(root.join(base), false, root, &mut files);
    }
    files
}

fn walk(dir: PathBuf, recurse: bool, root: &Path, files: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_name() == "__pycache__" {
            continue;
        }
        if path.is_dir() {
            if recurse {
                walk(path, true, root, files);
            }
        } else if path.extension() == Some(std::ffi::OsStr::new("py"))
            && let Ok(relative) = path.strip_prefix(root)
        {
            files.insert(relative.to_path_buf());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, disposition: Disposition, status: Status) -> Row {
        Row {
            path: PathBuf::from(path),
            disposition,
            rust: Vec::new(),
            reason: None,
            status,
            deleted: false,
        }
    }

    #[test]
    fn real_ledger_covers_the_tree() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root");
        let ledger = load(root).expect("ledger parses");
        let report = validate(root, &ledger);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.rows, report.scanned);
        assert!(report.rows > 0);
    }

    #[test]
    fn unlisted_python_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("tests")).unwrap();
        std::fs::write(dir.path().join("tests/test_a.py"), "").unwrap();
        let report = validate(dir.path(), &Ledger { row: vec![] });
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("test_a.py") && e.contains("missing from the ledger"))
        );
    }

    #[test]
    fn ledger_path_gone_requires_done_or_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let mut gone = row("speakerdesk/a.py", Disposition::Port, Status::Pending);
        let report = validate(
            dir.path(),
            &Ledger {
                row: vec![row("speakerdesk/a.py", Disposition::Port, Status::Pending)],
            },
        );
        assert!(report.errors.iter().any(|e| e.contains("missing on disk")));
        gone.status = Status::Done;
        let report = validate(dir.path(), &Ledger { row: vec![gone] });
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        let mut deleted = row("speakerdesk/a.py", Disposition::Port, Status::Pending);
        deleted.deleted = true;
        let report = validate(dir.path(), &Ledger { row: vec![deleted] });
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }

    #[test]
    fn deleted_row_with_existing_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("speakerdesk")).unwrap();
        std::fs::write(dir.path().join("speakerdesk/a.py"), "").unwrap();
        let mut deleted = row("speakerdesk/a.py", Disposition::Port, Status::Pending);
        deleted.deleted = true;
        let report = validate(dir.path(), &Ledger { row: vec![deleted] });
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("marked deleted but still exists"))
        );
    }

    #[test]
    fn drop_row_needs_decision_ref() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("speakerdesk")).unwrap();
        std::fs::write(dir.path().join("speakerdesk/a.py"), "").unwrap();
        let no_reason = row("speakerdesk/a.py", Disposition::Drop, Status::Pending);
        let report = validate(
            dir.path(),
            &Ledger {
                row: vec![no_reason],
            },
        );
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("requires a reason"))
        );
        let mut bad = row("speakerdesk/a.py", Disposition::Drop, Status::Pending);
        bad.reason = Some("no longer wanted".into());
        let report = validate(dir.path(), &Ledger { row: vec![bad] });
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("cite a locked decision"))
        );
        let mut good = row("speakerdesk/a.py", Disposition::Drop, Status::Pending);
        good.reason = Some("D11: dropped".into());
        let report = validate(dir.path(), &Ledger { row: vec![good] });
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }

    #[test]
    fn done_row_needs_existing_rust_paths() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("tests")).unwrap();
        std::fs::write(dir.path().join("tests/test_a.py"), "").unwrap();
        std::fs::create_dir_all(dir.path().join("crates/core/tests")).unwrap();
        std::fs::write(dir.path().join("crates/core/tests/a.rs"), "").unwrap();
        let mut done = row("tests/test_a.py", Disposition::Port, Status::Done);
        done.rust = vec![PathBuf::from("crates/core/tests/a.rs")];
        let report = validate(dir.path(), &Ledger { row: vec![done] });
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        let mut missing = row("tests/test_a.py", Disposition::Port, Status::Done);
        missing.rust = vec![PathBuf::from("crates/core/tests/b.rs")];
        let report = validate(dir.path(), &Ledger { row: vec![missing] });
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("missing Rust path"))
        );
    }

    #[test]
    fn duplicate_rows_fail() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("scripts")).unwrap();
        std::fs::write(dir.path().join("scripts/a.py"), "").unwrap();
        let report = validate(
            dir.path(),
            &Ledger {
                row: vec![
                    row("scripts/a.py", Disposition::Xtask, Status::Pending),
                    row("scripts/a.py", Disposition::Xtask, Status::Pending),
                ],
            },
        );
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("listed more than once"))
        );
    }
}
