//! `cargo xtask fetch-upstream`: clone the pinned upstream model sources into
//! `.cache/upstream/<name>` with a shallow fetch of the exact commit, verify
//! HEAD matches the pin, record their license files in
//! `.cache/upstream/LICENSES.md` and write the `mx.compile`/`mx.fast` usage
//! inventory to `.cache/upstream/mlx-usage.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};

struct Upstream {
    name: &'static str,
    url: &'static str,
    sha: &'static str,
    /// Repo-relative files/dirs containing the model code we port.
    inventory: &'static [&'static str],
}

const UPSTREAMS: &[Upstream] = &[
    Upstream {
        name: "mlx-audio",
        url: "https://github.com/Blaizzy/mlx-audio.git",
        sha: "feb25a37b07923bae556e59111995071d66afa0d",
        inventory: &[
            // Silero v6 VAD, Nemotron/Sortformer diarization, Whisper.
            "mlx_audio/vad/models/silero_vad",
            "mlx_audio/vad/models/sortformer",
            "mlx_audio/vad/models/nemotron_diarization",
            "mlx_audio/vad/utils.py",
            "mlx_audio/stt/models/whisper",
        ],
    },
    Upstream {
        name: "mlx-speech",
        url: "https://github.com/appautomaton/mlx-speech.git",
        sha: "4577ad28e6bef2ae2a11b8fa76a6cdb23bb2a9cc",
        inventory: &[
            // Cohere ASR.
            "src/mlx_speech/models/cohere_asr",
            "src/mlx_speech/models/reuse",
            "src/mlx_speech/generation/cohere_asr.py",
            "src/mlx_speech/asr/_adapters/cohere.py",
        ],
    },
];

const LOCAL_INVENTORY: &[&str] = &[
    "speakerdesk/mlx_ctc_components",
    "speakerdesk/mlx_voice_mil.py",
];

pub fn fetch(root: &Path) -> anyhow::Result<()> {
    let base = root.join(".cache/upstream");
    std::fs::create_dir_all(&base)?;
    for upstream in UPSTREAMS {
        ensure(root, &base, upstream)?;
    }
    write_licenses(&base)?;
    write_usage_inventory(root, &base)?;
    Ok(())
}

fn ensure(root: &Path, base: &Path, upstream: &Upstream) -> anyhow::Result<()> {
    let dir = base.join(upstream.name);
    if head(&dir)?.as_deref() == Some(upstream.sha) {
        println!("{}: already at {}", upstream.name, upstream.sha);
        return Ok(());
    }
    if dir.exists() && !dir.join(".git").exists() {
        bail!(
            "{} exists but is not a fetched repo; remove it and retry",
            dir.display()
        );
    }
    if !dir.exists() {
        git(root, &["init", "-q", &dir.to_string_lossy()])?;
    }
    let dir_arg = dir.to_string_lossy().to_string();
    git(
        root,
        &[
            "-C",
            &dir_arg,
            "fetch",
            "-q",
            "--depth",
            "1",
            upstream.url,
            upstream.sha,
        ],
    )?;
    git(
        root,
        &["-C", &dir_arg, "checkout", "-q", "--detach", "FETCH_HEAD"],
    )?;
    let head = head(&dir)?.unwrap_or_default();
    if head != upstream.sha {
        bail!(
            "{}: HEAD is {head}, expected {}",
            upstream.name,
            upstream.sha
        );
    }
    println!("{}: fetched {}", upstream.name, upstream.sha);
    Ok(())
}

fn head(dir: &Path) -> anyhow::Result<Option<String>> {
    if !dir.join(".git").exists() {
        return Ok(None);
    }
    let output = Command::new("git")
        .args(["-C", &dir.to_string_lossy(), "rev-parse", "HEAD"])
        .output()
        .context("git rev-parse failed")?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8_lossy(&output.stdout).trim().to_string(),
    ))
}

fn git(root: &Path, args: &[&str]) -> anyhow::Result<()> {
    let status = Command::new("git")
        .args(args)
        .current_dir(root)
        .status()
        .with_context(|| format!("git {}", args.join(" ")))?;
    if !status.success() {
        bail!("git {} failed", args.join(" "));
    }
    Ok(())
}

fn license_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if entry.file_name() != ".git" {
                    stack.push(path);
                }
            } else {
                let name = entry.file_name().to_string_lossy().to_uppercase();
                if name.starts_with("LICENSE")
                    || name.starts_with("LICENCE")
                    || name.starts_with("COPYING")
                    || name.starts_with("COPYRIGHT")
                    || name.starts_with("NOTICE")
                {
                    found.push(path);
                }
            }
        }
    }
    found.sort();
    found
}

fn write_licenses(base: &Path) -> anyhow::Result<()> {
    let mut out = String::from(
        "# Upstream licenses\n\n\
         Recorded by `cargo xtask fetch-upstream`; regenerated on every fetch.\n",
    );
    for upstream in UPSTREAMS {
        let dir = base.join(upstream.name);
        out.push_str(&format!("\n## {} @ {}\n", upstream.name, upstream.sha));
        let files = license_files(&dir);
        if files.is_empty() {
            out.push_str("\n(no license files found)\n");
        }
        for file in files {
            let relative = file.strip_prefix(&dir).unwrap_or(&file);
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("cannot read {}", file.display()))?;
            out.push_str(&format!(
                "\n### {}\n\n```\n{}```\n",
                relative.display(),
                text
            ));
            if !text.ends_with('\n') {
                out.push('\n');
            }
        }
    }
    let path = base.join("LICENSES.md");
    std::fs::write(&path, out).with_context(|| format!("cannot write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// `.py` files under each inventory path, excluding nested `tests` dirs.
fn inventory_files(root: &Path, paths: &[&str]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for relative in paths {
        let path = root.join(relative);
        if path.is_file() {
            files.push(path);
            continue;
        }
        let mut stack = vec![path];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    if entry.file_name() != "tests" && entry.file_name() != "__pycache__" {
                        stack.push(entry_path);
                    }
                } else if entry_path.extension() == Some(std::ffi::OsStr::new("py")) {
                    files.push(entry_path);
                }
            }
        }
    }
    files.sort();
    files
}

fn usage_lines(root: &Path, files: &[PathBuf]) -> Vec<String> {
    let mut hits = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let relative = file.strip_prefix(root).unwrap_or(file);
        for (index, line) in text.lines().enumerate() {
            if line.contains("mx.compile") || line.contains("mx.fast.") {
                hits.push(format!(
                    "{}:{}: {}",
                    relative.display(),
                    index + 1,
                    line.trim()
                ));
            }
        }
    }
    hits
}

fn write_usage_inventory(root: &Path, base: &Path) -> anyhow::Result<()> {
    let mut out = String::from(
        "# Upstream mx.compile / mx.fast usage inventory\n\n\
         Recorded by `cargo xtask fetch-upstream` (plan §6 requires mirroring\n\
         these call sites on the mlx-rs port).\n",
    );
    for upstream in UPSTREAMS {
        let dir = base.join(upstream.name);
        out.push_str(&format!("\n## {} @ {}\n\n", upstream.name, upstream.sha));
        let files = inventory_files(&dir, upstream.inventory);
        let hits = usage_lines(&dir, &files);
        if hits.is_empty() {
            out.push_str("(no mx.compile or mx.fast usage)\n");
        } else {
            for hit in &hits {
                out.push_str(&format!("- `{hit}`\n"));
            }
        }
    }
    out.push_str("\n## speakerdesk (in-repo)\n\n");
    let files = inventory_files(root, LOCAL_INVENTORY);
    let hits = usage_lines(root, &files);
    if hits.is_empty() {
        out.push_str("(no mx.compile or mx.fast usage)\n");
    } else {
        for hit in &hits {
            out.push_str(&format!("- `{hit}`\n"));
        }
    }
    let path = base.join("mlx-usage.md");
    std::fs::write(&path, out).with_context(|| format!("cannot write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}
