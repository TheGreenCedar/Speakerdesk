use std::fs;

use anyhow::{Context, bail};

use super::repo_root;

pub const TAURI_SHELL: &str = "desktop/src-tauri/src/main.rs";

/// Paths that still start the Python sidecar, given the Tauri shell source.
pub fn sidecar_spawn_paths(source: &str) -> Vec<&'static str> {
    if source.contains("tauri_plugin_shell") && source.contains(".sidecar(") {
        vec![TAURI_SHELL]
    } else {
        Vec::new()
    }
}

pub fn run() -> anyhow::Result<()> {
    let path = repo_root().join(TAURI_SHELL);
    let source = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let hits = sidecar_spawn_paths(&source);
    if hits.is_empty() {
        println!("python-free: {TAURI_SHELL}");
        return Ok(());
    }
    bail!(
        "check-python-free failed: Python sidecar spawn remains in {}",
        hits.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::{TAURI_SHELL, run, sidecar_spawn_paths};

    #[test]
    fn reports_tauri_shell_when_fixture_starts_plugin_sidecar() {
        let fixture = r#"
            use tauri_plugin_shell::{ShellExt, process::CommandEvent};
            let (mut rx, child) = app
                .shell()
                .sidecar("speakerdesk-runtime")
                .spawn()?;
        "#;
        let hits = sidecar_spawn_paths(fixture);
        assert!(
            hits.contains(&TAURI_SHELL),
            "expected {TAURI_SHELL} in {hits:?}"
        );
    }

    #[test]
    fn ignores_source_without_sidecar_startup() {
        assert!(sidecar_spawn_paths("fn main() {}\n").is_empty());
    }

    #[test]
    fn workspace_check_names_the_tauri_shell() {
        let error = run().expect_err("sidecar spawn still remains");
        assert!(error.to_string().contains(TAURI_SHELL), "{error}");
    }
}
