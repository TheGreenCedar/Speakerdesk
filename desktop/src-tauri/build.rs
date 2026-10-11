use std::path::{Path, PathBuf};

fn main() {
    provision_dev_resources();
    tauri_build::build()
}

// `tauri_build` requires every `bundle.externalBin` and `bundle.resources`
// path to exist, but the PyInstaller sidecar and the Swift capture helper are
// only produced by scripts/build_macos.sh. For plain cargo builds and
// `tauri dev`, write dev stand-ins: the sidecar stub delegates to the Python
// app exactly like scripts/run_local.sh, and the helper stub reports itself
// as unavailable. The packaging script overwrites them with the real
// artifacts before bundling.
fn provision_dev_resources() {
    let Ok(triple) = std::env::var("TARGET") else {
        return;
    };
    // The sidecar may run from binaries/ or from the target dir after tauri
    // dev copies it, so bake in the absolute repo root.
    let root = Path::new(&std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default())
        .parent()
        .and_then(Path::parent)
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let sidecar = PathBuf::from("binaries").join(format!("speakerdesk-runtime-{triple}"));
    write_if_missing(
        &sidecar,
        &format!(
            "#!/bin/sh\n\
             ROOT=\"{root}\"\n\
             PYTHON=\"$ROOT/.venv-package/bin/python\"\n\
             export DIAR_PYTHON=\"$PYTHON\" ASR_PYTHON=\"$PYTHON\" SPEAKERDESK_LIVE_PYTHON=\"$PYTHON\"\n\
             export SPEAKERDESK_MODELS=\"$ROOT/models\"\n\
             exec \"$PYTHON\" \"$ROOT/speakerdesk/app.py\"\n"
        ),
    );
    let helper = PathBuf::from("../capture/speakerdesk-capture");
    write_if_missing(
        &helper,
        "#!/bin/sh\n\
         echo 'speakerdesk-capture is built by scripts/build_macos.sh' >&2\n\
         exit 1\n",
    );
    write_if_missing(
        &PathBuf::from("../capture/speakerdesk-capture.capability.json"),
        "{\"helper_sha256\":\"\",\"export_contract\":\"\",\"capability_version\":\"\"}\n",
    );
}

fn write_if_missing(path: &Path, contents: &str) {
    if path.exists() {
        return;
    }
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() || std::fs::write(path, contents).is_err() {
        return;
    }
    #[cfg(unix)]
    if path.extension().is_none() {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755));
    }
}
