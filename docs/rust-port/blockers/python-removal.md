# Python removal

`desktop/src-tauri/src/main.rs` still launches the Python sidecar. `start_runtime` spawns `speakerdesk-runtime` through `tauri_plugin_shell` and still parses `SPEAKERDESK_UPDATE=` and `SPEAKERDESK_URL=` from that process's stdout.

Do not delete `speakerdesk/*.py`, `tests/*.py`, the requirements locks, or the sidecar spawn until that launch is gone. `cargo xtask check-python-free` fails and names `desktop/src-tauri/src/main.rs` while the spawn remains.
