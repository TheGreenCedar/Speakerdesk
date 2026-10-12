pub mod ci_changes;
pub mod fake_capture;
pub mod ownership;
pub mod pending;
pub mod probe;
pub mod python_free;
pub mod verify_bundle;

pub fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}
