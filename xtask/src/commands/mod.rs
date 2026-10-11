pub mod fake_capture;
pub mod ledger;
pub mod ownership;
pub mod pending;

pub fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}
