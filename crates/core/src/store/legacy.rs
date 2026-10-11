use std::path::Path;

use crate::error::StoreError;

/// One-way import into a new database. The legacy file is opened read-only.
/// The store agent fills this in. A failed import must delete only the partial file.
pub fn import_legacy(_legacy_db: &Path, _destination: &Path) -> Result<(), StoreError> {
    Err(StoreError::Unimplemented)
}
