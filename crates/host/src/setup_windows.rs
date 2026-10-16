//! Windows cannot exchange nonempty directories. Retain the old tree across both renames.
//! The setup lock and application setup lease exclude other writers and model work.
use speakerdesk_core::filesystem::rename;
use std::{fs, io, path::Path};

fn directory_exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() && !meta.is_symlink() => Ok(true),
        Ok(_) => Err(io::Error::other(
            "Model replacement path is linked or not a directory",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

pub(super) fn recover(root: &Path, ids: &[String]) -> io::Result<()> {
    let previous = root.join(".previous");
    if !directory_exists(&previous)? {
        return Ok(());
    }
    for id in ids {
        let name = super::relative(id)?;
        if name.components().count() != 1 {
            return Err(io::Error::other("Invalid model identity"));
        }
        let backup = previous.join(name);
        if !directory_exists(&backup)? {
            continue;
        }
        let target = root.join(name);
        if !directory_exists(&target)? {
            // An interruption between renames must restore the previous installation before load.
            rename(&backup, &target)?;
        }
        // A published target wins; keep its backup until a verified repair can safely remove it.
    }
    Ok(())
}

pub(super) fn replace(source: &Path, target: &Path) -> io::Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| io::Error::other("Missing model root"))?;
    let previous = parent.join(".previous");
    if !directory_exists(&previous)? {
        fs::create_dir(&previous)?;
    }
    let backup = previous.join(
        target
            .file_name()
            .ok_or_else(|| io::Error::other("Missing model identity"))?,
    );
    if directory_exists(&backup)? {
        // The caller has already verified and flushed the new tree, so an older backup is expendable.
        fs::remove_dir_all(&backup)?;
    }
    rename(target, &backup)?;
    if let Err(publication) = rename(source, target) {
        return match rename(&backup, target) {
            Ok(()) => Err(publication),
            Err(rollback) => Err(io::Error::other(format!(
                "Model publication failed: {publication}; restoring the previous model failed: {rollback}. Restart to recover the retained model."
            ))),
        };
    }
    fs::remove_dir_all(backup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_replacement_recovers_without_overwriting_a_published_tree() {
        let root = tempfile::tempdir().unwrap();
        let active = root.path().join("fixture");
        let backup = root.path().join(".previous/fixture");
        fs::create_dir_all(&backup).unwrap();
        fs::write(backup.join("weights"), b"previous").unwrap();
        recover(root.path(), &["fixture".into()]).unwrap();
        assert_eq!(fs::read(active.join("weights")).unwrap(), b"previous");
        assert!(!backup.exists());

        fs::create_dir_all(&backup).unwrap();
        fs::write(backup.join("weights"), b"older").unwrap();
        fs::write(active.join("weights"), b"published").unwrap();
        recover(root.path(), &["fixture".into()]).unwrap();
        assert_eq!(fs::read(active.join("weights")).unwrap(), b"published");
        assert_eq!(fs::read(backup.join("weights")).unwrap(), b"older");

        // A failed second rename rolls the current tree back without consuming a new staging tree.
        assert!(replace(&root.path().join("missing-staging"), &active).is_err());
        assert_eq!(fs::read(active.join("weights")).unwrap(), b"published");
        assert!(!backup.exists());
        fs::create_dir_all(&backup).unwrap();
        fs::remove_dir_all(&active).unwrap();
        fs::write(&active, b"occupied").unwrap();
        assert!(recover(root.path(), &["fixture".into()]).is_err());
        assert!(backup.is_dir());
    }
}
