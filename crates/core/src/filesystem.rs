//! Durable publication of owned, already-flushed files and prepared directories.
use std::{fs, io, path::Path};

pub fn sync_file(path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    let file = fs::OpenOptions::new().write(true).open(path)?;
    #[cfg(not(windows))]
    let file = fs::File::open(path)?;
    file.sync_all()
}

pub fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(not(windows))]
    {
        fs::File::open(path)?.sync_all()
    }
    #[cfg(windows)]
    {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Windows directory flushing is unavailable; publish with a write-through rename",
        ))
    }
}

/// Publish without replacing an existing destination. The source must be flushed first.
/// Unix retains the source hard link; Windows consumes the source. Callers clean up either case.
pub fn publish_new(source: &Path, destination: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        move_windows(source, destination, false)
    }
    #[cfg(not(windows))]
    {
        fs::hard_link(source, destination)?;
        sync_parent(destination)
    }
}

/// Replace a file or move a directory to a new name, on the same volume on Windows.
/// File contents and prepared directory contents must be flushed before publication.
pub fn rename(source: &Path, destination: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        move_windows(source, destination, !fs::symlink_metadata(source)?.is_dir())
    }
    #[cfg(not(windows))]
    {
        fs::rename(source, destination)?;
        sync_parent(destination)?;
        if source.parent() != destination.parent() {
            sync_parent(source)?;
        }
        Ok(())
    }
}

#[cfg(not(windows))]
fn sync_parent(path: &Path) -> io::Result<()> {
    sync_directory(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )
}

#[cfg(windows)]
fn move_windows(source: &Path, destination: &Path, replace: bool) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    let wide = |path: &Path| -> io::Result<Vec<u16>> {
        let name = path.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Publication path has no file name",
            )
        })?;
        // Canonicalizing only the parent retains leaf-link semantics and the long-path prefix.
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let path = fs::canonicalize(parent)?.join(name);
        let mut path: Vec<_> = path.as_os_str().encode_wide().collect();
        if path.contains(&0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Path contains NUL",
            ));
        }
        path.push(0);
        Ok(path)
    };
    let source = wide(source)?;
    let destination = wide(destination)?;
    let flags = MOVEFILE_WRITE_THROUGH
        | if replace {
            MOVEFILE_REPLACE_EXISTING
        } else {
            0
        };
    // No COPY_ALLOWED: cross-volume publication must fail rather than degrade to copy/delete.
    // https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-movefileexw
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
