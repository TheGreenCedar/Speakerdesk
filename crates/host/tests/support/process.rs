pub struct ObservedProcess {
    #[cfg(unix)]
    pub pid: libc::pid_t,
    #[cfg(windows)]
    handle: std::os::windows::io::OwnedHandle,
}

impl ObservedProcess {
    pub fn new(pid: u32) -> Self {
        #[cfg(unix)]
        {
            Self {
                pid: pid.try_into().unwrap(),
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::FromRawHandle;
            use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_SYNCHRONIZE};
            // Retain the running process identity so PID reuse cannot satisfy the exit check.
            let handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) };
            assert!(
                !handle.is_null(),
                "cannot observe child {pid}: {}",
                std::io::Error::last_os_error()
            );
            Self {
                // OpenProcess returned a handle owned by this observer.
                handle: unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(handle) },
            }
        }
    }

    pub fn has_exited(&self) -> bool {
        #[cfg(unix)]
        {
            if unsafe { libc::kill(self.pid, 0) } == 0 {
                return false;
            }
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ESRCH)
            );
            true
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::{
                Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT},
                System::Threading::WaitForSingleObject,
            };
            // The retained process handle stays valid until this observer is dropped.
            match unsafe { WaitForSingleObject(self.handle.as_raw_handle(), 0) } {
                WAIT_OBJECT_0 => true,
                WAIT_TIMEOUT => false,
                _ => panic!(
                    "cannot inspect child exit: {}",
                    std::io::Error::last_os_error()
                ),
            }
        }
    }
}
