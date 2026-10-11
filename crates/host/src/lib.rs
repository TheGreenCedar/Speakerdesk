//! Speakerdesk host: the axum app, auth/headers, static assets,
//! admission/update reservation, setup downloads, `InferenceSupervisor`
//! (implements `Models` over IPC), the capture supervisor, the meeting manager
//! and the job worker.
//!
//! The `speakerdesk-dev` binary serves browser dev mode and test fixtures.
