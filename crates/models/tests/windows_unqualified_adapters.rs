#![cfg(windows)]

use speakerdesk_models::windows_runtime::verify_model;
use std::path::Path;

#[test]
fn unqualified_ctc_and_whisper_cannot_acquire_model_admission() {
    for id in ["coarse-alignment", "whisper-language"] {
        let error = verify_model(Path::new("not-an-installed-model"), id)
            .expect_err("unqualified Windows graph acquired artifact admission");
        assert!(
            error
                .to_string()
                .contains(&format!("No admitted Windows artifact for {id}")),
            "unexpected admission failure: {error}"
        );
    }
}
