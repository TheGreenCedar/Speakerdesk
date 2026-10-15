#![cfg(windows)]

use speakerdesk_models::windows_runtime::verify_model;
use std::path::Path;

#[test]
fn qualification_inventory_does_not_advertise_production_readiness() {
    for id in ["coarse-alignment", "whisper-language"] {
        assert!(verify_model(Path::new("not-an-installed-model"), id).is_err());
    }
    use speakerdesk_core::ipc::{Request, Response};
    let reply = speakerdesk_models::dispatch::dispatch(
        Request::Hello {
            build_id: "qualification".into(),
        },
        &std::sync::atomic::AtomicBool::new(false),
    )
    .unwrap();
    let Response::Ready { device, models, .. } = reply else {
        panic!("expected readiness")
    };
    assert_eq!(device, "unavailable");
    assert!(models.is_empty());
}
