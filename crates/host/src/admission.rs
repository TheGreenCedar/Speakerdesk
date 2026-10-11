//! Update admission leases. The stdout `SPEAKERDESK_UPDATE=` frame is not used.
#[derive(Debug, Default)]
pub struct Admission;

impl Admission {
    pub fn new() -> Self {
        Self
    }
}
