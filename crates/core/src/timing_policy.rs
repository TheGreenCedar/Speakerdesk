//! Frozen identifiers from `alignment_text.py`. The timing agent must not rename them.
pub const PROVIDER_ID: &str = "omnilingual-ctc-mlx-supplied-text-v2";
pub const NORMALIZATION_POLICY_ID: &str = "omni_ctc_attached_ar_marks_ja_separator_v1";
pub const TIMING_UNIT_POLICY_ID: &str = "mixed_script_ctc_character_whitespace_v1";
pub const LEGACY_UNIT_POLICY_ID: &str = "whitespace_runs_v1";
pub const FUNCTIONAL_POLICY_ID: &str = "omni-ctc-mlx-f32-functional-v2-38ff6a225e75";
pub const UPSTREAM_NORMALIZATION_REVISION: &str = "81f51e224ce9e74b02cc2a3eaf21b2d91d743455";
pub const CTC_WEIGHTS_DIR: &str = "mlx-f32-st-v1";

/// Closed score-model identities. Functional timing support is separate from
/// measured timing accuracy; the Windows candidate has no calibrated language.
#[derive(Debug, PartialEq, Eq)]
pub struct Descriptor {
    pub model_sha256: &'static str,
    pub checkpoint: &'static str,
    pub provider_id: &'static str,
    pub provider_identity_sha256: &'static str,
    pub candidate_runtime: &'static str,
    pub functional_policy_id: &'static str,
    pub accuracy_languages: &'static [&'static str],
}

pub const MAC: Descriptor = Descriptor {
    model_sha256: crate::timing::MODEL,
    checkpoint: crate::alignment::CHECKPOINT,
    provider_id: PROVIDER_ID,
    provider_identity_sha256: crate::alignment::NATIVE_PROVIDER,
    candidate_runtime: "rust_mlx_dequantized_float32_GPU",
    functional_policy_id: FUNCTIONAL_POLICY_ID,
    accuracy_languages: &["en"],
};

pub const WINDOWS: Descriptor = Descriptor {
    model_sha256: "233ac8980dab9c142ef9fe91ab1d3b5d6309a9549f59a284e50199f7b1081699",
    checkpoint: "omnilingual-ctc-windowsml-f32-v1-233ac8980dab",
    provider_id: "omnilingual-ctc-windowsml-supplied-text-v1",
    // SHA-256 of provider/model/runtime/functional-policy strings, newline joined.
    provider_identity_sha256: "b69323bac5488fc9908ff0849629ef91937411633ed3de2cb09932d8e15c93b1",
    candidate_runtime: "rust_windowsml_2.4.89_directml_float32",
    functional_policy_id: "omni-ctc-windowsml-f32-functional-v1-233ac8980dab",
    accuracy_languages: &[],
};

pub fn descriptor_for_model(model: &str) -> Option<&'static Descriptor> {
    [&MAC, &WINDOWS]
        .into_iter()
        .find(|descriptor| descriptor.model_sha256 == model)
}

impl Descriptor {
    pub fn calibration_for_language(&self, language: &str) -> &'static str {
        if self.accuracy_languages.contains(&language) {
            crate::timing::CALIBRATION
        } else {
            self.functional_policy_id
        }
    }

    pub(crate) fn accepts_calibration(&self, calibration: &str) -> bool {
        calibration == self.functional_policy_id
            || (self.model_sha256 == MAC.model_sha256 && calibration == crate::timing::CALIBRATION)
    }
}
