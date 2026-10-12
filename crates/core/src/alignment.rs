//! Supplied-text timing. Model scores never replace the authoritative text.
use crate::{
    error::ChildError, ipc::CtcScoresOut, languages, memo::sha256_hex, models::Models, timing,
    timing_policy::*,
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use unicode_general_category::get_general_category;
use unicode_normalization::{UnicodeNormalization, char::canonical_combining_class};

pub const TOKENS_SHA256: &str = "a7a044c52cb29cbe8b0dc1953e92cefd4ca16b0ed968177b6beab21f9a7d0b31";
pub const CHECKPOINT: &str = "omnilingual-ctc-native-f32-v1-38ff6a225e75";
/// Runtime-owned identity, deliberately distinct from the historical Python source digest.
pub const NATIVE_PROVIDER: &str =
    "e82ab4bda54b01ce182babb0ac6b40e7fc1ec37095784c171e447d04e13221e0";
pub const MAX_SAMPLES: usize = 392000;

/// Attach timing to an unpublished candidate aligned from this exact crop.
pub fn attach(segment: &mut crate::document::Segment, alignment: Value) {
    use timing::{AudioAnchor, ReadingRow, ReadingWordEvidence, SpeakerActivity, TimedWord};
    let start = (segment.start * 16000.).round() as i64;
    let end = (segment.end * 16000.).round() as i64;
    let machine_revision = segment.machine_revision.unwrap_or(0);
    let audio_revision = segment.audio_revision.unwrap_or(0);
    let canonical_revision = segment
        .rest
        .get("canonical_machine_revision")
        .and_then(Value::as_i64)
        .unwrap_or(machine_revision);
    let epoch = segment
        .rest
        .get("language_epoch")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let identity = segment
        .canonical_utterance_id
        .clone()
        .unwrap_or_else(|| segment.id.clone());
    let anchor = json!({"start_sample":start,"end_sample":end});
    if alignment["raw_text"] != segment.text
        || alignment["text_sha256"] != sha256_hex(segment.text.as_bytes())
        || alignment["audio_anchor"] != anchor
        || alignment["provider_identity_sha256"] != NATIVE_PROVIDER
        || alignment["model_sha256"] != timing::MODEL
    {
        return;
    }
    segment.rest.insert(
        "audio_float32_sha256".into(),
        alignment["audio_float32_sha256"].clone(),
    );
    segment
        .rest
        .insert("reading_alignment".into(), alignment.clone());
    let Some(activity) = segment.speaker_activity.clone() else {
        return;
    };
    let Ok(activity) = serde_json::from_value::<SpeakerActivity>(activity) else {
        return;
    };
    let Some(words) = alignment["words"].as_array() else {
        return;
    };
    let words: Option<Vec<TimedWord>> = words
        .iter()
        .map(|w| {
            Some(TimedWord {
                text: w["text"].as_str()?.into(),
                start_char: w["start_char"].as_u64()? as usize,
                end_char: w["end_char"].as_u64()? as usize,
                start_sample: if w["status"] == "aligned" {
                    w["start_sample"].as_i64()
                } else {
                    None
                },
                end_sample: if w["status"] == "aligned" {
                    w["end_sample"].as_i64()
                } else {
                    None
                },
            })
        })
        .collect();
    let Some(words) = words else {
        return;
    };
    let row = ReadingRow {
        text: segment.text.clone(),
        start_sample: start,
        end_sample: end,
        machine_revision,
        audio_revision,
        protected_fields: segment.protected_fields.clone().unwrap_or_default(),
        text_audio_anchor: Some(AudioAnchor {
            start_sample: start,
            end_sample: end,
        }),
        reading_word_evidence: Some(ReadingWordEvidence {
            text_sha256: sha256_hex(segment.text.as_bytes()),
            machine_revision,
            audio_revision,
            retained_audio_revision: None,
            audio_anchor: AudioAnchor {
                start_sample: start,
                end_sample: end,
            },
            model_sha256: timing::MODEL.into(),
            calibration_id: alignment["score_calibration_id"]
                .as_str()
                .map(str::to_owned),
            frame_calibration_id: alignment["frame_calibration_id"]
                .as_str()
                .map(str::to_owned),
            timing_unit_policy_id: Some(TIMING_UNIT_POLICY_ID.into()),
            provider_id: Some(PROVIDER_ID.into()),
            provider_identity_sha256: Some(NATIVE_PROVIDER.into()),
            normalization_policy_id: Some(NORMALIZATION_POLICY_ID.into()),
            words,
        }),
        speaker_activity: Some(activity.clone()),
        installed_provider_identity: Some(NATIVE_PROVIDER.into()),
    };
    let turns = timing::project_turns(&row);
    if turns.is_empty() {
        return;
    }
    let activity_json = timing::activity_json(Some(&activity));
    let turns_json = timing::turns_json(&turns);
    segment.reading_turns = Some(serde_json::from_str(&turns_json).expect("serialized turns"));
    segment.speaker_activity =
        Some(serde_json::from_str(&activity_json).expect("serialized activity"));
    segment.canonical_utterance_id = Some(identity.clone());
    segment.machine_revision = Some(machine_revision);
    segment.audio_revision = Some(audio_revision);
    segment.rest.insert(
        "canonical_machine_revision".into(),
        json!(canonical_revision),
    );
    segment.rest.insert("language_epoch".into(), json!(epoch));
    segment
        .rest
        .insert("text_audio_anchor".into(), anchor.clone());
    segment.reading_turn_provenance = Some(
        json!({"utterance_id":identity,"canonical_machine_revision":canonical_revision,"machine_revision":machine_revision,"audio_revision":audio_revision,"language_epoch":epoch,"capture_source":segment.capture_source,
        "text_sha256":sha256_hex(segment.text.as_bytes()),"audio_anchor":anchor,"activity_sha256":sha256_hex(activity_json.as_bytes()),"turns_sha256":sha256_hex(turns_json.as_bytes()),
        "method":"ctc_units_temporal_nvidia_v3","transition_margin_samples":4000,"uncertainty_policy":"emission_inside_activity_no_competing_owner_in_margin",
        "model_sha256":timing::MODEL,"calibration_id":alignment["score_calibration_id"],"frame_calibration_id":alignment["frame_calibration_id"],
        "timing_unit_policy_id":TIMING_UNIT_POLICY_ID,"provider_id":PROVIDER_ID,"provider_identity_sha256":NATIVE_PROVIDER,"normalization_policy_id":NORMALIZATION_POLICY_ID}),
    );
}

pub struct Vocabulary {
    tokens: Vec<String>,
    lookup: HashMap<char, i32>,
}
impl Vocabulary {
    pub fn parse(contents: &str) -> Result<Self, String> {
        if sha256_hex(contents.as_bytes()) != TOKENS_SHA256 {
            return Err("CTC vocabulary digest differs".into());
        }
        Self::parse_tokens(contents)
    }
    fn parse_tokens(contents: &str) -> Result<Self, String> {
        let mut tokens = Vec::new();
        let mut seen = HashSet::new();
        for line in contents.lines() {
            let (token, id) = line.rsplit_once(' ').ok_or("invalid CTC vocabulary line")?;
            if token.is_empty()
                || id.parse::<usize>().ok() != Some(tokens.len())
                || !seen.insert(token)
            {
                return Err("CTC vocabulary must contain unique contiguous tokens".into());
            }
            tokens.push(token.to_string());
        }
        if tokens.len() < 5 || tokens[..5] != ["<s>", "<pad>", "</s>", "<unk>", " "] {
            return Err("invalid CTC reserved tokens".into());
        }
        let lookup = tokens
            .iter()
            .enumerate()
            .skip(4)
            .filter_map(|(id, token)| {
                let mut chars = token.chars();
                let c = chars.next()?;
                chars.next().is_none().then_some((c, id as i32))
            })
            .collect();
        Ok(Self { tokens, lookup })
    }

    pub fn align<M: Models + ?Sized>(
        &self,
        models: &M,
        pcm: &[f32],
        text: &str,
        start: i64,
        language: &str,
    ) -> Result<Value, ChildError> {
        if start < 0
            || pcm.is_empty()
            || pcm.len() > MAX_SAMPLES
            || pcm.iter().any(|v| !v.is_finite())
            || start.checked_add(pcm.len() as i64).is_none()
        {
            return Err(ChildError::failed(
                "invalid supplied-text audio range",
                false,
            ));
        }
        let prepared = self
            .prepare(text, language)
            .map_err(|e| ChildError::failed(e, false))?;
        let mut result = base(&prepared, start, pcm.len(), language);
        if !prepared.unsupported.is_empty() {
            result["reason"] = json!("unsupported_text");
        } else if !prepared.targets.iter().any(|t| t.character != ' ') {
            result["reason"] = json!("no_acoustic_target");
        } else {
            let mut ids = vec![0];
            for target in &prepared.targets {
                if !ids.contains(&target.id) {
                    ids.push(target.id);
                }
            }
            let scores = models.ctc_scores(pcm, &ids)?;
            self.materialize(&prepared, &scores, &mut result);
        }
        result["audio_float32_sha256"] = json!(pcm_sha256(pcm));
        Ok(result)
    }

    // Context::parse checks the acoustic cells; this checks their current input
    // identity and coverage against this vocabulary before saved-crop reuse.
    pub(crate) fn matches_input(
        &self,
        alignment: &Value,
        pcm: &[f32],
        text: &str,
        start: i64,
        language: &str,
    ) -> bool {
        if pcm.is_empty() || pcm.len() > MAX_SAMPLES || pcm.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let Ok(prepared) = self.prepare(text, language) else {
            return false;
        };
        if !prepared.unsupported.is_empty() {
            return false;
        }
        let Some(valid_frames) = pcm.len().checked_sub(400).map(|samples| samples / 320 + 1) else {
            return false;
        };
        let expected = base(&prepared, start, pcm.len(), language);
        if [
            "schema_version",
            "text_source",
            "offset_unit",
            "timing_kind",
            "language",
            "unsupported",
            "acoustic_aliases",
            "upstream_normalization_revision",
            "normalization_spans",
            "candidate_runtime",
            "qualified_scope",
            "timing_accuracy_calibrated_languages",
            "non_speech_proof",
            "linguistic_word_claim",
        ]
        .iter()
        .any(|key| alignment[*key] != expected[*key])
            || alignment["alignment_backend"] != "speakerdesk-rust-ctc-segmentation-1.7.4"
            || alignment["audio_float32_sha256"] != pcm_sha256(pcm)
        {
            return false;
        }
        let Some(characters) = alignment["characters"].as_array() else {
            return false;
        };
        let Some(words) = alignment["words"].as_array() else {
            return false;
        };
        let expected_words = expected["words"].as_array().expect("base words");
        characters.len() == prepared.targets.len()
            && characters
                .iter()
                .zip(&prepared.targets)
                .all(|(cell, target)| {
                    let separator = prepared.aliases.iter().any(|alias| {
                        alias["kind"] == "upstream_japanese_separator_alias"
                            && target.a <= alias["start_char"].as_u64().unwrap() as usize
                            && alias["end_char"].as_u64().unwrap() as usize <= target.b
                    });
                    cell["token_id"] == target.id
                        && cell["frame"]
                            .as_u64()
                            .is_some_and(|frame| frame < valid_frames as u64)
                        && cell["text"] == target.character.to_string()
                        && cell["start_char"] == target.a
                        && cell["end_char"] == target.b
                        && cell["start_utf16"] == utf16(&prepared.raw, target.a)
                        && cell["end_utf16"] == utf16(&prepared.raw, target.b)
                        && (cell["independent_acoustic_timestamp"] == false) == separator
                })
            && words.len() == expected_words.len()
            && words.iter().zip(expected_words).all(|(word, expected)| {
                [
                    "text",
                    "start_char",
                    "end_char",
                    "start_utf16",
                    "end_utf16",
                    "unit_kind",
                ]
                .iter()
                .all(|key| word[*key] == expected[*key])
            })
    }

    fn prepare(&self, text: &str, language: &str) -> Result<Prepared, String> {
        let raw: Vec<char> = text.chars().collect();
        if raw.len() > 4096 || !languages::is_supported(language) {
            return Err("unsupported supplied-text timing request".into());
        }
        let mut targets: Vec<Target> = Vec::new();
        let mut unsupported = Vec::new();
        let mut aliases = Vec::new();
        for Scalar { mut c, a, b } in normalized(&raw) {
            let category = get_general_category(c).abbreviation().as_bytes()[0];
            if arabic_mark(c) || c == 'ー' {
                let base = targets
                    .iter()
                    .rev()
                    .find(|t| t.character != ' ' && t.b <= a);
                let arabic = base.is_some_and(|t| {
                    arabic_mark(c)
                        && arabic_letter(t.character)
                        && raw[t.b..a].iter().all(|c| arabic_mark(*c))
                });
                let japanese = base.is_some_and(|t| {
                    c == 'ー' && kana(t.character) && raw[t.b..a].iter().all(|c| *c == 'ー')
                });
                if !arabic && !japanese {
                    unsupported.push(json!({"start_char":a,"end_char":b,"text":slice(&raw,a,b)}));
                    continue;
                }
                aliases.push(json!({"start_char":a,"end_char":b,"raw_text":slice(&raw,a,b),
                    "acoustic_token":if japanese { Some(" ") } else { None },
                    "kind":if japanese {"upstream_japanese_separator_alias"} else {"upstream_deleted_arabic_mark"},
                    "independent_acoustic_timestamp":false,"profile_id":NORMALIZATION_POLICY_ID}));
                if arabic {
                    continue;
                }
                c = ' ';
            }
            if c.is_whitespace() {
                if let Some(last) = targets.last_mut().filter(|t| t.character == ' ') {
                    last.b = b;
                    continue;
                }
                c = ' ';
            } else if category == b'P' {
                continue;
            }
            if !self.lookup.contains_key(&c) {
                let lower: Vec<char> = c.to_lowercase().collect();
                if lower.len() == 1 && lower[0] != c && self.lookup.contains_key(&lower[0]) {
                    c = lower[0];
                    aliases.push(json!({"start_char":a,"end_char":b,"raw_text":slice(&raw,a,b),"acoustic_token":c.to_string(),"kind":"supported_single_character_lowercase_alias"}));
                }
            }
            if let Some(&id) = self.lookup.get(&c) {
                targets.push(Target {
                    id,
                    character: c,
                    a,
                    b,
                });
            } else {
                unsupported.push(json!({"start_char":a,"end_char":b,"text":slice(&raw,a,b)}));
            }
        }
        if targets.len() > 1024 {
            return Err("too many supplied CTC targets".into());
        }
        while targets.last().is_some_and(|t| t.character == ' ') {
            targets.pop();
        }
        let first = targets
            .iter()
            .position(|t| t.character != ' ')
            .unwrap_or(targets.len());
        targets.drain(..first);
        Ok(Prepared {
            text: text.into(),
            raw,
            targets,
            unsupported,
            aliases,
        })
    }

    fn materialize(&self, p: &Prepared, scores: &CtcScoresOut, result: &mut Value) {
        let frames = scores.valid_frames as usize;
        let width = scores.token_ids.len();
        let n = (result["audio_anchor"]["end_sample"].as_i64().unwrap()
            - result["audio_anchor"]["start_sample"].as_i64().unwrap()) as usize;
        let valid = if n < 400 { 0 } else { (n - 400) / 320 + 1 };
        let columns: HashMap<i32, usize> = scores
            .token_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        if frames != valid
            || scores.audio_num_samples as usize != n
            || scores.frame_hop_samples != 320
            || scores.frame_origin_samples != 0
            || scores.model != timing::MODEL
            || scores.vocabulary_sha256 != TOKENS_SHA256
            || width == 0
            || columns.len() != width
            || !columns.contains_key(&0)
            || scores.gathered.len() != frames * width
            || scores
                .token_ids
                .iter()
                .any(|id| *id < 0 || *id as usize >= self.tokens.len())
            || p.targets.iter().any(|t| !columns.contains_key(&t.id))
            || scores.gathered.iter().any(|v| !v.is_finite() || *v > 0.)
            || scores
                .gathered
                .chunks(width.max(1))
                .any(|row| row.iter().map(|v| f64::from(*v).exp()).sum::<f64>() > 0.001f64.exp())
        {
            result["reason"] = json!("invalid_ctc_scores");
            return;
        }
        if frames == 0 {
            result["reason"] = json!("insufficient_acoustic_context");
            return;
        }
        let mut expanded = Vec::new();
        let mut positions = Vec::new();
        let mut separators = HashMap::new();
        let blank = columns[&0] as i64;
        for (i, target) in p.targets.iter().enumerate() {
            if i > 0 && target.id == p.targets[i - 1].id {
                separators.insert(i, 2 + expanded.len());
                expanded.push(blank);
            }
            positions.push(2 + expanded.len());
            expanded.push(columns[&target.id] as i64);
        }
        let mut config = timing::CtcSegmentationParameters::for_vocabulary(
            scores
                .token_ids
                .iter()
                .map(|id| self.tokens[*id as usize].clone())
                .collect(),
        );
        config.blank = blank;
        let ground = timing::prepare_token_list(blank, &[expanded]);
        let matrix: Vec<Vec<f32>> = scores
            .gathered
            .chunks_exact(width)
            .map(<[f32]>::to_vec)
            .collect();
        let Ok(aligned) = timing::ctc_segmentation(&config, &matrix, &ground.ground_truth) else {
            result["reason"] = json!("ctc_alignment_failed");
            return;
        };
        let frame_at = |position: usize, token: &str| -> Option<usize> {
            let &time = aligned.timings.get(position)?;
            if !time.is_finite() || time < 0. || (time - time.round()).abs() > 1e-7 {
                return None;
            }
            let frame = time.round() as usize;
            (aligned
                .states
                .get(frame)
                .is_some_and(|state| state == token)
                && frame < frames)
                .then_some(frame)
        };
        let mut characters = Vec::new();
        let mut previous = None;
        let start = result["audio_anchor"]["start_sample"].as_i64().unwrap();
        for (i, target) in p.targets.iter().enumerate() {
            let Some(frame) = frame_at(positions[i], &target.character.to_string())
                .filter(|f| previous.is_none_or(|last| *f > last))
            else {
                result["reason"] = json!("invalid_anchor_sequence");
                return;
            };
            let score = matrix[frame][columns[&target.id]];
            let repeated = separators.get(&i).is_none_or(|position| {
                frame_at(*position, &self.tokens[0]).is_some_and(|b| {
                    previous.is_some_and(|last| last < b)
                        && b < frame
                        && matrix[b][blank as usize] >= -20.
                })
            });
            let accepted = score >= -20. && repeated;
            let a = start + frame as i64 * 320;
            let b = a + 320;
            let separator = p.aliases.iter().any(|v| {
                v["kind"] == "upstream_japanese_separator_alias"
                    && target.a <= v["start_char"].as_u64().unwrap() as usize
                    && v["end_char"].as_u64().unwrap() as usize <= target.b
            });
            let mut value = json!({"text":target.character.to_string(),"token_id":target.id,"raw_text":slice(&p.raw,target.a,target.b),"start_char":target.a,"end_char":target.b,
                "start_utf16":utf16(&p.raw,target.a),"end_utf16":utf16(&p.raw,target.b),"frame":frame,"log_probability":score,
                "status":if accepted {"aligned"} else {"unresolved"},"start_sample":accepted.then_some(a),"end_sample":accepted.then_some(b)});
            if separator {
                value["independent_acoustic_timestamp"] = json!(false);
                value["acoustic_target_status"] = value["status"].clone();
                value["status"] = json!("not_independently_timed");
                value["target_cell_start_sample"] = value["start_sample"].take();
                value["target_cell_end_sample"] = value["end_sample"].take();
            }
            characters.push(value);
            previous = Some(frame);
        }
        for word in result["words"].as_array_mut().unwrap() {
            let a = word["start_char"].as_u64().unwrap();
            let b = word["end_char"].as_u64().unwrap();
            let contained = |v: &&Value| {
                a <= v["start_char"].as_u64().unwrap() && v["end_char"].as_u64().unwrap() <= b
            };
            let evidence: Vec<_> = characters
                .iter()
                .filter(contained)
                .filter(|v| v["independent_acoustic_timestamp"] != false)
                .collect();
            let aliases: Vec<_> = p.aliases.iter().filter(contained).cloned().collect();
            if !aliases.is_empty() {
                word["acoustic_aliases"] = json!(aliases);
            }
            if evidence.is_empty() {
                word["status"] = json!("not_acoustic");
            } else if evidence.iter().all(|v| v["status"] == "aligned") {
                word["status"] = json!("aligned");
                word["start_sample"] = evidence[0]["start_sample"].clone();
                word["end_sample"] = evidence.last().unwrap()["end_sample"].clone();
                word["minimum_token_log_probability"] = json!(
                    evidence
                        .iter()
                        .map(|v| v["log_probability"].as_f64().unwrap())
                        .fold(f64::INFINITY, f64::min)
                );
            }
        }
        let complete = characters
            .iter()
            .all(|c| c.get("acoustic_target_status").unwrap_or(&c["status"]) == "aligned")
            && result["words"]
                .as_array()
                .unwrap()
                .iter()
                .all(|w| w["status"] == "aligned");
        result["characters"] = json!(characters);
        result["complete"] = json!(complete);
        result["status"] = json!(if complete { "aligned" } else { "partial" });
        result["reason"] = if complete {
            Value::Null
        } else {
            json!("insufficient_acoustic_evidence")
        };
        result["alignment_backend"] = json!("speakerdesk-rust-ctc-segmentation-1.7.4");
    }
}

fn pcm_sha256(pcm: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(pcm.len() * 4);
    bytes.extend(pcm.iter().flat_map(|v| v.to_le_bytes()));
    sha256_hex(&bytes)
}

struct Target {
    id: i32,
    character: char,
    a: usize,
    b: usize,
}
struct Prepared {
    text: String,
    raw: Vec<char>,
    targets: Vec<Target>,
    unsupported: Vec<Value>,
    aliases: Vec<Value>,
}
#[derive(Clone)]
struct Scalar {
    c: char,
    a: usize,
    b: usize,
}

// Normalize only the open NFC segment; retain the reference's conservative spans
// when composition or reordered marks change an already-normalized suffix.
fn normalized(raw: &[char]) -> Vec<Scalar> {
    let mut result: Vec<Scalar> = Vec::new();
    let mut begin = 0;
    for (i, &c) in raw.iter().enumerate() {
        let tail: String = result[begin..]
            .iter()
            .map(|s| s.c)
            .chain(std::iter::once(c))
            .collect();
        let updated: Vec<char> = tail.nfc().collect();
        let common = result[begin..]
            .iter()
            .map(|s| s.c)
            .zip(&updated)
            .take_while(|(a, b)| a == *b)
            .count();
        let a = result[begin + common..]
            .iter()
            .map(|s| s.a)
            .min()
            .unwrap_or(i);
        result.truncate(begin + common);
        result.extend(updated[common..].iter().map(|&c| Scalar { c, a, b: i + 1 }));
        begin = result
            .iter()
            .rposition(|s| canonical_combining_class(s.c) == 0)
            .unwrap_or(0);
    }
    result
}
fn slice(raw: &[char], a: usize, b: usize) -> String {
    raw[a..b].iter().collect()
}
fn utf16(raw: &[char], end: usize) -> usize {
    raw[..end].iter().map(|c| c.len_utf16()).sum()
}
fn arabic_mark(c: char) -> bool {
    ('\u{064b}'..='\u{0652}').contains(&c)
}
fn arabic_letter(c: char) -> bool {
    matches!(c as u32, 0x61c|0x620..=0x63f|0x641..=0x64a|0x66e..=0x6d3|0x6d5|0x6ee..=0x6ef|0x6fa..=0x6fc|0x6ff|0x750..=0x77f|0x8a0..=0x8b4|0x8b6..=0x8c7|0xfb50..=0xfbb1|0xfbd3..=0xfbe9|0xfbfc..=0xfbff|0xfe80..=0xfef4)
}
fn kana(c: char) -> bool {
    c != 'ー' && matches!(c as u32,0x3041..=0x3096|0x30a1..=0x30fa|0xff66..=0xff9d)
}

pub(crate) fn calibration_for_language(language: &str) -> &'static str {
    if language == "en" {
        timing::CALIBRATION
    } else {
        FUNCTIONAL_POLICY_ID
    }
}

fn base(p: &Prepared, start: i64, samples: usize, language: &str) -> Value {
    let calibration = calibration_for_language(language);
    let words:Vec<_> = timing::raw_units(&p.text,TIMING_UNIT_POLICY_ID).expect("known policy").iter().map(|u| json!({"text":u.text,"start_char":u.start_char,"end_char":u.end_char,"start_utf16":u.start_utf16,"end_utf16":u.end_utf16,"unit_kind":u.unit_kind.as_str(),"status":"unresolved","start_sample":null,"end_sample":null})).collect();
    let spans: Vec<_> = p
        .aliases
        .iter()
        .filter(|v| v["kind"].as_str().unwrap().starts_with("upstream_"))
        .map(|v| {
            let mut v = v.clone();
            v["start_sample"] = Value::Null;
            v["end_sample"] = Value::Null;
            v["start_utf16"] = json!(utf16(&p.raw, v["start_char"].as_u64().unwrap() as usize));
            v["end_utf16"] = json!(utf16(&p.raw, v["end_char"].as_u64().unwrap() as usize));
            v
        })
        .collect();
    let english = language == "en"
        && spans.is_empty()
        && words.iter().all(|w| w["unit_kind"] == "whitespace_run");
    json!({"schema_version":1,"text_source":"cohere","raw_text":p.text,"text_sha256":sha256_hex(p.text.as_bytes()),"model_sha256":timing::MODEL,
        "offset_unit":"unicode_codepoint","timing_kind":"ctc_emission_cell_envelope","audio_anchor":{"start_sample":start,"end_sample":start+samples as i64},
        "status":"unresolved","complete":false,"reason":null,"words":words,"characters":[],"unsupported":p.unsupported,"acoustic_aliases":p.aliases,"non_speech_proof":false,
        "language":language,"normalization_policy_id":NORMALIZATION_POLICY_ID,"timing_unit_policy_id":TIMING_UNIT_POLICY_ID,"linguistic_word_claim":false,
        "upstream_normalization_revision":UPSTREAM_NORMALIZATION_REVISION,"normalization_spans":spans,"frame_calibration_id":calibration,"score_calibration_id":calibration,
        "provider_id":PROVIDER_ID,"provider_identity_sha256":NATIVE_PROVIDER,"candidate_runtime":"rust_mlx_dequantized_float32_GPU",
        "qualified_scope":if english {"bounded_ami_english_coarse_envelopes"} else {"multilingual_functional_ctc_units"},"timing_accuracy_calibrated_languages":if english {vec!["en"]} else {vec![]}})
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use unicode_normalization::UnicodeNormalization;

    fn replay(case: &Value) -> (Vocabulary, Prepared, CtcScoresOut, Value) {
        let ids: Vec<i32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
        let mut tokens: Vec<String> = (0..9812).map(|i| format!("unused-{i}")).collect();
        for (id, token) in ids.iter().zip(case["tokens"].as_array().unwrap()) {
            tokens[*id as usize] = token.as_str().unwrap().into();
        }
        let lookup = ids
            .iter()
            .filter_map(|id| {
                let text = &tokens[*id as usize];
                (text.chars().count() == 1).then(|| (text.chars().next().unwrap(), *id))
            })
            .collect();
        let vocabulary = Vocabulary { tokens, lookup };
        let prepared = vocabulary
            .prepare(
                case["text"].as_str().unwrap(),
                case["language"].as_str().unwrap(),
            )
            .unwrap();
        let samples = case["samples"].as_u64().unwrap() as usize;
        let result = base(
            &prepared,
            16000,
            samples,
            case["language"].as_str().unwrap(),
        );
        let scores = CtcScoresOut {
            valid_frames: ((samples - 400) / 320 + 1) as u32,
            audio_num_samples: samples as u32,
            frame_hop_samples: 320,
            frame_origin_samples: 0,
            token_ids: ids,
            model: timing::MODEL.into(),
            vocabulary_sha256: TOKENS_SHA256.into(),
            gathered: serde_json::from_value(case["gathered"].clone()).unwrap(),
        };
        (vocabulary, prepared, scores, result)
    }
    pub(crate) fn live_fixture() -> (Vocabulary, String, CtcScoresOut) {
        let all = cases();
        let case = &all["cases"][0];
        let (v, _, scores, _) = replay(case);
        (v, case["text"].as_str().unwrap().into(), scores)
    }
    fn cases() -> Value {
        serde_json::from_str(include_str!("../tests/testdata/alignment-reference.json")).unwrap()
    }
    pub(crate) fn functional_fixture(language: &str) -> Value {
        let cases = cases();
        let case = cases["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["language"] == language)
            .unwrap();
        let (vocabulary, prepared, scores, mut result) = replay(case);
        vocabulary.materialize(&prepared, &scores, &mut result);
        assert_eq!(result["complete"], true);
        result
    }

    #[test]
    fn independent_python_replays_preserve_text_offsets_aliases_and_cells() {
        for case in cases()["cases"].as_array().unwrap() {
            let (v, p, s, mut actual) = replay(case);
            if !p.unsupported.is_empty() {
                actual["reason"] = json!("unsupported_text");
            } else if p.targets.is_empty() {
                actual["reason"] = json!("no_acoustic_target");
            } else {
                v.materialize(&p, &s, &mut actual);
            }
            if actual["complete"] == true {
                let pcm = vec![0.; s.audio_num_samples as usize];
                actual["audio_float32_sha256"] = json!(pcm_sha256(&pcm));
                assert!(v.matches_input(
                    &actual,
                    &pcm,
                    &p.text,
                    16000,
                    case["language"].as_str().unwrap()
                ));
            }
            let expected = &case["expected"];
            for key in [
                "raw_text",
                "text_sha256",
                "audio_anchor",
                "status",
                "complete",
                "reason",
                "unsupported",
                "acoustic_aliases",
                "normalization_spans",
            ] {
                assert_eq!(actual[key], expected[key], "{} {key}", case["text"]);
            }
            for group in ["characters", "words"] {
                let actual = actual[group].as_array().unwrap();
                let expected = expected[group].as_array().unwrap();
                assert_eq!(actual.len(), expected.len(), "{} {group}", case["text"]);
                for (a, e) in actual.iter().zip(expected) {
                    for key in [
                        "text",
                        "raw_text",
                        "token_id",
                        "start_char",
                        "end_char",
                        "start_utf16",
                        "end_utf16",
                        "status",
                        "start_sample",
                        "end_sample",
                        "independent_acoustic_timestamp",
                        "acoustic_target_status",
                        "frame",
                    ] {
                        assert_eq!(a[key], e[key], "{} {group}.{key}", case["text"]);
                    }
                }
            }
        }
    }
    #[test]
    fn malformed_or_short_results_cannot_create_timing() {
        let all = cases();
        let case = &all["cases"][1];
        for mutation in 0..10 {
            let (v, p, mut s, mut result) = replay(case);
            match mutation {
                0 => s.audio_num_samples += 1,
                1 => s.frame_hop_samples = 321,
                2 => s.frame_origin_samples = 200,
                3 => s.model = "wrong".into(),
                4 => s.vocabulary_sha256 = "wrong".into(),
                5 => s.token_ids[0] = -1,
                6 => s.gathered[0] = f32::NAN,
                7 => s.gathered[0] = 0.1,
                8 => {
                    s.gathered.pop();
                }
                _ => s.gathered.fill(0.),
            }
            v.materialize(&p, &s, &mut result);
            assert_eq!(result["reason"], "invalid_ctc_scores");
            assert_eq!(result["complete"], false);
        }
        let (v, p, mut s, mut result) = replay(case);
        s.valid_frames = 0;
        s.audio_num_samples = 399;
        s.gathered.clear();
        result["audio_anchor"]["end_sample"] = json!(16399);
        v.materialize(&p, &s, &mut result);
        assert_eq!(result["reason"], "insufficient_acoustic_context");
        assert_eq!(result["raw_text"], "oo");
        assert_eq!(result["non_speech_proof"], false);
    }
    #[test]
    fn repeated_tokens_need_observed_blank_and_low_scores_are_not_renormalized() {
        for (case, blank_only) in [(&cases()["cases"][1], true), (&cases()["cases"][0], false)] {
            let (v, p, mut s, mut result) = replay(case);
            let width = s.token_ids.len();
            let blank = s.token_ids.iter().position(|id| *id == 0).unwrap();
            for row in s.gathered.chunks_mut(width) {
                for (i, score) in row.iter_mut().enumerate() {
                    if !blank_only || i == blank {
                        *score -= 25.;
                    }
                }
            }
            v.materialize(&p, &s, &mut result);
            assert_ne!(result["status"], "aligned");
            assert_eq!(result["complete"], false);
        }
    }
    #[test]
    fn unicode_composition_uses_maintained_normalization_without_changing_raw_offsets() {
        for text in [
            "Go 12.5 e\u{301}",
            "a\u{315}\u{300}",
            "\u{1100}\u{1161}\u{11a8}",
            "\u{301}a",
            "İ",
            "سَلَام",
            "コーヒー",
        ] {
            let raw: Vec<_> = text.chars().collect();
            let n = normalized(&raw);
            assert_eq!(
                n.iter().map(|s| s.c).collect::<String>(),
                text.nfc().collect::<String>()
            );
            assert!(n.iter().all(|s| s.a < s.b && s.b <= raw.len()));
        }
        let v = Vocabulary::parse_tokens("<s> 0\n<pad> 1\n</s> 2\n<unk> 3\n  4\né 5\n").unwrap();
        let p = v.prepare("e\u{301}", "en").unwrap();
        assert_eq!((p.targets[0].a, p.targets[0].b), (0, 2));
        assert!(Vocabulary::parse_tokens("<s> 0\n<pad> 1\n</s> 2\n<unk> 3\n  4\na 4\n").is_err());
    }

    #[test]
    fn context_ownership_accepts_normalized_targets_sharing_raw_character_spans() {
        // Real normalization/materialization, synthetic CTC emissions. Reordered
        // NFC marks share the original raw span; they are not malformed offsets.
        let vocabulary =
            Vocabulary::parse_tokens("<s> 0\n<pad> 1\n</s> 2\n<unk> 3\n  4\nà 5\n\u{315} 6\n")
                .unwrap();
        let text = "a\u{315}\u{300}";
        let prepared = vocabulary.prepare(text, "en").unwrap();
        assert_eq!(prepared.targets.len(), 2);
        assert_eq!(
            (prepared.targets[0].a, prepared.targets[0].b),
            (prepared.targets[1].a, prepared.targets[1].b)
        );
        let anchor = crate::live::AnchorRow {
            id: "normalized-context".into(),
            start_sample: 137,
            end_sample: 26 * 16000 + 137,
            machine_revision: 3,
            audio_revision: 4,
            language_epoch: 2,
        };
        let requests = crate::live::authoritative_requests(&anchor).unwrap();
        let parts = requests
            .iter()
            .map(|request| {
                let samples = (request.end_sample - request.start_sample) as usize;
                let frames = (samples - 400) / 320 + 1;
                let mut gathered = vec![-40.; frames * 3];
                for row in gathered.chunks_mut(3) {
                    row[0] = 0.;
                }
                for (index, cell) in [13 * 16000 + 137, 13 * 16000 + 137 + 640]
                    .into_iter()
                    .enumerate()
                {
                    let frame = ((cell - request.start_sample) / 320) as usize;
                    gathered[frame * 3..frame * 3 + 3].fill(-40.);
                    gathered[frame * 3 + index + 1] = 0.;
                }
                let scores = CtcScoresOut {
                    valid_frames: frames as u32,
                    audio_num_samples: samples as u32,
                    frame_hop_samples: 320,
                    frame_origin_samples: 0,
                    token_ids: vec![0, 5, 6],
                    model: timing::MODEL.into(),
                    vocabulary_sha256: TOKENS_SHA256.into(),
                    gathered,
                };
                let mut result = base(&prepared, request.start_sample, samples, "en");
                vocabulary.materialize(&prepared, &scores, &mut result);
                assert_eq!(result["complete"], true, "{result}");
                crate::live::TailPart {
                    request: request.clone(),
                    text: text.into(),
                    complete: true,
                    alignment: Some(result),
                }
            })
            .collect::<Vec<_>>();
        let selected = crate::live::assemble_authoritative(&anchor, &requests, &parts).unwrap();
        assert!(selected.complete, "{:?}", selected.reason);
        assert_eq!(selected.text.as_deref(), Some(text));
        assert_eq!(selected.words[0].end_char, 3);
    }

    #[test]
    fn ordinary_import_publishes_usable_timing_from_independent_ctc_replay() {
        let all = cases();
        let case = &all["cases"][0];
        let (vocabulary, _, scores, _) = replay(case);
        let samples = scores.audio_num_samples as usize;
        let duration = samples as f64 / 16000.;
        let mut models = crate::import::tests::Scripted::new(
            vec![crate::ipc::Turn {
                start: 0.,
                end: duration,
                speaker: "speaker_0".into(),
            }],
            vec![(0, samples)],
        );
        models.alignment = Some((case["text"].as_str().unwrap().into(), scores));
        let pcm: Vec<f32> = (0..samples)
            .map(|i| ((i % 17) as f32 - 8.) / 1000.)
            .collect();
        let document =
            crate::import::import_recording(&models, &pcm, "en", Some(&vocabulary)).unwrap();
        assert_eq!(document.segments.len(), 1);
        let segment = &document.segments[0];
        assert_eq!(segment.text, case["text"].as_str().unwrap());
        assert!(
            segment
                .reading_turns
                .as_ref()
                .is_some_and(|turns| !turns.as_array().unwrap().is_empty())
        );
        assert!(segment.reading_turn_provenance.is_some());
        let mut reading: timing::ReadingSegment =
            serde_json::from_value(serde_json::to_value(segment).unwrap()).unwrap();
        reading.installed_provider_identity = Some(NATIVE_PROVIDER.into());
        assert_eq!(timing::validated_turns(&reading), reading.reading_turns);
        assert!(!segment.rest.contains_key("decode_context"));
        assert_eq!(
            segment.rest["reading_alignment"]["audio_anchor"],
            json!({"start_sample":0,"end_sample":samples})
        );
        assert!(
            !document.provenance.as_ref().unwrap()["timing"]
                .as_str()
                .unwrap()
                .contains("no word alignment")
        );
        let text = crate::export::export(&document, "txt").unwrap().body;
        assert!(text.contains("Speaker 1:"));
    }

    #[test]
    fn native_projection_survives_serialization_and_exports_reject_stale_or_forged_evidence() {
        let all = cases();
        let case = &all["cases"][0];
        let (v, p, s, mut result) = replay(case);
        v.materialize(&p, &s, &mut result);
        let end = 1. + s.audio_num_samples as f64 / 16000.;
        let mut segment: crate::document::Segment=serde_json::from_value(json!({
            "id":"seg-0","start":1.,"end":end,"speaker":"unassigned","text":case["text"],"language":"en",
            "machine_revision":7,"audio_revision":11,"canonical_machine_revision":7,"language_epoch":3,
            "capture_source":{"source_id":"system","source_revision":1,"catalog_sha256":"retained-catalog"},
            "speaker_activity":{"audio_revision":11,"observed_end_sample":(end*16000.).round() as i64,
                "regions":[{"start_sample":16000,"end_sample":(end*16000.).round() as i64,"speakers":["speaker_0"]}]}
        })).unwrap();
        attach(&mut segment, result);
        assert_eq!(segment.machine_revision, Some(7));
        assert_eq!(segment.audio_revision, Some(11));
        assert_eq!(segment.rest["language_epoch"], 3);
        assert!(segment.reading_turns.is_some());
        let document:crate::document::Document=serde_json::from_value(json!({"schema_version":1,"segments":[segment],"speakers":{"unassigned":"Unknown","speaker_0":"Alice"}})).unwrap();
        let roundtrip: crate::document::Document =
            serde_json::from_str(&serde_json::to_string(&document).unwrap()).unwrap();
        let export = crate::export::export(&roundtrip, "txt").unwrap().body;
        assert!(export.contains("Alice:"), "{export}");
        assert!(export.contains(case["text"].as_str().unwrap()));
        for changes in [
            json!({"text":"Edited words"}),
            json!({"start":1.01}),
            json!({"speaker":"speaker_0"}),
        ] {
            let edited = crate::edit::edit_passage(crate::edit::PassageEditInput {
                status: "ready",
                duration: end,
                revision: 1,
                document: Some(&roundtrip),
                segment_id: "seg-0",
                body: &json!({"segment_revision":7,"changes":changes}),
                now: 1.,
            })
            .unwrap();
            let late =
                crate::live::apply_canonical_candidate(&edited.document, &roundtrip.segments[0]);
            assert!(late.segments[0].reading_turns.is_none());
            assert!(late.segments[0].reading_turn_provenance.is_none());
            assert!(!late.segments[0].rest.contains_key("reading_alignment"));
        }
        let mut stale = roundtrip.clone();
        stale.segments[0].text = "Edited words".into();
        let export = crate::export::export(&stale, "txt").unwrap().body;
        assert!(export.contains("Unknown: Edited words"), "{export}");
        let mut forged = roundtrip;
        forged.segments[0].reading_turn_provenance.as_mut().unwrap()["provider_identity_sha256"] =
            json!("forged");
        forged.segments[0]
            .rest
            .insert("installed_provider_identity".into(), json!("forged"));
        let export = crate::export::export(&forged, "txt").unwrap().body;
        assert!(export.contains("Unknown:"), "{export}");
        assert!(!export.contains("Alice:"), "{export}");
    }
}
