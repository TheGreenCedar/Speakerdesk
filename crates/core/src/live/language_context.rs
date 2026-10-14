//! Main-language counts, context selection and publication witness checks.
//!
//! Only accepted direct acoustic observations vote. Borrowed decisions retain
//! their exact witnesses and are checked again before publication.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde_json::Value;

use crate::document::Segment;
use crate::language::{LanguageContext, LanguageContextAnchor, LanguageContextEvidence};
use crate::languages;

use super::{RATE, round_samples, segment_bounds, sha256_hex};

/// Compact direct-acoustic language counts owned by one Engine or accuracy
/// operation. Borrowed/manual decisions never vote. Physical probe memberships
/// let accepted replacements update counts without copying transcript history.
#[derive(Default)]
pub struct MainLanguageSummary {
    groups: BTreeMap<String, MainLanguageGroup>,
}
#[derive(Default)]
struct MainLanguageGroup {
    through: i64,
    observations: BTreeMap<LanguageRange, Vec<LanguageVote>>,
    selected: BTreeMap<LanguageRange, LanguageVote>,
    meeting: BTreeMap<String, MainLanguageCount>,
    speakers: BTreeMap<String, BTreeMap<String, MainLanguageCount>>,
    compacted_before: i64,
    archived_through: i64,
}
#[derive(Default)]
struct MainLanguageCount {
    supports: BTreeSet<LanguageRange>,
    archived: usize,
    witnesses: Vec<LanguageVote>,
}
type LanguageRange = (i64, i64);
#[derive(Clone)]
struct LanguageVote {
    language: String,
    speaker_track: Option<String>,
    anchor: LanguageContextAnchor,
}
type LanguageVoteIdentity<'a> = (
    &'a str,
    Option<&'a str>,
    (
        &'a str,
        i64,
        Option<&'a str>,
        i64,
        i64,
        &'a str,
        Option<usize>,
    ),
);
fn language_vote_identity(vote: &LanguageVote) -> LanguageVoteIdentity<'_> {
    (
        &vote.language,
        vote.speaker_track.as_deref(),
        (
            &vote.anchor.utterance_id,
            vote.anchor.row_start_sample,
            vote.anchor.speaker_track_id.as_deref(),
            vote.anchor.start_sample,
            vote.anchor.end_sample,
            &vote.anchor.probe_sha256,
            vote.anchor.probe_index,
        ),
    )
}
fn same_language_vote(left: &LanguageVote, right: &LanguageVote) -> bool {
    language_vote_identity(left) == language_vote_identity(right)
}
impl MainLanguageGroup {
    fn reselect(&mut self, first: LanguageRange, last: LanguageRange) {
        let mut through = self
            .selected
            .range(..first)
            .next_back()
            .map_or(self.archived_through, |(range, _)| range.1)
            .max(self.archived_through);
        let counts = |counts: &mut BTreeMap<String, MainLanguageCount>,
                      language: &str,
                      range: LanguageRange,
                      add: bool| {
            if add {
                let count = counts.entry(language.to_owned()).or_default();
                count.supports.insert(range);
            } else if let Some(count) = counts.get_mut(language) {
                count.supports.remove(&range);
                if count.supports.is_empty() && count.archived == 0 {
                    counts.remove(language);
                }
            }
        };
        for (&range, votes) in self.observations.range(first..) {
            let was_selected = self.selected.contains_key(&range);
            if let Some(vote) = self.selected.remove(&range) {
                let language = vote.language;
                let track = vote.speaker_track;
                counts(&mut self.meeting, &language, range, false);
                if let Some(track) = track {
                    counts(
                        self.speakers.entry(track).or_default(),
                        &language,
                        range,
                        false,
                    );
                }
            }
            let vote = votes.first().filter(|first| {
                range.0 >= through
                    && votes.iter().all(|vote| {
                        vote.language == first.language && vote.speaker_track == first.speaker_track
                    })
            });
            if let Some(vote) = vote {
                self.selected.insert(range, vote.clone());
                counts(&mut self.meeting, &vote.language, range, true);
                if let Some(track) = &vote.speaker_track {
                    counts(
                        self.speakers.entry(track.clone()).or_default(),
                        &vote.language,
                        range,
                        true,
                    );
                }
                through = range.1;
                // An unchanged selected interval restores the old cutoff, so
                // later overlap decisions cannot depend on this replacement.
                if range > last && was_selected {
                    break;
                }
            }
        }
        self.through = self
            .selected
            .keys()
            .next_back()
            .map_or(self.archived_through, |range| range.1)
            .max(self.archived_through);
    }

    fn compact_before(&mut self, through: i64) {
        if through <= self.compacted_before {
            return;
        }
        let old: Vec<_> = self
            .selected
            .iter()
            .filter(|(range, _)| range.1 <= through)
            .map(|(range, vote)| (*range, vote.clone()))
            .collect();
        let archive = |count: &mut MainLanguageCount, range: LanguageRange, vote: &LanguageVote| {
            if count.supports.remove(&range) {
                count.archived += 1;
                count.witnesses.push(vote.clone());
                if count.witnesses.len() > 2 {
                    count.witnesses.remove(0);
                }
            }
        };
        for (range, vote) in old {
            archive(
                self.meeting
                    .get_mut(&vote.language)
                    .expect("selected language"),
                range,
                &vote,
            );
            if let Some(track) = &vote.speaker_track {
                archive(
                    self.speakers
                        .get_mut(track)
                        .expect("selected speaker")
                        .get_mut(&vote.language)
                        .expect("selected language"),
                    range,
                    &vote,
                );
            }
            self.selected.remove(&range);
            self.archived_through = self.archived_through.max(range.1);
        }
        self.observations.retain(|range, _| range.1 > through);
        self.compacted_before = through;
    }
}
pub(super) fn summary_key(row: &Segment) -> Option<String> {
    Some(
        serde_json::json!([
            row.rest.get("language_epoch")?.as_i64()?,
            row.capture_source,
            row.rest.get("stream_generation")
        ])
        .to_string(),
    )
}
fn speaker_track(row: &Segment) -> Option<String> {
    let tracks = row.rest.get("source_speaker_candidates")?.as_array()?;
    (tracks.len() == 1)
        .then(|| tracks[0].as_str().map(str::to_owned))
        .flatten()
        .filter(|id| !id.is_empty() && id != "unknown")
}
impl MainLanguageSummary {
    pub(super) fn observed_through(&self, key: &str) -> i64 {
        self.groups.get(key).map_or(0, |group| group.through)
    }

    pub(super) fn retain_ownership(
        &mut self,
        epochs: &HashSet<i64>,
        source: Option<&Value>,
        stream: u32,
    ) {
        self.groups.retain(|key, _| {
            serde_json::from_str::<Value>(key).ok().is_some_and(|key| {
                key[0].as_i64().is_some_and(|epoch| epochs.contains(&epoch))
                    && key[1] == source.unwrap_or(&Value::Null).clone()
                    && key[2] == stream
            })
        });
    }

    pub(super) fn compact_before(&mut self, through: i64) {
        for group in self.groups.values_mut() {
            group.compact_before(through);
        }
    }
    pub(super) fn witness_rows(&self) -> HashSet<String> {
        let mut rows = HashSet::new();
        for group in self.groups.values() {
            for counts in std::iter::once(&group.meeting).chain(group.speakers.values()) {
                for count in counts.values() {
                    for vote in &count.witnesses {
                        rows.insert(vote.anchor.utterance_id.clone());
                    }
                    for range in count.supports.iter().rev().take(2) {
                        if let Some(vote) = group.selected.get(range) {
                            rows.insert(vote.anchor.utterance_id.clone());
                        }
                    }
                }
            }
        }
        rows
    }
    pub(super) fn needs_rebuild(&self, prior: Option<&Segment>, accepted: &Segment) -> bool {
        let old = main_language_votes(prior.map_or(&[][..], std::slice::from_ref));
        let new = main_language_votes(std::slice::from_ref(accepted));
        let old_ids: BTreeSet<_> = old
            .iter()
            .map(|(_, vote)| language_vote_identity(vote))
            .collect();
        let new_ids: BTreeSet<_> = new
            .iter()
            .map(|(_, vote)| language_vote_identity(vote))
            .collect();
        old.iter()
            .filter(|(_, vote)| !new_ids.contains(&language_vote_identity(vote)))
            .chain(
                new.iter()
                    .filter(|(_, vote)| !old_ids.contains(&language_vote_identity(vote))),
            )
            .any(|(key, vote)| {
                self.groups
                    .get(key)
                    .is_some_and(|group| vote.anchor.start_sample < group.compacted_before)
            })
    }
    pub fn from_segments(rows: &[Segment]) -> Self {
        let mut summary = Self::default();
        for (key, vote) in main_language_votes(rows) {
            let range = (vote.anchor.start_sample, vote.anchor.end_sample);
            summary
                .groups
                .entry(key)
                .or_default()
                .observations
                .entry(range)
                .or_default()
                .push(vote);
        }
        for group in summary.groups.values_mut() {
            group.reselect((0, 0), (i64::MAX, i64::MAX));
        }
        summary
    }
    pub(super) fn add(
        &mut self,
        key: String,
        language: String,
        speaker_track: Option<String>,
        anchor: LanguageContextAnchor,
    ) {
        let vote = LanguageVote {
            language,
            speaker_track,
            anchor,
        };
        let group = self.groups.entry(key).or_default();
        let range = (vote.anchor.start_sample, vote.anchor.end_sample);
        if range.0 < group.through {
            return;
        }
        group.observations.entry(range).or_default().push(vote);
        group.reselect(range, range);
    }
    /// Called only for accepted publication. Ordinary disjoint replacements
    /// touch their own probes; overlap changes walk only affected memberships.
    pub fn accept_publication(&mut self, prior: Option<&Segment>, accepted: &Segment) {
        let old = main_language_votes(prior.map_or(&[][..], std::slice::from_ref));
        let new = main_language_votes(std::slice::from_ref(accepted));
        let old_identities: BTreeSet<_> = old
            .iter()
            .map(|(key, vote)| (key.as_str(), language_vote_identity(vote)))
            .collect();
        let new_identities: BTreeSet<_> = new
            .iter()
            .map(|(key, vote)| (key.as_str(), language_vote_identity(vote)))
            .collect();
        let mut changed = BTreeMap::new();
        for (remove, observations, other) in [
            (true, &old, &new_identities),
            (false, &new, &old_identities),
        ] {
            for (key, vote) in observations {
                if other.contains(&(key.as_str(), language_vote_identity(vote))) {
                    continue;
                }
                let range = (vote.anchor.start_sample, vote.anchor.end_sample);
                let group = self.groups.entry(key.clone()).or_default();
                let votes = group.observations.entry(range).or_default();
                if remove {
                    votes.retain(|current| !same_language_vote(current, vote));
                } else if !votes
                    .iter()
                    .any(|current| same_language_vote(current, vote))
                {
                    votes.push(vote.clone());
                }
                let bounds = changed.entry(key.clone()).or_insert((range, range));
                bounds.0 = bounds.0.min(range);
                bounds.1 = bounds.1.max(range);
            }
        }
        for (key, (first, last)) in changed {
            let group = self.groups.get_mut(&key).expect("changed group");
            group.reselect(first, last);
            let removed: Vec<_> = group
                .observations
                .range(first..=last)
                .filter(|(_, votes)| votes.is_empty())
                .map(|(range, _)| *range)
                .collect();
            for range in removed {
                group.observations.remove(&range);
            }
        }
    }
    pub fn context(&self, rows: &[Segment], target: &Segment) -> Option<LanguageContext> {
        let mut context = self.context_for(target)?;
        // Row revisions may advance as more direct words are accepted. Rebind
        // only exact surviving acoustic receipts; protected/source edits fail.
        for anchor in &mut context.evidence.as_mut()?.anchors {
            let row = context_anchor_row(rows, &anchor.utterance_id, anchor.row_start_sample)?;
            anchor.machine_revision = row.machine_revision.unwrap_or(0);
            anchor.audio_revision = row.audio_revision.unwrap_or(0);
        }
        let mut candidate = target.clone();
        candidate.language_detection = Some(
            serde_json::json!({"context_evidence": crate::import::context_evidence_value(context.evidence.as_ref()?) }),
        );
        borrowed_context_current(rows, &candidate).then_some(context)
    }
    pub fn has_context(&self, target: &Segment) -> bool {
        self.context_for(target).is_some()
    }
    pub(super) fn context_for(&self, target: &Segment) -> Option<LanguageContext> {
        if target.rest.get("language_mode").and_then(Value::as_str) != Some(languages::AUTOMATIC) {
            return None;
        }
        let group = self.groups.get(&summary_key(target)?)?;
        let choose = |counts: &BTreeMap<String, MainLanguageCount>, track: Option<String>| {
            let total: usize = counts
                .values()
                .map(|count| count.supports.len() + count.archived)
                .sum();
            let (language, count) = counts
                .iter()
                .max_by_key(|(_, count)| count.supports.len() + count.archived)?;
            let selected = count.supports.len() + count.archived;
            if selected < 2 || selected * 2 <= total {
                return None;
            }
            let mut anchors =
                count
                    .witnesses
                    .iter()
                    .map(|vote| vote.anchor.clone())
                    .chain(count.supports.iter().rev().take(2).filter_map(|range| {
                        group.selected.get(range).map(|vote| vote.anchor.clone())
                    }))
                    .collect::<Vec<_>>();
            anchors.sort_by_key(|a| (a.start_sample, a.end_sample));
            if anchors.len() > 2 {
                anchors.drain(..anchors.len() - 2);
            }
            Some(LanguageContext {
                language: language.clone(),
                end_sample: anchors.iter().map(|a| a.end_sample).max()?,
                evidence: Some(LanguageContextEvidence {
                    scope: if track.is_some() {
                        "speaker_main_language"
                    } else {
                        "meeting_main_language"
                    },
                    speaker_track_id: track,
                    language_epoch: target.rest["language_epoch"].as_i64()?,
                    interpretation_generation: 0,
                    capture_source: target.capture_source.clone(),
                    stream_generation: target.rest.get("stream_generation").and_then(Value::as_i64),
                    anchors,
                }),
            })
        };
        speaker_track(target)
            .and_then(|track| {
                group
                    .speakers
                    .get(&track)
                    .and_then(|counts| choose(counts, Some(track.clone())))
            })
            .or_else(|| choose(&group.meeting, None))
    }
}
/// Convenience for one isolated saved request. Accuracy workers retain one
/// summary for their entire operation instead of rebuilding it per passage.
pub fn saved_language_context(rows: &[Segment], target: &Segment) -> Option<LanguageContext> {
    MainLanguageSummary::from_segments(rows).context(rows, target)
}
fn main_language_votes(rows: &[Segment]) -> Vec<(String, LanguageVote)> {
    let mut votes = Vec::new();
    for row in rows {
        let Ok((row_start, row_end)) = segment_bounds(row) else {
            continue;
        };
        let accepted_end = row
            .rest
            .get("canonical_text_end_sample")
            .and_then(Value::as_i64)
            .unwrap_or(row_end);
        let prefix = row_start < accepted_end && accepted_end < row_end;
        if row.separation.is_some()
            || row.canonical_utterance_id.is_none()
            || row.text.trim().is_empty()
            || row
                .protected_fields
                .as_ref()
                .is_some_and(|fields| fields.iter().any(|field| field != "text"))
            || (!prefix
                && row
                    .transcription_review
                    .as_ref()
                    .is_some_and(|v| v["partial_text"] == true))
            || (!prefix
                && row
                    .rest
                    .get("canonical_unresolved")
                    .is_some_and(|v| !v.is_null()))
            || (!prefix && row.rest.get("audio_state").is_some_and(|v| !v.is_null()))
            || row.rest.get("language_mode").and_then(Value::as_str) != Some(languages::AUTOMATIC)
        {
            continue;
        }
        let Some(acoustic) = &row.acoustic_evidence else {
            continue;
        };
        if acoustic["complete"] != true
            || acoustic["decision"] != "speech"
            || acoustic["evidence_error"] != false
            || acoustic["start_sample"]
                .as_i64()
                .is_none_or(|a| a > row_start)
            || acoustic["end_sample"].as_i64().is_none_or(|b| b < row_end)
            || !matches!(
                row.rest.get("canonical_state").and_then(Value::as_str),
                Some("sealed" | "open")
            )
            || (row.rest["canonical_state"] == "open"
                && row
                    .rest
                    .get("canonical_text_end_sample")
                    .and_then(Value::as_i64)
                    .is_none())
        {
            continue;
        }
        let Some(probes) = row
            .language_detection
            .as_ref()
            .and_then(|v| v["probes"].as_array())
        else {
            continue;
        };
        let detection = row.language_detection.as_ref().expect("probes checked");
        if detection["mode"] == "manual" {
            continue;
        }
        let witnesses = detection["context_witnesses"]
            .as_array()
            .map_or(&[][..], Vec::as_slice);
        for (probe_index, probe) in probes
            .iter()
            .enumerate()
            .map(|(i, p)| (Some(i), p))
            .chain(witnesses.iter().take(2).map(|p| (None, p)))
        {
            let (Some(a), Some(b), Some(language)) = (
                probe["start_sample"].as_i64(),
                probe["end_sample"].as_i64(),
                probe["language"].as_str(),
            ) else {
                continue;
            };
            let decision = &probe["decision"];
            if a < row_start
                || b <= a
                || b - a > (crate::language::PROBE_SECONDS * RATE as f64).ceil() as i64
                || b > row_end
                || b > accepted_end
                || acoustic["uncertain_regions"]
                    .as_array()
                    .is_some_and(|regions| {
                        regions.iter().any(|region| {
                            region["start_sample"]
                                .as_i64()
                                .is_some_and(|start| start < b)
                                && region["end_sample"].as_i64().is_some_and(|end| end > a)
                        })
                    })
                || !languages::is_supported(language)
                || decision["mode"] != "auto"
                || decision["reason"] != "detected"
                || decision["detector_error"] == true
                || !decision["context_evidence"].is_null()
                || !crate::language::is_confident_detection(
                    decision["probability"].as_f64().unwrap_or(0.),
                    decision["margin"].as_f64().unwrap_or(0.),
                )
            {
                continue;
            }
            let vote = LanguageVote {
                language: language.to_owned(),
                speaker_track: speaker_track(row),
                anchor: LanguageContextAnchor {
                    utterance_id: row.canonical_utterance_id.clone().expect("checked"),
                    row_start_sample: row_start,
                    speaker_track_id: speaker_track(row),
                    start_sample: a,
                    end_sample: b,
                    machine_revision: row.machine_revision.unwrap_or(0),
                    audio_revision: row.audio_revision.unwrap_or(0),
                    probe_sha256: sha256_hex(
                        serde_json::to_string(probe).unwrap_or_default().as_bytes(),
                    ),
                    probe_index,
                },
            };
            if let Some(key) = summary_key(row) {
                votes.push((key, vote));
            }
        }
    }
    votes
}

fn choose_language_context(
    votes: Vec<LanguageVote>,
    target_track: Option<&str>,
    epoch: i64,
    capture_source: Option<Value>,
    stream_generation: Option<i64>,
) -> Option<LanguageContext> {
    let mut ranges: BTreeMap<LanguageRange, Option<LanguageVote>> = BTreeMap::new();
    for vote in votes {
        let key = (vote.anchor.start_sample, vote.anchor.end_sample);
        match ranges.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(Some(vote));
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if entry.get().as_ref().is_some_and(|prior| {
                    prior.language != vote.language || prior.speaker_track != vote.speaker_track
                }) {
                    entry.insert(None);
                }
            }
        }
    }
    let mut through = -1;
    let votes: Vec<_> = ranges
        .into_values()
        .flatten()
        .filter(|vote| {
            if vote.anchor.start_sample < through {
                return false;
            }
            through = vote.anchor.end_sample;
            true
        })
        .collect();
    let choose = |speaker: Option<&str>| {
        let eligible: Vec<_> = votes
            .iter()
            .filter(|vote| speaker.is_none_or(|id| vote.speaker_track.as_deref() == Some(id)))
            .collect();
        let mut counts = BTreeMap::new();
        for vote in &eligible {
            *counts.entry(vote.language.as_str()).or_insert(0usize) += 1;
        }
        let (language, count) = counts.into_iter().max_by_key(|(_, count)| *count)?;
        if count < 2 || count * 2 <= eligible.len() {
            return None;
        }
        let anchors: Vec<_> = eligible
            .iter()
            .filter(|vote| vote.language == language)
            .rev()
            .take(2)
            .map(|vote| vote.anchor.clone())
            .collect();
        Some(LanguageContext {
            language: language.into(),
            end_sample: anchors.iter().map(|a| a.end_sample).max()?,
            evidence: Some(LanguageContextEvidence {
                scope: if speaker.is_some() {
                    "speaker_main_language"
                } else {
                    "meeting_main_language"
                },
                speaker_track_id: speaker.map(str::to_owned),
                language_epoch: epoch,
                interpretation_generation: 0,
                capture_source: capture_source.clone(),
                stream_generation,
                anchors,
            }),
        })
    };
    target_track
        .and_then(|id| choose(Some(id)))
        .or_else(|| choose(None))
}

fn preceding_probe_values(probes: &[Value], start: i64) -> impl Iterator<Item = &Value> {
    // The owning Refinement keeps this snapshot ordered by probe end, including
    // overlapping contexts. Lookup touches only the preceding 60-second suffix.
    let through =
        probes.partition_point(|probe| probe["end_sample"].as_i64().unwrap_or(i64::MAX) <= start);
    probes[..through]
        .iter()
        .rev()
        .take_while(move |probe| {
            probe["end_sample"]
                .as_i64()
                .is_some_and(|end| start - end <= crate::language::RECENT_CONTEXT_SAMPLES)
        })
        .filter(move |probe| {
            probe["end_sample"].as_i64().is_some_and(|end| {
                (0..=crate::language::RECENT_CONTEXT_SAMPLES).contains(&(start - end))
            })
        })
}

pub(super) fn same_anchor_window_context(
    original: &Segment,
    probes: &[Value],
    start: i64,
    internal: bool,
) -> Option<LanguageContext> {
    let (first, last) = segment_bounds(original).ok()?;
    let mut votes = Vec::new();
    for probe in preceding_probe_values(probes, start) {
        let (Some(a), Some(b), Some(language)) = (
            probe["start_sample"].as_i64(),
            probe["end_sample"].as_i64(),
            probe["language"].as_str(),
        ) else {
            continue;
        };
        let decision = &probe["decision"];
        if a < first
            || b <= a
            || b > last
            || !languages::is_supported(language)
            || decision["mode"] != "auto"
            || decision["reason"] != "detected"
            || decision["detector_error"] == true
            || !decision["context_evidence"].is_null()
            || !crate::language::is_confident_detection(
                decision["probability"].as_f64().unwrap_or(0.),
                decision["margin"].as_f64().unwrap_or(0.),
            )
        {
            continue;
        }
        votes.push(LanguageVote {
            language: language.into(),
            speaker_track: None,
            anchor: LanguageContextAnchor {
                utterance_id: original.canonical_utterance_id.clone()?,
                row_start_sample: first,
                speaker_track_id: speaker_track(original),
                start_sample: a,
                end_sample: b,
                machine_revision: original.machine_revision.unwrap_or(0),
                audio_revision: original.audio_revision.unwrap_or(0),
                probe_sha256: sha256_hex(probe.to_string().as_bytes()),
                probe_index: None,
            },
        });
    }
    let mut context = choose_language_context(
        votes,
        None,
        original.rest["language_epoch"].as_i64()?,
        original.capture_source.clone(),
        original
            .rest
            .get("stream_generation")
            .and_then(Value::as_i64),
    )?;
    if internal {
        context.evidence.as_mut()?.scope = "same_anchor_direct_windows";
    }
    Some(context)
}

fn context_anchor_row<'a>(rows: &'a [Segment], id: &str, start: i64) -> Option<&'a Segment> {
    let from = rows.partition_point(|row| round_samples(row.start) < start);
    rows[from..]
        .iter()
        .take_while(|row| round_samples(row.start) == start)
        .find(|row| row.canonical_utterance_id.as_deref() == Some(id))
}

/// Recheck only the selected direct witnesses before publishing borrowed text.
/// Display-name changes are neutral; source/epoch/track/edit changes supersede it.
pub fn borrowed_context_current(rows: &[Segment], candidate: &Segment) -> bool {
    let Some(detection) = &candidate.language_detection else {
        return true;
    };
    let mut decisions = vec![detection];
    if let Some(probes) = detection["probes"].as_array() {
        decisions.extend(probes.iter().map(|probe| &probe["decision"]));
    }
    for evidence in decisions
        .into_iter()
        .filter_map(|decision| decision.get("context_evidence"))
    {
        let Some(anchors) = evidence["anchors"].as_array() else {
            return false;
        };
        for anchor in anchors {
            let Some(id) = anchor["utterance_id"].as_str() else {
                return false;
            };
            let Some(row) =
                context_anchor_row(rows, id, anchor["row_start_sample"].as_i64().unwrap_or(-1))
            else {
                return false;
            };
            if row
                .protected_fields
                .as_ref()
                .is_some_and(|fields| fields.iter().any(|field| field != "text"))
                // Append-only audio growth advances the row revision while
                // leaving the selected acoustic prefix unchanged. Geometry and
                // its exact direct receipt remain the publication authority.
                || row.audio_revision.unwrap_or(0)
                    < anchor["audio_revision"].as_i64().unwrap_or(i64::MAX)
                || anchor["start_sample"].as_i64().is_none_or(|start| {
                    start < round_samples(row.start)
                })
                || anchor["end_sample"].as_i64().is_none_or(|end| {
                    end > round_samples(row.end)
                        || (evidence["scope"] != "same_anchor_direct_windows"
                            && end > row.rest["canonical_text_end_sample"].as_i64().unwrap_or(-1))
                })
                || row.capture_source.as_ref().unwrap_or(&Value::Null)
                    != &evidence["capture_source"]
                || row.rest.get("language_epoch") != evidence.get("language_epoch")
                || row.rest.get("stream_generation").unwrap_or(&Value::Null)
                    != &evidence["stream_generation"]
                || row.rest.get("language_mode").and_then(Value::as_str)
                    != Some(languages::AUTOMATIC)
                || row.language_detection.as_ref().is_some_and(|d| d["mode"] == "manual")
                || serde_json::json!(speaker_track(row)) != anchor["speaker_track_id"]
                || (evidence["scope"] == "speaker_main_language"
                    && row.rest["source_speaker_candidates"]
                        != serde_json::json!([evidence["speaker_track_id"]]))
            {
                return false;
            }
            let receipt = if evidence["scope"] == "same_anchor_direct_windows" {
                if candidate.canonical_utterance_id.as_deref() != Some(id) {
                    return false;
                }
                candidate.language_detection.as_ref()
            } else {
                row.language_detection.as_ref()
            };
            let Some(probes) = receipt.and_then(|d| d["probes"].as_array()) else {
                return false;
            };
            let matches = |probe: &Value| {
                probe["start_sample"] == anchor["start_sample"]
                    && probe["end_sample"] == anchor["end_sample"]
                    && sha256_hex(probe.to_string().as_bytes()) == anchor["probe_sha256"]
            };
            let exact = receipt
                .and_then(|d| d["context_witnesses"].as_array())
                .is_some_and(|witnesses| witnesses.iter().take(2).any(matches))
                || if let Some(index) = anchor["probe_index"].as_u64() {
                    probes.get(index as usize).is_some_and(matches)
                } else {
                    probes.iter().rev().any(matches)
                };
            if !exact {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value};

    use crate::document::Segment;
    use crate::ipc::LanguageProbeOut;
    use crate::speech;

    use super::super::tests::{FakeModels, TestEngine, loud, notice, revisions, speech_passage};
    use super::super::{
        EngineConfig, EngineEvent, LiveCommand, RATE, apply_canonical_candidate, empty_document,
        publish_saved_passage, refine_saved_passage_with_context, round_samples,
    };
    use super::{MainLanguageSummary, borrowed_context_current, saved_language_context};

    fn accepted_language_row(id: &str, start: i64, end: i64) -> Segment {
        let mut segment = crate::transcript::bare_segment(
            id,
            start as f64 / RATE as f64,
            end as f64 / RATE as f64,
            "speaker_0",
            "Direct accepted words",
        );
        segment.canonical_utterance_id = Some(id.into());
        segment.language = Some("en".into());
        segment.capture_source = Some(
            serde_json::json!({"catalog_sha256":"owned","source_id":"microphone","source_revision":1}),
        );
        segment.rest.extend(
            serde_json::from_value::<Map<String, Value>>(serde_json::json!({
                "start_sample":start,"end_sample":end,"canonical_text_end_sample":end,
                "language_mode":"auto","language_epoch":0,"stream_generation":0,
                "canonical_state":"sealed","source_speaker_candidates":["speaker_0"]
            }))
            .unwrap(),
        );
        let passage = speech_passage(
            0.,
            (end - start) as f64 / RATE as f64,
            "Direct accepted words",
            start,
        );
        segment.acoustic_evidence = Some(speech::admission_value(&passage.acoustic_evidence));
        let mut detection = passage.language_detection;
        detection.probes = vec![crate::language::ProbeEvent {
            start_sample: start,
            end_sample: end,
            language: Some("en".into()),
            decision: detection.clone(),
            review: false,
        }];
        segment.language_detection = Some(crate::import::detection_value(&detection));
        segment
    }

    #[test]
    fn compact_language_totals_survive_long_pause_and_keep_bounded_witnesses() {
        let mut engine = TestEngine::new(
            EngineConfig::new("language-memory", "auto"),
            FakeModels::english(),
        )
        .unwrap();
        for _ in 0..80 {
            for pcm in [loud(RATE as usize), vec![0.; RATE as usize]] {
                let start = engine.recording.writer.frames() as i64;
                engine.append_pcm(&pcm);
                engine.handle(notice(start, start + RATE, 0)).unwrap();
            }
        }
        for _ in 0..140 {
            let start = engine.recording.writer.frames() as i64;
            engine.append_pcm(&vec![0.; RATE as usize]);
            engine.handle(notice(start, start + RATE, 0)).unwrap();
        }
        let group = engine.main_language.groups.values().next().unwrap();
        assert!(group.meeting["en"].archived >= 70);
        assert_eq!(group.meeting["en"].witnesses.len(), 2);
        assert!(group.observations.len() <= 2);
        assert!(engine.book.rows.len() <= 4);
        engine
            .models
            .probe_scores
            .lock()
            .unwrap()
            .push_back(Ok(LanguageProbeOut {
                codes: vec!["en".into(), "fr".into()],
                probabilities: vec![0.4, 0.6],
            }));
        let start = engine.recording.writer.frames() as i64;
        engine.append_pcm(&loud(RATE as usize));
        engine.handle(notice(start, start + RATE, 0)).unwrap();
        engine.append_pcm(&vec![0.; RATE as usize]);
        let events = engine
            .handle(notice(start + RATE, start + 2 * RATE, 0))
            .unwrap();
        let row = revisions(&events).into_iter().last().unwrap();
        assert_eq!(row.language.as_deref(), Some("en"));
        assert_eq!(
            row.language_detection.as_ref().unwrap()["context_evidence"]["scope"],
            "speaker_main_language"
        );
    }

    #[test]
    fn growing_live_language_donor_keeps_its_exact_accepted_probe_prefix() {
        let mut engine = TestEngine::new(
            EngineConfig::new("growing-language-donor", "auto"),
            FakeModels::english(),
        )
        .unwrap();
        engine.append_pcm(&loud((7 * RATE) as usize));
        let mut document = empty_document();
        for event in engine.handle(notice(0, 7 * RATE, 0)).unwrap() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                document = apply_canonical_candidate(&document, &candidate);
            }
        }
        let original = document.segments[0].clone();
        assert!(!original.text.is_empty());
        assert!(
            original.language_detection.as_ref().unwrap()["probes"]
                .as_array()
                .unwrap()
                .len()
                >= 2
        );
        engine
            .models
            .probe_scores
            .lock()
            .unwrap()
            .extend((0..4).map(|_| {
                Ok(LanguageProbeOut {
                    codes: vec!["ru".into(), "en".into()],
                    probabilities: vec![0.51, 0.49],
                })
            }));
        engine.append_pcm(&loud((3 * RATE) as usize));
        for event in engine.handle(notice(7 * RATE, 10 * RATE, 0)).unwrap() {
            if let EngineEvent::CanonicalRevision { candidate, .. } = event {
                assert!(borrowed_context_current(&document.segments, &candidate));
                document = apply_canonical_candidate(&document, &candidate);
            }
        }
        assert!(document.segments[0].audio_revision > original.audio_revision);
        assert_eq!(
            document.segments[0].language_detection.as_ref().unwrap()["reason"],
            "recent_context"
        );
        assert_eq!(
            document.segments[0].language_detection.as_ref().unwrap()["context_witnesses"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(
            document.segments[0].language_detection.as_ref().unwrap()["context_witnesses"]
                .as_array()
                .unwrap()
                .iter()
                .all(|probe| probe["decision"]["reason"] == "detected"
                    && probe["decision"]["context_evidence"].is_null())
        );
        engine.models.probe_scores.lock().unwrap().clear();
        engine
            .models
            .probe_scores
            .lock()
            .unwrap()
            .extend((0..4).map(|_| {
                Ok(LanguageProbeOut {
                    codes: vec!["ru".into(), "en".into()],
                    probabilities: vec![0.51, 0.49],
                })
            }));
        let stopped = engine.handle(LiveCommand::Stop).unwrap();
        let candidates = revisions(&stopped);
        let borrowed = candidates
            .iter()
            .rev()
            .find(|row| !row.text.is_empty())
            .unwrap();
        assert_eq!(borrowed.language.as_deref(), Some("en"));
        assert_eq!(
            borrowed.language_detection.as_ref().unwrap()["reason"],
            "recent_context"
        );
        assert!(
            borrowed_context_current(&document.segments, borrowed),
            "ordinary owned append keeps exact earlier acoustic evidence"
        );
        let mut duplicate = (*borrowed).clone();
        let detection = duplicate.language_detection.as_mut().unwrap();
        let direct = detection["context_witnesses"].as_array().unwrap().clone();
        detection["probes"].as_array_mut().unwrap().extend(direct);
        let original_summary = MainLanguageSummary::from_segments(std::slice::from_ref(borrowed));
        let duplicate_summary = MainLanguageSummary::from_segments(&[duplicate]);
        assert_eq!(
            original_summary.groups.values().next().unwrap().meeting["en"]
                .supports
                .len(),
            duplicate_summary.groups.values().next().unwrap().meeting["en"]
                .supports
                .len(),
            "one-time sealed summary never counts the same physical direct probe twice"
        );
        let mut shrunk = document.segments.clone();
        let supports = borrowed.language_detection.as_ref().unwrap()["context_evidence"]["anchors"]
            .as_array()
            .unwrap();
        shrunk[0].end = supports
            .iter()
            .map(|anchor| anchor["end_sample"].as_i64().unwrap())
            .min()
            .unwrap() as f64
            / RATE as f64
            - 0.001;
        assert!(
            !borrowed_context_current(&shrunk, borrowed),
            "retained JSON cannot authorize a probe outside shrunken owned audio"
        );
        let mut retimed = document.segments.clone();
        retimed[0].start += 0.001;
        assert!(!borrowed_context_current(&retimed, borrowed));
    }

    #[test]
    fn accepted_language_publications_refresh_counts_and_physical_memberships() {
        let row = accepted_language_row;
        let set_language = |row: &mut Segment, language: &str| {
            row.language = Some(language.into());
            row.language_detection.as_mut().unwrap()["probes"][0]["language"] =
                Value::from(language);
        };
        for case in [
            "majority",
            "borrowed",
            "spelling",
            "protected",
            "source",
            "epoch",
            "stream",
            "duplicate",
            "conflict",
            "overlap",
        ] {
            let mut rows: Vec<_> = (0..5)
                .map(|index| {
                    let mut row = row(
                        &format!("direct-{index}"),
                        index * 3 * RATE,
                        (index + 1) * 3 * RATE,
                    );
                    if index >= 3 {
                        set_language(&mut row, "de");
                    }
                    row.rest.insert(
                        "source_speaker_candidates".into(),
                        serde_json::json!(["speaker_0"]),
                    );
                    row
                })
                .collect();
            let mut target = row("target", 24 * RATE, 27 * RATE);
            target.rest.insert(
                "source_speaker_candidates".into(),
                serde_json::json!(["new-speaker"]),
            );
            if matches!(case, "duplicate" | "conflict" | "overlap") {
                let mut extra = row("extra", 0, 3 * RATE);
                if case == "overlap" {
                    extra = row("extra", RATE, 4 * RATE);
                }
                rows.push(extra);
            }
            let mut summary = MainLanguageSummary::from_segments(&rows);
            assert_eq!(
                summary.context(&rows, &target).unwrap().language,
                "en",
                "{case}"
            );
            let index = if matches!(case, "duplicate" | "conflict") {
                5
            } else {
                0
            };
            let prior = rows[index].clone();
            match case {
                "majority" | "conflict" => set_language(&mut rows[index], "de"),
                "borrowed" | "duplicate" | "overlap" => {
                    rows[index].language_detection.as_mut().unwrap()["probes"][0]["decision"]["reason"] =
                        Value::from("recent_context");
                }
                "spelling" => {
                    rows[index].text = "Protected spelling".into();
                    rows[index].protected_fields = Some(vec!["text".into()]);
                    rows[index].machine_revision = Some(7);
                }
                "protected" => rows[index].protected_fields = Some(vec!["speaker".into()]),
                "source" => {
                    rows[index].capture_source.as_mut().unwrap()["source_id"] =
                        Value::from("system")
                }
                "epoch" => {
                    rows[index]
                        .rest
                        .insert("language_epoch".into(), Value::from(1));
                }
                "stream" => {
                    rows[index]
                        .rest
                        .insert("stream_generation".into(), Value::from(1));
                }
                _ => unreachable!(),
            }
            summary.accept_publication(Some(&prior), &rows[index]);
            let expected = match case {
                "majority" => Some("de"),
                "spelling" | "duplicate" => Some("en"),
                _ => None,
            };
            assert_eq!(
                summary
                    .context(&rows, &target)
                    .map(|context| context.language),
                expected.map(str::to_owned),
                "{case}"
            );
            assert_eq!(
                summary.context(&rows, &target),
                saved_language_context(&rows, &target),
                "{case}: incremental and initial ownership agree"
            );
            // Repeated receipt/revision publication cannot mint acoustic votes.
            summary.accept_publication(Some(&rows[index]), &rows[index]);
            assert_eq!(
                summary.context(&rows, &target),
                saved_language_context(&rows, &target),
                "{case}: repeat publication"
            );
        }

        let mut rows = Vec::new();
        let mut summary = MainLanguageSummary::default();
        let mut target = row("empty-target", 18 * RATE, 21 * RATE);
        for index in 0..2 {
            rows.push(row(
                &format!("new-{index}"),
                index * 3 * RATE,
                (index + 1) * 3 * RATE,
            ));
            summary.accept_publication(None, rows.last().unwrap());
            assert_eq!(summary.context(&rows, &target).is_some(), index == 1);
        }
        for index in 2..4 {
            let mut accepted = row(
                &format!("new-{index}"),
                index * 3 * RATE,
                (index + 1) * 3 * RATE,
            );
            set_language(&mut accepted, "de");
            accepted.rest.insert(
                "source_speaker_candidates".into(),
                serde_json::json!(["new-speaker"]),
            );
            rows.push(accepted);
            summary.accept_publication(None, rows.last().unwrap());
        }
        target.rest.insert(
            "source_speaker_candidates".into(),
            serde_json::json!(["new-speaker"]),
        );
        let context = summary.context(&rows, &target).unwrap();
        assert_eq!(context.language, "de");
        assert_eq!(context.evidence.unwrap().scope, "speaker_main_language");
        assert_eq!(
            summary.context(&rows, &target),
            saved_language_context(&rows, &target)
        );
    }

    #[test]
    fn saved_language_context_requires_independent_fresh_owned_acoustic_evidence() {
        let row = accepted_language_row;
        for case in [
            "qualified",
            "foreign_overlap",
            "duplicate",
            "overlap",
            "conflict",
            "source",
            "stream",
            "epoch",
            "manual",
            "borrowed",
            "partial",
            "acoustic",
            "protected",
            "stale",
            "future",
        ] {
            let mut first = row("first", 0, 3 * RATE);
            let mut second = row("second", 3 * RATE, 6 * RATE);
            let mut target = row("target", 9 * RATE, 12 * RATE);
            let mut extra = Vec::new();
            match case {
                "foreign_overlap" => {
                    first = row("recent-long", 0, 76 * RATE);
                    let mut a = first.language_detection.as_ref().unwrap()["probes"][0].clone();
                    a["start_sample"] = Value::from(70 * RATE);
                    a["end_sample"] = Value::from(73 * RATE);
                    let mut b = a.clone();
                    b["start_sample"] = Value::from(73 * RATE);
                    b["end_sample"] = Value::from(76 * RATE);
                    first.language_detection.as_mut().unwrap()["probes"] =
                        serde_json::json!([a, b]);
                    second = row("foreign-short", 5 * RATE, 10 * RATE);
                    second.capture_source.as_mut().unwrap()["source_id"] = Value::from("system");
                    target = row("target", 77 * RATE, 80 * RATE);
                }
                "duplicate" => second = row("copy", 0, 3 * RATE),
                "overlap" => second = row("overlap", RATE, 4 * RATE),
                "conflict" => {
                    let mut copy = first.clone();
                    copy.language_detection.as_mut().unwrap()["probes"][0]["language"] =
                        Value::from("de");
                    extra.push(copy);
                }
                "source" => {
                    second.capture_source.as_mut().unwrap()["source_id"] = Value::from("system")
                }
                "stream" => {
                    second
                        .rest
                        .insert("stream_generation".into(), Value::from(1));
                }
                "epoch" => {
                    second.rest.insert("language_epoch".into(), Value::from(1));
                }
                "manual" => {
                    second
                        .rest
                        .insert("language_mode".into(), Value::from("en"));
                }
                "borrowed" => {
                    second.language_detection.as_mut().unwrap()["probes"][0]["decision"]["reason"] =
                        Value::from("recent_context")
                }
                "partial" => {
                    second.transcription_review = Some(serde_json::json!({"partial_text":true}))
                }
                "acoustic" => {
                    second.acoustic_evidence.as_mut().unwrap()["complete"] = Value::Bool(false)
                }
                "protected" => second.protected_fields = Some(vec!["text".into()]),
                "stale" => target = row("target", 73 * RATE, 76 * RATE),
                "future" => {
                    first = row("first", 3 * RATE, 6 * RATE);
                    second = row("second", 6 * RATE, 9 * RATE);
                    target = row("target", 0, 3 * RATE);
                }
                _ => {}
            }
            let mut rows = vec![first, second];
            rows.extend(extra);
            let context = saved_language_context(&rows, &target);
            assert_eq!(
                context.is_some(),
                matches!(
                    case,
                    "qualified" | "foreign_overlap" | "stale" | "future" | "protected"
                ),
                "{case}"
            );
            if let Some(context) = context {
                assert_eq!(context.language, "en");
                let evidence = context.evidence.unwrap();
                assert_eq!(evidence.anchors.len(), 2);
                assert_eq!(evidence.speaker_track_id.as_deref(), Some("speaker_0"));
                assert_eq!(evidence.capture_source, target.capture_source);
                if case == "foreign_overlap" {
                    continue;
                }
                // The saved crop clock is translated at the real router seam.
                let models = FakeModels::english();
                models.probe_scores.lock().unwrap().extend((0..2).map(|_| {
                    Ok(LanguageProbeOut {
                        codes: vec!["ru".into(), "en".into()],
                        probabilities: vec![0.51, 0.49],
                    })
                }));
                target.language = None;
                let decoded = refine_saved_passage_with_context(
                    &models,
                    &target,
                    &loud((3 * RATE) as usize),
                    None,
                    saved_language_context(&rows, &target),
                )
                .unwrap();
                assert_eq!(decoded.language.as_deref(), Some("en"));
                let decision = decoded.language_detection.clone().unwrap();
                assert_eq!(
                    decision["context_end_sample"],
                    if case == "future" { 9 * RATE } else { 6 * RATE }
                );
                assert_eq!(
                    decision["context_evidence"]["scope"],
                    "speaker_main_language"
                );
                assert_eq!(
                    decision["probes"][0]["start_sample"],
                    round_samples(target.start)
                );
                if matches!(case, "stale" | "future") {
                    continue;
                }
                let mut candidate = decoded.clone();
                candidate.language_detection = Some(decision);
                assert!(borrowed_context_current(&rows, &candidate));
                let mut spelling = rows.clone();
                spelling[0].text = "Protected donor spelling".into();
                spelling[0].protected_fields = Some(vec!["text".into()]);
                spelling[0].machine_revision = Some(spelling[0].machine_revision.unwrap_or(0) + 1);
                assert!(borrowed_context_current(&spelling, &candidate));
                assert!(
                    saved_language_context(&spelling, &target).is_some(),
                    "exactly two immutable acoustic supports survive spelling-only edits"
                );
                for change in ["speaker", "source", "epoch", "probe"] {
                    let mut changed = rows.clone();
                    match change {
                        "text" => {
                            changed[0].text = "Protected donor edit".into();
                            changed[0].protected_fields = Some(vec!["text".into()]);
                        }
                        "speaker" => {
                            changed[0].speaker = "different-track".into();
                            changed[0].protected_fields = Some(vec!["speaker".into()]);
                        }
                        "source" => {
                            changed[0].capture_source.as_mut().unwrap()["source_revision"] =
                                Value::from(2)
                        }
                        "epoch" => {
                            changed[0]
                                .rest
                                .insert("language_epoch".into(), Value::from(1));
                        }
                        "revision" => changed[0].machine_revision = Some(1),
                        "probe" => {
                            changed[0].language_detection.as_mut().unwrap()["probes"][0]["decision"]
                                ["probability"] = Value::from(0.95)
                        }
                        _ => unreachable!(),
                    }
                    assert!(
                        !borrowed_context_current(&changed, &candidate),
                        "selected donor {change}"
                    );
                    let mut document = crate::transcript::bare_document(
                        &[("speaker_0", "Renamed speaker")],
                        [changed, vec![target.clone()]].concat(),
                    );
                    assert!(
                        !publish_saved_passage(&mut document, &target, &candidate).unwrap(),
                        "selected donor {change}"
                    );
                    let mut fresh = crate::transcript::bare_document(
                        &[("speaker_0", "Renamed speaker")],
                        [rows.clone(), vec![target.clone()]].concat(),
                    );
                    assert!(
                        publish_saved_passage(&mut fresh, &target, &candidate).unwrap(),
                        "unchanged/fresh retry after {change}"
                    );
                }
            }
        }
    }
}
