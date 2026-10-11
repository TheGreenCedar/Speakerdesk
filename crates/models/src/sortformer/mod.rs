//! Nemotron-3 diarization. Upstream: `Blaizzy/mlx-audio@feb25a37`.
//! Weights: `mlx-community/Nemotron-3-Diarization@59ed2db`.
use std::path::Path;

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{DiarFeedOut, DiarPreset, Handle, Turn};

#[cfg(target_os = "macos")]
pub mod network;

pub const WEIGHT_PIN: &str =
    "mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f";

pub fn diar_open(_preset: DiarPreset) -> Result<Handle, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diar_feed(_handle: Handle, _pcm: &[f32], _last: bool) -> Result<DiarFeedOut, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diar_close(_handle: Handle) -> Result<(), ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diarize_file(_path: &Path) -> Result<Vec<Turn>, ChildError> {
    Err(ChildError::Unimplemented)
}

pub fn diarize_window(_pcm: &[f32]) -> Result<Vec<Turn>, ChildError> {
    Err(ChildError::Unimplemented)
}

/// Speakerdesk calls mlx-audio with these post-process settings
/// (`live_refinement.py`). Nemotron-3's speaker cap is 8 (`pipeline.py`).
const ACTIVITY_THRESHOLD: f32 = 0.5;
const MIN_DURATION_SECONDS: f64 = 0.0;
const MERGE_GAP_SECONDS: f64 = 0.0;
const MAX_SPEAKERS: usize = 8;
/// 160-sample hop at 16 kHz.
const FRAME_SECONDS: f64 = 0.01;

/// Turn a `[frame][speaker]` probability grid into turns.
///
/// Each row is one 10 ms frame. A column is active when its probability is
/// strictly above [`ACTIVITY_THRESHOLD`]. Runs shorter than
/// [`MIN_DURATION_SECONDS`] are dropped, and a gap of at most
/// [`MERGE_GAP_SECONDS`] joins neighboring runs of the same speaker. Both
/// durations are zero here, so every active run is kept and a silent frame
/// splits the speaker. Labels are zero-based `speaker_N`. Columns past the
/// eighth speaker are ignored.
pub fn frames_to_turns(frames: &[impl AsRef<[f32]>]) -> Vec<Turn> {
    let speakers = frames
        .iter()
        .map(|frame| frame.as_ref().len().min(MAX_SPEAKERS))
        .max()
        .unwrap_or(0);
    let mut spans = Vec::new();
    for speaker in 0..speakers {
        let mut runs = Vec::new();
        let mut begin = None;
        for (index, frame) in frames.iter().enumerate() {
            let probability = frame.as_ref().get(speaker).copied().unwrap_or(0.0);
            if probability > ACTIVITY_THRESHOLD {
                if begin.is_none() {
                    begin = Some(index);
                }
            } else if let Some(start) = begin.take() {
                keep_run(&mut runs, start, index);
            }
        }
        if let Some(start) = begin {
            keep_run(&mut runs, start, frames.len());
        }
        for (start, end) in merge_runs(runs) {
            spans.push((speaker, start, end));
        }
    }
    spans.sort_by_key(|&(speaker, start, _end)| (start, speaker));
    spans
        .into_iter()
        .map(|(speaker, start, end)| Turn {
            start: start as f64 * FRAME_SECONDS,
            end: end as f64 * FRAME_SECONDS,
            speaker: format!("speaker_{speaker}"),
        })
        .collect()
}

fn keep_run(runs: &mut Vec<(usize, usize)>, start: usize, end: usize) {
    let duration = (end - start) as f64 * FRAME_SECONDS;
    if duration >= MIN_DURATION_SECONDS {
        runs.push((start, end));
    }
}

fn merge_runs(runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut merged = Vec::with_capacity(runs.len());
    for (start, end) in runs {
        if let Some((previous_start, previous_end)) = merged.last().copied() {
            let gap = (start - previous_end) as f64 * FRAME_SECONDS;
            if gap <= MERGE_GAP_SECONDS {
                merged.pop();
                merged.push((previous_start, end));
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

#[cfg(test)]
mod tests {
    use speakerdesk_core::ipc::Turn;

    use super::frames_to_turns;

    /// Two turns match when they name the same speaker and share both edges.
    fn identical(left: &[Turn], right: &[Turn]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.speaker == b.speaker && a.start == b.start && a.end == b.end)
    }

    fn turn(speaker: &str, start: f64, end: f64) -> Turn {
        Turn {
            start,
            end,
            speaker: speaker.to_string(),
        }
    }

    #[test]
    fn synthetic_10ms_grid_matches_speaker_edges() {
        // Columns are speaker 0..8. Each row is one 10 ms frame.
        // Speaker 0 speaks, pauses one frame, speaks again, then overlaps speaker 7.
        // 0.5 is inactive. Column 8 is above threshold and must be clipped.
        let silent = [0.0; 9];
        let mut grid = vec![silent; 11];
        for frame in [1, 2, 3, 5, 6] {
            grid[frame][0] = 0.9;
        }
        grid[7][0] = 0.5;
        grid[7][1] = 0.51;
        for row in grid.iter_mut().skip(8) {
            row[0] = 1.0;
            row[7] = 0.75;
            row[8] = 1.0;
        }

        let turns = frames_to_turns(&grid);
        let expected = [
            turn("speaker_0", 0.01, 0.04),
            turn("speaker_0", 0.05, 0.07),
            turn("speaker_1", 0.07, 0.08),
            turn("speaker_0", 0.08, 0.11),
            turn("speaker_7", 0.08, 0.11),
        ];
        assert!(
            identical(&turns, &expected),
            "speakers and edges diverged: {turns:?}"
        );
    }

    #[test]
    fn empty_and_subthreshold_grids_have_no_turns() {
        assert!(frames_to_turns(&[] as &[[f32; 8]]).is_empty());
        let below = [[0.5_f32; 8], [0.0; 8]];
        assert!(frames_to_turns(&below).is_empty());
    }
}
