"""Offline labelled evaluation. Never enrolls People or updates meeting documents."""
import math
import hashlib
from pathlib import Path
from voice_profiles import Calibration, VoiceClip, make_profile, match_from_ranked, rank_matches


def fixture_groups(manifest, manifest_path, *, maximum_clips):
    if (manifest.get('schema_version') != 1 or not manifest.get('dataset_id')
            or manifest.get('fixture_source') not in ('synthetic', 'consented', 'licensed')):
        raise ValueError('Labelled fixtures need a dataset ID and synthetic, consented or licensed provenance.')
    root = (Path(manifest_path).parent/manifest['fixture_root']).resolve()
    groups = manifest.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('Supply labelled recording groups.')
    seen_ids, splits, all_clips, count, hashes = set(), {}, {}, 0, {}
    audio_bytes = 0
    result = []
    for group in groups:
        gid, recording = group['id'], group['recording_id']
        split, speaker = group['split'], group['speaker_id']
        if (not isinstance(gid, str) or not gid or gid in seen_ids or not isinstance(recording, str)
                or not recording or split not in ('enrollment', 'calibration', 'held_out', 'smoke')
                or speaker is not None and (not isinstance(speaker, str) or not speaker)):
            raise ValueError('Use unique groups, recording IDs, split names and known or unknown labels.')
        seen_ids.add(gid)
        audio = (root/group['file']).resolve()
        if not audio.is_relative_to(root) or not audio.is_file():
            raise ValueError('Fixture audio must exist inside its declared root.')
        if audio not in hashes:
            audio_bytes += audio.stat().st_size
            if audio.stat().st_size > 64*1024*1024 or audio_bytes > 256*1024*1024:
                raise ValueError('Use bounded pilot recordings: 64 MiB per file and 256 MiB total.')
            digest = hashlib.sha256()
            with audio.open('rb') as handle:
                for chunk in iter(lambda: handle.read(512*1024), b''):
                    digest.update(chunk)
            hashes[audio] = digest.hexdigest()
        for key in (('recording', recording), ('file', str(audio)), ('content', hashes[audio])):
            if key in splits and splits[key] != split:
                raise ValueError('A recording cannot leak between enrollment, calibration and held-out sets.')
            splits[key] = split
        clips = group['clips']
        if not isinstance(clips, list) or not 2 <= len(clips) <= 12:
            raise ValueError('Each query or reference needs 2–12 separate clean passages.')
        selected = []
        for index, clip in enumerate(clips):
            start, end = clip['start'], clip['end']
            if (clip.get('clean') is not True or not isinstance(start, (int, float))
                    or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end or not 2 <= end-start <= 10):
                raise ValueError('Every fixture passage needs a reviewed clean label and 2–10 second bounds.')
            selected.append(VoiceClip(recording, gid, f'{gid}:{index}', start, end))
            spans = all_clips.setdefault(str(audio), [])
            if any(start < previous_end and previous_start < end for previous_start, previous_end in spans):
                raise ValueError('Fixture passages must be distinct and nonoverlapping.')
            spans.append((start, end))
        count += len(selected)
        if count > maximum_clips:
            raise ValueError('Fixture selection exceeds this run’s bounded clip count.')
        result.append({**group, 'audio': audio, 'audio_sha256': hashes[audio], 'selected': selected})
    return result


def extract_groups(backend, groups):
    results, metrics = [], []
    for group in groups:
        vectors = []
        for clip in group['selected']:
            embedding = backend.embed(group['audio'], clip)
            if embedding.clean is not True:
                raise ValueError(f'Fixture passage {clip.segment_id} failed waveform quality checks.')
            vectors.append(embedding.vector)
            metrics.append({'group': group['id'], 'segment_id': clip.segment_id, **(embedding.metrics or {})})
        results.append({**group, 'vectors': vectors})
    return results, metrics


def reference_profiles(model, groups):
    references = {}
    for group in groups:
        if group['split'] == 'enrollment':
            if group['speaker_id'] is None:
                raise ValueError('Enrollment reference fixtures must have known labels.')
            bucket = references.setdefault(group['speaker_id'], {'vectors': [], 'clips': []})
            bucket['vectors'].extend(group['vectors']); bucket['clips'].extend(group['selected'])
    if len(references) < 2:
        raise ValueError('Calibration needs at least two labelled reference speakers.')
    return [{'person_id': speaker, 'version': 'offline-fixture', **make_profile(model, data['vectors'], data['clips'])}
            for speaker, data in references.items()]


def trial_counts(model, policy, queries, profiles, ranks):
    counts = {'genuine_trials': 0, 'impostor_trials': 0, 'false_accepts': 0, 'false_rejects': 0,
              'wrong_identities': 0}
    enrolled = {p['person_id'] for p in profiles}
    for query in queries:
        expected = query['speaker_id']
        if expected is not None and expected not in enrolled:
            raise ValueError('Non-enrolled fixture speakers must be explicitly labelled unknown.')
        match = match_from_ranked(model, policy, ranks[query['id']])
        if expected is None:
            counts['impostor_trials'] += 1
            counts['false_accepts'] += match is not None
        else:
            counts['genuine_trials'] += 1
            counts['false_rejects'] += match is None or match['person_id'] != expected
            counts['wrong_identities'] += match is not None and match['person_id'] != expected
    if not counts['genuine_trials'] or not counts['impostor_trials']:
        raise ValueError('Each evaluation set requires recurring known speakers and unknown speakers.')
    return {**counts, 'false_accept_rate': counts['false_accepts']/counts['impostor_trials'],
            'false_reject_rate': counts['false_rejects']/counts['genuine_trials']}


def calibrate(model, groups, dataset_id, *, maximum_far, maximum_frr):
    """Select from observed calibration scores, then evaluate untouched held-out queries."""
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (maximum_far, maximum_frr)):
        raise ValueError('Specify reviewed error-rate limits between zero and one.')
    profiles = reference_profiles(model, groups)
    calibration = [g for g in groups if g['split'] == 'calibration']
    held_out = [g for g in groups if g['split'] == 'held_out']
    ranks = {g['id']: rank_matches(model, g['vectors'], profiles) for g in calibration}
    thresholds, margins = set(), set()
    for query in calibration:
        ranked = ranks[query['id']]
        best, minimum, _ = ranked[0]
        gap = best-ranked[1][0]
        for values, score in ((thresholds, minimum), (margins, gap)):
            for boundary in (score, math.nextafter(score, math.inf)):
                if 0 < boundary <= 1:
                    values.add(boundary)
    if len(thresholds)*len(margins) > 100000:
        raise ValueError('Calibration search exceeds its bounded candidate count; review a smaller pilot first.')
    feasible = []
    for threshold in sorted(thresholds):
        for margin in sorted(margins):
            provisional = Calibration(model, dataset_id+':calibration', threshold, margin, 1, 1, 0., 0.)
            counts = trial_counts(model, provisional, calibration, profiles, ranks)
            if (counts['false_accept_rate'] <= maximum_far and counts['false_reject_rate'] <= maximum_frr
                    and counts['wrong_identities'] == 0):
                feasible.append((counts['false_reject_rate'], counts['false_accept_rate'], -threshold, -margin, counts))
    if not feasible:
        raise ValueError('No observed threshold and margin satisfy the requested calibration limits.')
    _, _, negative_threshold, negative_margin, counts = min(feasible, key=lambda row: row[:4])
    policy = Calibration(model, dataset_id+':calibration', -negative_threshold, -negative_margin,
                         counts['genuine_trials'], counts['impostor_trials'],
                         counts['false_accept_rate'], counts['false_reject_rate'])
    held_ranks = {g['id']: rank_matches(model, g['vectors'], profiles) for g in held_out}
    held = trial_counts(model, policy, held_out, profiles, held_ranks)
    passed = (held['false_accept_rate'] <= maximum_far and held['false_reject_rate'] <= maximum_frr
              and held['wrong_identities'] == 0)
    return policy, {**held, 'dataset_id': dataset_id+':held-out', 'meets_error_limits': passed}
