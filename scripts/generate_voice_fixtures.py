"""Generate bounded synthetic development fixtures from verified local macOS voices.

Uses Apple's installed say and afconvert; never downloads assets, captures audio,
enrolls a Person, or imports/runs the speaker model. Run with a new output directory
outside this checkout and Python 3.10 or later. Output is reproducible for the recorded macOS/voice versions;
OS voice updates can change the audio bytes.
"""
import argparse
from array import array
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave


RATE = 16_000
DISK_FLOOR = 40_000_000_000
MAX_BYTES = 20_000_000  # Includes generated audio, metadata and temporary files.
MAX_CLIPS = 80
DATASET_ID = 'macos-compact-tts-development-v1'
MAUI_ROOT = Path('/System/Library/PrivateFrameworks/TextToSpeechMauiSupport.framework/Versions/A/Resources/TTSResources')
ASSET_ROOT = Path('/System/Library/AssetsV2/com_apple_MobileAsset_TTSAXResourceModelAssets')

# Restrict selection to ordinary compact voices with preinstalled local resources.
# No novelty voices, neural voice inference, voice download or quality upgrade.
VOICES = (
    ('samantha', 'Samantha (English (US))', 'en-US', 'Samantha', 'known'),
    ('daniel', 'Daniel (English (UK))', 'en-GB', 'Daniel', 'known'),
    ('karen', 'Karen', 'en-AU', 'Karen', 'known'),
    ('rishi', 'Rishi', 'en-IN', 'Rishi', 'known'),
    ('moira', 'Moira', 'en-IE', 'Moira', 'unknown_recurring'),
    ('tessa', 'Tessa', 'en-ZA', 'Tessa', 'unknown_recurring'),
    ('alice', 'Alice', 'it-IT', 'Alice', 'unknown_unseen'),
    ('anna', 'Anna', 'de-DE', 'Anna', 'unknown_unseen'),
)

# All 72 passages are different; none of the scripts is reused across splits.
TEXTS = {
    'enrollment': (
        'The planning meeting starts after lunch, and each team will bring a short list of priorities.',
        'Please leave enough time for questions before we agree on the schedule for the next release.',
        'Our first draft explains the customer problem, then describes the changes we expect to make.',
        'A clear owner for each action will help us finish the work before the next review.',
        'We should compare the latest figures with last month before deciding where to spend the budget.',
        'The new support guide includes a simple checklist and several examples from common customer questions.',
        'I can prepare the notes this afternoon if someone else checks the final numbers before tomorrow.',
        'Every department should have an opportunity to comment on the proposal before the document is approved.',
        'The design team is testing three small changes to help visitors find the information they need.',
        'We have collected the open questions and will discuss them with the project lead on Thursday.',
        'The workshop should begin with a short demonstration, followed by a practical exercise in small groups.',
        'Keeping the instructions concise will make it easier for new colleagues to complete their first task.',
        'The delivery estimate includes time for inspection, packaging, and a final check at the receiving office.',
        'Please record the decision in the shared notes so the afternoon team can follow the same process.',
        'Our regular review will focus on the remaining risks and the steps required to resolve them.',
        'We can move the discussion to Friday if the updated report needs another day of careful review.',
    ),
    'calibration': (
        'The train arrived early this morning, so we had time to walk through the quiet station square.',
        'A small bakery near the bridge sells fresh bread and opens before most of the nearby shops.',
        'I packed a light jacket because the forecast suggested a cool breeze later in the evening.',
        'The path beside the river passes several gardens before reaching a broad field beyond the town.',
        'We found a shaded table outside the library and spent an hour discussing the books we borrowed.',
        'The museum has added a new display about the tools used by local craftspeople many years ago.',
        'A volunteer at the entrance explained how to follow the marked route around the restored building.',
        'The afternoon clouds cleared just before sunset, revealing the hills on the far side of the valley.',
        'There is a comfortable waiting area upstairs where passengers can read while they wait for boarding.',
        'The visitor map shows several walking routes, including a gentle loop that returns to the main gate.',
        'I noticed a handwritten sign announcing a community concert in the park at the end of June.',
        'We decided to take the longer route home because the narrow street was closed for repairs.',
        'A row of young trees now separates the playground from the busy road along the eastern boundary.',
        'The cafe serves breakfast until noon, with a few simple choices that change during the week.',
        'We saved the directions before leaving because mobile reception becomes unreliable further into the countryside.',
        'The guide recommended bringing comfortable shoes and enough water for the steady climb to the lookout.',
        'The old clock above the market entrance has been repaired and now rings at the correct hour.',
        'Two wide windows bring plenty of daylight into the reading room even on an overcast winter morning.',
        'The ferry crosses the bay several times a day, but the last departure is earlier on Sundays.',
        'A careful inspection of the bicycle revealed that the rear brake needed a small adjustment before riding.',
        'The gardener suggested planting the herbs near the kitchen door where they would receive the morning sun.',
        'We chose a quiet corner of the campsite with level ground and a clear view of the lake.',
        'The exhibition includes a series of photographs showing how the neighborhood changed over the last century.',
        'I will return the borrowed umbrella tomorrow when I pass the office on my way to the station.',
    ),
    'held_out': (
        'The revised proposal gives the operations team a full week to check the process before the public launch.',
        'We should send the agenda in advance so everyone can prepare an example from their recent work.',
        'The quality report separates the issues that need immediate attention from those planned for a later release.',
        'I have reserved a smaller meeting room because only the project owners need to attend this discussion.',
        'The latest survey asks customers which parts of the service were clear and which needed more explanation.',
        'A short summary at the beginning of the document will help readers decide where to look for detail.',
        'Our finance colleague will review the purchase request after confirming that the supplier information is complete.',
        'The training session includes time to practice the new workflow and discuss any questions with the instructor.',
        'We need an updated inventory before ordering replacement parts for the equipment in the northern workshop.',
        'The research team has arranged several interviews with people who use the service regularly during the week.',
        'A consistent naming convention will help the support team find older records without searching several different folders.',
        'The maintenance window should be scheduled when the fewest customers are likely to need access to the system.',
        'I reviewed the draft announcement and added a sentence explaining how customers can contact the help desk.',
        'The project board now shows the completed tasks and the remaining dependencies for each planned delivery.',
        'We can divide the larger task into two smaller changes so reviewers can assess each part more easily.',
        'The office manager has requested a final count of attendees before confirming the catering order for Wednesday.',
        'A colleague noticed that the printed instructions use an older address and should be corrected before distribution.',
        'The weekly update will include a brief explanation of the delay and the next date we can confirm.',
        'The supplier offered to demonstrate the equipment at our office before we choose which configuration to purchase.',
        'Please check that the contact details are current before passing the request to the customer success team.',
        'We should review the findings together after the analysts finish checking the unusual entries in the report.',
        'The reception team keeps a spare copy of the building map for visitors who arrive outside normal hours.',
        'The prototype allows readers to adjust the display size while keeping the navigation in the same familiar position.',
        'A small group will test the instructions next week and report any steps that are difficult to follow.',
        'The warehouse supervisor has moved the damaged packages to a separate area until the inspection is complete.',
        'Our next conversation should focus on the assumptions that changed rather than repeat the entire project history.',
        'The revised checklist asks the reviewer to confirm both the date and the destination before approving a shipment.',
        'I will prepare an example using the new format so the rest of the team can see the expected result.',
        'The building caretaker recommended using the side entrance while repairs continue near the main reception desk.',
        'Each team member should record a short handover note before leaving for the scheduled break next month.',
        'The final presentation will explain the recommendation, the supporting evidence, and the questions that still need answers.',
        'We have enough time to review the revised plan tomorrow if the updated estimates arrive before this evening.',
    ),
}
CONDITIONS = {
    'enrollment': ((178, .82, 'direct'), (186, .72, 'lowpass_4800hz')),
    'calibration': ((170, .74, 'highpass_80hz'), (191, .88, 'direct')),
    'held_out': ((183, .78, 'lowpass_4800hz'), (174, .66, 'highpass_80hz')),
}
LIMITATION = ('Synthetic development pilot only. These OS TTS voices and simulated sessions do not establish '
              'real-human cross-meeting recognition accuracy, confidence intervals, acoustic overlap handling, '
              'microphone/noise robustness or deployment coverage. Alice and Anna are non-English compact '
              'voices reading English scripts; their accent may make unseen rejection easier. Clean labels '
              'mean a single generated voice with no mixed audio plus waveform checks, not human acoustic review.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


class Budget:
    def __init__(self, root):
        self.root = root
        self.initial_free = shutil.disk_usage(root.parent).free
        self.minimum_free = self.initial_free
        self.maximum_tree = 0
        if self.minimum_free < DISK_FLOOR + MAX_BYTES:
            raise ValueError('Need the 40 GB disk floor plus at most 20 MB for fixture generation.')

    def check(self):
        free = shutil.disk_usage(self.root.parent).free
        self.minimum_free = min(self.minimum_free, free)
        self.maximum_tree = max(self.maximum_tree, tree_bytes(self.root))
        if free < DISK_FLOOR:
            raise ValueError('Fixture generation breached the 40,000,000,000-byte free disk floor.')
        if self.maximum_tree > MAX_BYTES:
            raise ValueError('Fixture generation exceeded its 20,000,000-byte total output/scratch budget.')

    def run(self, command):
        self.check()
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    self.check()
                    if time.monotonic() - started > 15:
                        raise ValueError('A local audio utility exceeded its 15-second wall-time budget.')
                    time.sleep(.1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            self.check()
            log.seek(0)
            message = log.read(4000).decode(errors='replace')
            if process.returncode or 'sandbox_extension_issue_file failed' in message:
                raise ValueError(f'{command[0]} failed: {message or process.returncode}')


def installed_voices():
    listing = subprocess.run(['/usr/bin/say', '-v', '?'], capture_output=True, text=True, check=True, timeout=10).stdout
    names = {m.group(1).strip() for line in listing.splitlines()
             if (m := re.match(r'^(.*?)\s+[a-z]{2}_[A-Z]{2}\s+#', line))}
    # Match the exact installed compact identifiers, rather than trusting name aliases.
    jxa = """ObjC.import('AppKit');
const ids = ObjC.deepUnwrap($.NSSpeechSynthesizer.availableVoices);
JSON.stringify(ids.map(id => ObjC.deepUnwrap($.NSSpeechSynthesizer.attributesForVoice(id)))
.map(r => ({name:r.VoiceName,id:r.VoiceIdentifier,locale:r.VoiceLocaleIdentifier})));"""
    inventory = json.loads(subprocess.run(['/usr/bin/osascript', '-l', 'JavaScript', '-e', jxa],
                                         capture_output=True, text=True, check=True, timeout=10).stdout)
    assets = []
    if ASSET_ROOT.is_dir():
        for info in sorted(ASSET_ROOT.glob('*.asset/Info.plist')):
            with info.open('rb') as handle:
                properties = plistlib.load(handle).get('MobileAssetProperties', {})
            assets.append((info.parent, properties))
    evidence = []
    for key, name, locale, base_name, role in VOICES:
        voice_id = f'com.apple.voice.compact.{locale}.{base_name}'
        matching = [r for r in inventory if r['id'] == voice_id]
        if not matching or matching[0]['name'] not in names:
            raise ValueError(f'Exact installed compact voice unavailable: {name}. Do not download a replacement.')
        # macOS may add a language suffix to display names; the compact ID stays fixed.
        name = matching[0]['name']
        resource_root = MAUI_ROOT / locale / base_name
        info = resource_root / 'Info.plist'
        if not info.is_file():
            raise ValueError(f'Preinstalled local metadata missing for {name}; refusing synthesis.')
        with info.open('rb') as handle:
            properties = plistlib.load(handle).get('MobileAssetProperties', {})
        if (properties.get('Name') != base_name or properties.get('Type') != 'Maui'
                or locale not in properties.get('Languages', [])):
            raise ValueError(f'Unexpected preinstalled voice metadata for {name}; refusing synthesis.')
        data = sorted((resource_root / 'Contents').glob('*.dat'))
        # Some compact voices have a local asset in addition to their OS super-compact payload.
        for asset_root, asset_properties in assets:
            if asset_properties.get('VoiceId') == voice_id:
                data += sorted((asset_root / 'AssetData' / 'Contents').glob('*.dat'))
        if not data or sum(p.stat().st_size for p in data) < 100_000:
            raise ValueError(f'Local voice payload missing for {name}; refusing synthesis.')
        evidence.append({'key': key, 'say_name': name, 'voice_id': voice_id, 'locale': locale, 'role': role,
                         'metadata': str(info), 'metadata_sha256': digest(info),
                         'local_data_files': [{'path': str(p), 'bytes': p.stat().st_size} for p in data]})
    return listing, evidence


def read_pcm(path):
    with wave.open(str(path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, RATE, 'NONE'):
            raise ValueError('afconvert must produce mono PCM16 at 16 kHz.')
        samples = array('h', audio.readframes(audio.getnframes()))
    if sys.byteorder != 'little':
        samples.byteswap()
    return samples


def passage(samples, gain, channel):
    active = [i for i, value in enumerate(samples) if abs(value) > 32]
    if not active:
        raise ValueError('Synthetic passage is silent.')
    # Retain 80 ms around speech; never crop a longer sentence to meet clip bounds.
    samples = samples[max(0, active[0] - 1280):min(len(samples), active[-1] + 1281)]
    if not 2 * RATE <= len(samples) <= 10 * RATE:
        raise ValueError(f'A complete utterance must last 2–10 seconds, got {len(samples)/RATE:.3f}.')
    processed = array('h')
    previous_input = previous_output = 0.
    lowpass_alpha = 1 - math.exp(-2 * math.pi * 4800 / RATE)
    highpass_alpha = math.exp(-2 * math.pi * 80 / RATE)
    for value in samples:
        if channel == 'lowpass_4800hz':
            output = previous_output + lowpass_alpha * (value - previous_output)
        elif channel == 'highpass_80hz':
            output = highpass_alpha * (previous_output + value - previous_input)
        elif channel == 'direct':
            output = float(value)
        else:
            raise ValueError('Unknown channel condition.')
        previous_input, previous_output = float(value), output
        value = round(output * gain)
        if not -32766 <= value <= 32766:
            raise ValueError('Synthetic channel processing clipped the passage.')
        processed.append(value)
    rms = math.sqrt(sum(value * value for value in processed) / len(processed)) / 32768
    peak = max(abs(value) for value in processed) / 32768
    if rms < .001 or peak >= .999:
        raise ValueError('Synthetic passage failed silence/clipping checks.')
    return processed, {'seconds': len(processed) / RATE, 'rms': rms, 'peak': peak}


def write_pcm(path, samples):
    samples = array('h', samples)
    if sys.byteorder != 'little':
        samples.byteswap()
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(samples.tobytes())


def validate_manifest(manifest, path):
    # This import only validates fixture labels/bounds/hashes; it never loads a model.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
    from voice_measurement import fixture_groups
    groups = fixture_groups(manifest, path, maximum_clips=MAX_CLIPS)
    if len(groups) != 36 or sum(len(g['selected']) for g in groups) != 72:
        raise ValueError('Unexpected synthetic pilot group or passage count.')
    if len({g['recording_id'] for g in groups}) != len(groups) or len({g['audio_sha256'] for g in groups}) != len(groups):
        raise ValueError('Every synthetic session must have unique recording ID and bytes.')
    return groups


def generate(output):
    if platform.system() != 'Darwin':
        raise ValueError('This generator requires macOS and installed Apple compact voices.')
    checkout = Path(__file__).resolve().parents[1]
    output = output.expanduser().resolve()
    if output.is_relative_to(checkout) or output.exists() or not output.parent.is_dir():
        raise ValueError('Choose a new directory outside the Git checkout with an existing parent.')
    for executable in ('/usr/bin/say', '/usr/bin/afconvert', '/usr/bin/osascript'):
        if not Path(executable).is_file():
            raise ValueError(f'Required installed Apple utility unavailable: {executable}')
    budget = Budget(output)
    listing, voices = installed_voices()
    if len(set(sum((list(texts) for texts in TEXTS.values()), []))) != 72:
        raise ValueError('All passage scripts must be distinct across the dataset.')
    output.mkdir()
    started = time.monotonic()
    try:
        (output / '.gitignore').write_text('*\n')
        (output / 'say-installed-voices.txt').write_text(listing)
        (output / 'voice-provenance.json').write_text(json.dumps(voices, indent=2) + '\n')
        groups, passage_metrics = [], []
        with tempfile.TemporaryDirectory(prefix='.scratch-', dir=output) as scratch_name:
            scratch = Path(scratch_name)
            for split, scripts in TEXTS.items():
                split_root = output / split
                split_root.mkdir()
                cursor = 0
                for voice in voices:
                    if voice['role'] != 'known' and (split == 'enrollment' or
                            voice['role'] == 'unknown_unseen' and split != 'held_out'):
                        continue
                    for session, (rate, gain, channel) in enumerate(CONDITIONS[split], 1):
                        gid = f'{split}-{voice["key"]}-session-{session}'
                        samples, clips, texts = array('h'), [], []
                        for passage_index in range(2):
                            text = scripts[cursor]
                            cursor += 1
                            source, converted = scratch / 'passage.aiff', scratch / 'passage.wav'
                            # Voice IDs came from the installed inventory and locally verified payloads.
                            budget.run(['/usr/bin/say', '-v', voice['voice_id'], '-r', str(rate),
                                        '-o', str(source), text])
                            budget.run(['/usr/bin/afconvert', str(source), str(converted),
                                        '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1'])
                            processed, metrics = passage(read_pcm(converted), gain, channel)
                            source.unlink()
                            converted.unlink()
                            start = len(samples) / RATE
                            samples.extend(processed)
                            clips.append({'start': start, 'end': len(samples) / RATE, 'clean': True})
                            texts.append(text)
                            passage_metrics.append({'group': gid, 'passage': passage_index, **metrics})
                            if passage_index == 0:
                                samples.extend([0] * (RATE // 4))
                            budget.check()
                        audio = split_root / f'{gid}.wav'
                        budget.check()
                        write_pcm(audio, samples)
                        budget.check()
                        groups.append({'id': gid, 'recording_id': DATASET_ID + ':' + gid,
                                       'speaker_id': 'synthetic:' + voice['key'] if voice['role'] == 'known' else None,
                                       'split': split, 'file': str(audio.relative_to(output)), 'clips': clips,
                                       'synthetic_voice': voice['key'], 'session': session,
                                       'conditions': {'words_per_minute': rate, 'gain': gain, 'channel': channel},
                                       'passage_texts': texts})
                        print(f'Generated {gid}: 2 passages', flush=True)
                if cursor != len(scripts):
                    raise ValueError('Script count does not match the planned split.')
        manifest = {'schema_version': 1, 'dataset_id': DATASET_ID, 'fixture_source': 'synthetic',
                    'fixture_root': '.', 'groups': groups, 'limitation': LIMITATION}
        manifest_path = output / 'manifest.json'
        validated = validate_manifest(manifest, manifest_path)
        audio_bytes = sum(g['audio'].stat().st_size for g in validated)
        durations = [m['seconds'] for m in passage_metrics]
        report = {'schema_version': 1, 'dataset_id': DATASET_ID, 'fixture_source': 'synthetic',
                  'generator_sha256': digest(__file__), 'macos_version': platform.mac_ver()[0],
                  'voice_count': len(voices), 'recurring_known_voices': 4, 'recurring_unknown_voices': 2,
                  'entirely_unseen_held_out_voices': 2, 'session_groups': len(groups), 'embedding_clips': len(durations),
                  'groups_by_split': dict(Counter(g['split'] for g in groups)),
                  'clips_by_split': dict(Counter(g['split'] for g in groups for _ in g['clips'])),
                  'audio_bytes': audio_bytes, 'total_audio_seconds': sum(g['clips'][-1]['end'] for g in groups),
                  'passage_seconds_min': min(durations), 'passage_seconds_max': max(durations),
                  'rms_min': min(m['rms'] for m in passage_metrics), 'peak_max': max(m['peak'] for m in passage_metrics),
                  'elapsed_seconds': time.monotonic() - started, 'free_disk_before_bytes': budget.initial_free,
                  'disk_floor_bytes': DISK_FLOOR, 'maximum_generated_tree_bytes': budget.maximum_tree,
                  'audio_format': {'channels': 1, 'sample_rate': RATE, 'bits_per_sample': 16, 'encoding': 'PCM'},
                  'unique_recording_ids': len({g['recording_id'] for g in groups}),
                  'unique_audio_sha256': len({g['audio_sha256'] for g in validated}),
                  'waveform_checks': passage_metrics,
                  'recordings': [{'file': g['file'], 'bytes': g['audio'].stat().st_size,
                                  'sha256': g['audio_sha256']} for g in validated], 'limitation': LIMITATION}
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        budget.check()
        report['minimum_observed_free_disk_bytes'] = budget.minimum_free
        report['free_disk_after_bytes'] = shutil.disk_usage(output).free
        report['disk_headroom_after_bytes'] = report['free_disk_after_bytes'] - DISK_FLOOR
        report['maximum_generated_tree_bytes'] = budget.maximum_tree
        (output / 'generation-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        budget.check()
        print(json.dumps({'manifest': str(manifest_path), 'groups': len(groups), 'clips': len(durations),
                          'audio_bytes': audio_bytes, 'total_tree_bytes': tree_bytes(output),
                          'free_disk_bytes': shutil.disk_usage(output).free}, indent=2))
        return output
    except BaseException:
        # Own only this previously nonexistent directory; keep failures from looking complete.
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error('Use an already installed Python 3.10 or later for the fixture schema validator.')
    try:
        generate(args.output_dir)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, IndexError, wave.Error) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
