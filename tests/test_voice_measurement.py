"""Independent synthetic-vector calibration and fixture-leakage regressions."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from voice_measurement import calibrate, fixture_groups
from voice_profiles import VoiceClip, VoiceModel
MODEL = VoiceModel('cpu-fixture', 'not-a-real-model', 'a'*64, 2)


def group(gid, split, speaker, vector):
    return {'id': gid, 'recording_id': gid, 'split': split, 'speaker_id': speaker, 'vectors': [vector, vector],
            'selected': [VoiceClip(gid, gid, f'{gid}:0', 0, 2), VoiceClip(gid, gid, f'{gid}:1', 3, 5)]}


def labelled_groups():
    return [group('ref-a', 'enrollment', 'a', (1., 0.)), group('ref-b', 'enrollment', 'b', (0., 1.)),
            group('cal-a', 'calibration', 'a', (.95, .3122498999)), group('cal-b', 'calibration', 'b', (.2, .9797958971)),
            group('cal-unknown', 'calibration', None, (.707, .707)), group('held-a', 'held_out', 'a', (1., 0.)),
            group('held-b', 'held_out', 'b', (0., 1.)), group('held-unknown', 'held_out', None, (.707, .707))]


class CalibrationTests(unittest.TestCase):
    def test_measured_policy_rejects_ambiguous_unknown_and_passes_recurring_speakers(self):
        policy, held = calibrate(MODEL, labelled_groups(), 'cpu-pilot', maximum_far=0., maximum_frr=0.)
        self.assertGreater(policy.threshold,.707);self.assertLess(policy.threshold,.95)
        self.assertGreater(policy.margin,0.);self.assertLess(policy.margin,.6378)
        self.assertEqual((policy.genuine_trials, policy.impostor_trials), (2, 1))
        self.assertEqual((held['false_accepts'], held['false_rejects'], held['wrong_identities']), (0, 0, 0))
        self.assertTrue(held['meets_error_limits'])

    def test_held_out_failure_is_reported_without_retuning(self):
        groups = labelled_groups(); original, _ = calibrate(MODEL, groups, 'cpu', maximum_far=0., maximum_frr=0.)
        groups[-1]['vectors'] = [(1., 0.), (1., 0.)]
        frozen, held = calibrate(MODEL, groups, 'cpu', maximum_far=0., maximum_frr=0.)
        self.assertEqual(frozen, original); self.assertEqual(held['false_accept_rate'], 1.)
        self.assertFalse(held['meets_error_limits'])

    def test_wrong_identity_and_missing_unknown_trials_cannot_pass(self):
        groups = labelled_groups(); groups[2]['vectors'] = [(0., 1.), (0., 1.)]
        with self.assertRaisesRegex(ValueError, 'No observed'):
            calibrate(MODEL, groups, 'cpu', maximum_far=0., maximum_frr=0.)
        groups = [g for g in labelled_groups() if g['id'] != 'cal-unknown']
        with self.assertRaisesRegex(ValueError, 'unknown speakers'):
            calibrate(MODEL, groups, 'cpu', maximum_far=0., maximum_frr=0.)

    def test_fixture_consent_cleanliness_budget_and_recording_split_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'one.wav').write_bytes(b'fixture loader does not infer')
            manifest = {'schema_version': 1, 'dataset_id': 'cpu', 'fixture_source': 'synthetic', 'fixture_root': '.',
                        'groups': [{'id': 'one', 'recording_id': 'one', 'speaker_id': 'a', 'split': 'enrollment', 'file': 'one.wav',
                                    'clips': [{'start': 0, 'end': 2, 'clean': True}, {'start': 3, 'end': 5, 'clean': True}]}]}
            self.assertEqual(len(fixture_groups(manifest, root/'manifest.json', maximum_clips=2)), 1)
            for changed in [{'fixture_source': 'user-recording-without-consent'}, {'dataset_id': ''}]:
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    fixture_groups({**manifest, **changed}, root/'manifest.json', maximum_clips=2)
            with self.assertRaisesRegex(ValueError, 'bounded clip'):
                fixture_groups(manifest, root/'manifest.json', maximum_clips=1)
            leak = copy.deepcopy(manifest); leak['groups'].append({**leak['groups'][0], 'id': 'two', 'split': 'held_out'})
            with self.assertRaisesRegex(ValueError, 'leak'): fixture_groups(leak, root/'manifest.json', maximum_clips=4)
            (root/'copy.wav').write_bytes((root/'one.wav').read_bytes())
            leak['groups'][1].update(file='copy.wav', recording_id='different-label')
            with self.assertRaisesRegex(ValueError, 'leak'): fixture_groups(leak, root/'manifest.json', maximum_clips=4)
            overlap = copy.deepcopy(manifest); overlap['groups'][0]['clips'][1].update(start=1, end=3)
            with self.assertRaisesRegex(ValueError, 'nonoverlapping'): fixture_groups(overlap, root/'manifest.json', maximum_clips=2)
            dirty = copy.deepcopy(manifest); dirty['groups'][0]['clips'][0]['clean'] = False
            with self.assertRaisesRegex(ValueError, 'reviewed clean'): fixture_groups(dirty, root/'manifest.json', maximum_clips=2)


if __name__ == '__main__': unittest.main()
