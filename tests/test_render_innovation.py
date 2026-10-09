"""Observable causal output/retention controls; no model execution."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
import hashlib
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from render_innovation import CausalInnovation, HISTORY, InnovationSourceSink, InnovationReceiptWriter, POLICY_SHA256
from audio import pcm16_bytes


class StreamingInnovationTests(unittest.TestCase):
    def setUp(self):
        self.far=np.random.default_rng(17).normal(0,.1,64000).astype(np.float32)
        self.raw=np.zeros_like(self.far); self.raw[1232:]=.18*self.far[:-1232]

    def run_stream(self,raw=None,far=None,packet=496,epochs=None,covered=None):
        raw=self.raw if raw is None else raw; far=self.far if far is None else far
        cover=np.ones(len(raw),bool) if covered is None else covered
        s=CausalInnovation(); out=[]; records=[]
        for a in range(0,len(raw),packet):
            b=min(len(raw),a+packet)
            epoch=epochs(a) if epochs else 0
            y,r=s.process(a,epoch,raw[a:b],np.repeat(far[a:b,None],2,axis=1),raw[a:b],cover[a:b])
            out.append(y); records.extend(r)
        return np.concatenate(out),records,s

    def test_initial_native_prefix_is_exact_and_every_used_path_precedes_output(self):
        y,rows,s=self.run_stream()
        np.testing.assert_array_equal(y[:HISTORY],self.raw[:HISTORY])
        self.assertLess(float(np.sqrt(np.mean(y[HISTORY:]**2))),1e-7)
        for row in rows:
            if row['path']:self.assertLessEqual(row['path']['observed_end'],row['start_sample'])

    def test_packet_partition_and_invalid_sample_are_causal(self):
        raw=self.raw.copy(); raw[30011]=1
        a,_,_=self.run_stream(raw=raw,packet=37)
        b,_,_=self.run_stream(raw=raw,packet=752)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a[30011],raw[30011])

    def test_first_frame_normal_and_quiet_overlap_do_not_create_false_support(self):
        near=np.random.default_rng(23).normal(0,.03,len(self.raw)).astype(np.float32)
        for level in (1.,.5):
            raw=self.raw+level*near; y,rows,_=self.run_stream(raw=raw)
            np.testing.assert_array_equal(y,raw)
            self.assertTrue(all(r['method']=='native_fallback' for r in rows))

    def test_correlated_near_after_initial_fit_cannot_refit_gain(self):
        raw=self.raw.copy(); raw[20000:]+=.004*self.far[20000-1232:-1232]
        y,_,s=self.run_stream(raw=raw)
        near=.004*self.far[20000-1232:-1232]
        np.testing.assert_allclose(y[20000:],near,atol=1e-7,rtol=1e-5)
        self.assertAlmostEqual(s.frozen['gain'],.18,places=6)

    def test_genuine_repeat_after_render_ends_is_preserved_exactly(self):
        far=self.far.copy(); far[20000:]=0
        raw=np.zeros_like(far); raw[1232:]=.18*far[:-1232]
        raw[32000:40000]=self.far[3000:11000]
        y,_,_=self.run_stream(raw=raw,far=far)
        np.testing.assert_array_equal(y[32000:40000],raw[32000:40000])

    def test_explicit_epoch_discards_prior_without_reusing_initial_words(self):
        y,rows,_=self.run_stream(packet=2000,epochs=lambda a:int(a>=30000))
        np.testing.assert_array_equal(y[30000:30000+HISTORY],self.raw[30000:30000+HISTORY])
        self.assertTrue(all(r['method']=='native_fallback' for r in rows if 30000<=r['start_sample']<44000))

    def test_reference_gap_invalidates_prior_and_requires_new_support(self):
        covered=np.ones(len(self.raw),bool); covered[30000:32000]=False
        y,rows,_=self.run_stream(covered=covered,packet=2000)
        np.testing.assert_array_equal(y[30000:46000],self.raw[30000:46000])
        self.assertTrue(any(r['reason']=='uncovered_nonfinite_or_clipped' for r in rows))

    def test_same_epoch_changed_delay_declines_and_cannot_absorb_genuine_near_repeat(self):
        raw=self.raw.copy(); raw[32000:]=.18*self.far[32000-2400:-2400]
        y,rows,s=self.run_stream(raw=raw,packet=2000)
        # A new path is observable only after the first changed window. This
        # retained 125ms transition is an explicit limitation, not a safety proof.
        # These SAME samples may be changed echo or genuine delayed near speech.
        # Without an independent capture epoch, qualification cannot decide.
        np.testing.assert_array_equal(y[34000:],raw[34000:])
        self.assertFalse(any(r['path'] and r['path']['lag']==2400 for r in rows))
        self.assertAlmostEqual(s.frozen['gain'],.18,places=6)
        self.assertEqual(s.frozen['lag'],1232)

    def test_explicit_route_epoch_can_qualify_new_delay_only_after_preceding_support(self):
        raw=self.raw.copy(); raw[32000:]=.18*self.far[32000-2400:-2400]
        y,rows,_=self.run_stream(raw=raw,packet=2000,epochs=lambda a:int(a>=32000))
        np.testing.assert_array_equal(y[32000:46000],raw[32000:46000])
        changed=[r for r in rows if r['path'] and r['path']['lag']==2400]
        self.assertTrue(changed);self.assertEqual(changed[0]['start_sample'],46000)
        self.assertLess(float(np.sqrt(np.mean(y[46000:]**2))),1e-7)

    def test_expired_support_cannot_refit_correlated_near_into_echo(self):
        far=np.tile(self.far,9)
        raw=np.zeros_like(far); raw[1232:]=.18*far[:-1232]
        raw[20000:]+=.004*far[20000-1232:-1232]
        y,rows,s=self.run_stream(raw=raw,far=far,packet=2000)
        self.assertTrue(any(r['reason']=='expired_support' for r in rows))
        np.testing.assert_array_equal(y[500000:],raw[500000:])
        self.assertAlmostEqual(s.frozen['gain'],.18,places=6)

    def test_two_different_render_paths_cannot_choose_arbitrary_channel(self):
        ref=np.column_stack((self.far,np.concatenate((self.far[8:],np.zeros(8,np.float32)))))
        s=CausalInnovation()
        y,rows=s.process(0,0,self.raw,ref,self.raw,np.ones(len(self.raw),bool))
        np.testing.assert_array_equal(y,self.raw)
        self.assertTrue(all(r['method']=='native_fallback' for r in rows))

    def test_retained_receipts_pin_native_and_selected_pcm_and_refuse_future_fit(self):
        selected,rows,_=self.run_stream(packet=len(self.raw))
        tracks=dict(microphone=self.raw,system=self.far,microphone_native=self.raw,
            microphone_clean=selected,innovation_reference=np.repeat(self.far[:,None],2,axis=1),
            innovation_coverage=np.ones(len(self.raw),bool),innovation_decisions=rows,
            innovation_native_mix=np.clip(self.raw+self.far,-1,1),
            innovation_selected_mix=np.clip(selected+self.far,-1,1))
        with tempfile.TemporaryDirectory() as folder:
            writer=InnovationReceiptWriter(folder)
            bad={**tracks,'innovation_decisions':[dict(rows[0],policy_sha256='wrong')]}
            with self.assertRaises(ValueError):writer.write(0,bad)
            self.assertEqual(writer.end,0)
            import copy
            future=copy.deepcopy(rows)
            active=next(row for row in future if row['path'])
            active['path']['observed_end']=active['start_sample']+1
            with self.assertRaises(ValueError):writer.write(0,{**tracks,'innovation_decisions':future})
            self.assertEqual(writer.end,0)
            writer.write(0,tracks);writer.close()
            summary=json.loads((Path(folder)/'innovation-evidence.json').read_text())
            self.assertEqual(summary['samples'],len(self.raw))
            self.assertEqual(summary['policy_sha256'],POLICY_SHA256)
            self.assertFalse(summary['production_admissible'])
            for name,array in (('microphone_native',self.raw),('microphone_clean',selected)):
                self.assertEqual(summary['pcm_sha256'][name],hashlib.sha256(pcm16_bytes(array)).hexdigest())
            with wave.open(str(Path(folder)/'microphone_native.wav')) as w:
                self.assertEqual(w.readframes(w.getnframes()),pcm16_bytes(self.raw))

    def test_selected_mix_and_native_mix_are_separate_and_consistent(self):
        emitted=[]; sink=InnovationSourceSink(lambda mix,tracks:emitted.append((mix,tracks)))
        for a in range(0,len(self.raw),2000):
            b=a+2000; raw=self.raw[a:b]; far=self.far[a:b]
            tracks=dict(microphone=raw,system=far,system_reference=np.repeat(far[:,None],2,axis=1),
                microphone_present=np.ones(len(raw),bool),system_present=np.ones(len(raw),bool))
            sink.retain(tracks,a,0)
            sink.emit(np.clip(raw+far,-1,1),dict(microphone=raw,system=far,microphone_clean=raw))
        self.assertEqual(sink.queued,0)
        for mixed,tracks in emitted:
            np.testing.assert_array_equal(mixed,np.clip(tracks['system']+tracks['microphone_clean'],-1,1))
            np.testing.assert_array_equal(tracks['innovation_native_mix'],np.clip(tracks['system']+tracks['microphone_native'],-1,1))
        self.assertTrue(any(not np.array_equal(mixed,t['innovation_native_mix']) for mixed,t in emitted))


if __name__=='__main__':unittest.main()
