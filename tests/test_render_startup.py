"""Bounded unpublished prefix controls; no models, playback or devices."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from render_startup import StartupSourceSink,StartupReceiptWriter,verify_prefix,POLICY_SHA256
from audio import pcm16_bytes
from capture_sources import catalog,validate_catalog,binding


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.far=np.random.default_rng(17).normal(0,.1,64000).astype(np.float32)
        self.raw=np.zeros_like(self.far);self.raw[1232:]=.18*self.far[:-1232]
        self.timer=patch('render_startup.threading.Timer');self.timer.start();self.addCleanup(self.timer.stop)

    def stream(self,raw=None,packet=496,epochs=None,cover=None,finish=True,render_epochs=None):
        raw=self.raw if raw is None else raw;out=[];records=[];horizons=[]
        def publish(mix,tracks):
            out.append(tracks['microphone_clean'].copy());records.extend(tracks['innovation_decisions'])
            horizons.append((sink.selector.end,len(np.concatenate(out))))
            np.testing.assert_array_equal(mix,np.clip(self.far[len(np.concatenate(out))-len(mix):len(np.concatenate(out))]+tracks['microphone_clean'],-1,1))
        sink=StartupSourceSink(publish)
        for a in range(0,len(raw),packet):
            b=min(a+packet,len(raw));coverage=np.ones(b-a,bool) if cover is None else cover[a:b]
            tracks=dict(microphone=raw[a:b],system_reference=np.repeat(self.far[a:b,None],2,axis=1),
                microphone_present=coverage,system_present=coverage)
            if render_epochs is not None:tracks['system_reference_epoch']=np.full(b-a,render_epochs(a),dtype=np.int64)
            sink.retain(tracks,a,epochs(a) if epochs else 0)
            sink.emit(np.clip(self.far[a:b]+raw[a:b],-1,1),dict(microphone=raw[a:b],system=self.far[a:b],microphone_clean=raw[a:b]))
        if finish:sink.flush()
        return np.concatenate(out) if out else np.empty(0,np.float32),records,sink,horizons

    def test_playback_from_start_removes_known_echo_without_padding_unknown_reference(self):
        y,rows,sink,h=self.stream()
        np.testing.assert_array_equal(y[:1232],self.raw[:1232])
        self.assertLess(float(np.sqrt(np.mean(y[1232:]**2))),1e-7)
        self.assertEqual(h[0][1],14000)
        self.assertTrue(any(r['method']=='startup_innovation' for r in rows))

    def test_packet_partition_has_identical_publication_samples_and_no_rewrites(self):
        a,_,_,_=self.stream(packet=37);b,_,_,_=self.stream(packet=4000)
        np.testing.assert_array_equal(a,b)

    def test_first_frame_quiet_and_normal_speech_pay_cap_and_keep_native_prefix(self):
        near=np.random.default_rng(23).normal(0,.03,len(self.raw)).astype(np.float32)
        for gain in (1.,.5):
            raw=self.raw+gain*near;y,_,sink,h=self.stream(raw=raw)
            np.testing.assert_array_equal(y,raw);self.assertEqual(h[0][1],32000)
            self.assertEqual(sink.publication_log[0]['reason'],'source_cap')

    def test_prefix_route_change_preserves_mismatching_old_path(self):
        raw=self.raw.copy();raw[10000:]=.18*self.far[10000-2400:-2400]
        y,_,_,_=self.stream(raw=raw)
        np.testing.assert_array_equal(y[:10000],raw[:10000])

    def test_reference_gap_flushes_native_and_cannot_cross_held_support(self):
        cover=np.ones(len(self.raw),bool);cover[7000]=False
        y,rows,sink,_=self.stream(cover=cover)
        np.testing.assert_array_equal(y[:14000],self.raw[:14000])
        self.assertEqual(sink.publication_log[0]['reason'],'reference_gap')
        self.assertFalse(any(r['method']=='startup_innovation' for r in rows))

    def test_EOF_and_epoch_flush_preserve_unqualified_samples_once(self):
        y,_,sink,_=self.stream(raw=self.raw[:4000]);np.testing.assert_array_equal(y,self.raw[:4000])
        self.assertEqual(sink.publication_log[0]['reason'],'epoch_or_EOF')
        y,rows,_,_=self.stream(packet=2000,epochs=lambda a:int(a>=8000))
        np.testing.assert_array_equal(y[:8000],self.raw[:8000]);self.assertEqual(len(y),len(self.raw))
        self.assertEqual({r['epoch'] for r in rows},{0,1})

    def test_qualified_new_route_still_exposes_unknown_reference_prefix(self):
        # A qualified second path cannot retroactively supply reference history
        # across the epoch. This explains a real retained ASR ghost, rather than
        # asserting that successful later cancellation qualifies the prefix.
        raw=self.raw.copy();raw[32000:]=.18*self.far[32000-2400:-2400]
        y,rows,sink,_=self.stream(raw=raw,packet=2000,epochs=lambda a:int(a>=32000))
        np.testing.assert_array_equal(y[32000:34400],raw[32000:34400])
        self.assertGreater(float(np.sqrt(np.mean(y[32000:34400]**2))),.01)
        self.assertLess(float(np.sqrt(np.mean(y[34400:]**2))),1e-7)
        prefix=[r for r in rows if r['start_sample']<34400 and r['end_sample']>32000]
        self.assertTrue(prefix)
        self.assertTrue(all(r['method']=='native_fallback' and r['reason']=='unknown_prefix_reference' for r in prefix))
        self.assertEqual(sink.publication_log[-1]['epoch'],1)
        self.assertEqual(sink.publication_log[-1]['reason'],'qualified_path')

    def test_microphone_reset_uses_only_continuous_explicit_render_history(self):
        raw=self.raw.copy();raw[32000:]=.18*self.far[32000-2400:-2400]
        y,rows,_,_=self.stream(raw=raw,packet=2000,epochs=lambda a:int(a>=32000),render_epochs=lambda a:0)
        self.assertLess(float(np.sqrt(np.mean(y[32000:34400]**2))),1e-7)
        evidence=[r['reference_history'] for r in rows if r.get('reference_history')]
        self.assertTrue(evidence)
        self.assertTrue(all(h['end_sample']==32000 and h['render_epoch']==0 for h in evidence))
        # Same PCM but changed system route is not continuous reference truth.
        stale,rows,_,_=self.stream(raw=raw,packet=2000,epochs=lambda a:int(a>=32000),render_epochs=lambda a:int(a>=32000))
        np.testing.assert_array_equal(stale[32000:34400],raw[32000:34400])
        self.assertFalse(any(r.get('reference_history') for r in rows))

    def test_immediate_near_after_microphone_reset_keeps_native_prefix(self):
        echo=self.raw.copy();echo[32000:]=.18*self.far[32000-2400:-2400]
        near=np.random.default_rng(31).normal(0,.03,2000).astype(np.float32)
        for gain in (1.,.1):
            with self.subTest(gain=gain):
                raw=echo.copy();raw[32000:34000]+=gain*near
                y,rows,_,_=self.stream(raw=raw,packet=2000,epochs=lambda a:int(a>=32000),render_epochs=lambda a:0)
                np.testing.assert_array_equal(y[32000:34000],raw[32000:34000])
                self.assertLess(float(np.sqrt(np.mean(y[36000:]**2))),1e-7)

    def test_render_gap_cannot_be_hidden_by_microphone_epoch_change(self):
        raw=self.raw.copy();raw[32000:]=.18*self.far[32000-2400:-2400]
        covered=np.ones(len(raw),bool);covered[30000:32000]=False
        y,rows,_,_=self.stream(raw=raw,packet=2000,epochs=lambda a:int(a>=32000),cover=covered,render_epochs=lambda a:0)
        np.testing.assert_array_equal(y[32000:34400],raw[32000:34400])
        self.assertFalse(any(r.get('reference_history') for r in rows))

    def test_wall_timeout_releases_native_and_does_not_wait_for_more_capture(self):
        out=[];sink=StartupSourceSink(lambda _,t:out.append(t['microphone_clean'].copy()))
        raw=self.raw[:4000];ref=np.repeat(self.far[:4000,None],2,axis=1)
        sink.retain(dict(microphone=raw,system_reference=ref,microphone_present=np.ones(4000,bool),system_present=np.ones(4000,bool)),0,0)
        sink.emit(raw+self.far[:4000],dict(microphone=raw,system=self.far[:4000],microphone_clean=raw))
        self.assertFalse(out)
        with patch('render_startup.time.monotonic',return_value=sink.started+2.):sink._timeout(sink.generation)
        np.testing.assert_array_equal(out[0],raw)
        sink.flush();self.assertEqual(len(out),1)

    def test_old_awakened_timer_cannot_release_a_new_epoch_hold(self):
        out=[];sink=StartupSourceSink(lambda _,t:out.append(t['microphone_clean'].copy()))
        def send(a,epoch):
            raw=self.raw[a:a+4000];far=self.far[a:a+4000]
            sink.retain(dict(microphone=raw,system_reference=np.repeat(far[:,None],2,axis=1),
                microphone_present=np.ones(4000,bool),system_present=np.ones(4000,bool)),a,epoch)
            sink.emit(raw+far,dict(microphone=raw,system=far,microphone_clean=raw))
        send(0,0);old=sink.generation;send(4000,1)
        self.assertEqual(len(out),1);new=sink.generation;self.assertNotEqual(old,new)
        with patch('render_startup.time.monotonic',return_value=sink.started+3.):sink._timeout(old)
        self.assertEqual(len(out),1);self.assertEqual(sink.held_count,4000)
        with patch('render_startup.time.monotonic',return_value=sink.started+2.):sink._timeout(new)
        self.assertEqual(len(out),2);sink.flush()

    def test_timer_publication_failure_surfaces_on_capture_flush(self):
        def fail(_,tracks):raise OSError('Retained disk write failed')
        sink=StartupSourceSink(fail);raw=self.raw[:4000];far=self.far[:4000]
        sink.retain(dict(microphone=raw,system_reference=np.repeat(far[:,None],2,axis=1),
            microphone_present=np.ones(4000,bool),system_present=np.ones(4000,bool)),0,0)
        sink.emit(raw+far,dict(microphone=raw,system=far,microphone_clean=raw))
        with patch('render_startup.time.monotonic',return_value=sink.started+2.):sink._timeout(sink.generation)
        with self.assertRaisesRegex(OSError,'disk write'):sink.flush()

    def test_receipts_and_catalog_bind_same_selected_live_and_saved_PCM(self):
        with tempfile.TemporaryDirectory() as folder:
            writer=StartupReceiptWriter(folder);saved=wave.open(str(Path(folder)/'microphone_clean.wav'),'wb')
            saved.setparams((1,2,16000,0,'NONE','none'));live=[]
            def publish(_,tracks):
                data=pcm16_bytes(tracks['microphone_clean']);live.append(data);saved.writeframes(data)
                writer.write(writer.end,tracks)
            sink=StartupSourceSink(publish)
            for a in range(0,len(self.raw),4000):
                b=a+4000;raw=self.raw[a:b];far=self.far[a:b]
                sink.retain(dict(microphone=raw,system_reference=np.repeat(far[:,None],2,axis=1),microphone_present=np.ones(4000,bool),system_present=np.ones(4000,bool)),a,0)
                sink.emit(raw+far,dict(microphone=raw,system=far,microphone_clean=raw))
            sink.flush();saved.close();writer.close()
            with wave.open(str(Path(folder)/'microphone_clean.wav'),'rb') as w:self.assertEqual(w.readframes(w.getnframes()),b''.join(live))
            summary=json.loads((Path(folder)/'innovation-evidence.json').read_text());self.assertEqual(summary['policy_sha256'],POLICY_SHA256)
            c=catalog('a'*32,innovation=True,startup=True);self.assertEqual(validate_catalog(c,'a'*32),c)
            self.assertNotEqual(binding(c,'microphone_clean'),binding(catalog('a'*32,innovation=True),'microphone_clean'))
            altered=json.loads(json.dumps(c));altered['derivation']['parameters_sha256']='0'*64
            with self.assertRaises(ValueError):validate_catalog(altered,'a'*32)


if __name__=='__main__':unittest.main()
