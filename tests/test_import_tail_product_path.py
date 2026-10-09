"""Independent real upload/run/API/renderer controls; no neural inference."""
import io, json, os, re, subprocess, sys, tempfile, time, types, unittest, wave
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

SOURCE = Path(os.environ.get('FAINT_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
FIXTURE = Path(os.environ.get('FAINT_RECEIPTS', SOURCE/'tests/metadata/import_tail_receipts.json'))
EVIDENCE = Path(os.environ['FAINT_REVIEW_EVIDENCE']) if os.environ.get('FAINT_REVIEW_EVIDENCE') else None
sys.path[:0] = [str(SOURCE/'speakerdesk'), str(SOURCE/'tests')]
from app import create_app
import inference_worker
from speech_admission import SpeechFrames
from reading_turns import validated_turns
from test_language_detection import model_modules
from pcm_peer import varying_pcm
RECEIPTS = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else None
RATE = 16000

class ReceiptSpeech:
    def __init__(self, records): self.records = records
    def session(self, start_sample=0, *, archive=None):
        ledger = SpeechFrames(start_sample)
        for a,b,_,_,observation in self.records:
            ledger.append(a,b,max(observation['raw_probability'],observation['normalized_probability']),observation=observation)
        assert [list(frame) for frame in ledger.frames] == self.records
        session = types.SimpleNamespace(received=start_sample,evidence=ledger)
        def feed(audio,start,*,final=False):
            assert start == session.received
            session.received += len(audio)
            if final: assert session.received == ledger.end_sample
        session.feed=feed
        return session

@unittest.skipUnless(RECEIPTS, 'Retained synthetic import tail observations required')
class ImportTailProductPath(unittest.TestCase):
    def imported(self, name, case_name='auto-real-thanks', turns=None, texts=None, alignment_provider=None, expected_status='ready'):
        case=RECEIPTS[case_name];count=case['frames'][-1][1]
        pcm=varying_pcm(count,3,dtype='<i2')
        pcm[max(f[1] for f in case['frames'] if f[3]):]=0
        wav=io.BytesIO()
        with wave.open(wav,'wb') as out:
            out.setparams((1,2,RATE,0,'NONE','none'));out.writeframes(pcm.tobytes())
        wav.seek(0)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoint=root/'checkpoint';checkpoint.mkdir()
            (checkpoint/'model.safetensors').touch()
            config=dict(diar_path='cpu',cohere_path=str(checkpoint),speech_path='cpu',lid_path='cpu',
                        diar_python=sys.executable,asr_python=sys.executable,diar_kind='nemotron',device='mlx')
            if alignment_provider is not None:config['alignment_path']='contract-test-provider'
            asr=Mock()
            continuous=RECEIPTS['gpu_continuous_probe']['raw_text']
            def decode(samples,**kwargs):
                if texts: text=texts[asr.transcribe.call_count-1]
                elif case_name=='auto-real-thanks': text=continuous if len(samples)>=44800 else 'you'
                else: text=case['result']['words'][0]['text']
                return types.SimpleNamespace(text='  '+text+'  ',tokens=[1])
            asr.transcribe.side_effect=decode
            detector=Mock();detector.detect.return_value=dict(en=.99,fr=.01)
            def worker(_python,task,request,_folder):
                if task=='diarize': return {'turns':case['diarization'] if turns is None else turns}
                return inference_worker.run(task,request)
            def wait(jid,active):
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    job=client.get('/api/jobs/'+jid).json
                    if job['status'] not in active: return job
                    time.sleep(.005)
                self.fail('Controlled import did not finish')
            app=create_app(root/'saved-home');client=app.test_client()
            token=re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').get_data(as_text=True)).group(1)
            headers={'X-Speakerdesk-Token':token}
            try:
                with patch.dict(sys.modules,model_modules(asr)), patch('inference_worker.check_memory'), \
                     patch('inference_worker.importlib.metadata.version',return_value='CPU boundary peer'), \
                     patch('speech_admission.SileroModel',return_value=ReceiptSpeech(case['frames'])), \
                     patch('language_detection.WhisperLanguageDetector',return_value=detector), \
                     patch('pipeline.model_config',return_value=config), patch('app.model_config',return_value=config), \
                     patch('coarse_alignment.CoarseAlignment',return_value=alignment_provider,
                           side_effect=alignment_provider if isinstance(alignment_provider,Exception) else None), \
                     patch('app.preflight',return_value=[]), patch('pipeline.preflight',return_value=[]), \
                     patch('pipeline.run_worker',side_effect=worker):
                    response=client.post('/api/jobs',headers=headers,data={'files':(wav,'synthetic-control.wav'),'language':'auto'})
                    response.request.input_stream.close();self.assertEqual(response.status_code,201)
                    jid=response.json[0]['id'];self.assertEqual(wait(jid,{'preparing'})['status'],'uploaded')
                    self.assertEqual(client.post('/api/jobs/'+jid+'/run',headers=headers).status_code,202)
                    job=wait(jid,{'queued','processing'});self.assertEqual(job['status'],expected_status,job.get('message'))
                    if expected_status=='failed':
                        self.assertIsNone(job.get('document'))
                        return [],dict(ASR_calls=asr.transcribe.call_count,job=job,neural_models_executed=False)
                    exported=client.get('/api/jobs/'+jid+'/export/txt').get_data(as_text=True)
                tag=os.environ.get('FAINT_RUN_TAG','review')+'-'+name
                evidence_dir=EVIDENCE or root
                payload=evidence_dir/(tag+'-job.json');payload.write_text(json.dumps(job,indent=2)+'\n')
                script="""
const fs=require('node:fs'),assert=require('node:assert/strict');
const [root,payload]=process.argv.slice(1),{frontend}=require(root+'/tests/support/frontend_dom.cjs');
const job=JSON.parse(fs.readFileSync(payload)),f=frontend(root);f.seed(0);f.context.job=job;
f.run('selected=structuredClone(job);doc=structuredClone(selected.document);window.original=JSON.stringify(doc);renderEditor()');
const rows=f.document.getElementById('segments').children.map(c=>({speaker:c.dataset.speaker,text:c.querySelector('textarea')?.value,
 label:c.querySelector('select')?.children.find(o=>o.selected)?.textContent,
 readingLabels:c.querySelectorAll('.reading-turn-speaker').map(t=>t.textContent)}));
assert.equal(f.run('JSON.stringify(doc)===original'),true);console.log(JSON.stringify(rows));
"""
                rendered=json.loads(subprocess.run(['node','-e',script,str(SOURCE),str(payload)],check=True,capture_output=True,text=True,timeout=20).stdout)
                rows=[s for s in job['document']['segments'] if s['text']]
                summary=dict(ASR_calls=asr.transcribe.call_count,words=[dict(text=s['text'],speaker=s['speaker'],
                    cohere_raw_text=s.get('cohere_raw_text'),activity_regions=s.get('activity_regions'),
                    speaker_candidates=s.get('speaker_candidates'),reading_turns=validated_turns(s)) for s in rows],
                    rendered=rendered,exported_txt=exported,diarization=job['document']['diarization'],neural_models_executed=False)
                (evidence_dir/(tag+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
                return rows,summary
            finally:
                app.extensions['speakerdesk']['meetings'].close()
                app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)

    def test_continuous_tail_keeps_previously_known_sentence_attributable(self):
        # Fabricated timing explicitly tests the producer/consumer contract;
        # independent actual GPU evidence is qualified outside this CPU suite.
        from test_import_reading_worker import ContractAligner
        rows,evidence=self.imported('known-sentence',alignment_provider=ContractAligner())
        sentence=RECEIPTS['gpu_continuous_probe']['raw_text']
        row=next(s for s in rows if s['text']==sentence)
        self.assertEqual(row['cohere_raw_text'],'  '+sentence+'  ')
        self.assertEqual(evidence['diarization'],[dict(start=0.,end=2.8000000000000003,speaker='speaker_0')])
        assigned=(row['speaker']=='speaker_0' or bool(validated_turns(row)) and
                  all(t['speaker']=='speaker_0' for t in validated_turns(row)))
        self.assertTrue(assigned,'The complete previously known sentence lost its speaker; the API and shipped renderer have no qualified reading turns to retain it')

    def test_A_B_A_boundaries_and_raw_text_survive_saved_API_and_renderer(self):
        texts=['A opening.','B response.','A return.']
        rows,evidence=self.imported('A-B-A',turns=['0 1 speaker_0','1 2 speaker_1','2 3.9698125 speaker_0'],texts=texts)
        self.assertEqual([s['text'] for s in rows],texts)
        self.assertEqual([s['speaker'] for s in rows],['speaker_0','speaker_1','speaker_0'])
        self.assertEqual([r['speaker'] for r in evidence['rendered']],['speaker_0','speaker_1','speaker_0'])

    def test_unavailable_required_provider_does_not_publish_incomplete_import(self):
        rows,evidence=self.imported('unavailable-provider',alignment_provider=FileNotFoundError('Research weights unavailable'),expected_status='failed')
        self.assertEqual(evidence['ASR_calls'],1)
        self.assertEqual(rows,[])
        self.assertIn('Required transcript timing failed to load',evidence['job']['message'])
        self.assertIsNone(evidence['job'].get('document'))

    def test_true_overlap_stays_local_and_tail_unassigned(self):
        rows,evidence=self.imported('overlap',turns=['0 2.8 speaker_0','0 2.8 speaker_1'],texts=['Two people.','you'])
        self.assertEqual([s['speaker'] for s in rows],['overlap','unassigned'])
        self.assertEqual([r['speaker'] for r in evidence['rendered']],['overlap','unassigned'])
        self.assertEqual([s['text'] for s in rows],['Two people.','you'])

    def test_quiet_known_words_and_zero_tail_keep_speaker(self):
        rows,evidence=self.imported('quiet',case_name='auto-quiet')
        self.assertEqual([s['speaker'] for s in rows],['speaker_0'])
        self.assertEqual(rows[0]['text'],RECEIPTS['auto-quiet']['result']['words'][0]['text'])
        self.assertEqual([r['speaker'] for r in evidence['rendered']],['speaker_0'])
