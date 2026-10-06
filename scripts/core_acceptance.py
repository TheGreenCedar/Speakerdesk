"""Fail-closed core acoustic release policy; no models or publishing in validation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile
import unicodedata
import subprocess
import wave
import array
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
HARNESSES=('scripts/packaged_replay.py','scripts/replay_capture.py','scripts/acceptance_fixtures.py','scripts/core_acceptance.py',
           'speakerdesk/admission_receipt.py','speakerdesk/speech_admission.py')
GROUPS={'silence','background_noise','quiet_speech','brief_speech','genuine_thanks','sentence_continuity','overlap','language_fallback','protected_corrections'}

def require(value,message):
    if not value:raise ValueError(message)

def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024**2):digest.update(block)
    return digest.hexdigest()

def assignment(path,name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return node.value
    raise ValueError('Missing model identity: '+name)

def models(root=ROOT):
    values=assignment(root/'speakerdesk/model_setup.py','SPECS')
    imported={'LID_SPEC':'language_detection.py','SILERO_SPEC':'speech_admission.py',
              'ALIGNMENT_SPEC':'alignment_artifact.py'}
    specs=[]
    for value in values.elts:
        if isinstance(value,ast.Name):
            require(value.id in imported,'Unresolved model identity: '+value.id)
            value=assignment(root/'speakerdesk'/imported[value.id],value.id)
        specs.append(ast.literal_eval(value))
    return {s['directory']:{**{k:s[k] for k in ('repo','revision','bytes','sha256','files')},
                           **({'weight_file':s['weight_file']} if 'weight_file' in s else {})} for s in specs}

def model_file_pins(root=ROOT):
    pinned=json.loads((root/'tests/acceptance/model-metadata.json').read_text())['models'];result={}
    for name,spec in models(root).items():
        require(pinned[name]['repo']==spec['repo'] and pinned[name]['revision']==spec['revision'],'Model metadata revision differs')
        hashes=dict(pinned[name]['file_sha256']);hashes[spec.get('weight_file','model.safetensors')]=spec['sha256']
        require(set(hashes)==set(spec['files']),'Model metadata inventory differs')
        result[name]=hashes
    return result

def suite(root=ROOT):
    paths=[root/'tests/acceptance/recipes.json',root/'tests/acceptance/holdout-recipes.json']
    values=[json.loads(p.read_text()) for p in paths]
    cases=list(values[0]['cases'])
    for raw in values[1]['recipes']:
        case=dict(raw,language=raw['language_mode'],expect=dict(raw['expected']))
        case['groups']=list(raw.get('groups',['sentence_continuity']))
        if 'gratitude' in raw['id']:case['groups']=['genuine_thanks','sentence_continuity']
        if 'preceding_context_case' in raw:case['groups'].append('language_fallback')
        case['parts']=[dict(p,kind='zero',seconds=p['duration_ms']/1000) if p['kind']=='pause' else
                       dict(p,voice=raw['voiceid'],rate=raw['rate_words_per_minute']) for p in raw['parts']]
        cases.append(case)
    ids=[case['id'] for case in cases]
    require(len(ids)==len(set(ids)),'Duplicate acoustic case')
    require(set(values[0]['required_groups'])==GROUPS,'Required core groups changed')
    require(all(case['partition'] in ('tuning','holdout') for case in cases),'Invalid fixture partition')
    require(GROUPS<={group for case in cases for group in case['groups']},'Core coverage incomplete')
    require(any(case['partition']=='holdout' and 'sentence_continuity' in case['groups'] for case in cases),'Independent sentence holdout missing')
    require(any(case['partition']=='holdout' and 'genuine_thanks' in case['groups'] for case in cases),'Independent genuine-words holdout missing')
    identity={p.name:digest_file(p) for p in paths}
    return cases,identity

def words(text):
    return re.findall(r'\w+', ''.join(c for c in unicodedata.normalize('NFKD',text.casefold())
                      if not unicodedata.combining(c)))

def transport_pcm(path):
    with wave.open(str(path),'rb') as stream:
        require((stream.getnchannels(),stream.getsampwidth(),stream.getframerate())==(1,2,16000),'Invalid replay PCM')
        # Replay emits exactly representable PCM16/32768 float32. The capture
        # writer now rounds the inverse and saturates; every source PCM16 bit
        # survives. Keep the original0.5.1 evaluator/results unchanged elsewhere.
        return stream.readframes(stream.getnframes())

def established_context(observed,boundary_sample):
    # Pure production reconciliation policy: no app, model, or MLX import.
    sys.path.insert(0,str(ROOT/'speakerdesk'))
    from meeting_refinement import preceding_language_context
    return preceding_language_context(observed.get('document',{}).get('segments',[]),boundary_sample,observed.get('language_epoch',0))

def pause_acknowledged(job,request_id,through_sample):
    receipt=job.get('pause_flush') or {}
    available=receipt.get('available_sample');speech=receipt.get('speech_observed_sample')
    return (isinstance(request_id,str) and bool(request_id) and receipt.get('request_id')==request_id
        and type(through_sample) is int and through_sample>=0
        and receipt.get('state')=='complete' and type(receipt.get('through_sample')) is int and receipt['through_sample']==through_sample
        and type(receipt.get('received_sample')) is int and receipt['received_sample']==through_sample
        and type(available) is int and type(speech) is int and 0<=available<=speech<=through_sample
        and receipt.get('deferred_audio')==([{'start_sample':available,'end_sample':through_sample}] if available<through_sample else [])
        and type(receipt.get('fast_sequence')) is int and receipt['fast_sequence']==job.get('last_fast_sequence',0))

def evaluate(recipe, trace):
    """Recompute mandatory assertions from primary production documents."""
    checks={};expected=recipe['expect']
    for stage in ('provisional','refined'):
        job=trace.get(stage,{})
        rows=job.get('document',{}).get('segments',[])
        primary=[row for row in rows if row.get('text','').strip()]
        tokens=words(' '.join(row['text'] for row in primary))
        if 'exact_text' in expected:checks[stage+'_exact_text']=tokens==words(expected['exact_text'])
        negative=bool(expected.get('no_words') and job.get('canonical_utterances'))
        if negative:
            inspected=canonical_admission(trace,stage)
            checks[stage+'_observed']=isinstance(job.get('id'),str) and inspected
            checks[stage+'_local_models']=inspected
        else:
            checks[stage+'_observed']=isinstance(job.get('id'),str) and bool(job.get('last_fast_sequence'))
            checks[stage+'_local_models']=job.get('document',{}).get('provenance',{}).get('kind')=='local_inference'
        if expected.get('no_words'):checks[stage+'_no_words']=not tokens
        if expected.get('terms'):
            offset=0;complete=True
            for term in expected['terms']:
                for token in words(term):
                    try:offset=tokens.index(token,offset)+1
                    except ValueError:complete=False
            checks[stage+'_coverage']=complete
        joined=' '+' '.join(tokens)+' '
        for index,phrase in enumerate(expected.get('required_phrases',[])):
            checks[f'{stage}_phrase_{index}']=' '+' '.join(words(phrase))+' ' in joined
        occurrences=list(expected.get('phrase_occurrences',[]))
        if 'minimum_thank_you_count' in expected:
            occurrences.append({'phrase':'thank you','minimum':expected['minimum_thank_you_count']})
        for index,item in enumerate(occurrences):
            phrase=words(item['phrase']);count=sum(tokens[i:i+len(phrase)]==phrase for i in range(len(tokens)))
            checks[f'{stage}_occurrences_{index}']=item.get('minimum',0)<=count<=item.get('maximum',float('inf'))
        if 'maximum_word_rows' in expected:checks[stage+'_continuity']=len(primary)<=expected['maximum_word_rows']
        if 'reading_turns' in expected:checks[stage+'_turns']=len(primary)==expected['reading_turns']
        if expected.get('language'):checks[stage+'_language']=bool(primary) and all(r.get('language')==expected['language'] for r in primary)
        if expected.get('mixed_evidence'):
            mixed=[r for r in rows if any(len(a.get('speakers',[]))>1 for a in r.get('activity_regions',[]))
                   or len(r.get('speaker_candidates',[]))>1]
            checks[stage+'_overlap_observed']=bool(mixed)
            checks[stage+'_mixed_not_clean']=bool(mixed) and all(r.get('voice_eligible') is False for r in mixed)
        if expected.get('recent_context'):
            probes=[p for r in rows for p in r.get('language_detection',{}).get('probes',[r.get('language_detection',{})])]
            checks[stage+'_actual_context_fallback']=any(p.get('decision',p).get('reason')=='recent_context' for p in probes)
    final=trace.get('refined',{})
    checks['provisional_paused_before_refinement']=trace.get('provisional',{}).get('status')=='paused' and trace.get('provisional',{}).get('rolling_refinement',{}).get('cancelled') is True
    paused=trace.get('provisional',{})
    checks['provisional_pause_acknowledged']=pause_acknowledged(paused,trace.get('pause_request_id'),trace.get('input_frames',-1))
    checks['same_job']=final.get('id')==trace.get('job_id')==trace.get('provisional',{}).get('id')
    if expected.get('no_words') and final.get('canonical_utterances'):
        uncertainty=final.get('canonical_uncertain_samples')
        checks['negative_audio_inspected']=(canonical_admission(trace,'refined') and final.get('status')=='ready'
            and not final.get('rolling_sources') and not final.get('refinement_unresolved')
            and ((uncertainty==0 and final.get('refinement_status')=='complete'
                  and final.get('rolling_refinement',{}).get('completed_sample',0)>=trace.get('input_frames',float('inf')))
                 or (type(uncertainty) is int and uncertainty>0 and final.get('refinement_status')=='unresolved'
                     and final.get('rolling_refinement',{}).get('completed_sample',0)<trace.get('input_frames',0))))
    else:
        checks['refinement_executed']=final.get('status')=='ready' and final.get('refinement_status')=='complete' and not final.get('refinement_unresolved') and final.get('rolling_refinement',{}).get('completed_sample',0)>=trace.get('input_frames',float('inf'))
    saved=trace.get('saved_audio',{})
    checks['audio_preserved']=saved.get('frames')==trace.get('input_frames') and isinstance(saved.get('pcm_sha256'),str) and saved['pcm_sha256']==saved.get('expected_pcm_sha256')
    if expected.get('protected_correction'):
        edit=trace.get('edit',{});row=next((r for r in final.get('document',{}).get('segments',[]) if r.get('id')==edit.get('id')), {})
        checks['protected_correction']=row.get('text')==recipe['correction'] and 'text' in row.get('protected_fields',[]) and bool(edit.get('machine_revision'))
    if recipe.get('preceding_context_case'):
        predecessor=trace.get('preceding',{})
        observed=predecessor.get('observed',{});boundary=trace.get('slice_start_sample',0)
        context=established_context(observed,boundary)
        checks['actual_preceding_context']=predecessor.get('case')==recipe['preceding_context_case'] and predecessor.get('job_id')==observed.get('id')==final.get('id') and boundary>0 and context is not None and context.get('language')=='en' and context==predecessor.get('context')
        rows=final.get('document',{}).get('segments',[])
        events=[(p.get('decision',p),p.get('language',r.get('language'))) for r in rows for p in r.get('language_detection',{}).get('probes',[r.get('language_detection',{})])]
        checks['actual_language_contradiction']=any(e.get('reason')=='detected' and language==expected['language'] for e,language in events)
    return checks


def canonical_admission(trace,stage):
    """Only a bound actual-inspection receipt can qualify canonical no-word audio.

    A Pause may leave an explicit sub-frame tail; Stop must inspect every sample.
    Uncertainty remains uncertainty. This never substitutes for speech assertions.
    """
    sys.path.insert(0,str(ROOT/'speakerdesk'))
    from admission_receipt import validate_receipt
    job=trace.get(stage,{})
    try:
        if stage=='provisional':
            pause=job.get('pause_flush') or {};receipt=pause.get('admission_receipt')
            if not pause_acknowledged(job,trace.get('pause_request_id'),trace.get('input_frames')):return False
            validate_receipt(receipt,job.get('admission_execution') or {},phase='pause',
                request_id=trace.get('pause_request_id'),received_sample=trace['input_frames'],
                observed_sample=pause.get('speech_observed_sample'),
                pcm_sha256=trace.get('admission_pcm',{}).get(stage))
        else:
            receipt=job.get('capture_admission');request=job.get('capture_inspection_request') or {}
            if request.get('through_sample')!=trace.get('input_frames'):return False
            validate_receipt(receipt,job.get('admission_execution') or {},phase='stop',
                request_id=request.get('request_id'),received_sample=trace['input_frames'],
                observed_sample=job.get('canonical_observed_sample'),
                uncertain_samples=job.get('canonical_uncertain_samples'),
                pcm_sha256=trace.get('admission_pcm',{}).get(stage))
            if job.get('admission_execution')!=trace.get('provisional',{}).get('admission_execution'):return False
        return (job.get('id')==trace.get('job_id') and isinstance(trace.get('admission_pcm',{}).get(stage),str)
                and receipt['speech_samples']==0 and receipt['decision'] in ('no_speech','uncertain'))
    except (ValueError,TypeError,KeyError,AttributeError):return False

def artifact(path,expected):
    require(path.is_file() and not path.is_symlink(),'Evidence file unavailable: '+path.name)
    require(type(expected.get('bytes')) is int and expected['bytes']>=0 and path.stat().st_size==expected['bytes'],'Evidence size differs: '+path.name)
    require(digest_file(path)==expected.get('sha256'),'Evidence hash differs: '+path.name)

def zip_bundle(zipped):
    """Canonical inventory avoids accepting a different signed app with one matching executable."""
    inventory={}
    for item in zipped.infolist():
        parts=Path(item.filename).parts
        if item.is_dir() or not parts or not parts[0].endswith('.app'):continue
        require('..' not in parts and not item.filename.startswith('/'),'Unsafe signed archive member')
        require(item.filename not in inventory,'Duplicate signed archive member')
        with zipped.open(item) as stream:inventory[item.filename]=hashlib.file_digest(stream,'sha256').hexdigest()
    require(inventory,'Signed app inventory empty')
    return inventory

def bundle_identity(inventory):
    return hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate(report_path,manifest_path,*,root=ROOT):
    """Every mandatory case must be an actual packaged run on exact signed bytes."""
    report_path=Path(report_path);manifest_path=Path(manifest_path)
    require(report_path.is_file(),'Required packaged core acceptance report missing')
    report=json.loads(report_path.read_text());manifest=json.loads(manifest_path.read_text())
    require(manifest.get('product')=='Speakerdesk' and manifest.get('signing')=='developer-id' and manifest.get('notarized') is True,'Exact signed candidate required')
    require(re.fullmatch(r'[0-9a-f]{40}',manifest.get('source_commit','')) is not None,'Invalid candidate source')
    require(report.get('schema_version')==1 and report.get('scope')=='packaged_production_replay','Source/mock/browser evidence cannot satisfy packaged acoustics')
    require(report.get('source_commit')==manifest['source_commit'],'Core report source differs')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    require(head==manifest['source_commit'],'Validate against the exact candidate source checkout')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(),'Candidate source checkout changed')
    require(report.get('status')=='passed' and report.get('complete') is True and report.get('models_executed') is True,'Core acceptance failed, skipped, or incomplete')
    require(report.get('private_audio') is False and report.get('native_capture') is False,'Only authorized synthetic replay qualifies')
    require(report.get('model_substitution') is False and report.get('probabilities_modified') is False,'Substitute model/evidence rejected')
    require(report.get('runtime_scope')=='frozen_packaged_backend_live_and_refinement','Production packaged live/refinement route unverified')
    require(report.get('model_pins')==models(root),'Core model pins differ')
    require(report.get('model_files')==model_file_pins(root),'Loaded checkpoint or configuration metadata differs')
    cases,suite_hashes=suite(root)
    require(report.get('suite_sha256')==suite_hashes,'Fixture suite changed; rerun required')
    require(report.get('harness_sha256')=={name:digest_file(root/name) for name in HARNESSES},'Acceptance harness changed; rerun required')
    archive=next((item for item in manifest['files'] if item['filename'].endswith('.app.zip')),None)
    require(archive is not None and report.get('package_sha256')==archive['sha256'],'Core acceptance package differs')
    payload=manifest_path.parent/archive['filename'];artifact(payload,archive)
    require(report.get('signature_verified') is True and isinstance(report.get('app_cdhash'),str) and re.fullmatch(r'[0-9a-f]{40,64}',report['app_cdhash']) is not None,'Signed execution identity missing')
    with zipfile.ZipFile(payload) as zipped:
        require(report.get('bundle_sha256')==bundle_identity(zip_bundle(zipped)),'Executed bundle differs from signed payload')
        members=[item for item in zipped.infolist() if item.filename.endswith('/Contents/MacOS/speakerdesk-runtime')]
        require(len(members)==1 and 0<members[0].file_size<=500*1024**2,'Unexpected packaged runtime inventory')
        digest=hashlib.sha256()
        with zipped.open(members[0]) as stream:
            while block:=stream.read(1024**2):digest.update(block)
        require(report.get('runtime_sha256')==digest.hexdigest(),'Executed runtime differs from signed payload')
    observed=report.get('cases',[])
    require(len(observed)==len(cases) and {c.get('id') for c in observed}=={c['id'] for c in cases},'Missing or extra core cases')
    by_id={c['id']:c for c in observed}
    for recipe in cases:
        case=by_id[recipe['id']]
        require(case.get('partition')==recipe['partition'] and case.get('groups')==recipe['groups'],'Core case identity differs')
        require(case.get('status')=='passed' and case.get('stages')==['provisional','refined'],'Required core case not run/passed: '+recipe['id'])
        require(case.get('assertions') and all(value is True for value in case['assertions'].values()),'Core assertion failed: '+recipe['id'])
        for key in ('input_wav','saved_wav','trace','synthesis'):
            evidence=case.get(key,{})
            require(isinstance(evidence.get('file'),str),'Core raw evidence missing')
            relative=Path(evidence['file']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe evidence path')
            artifact(report_path.parent/relative,evidence)
        raw=json.loads((report_path.parent/case['trace']['file']).read_text())
        synthesis=json.loads((report_path.parent/case['synthesis']['file']).read_text())
        require(synthesis.get('recipe_id')==recipe['id'],'Synthetic fixture recipe differs')
        require(raw.get('synthesis')==synthesis,'Synthetic fixture trace differs')
        with wave.open(str(report_path.parent/case['input_wav']['file']),'rb') as pcm:
            require((pcm.getnchannels(),pcm.getsampwidth(),pcm.getframerate())==(1,2,16000) and pcm.getnframes()==raw.get('input_frames'),'Core input PCM identity differs')
        with wave.open(str(report_path.parent/case['saved_wav']['file']),'rb') as pcm:
            require((pcm.getnchannels(),pcm.getsampwidth(),pcm.getframerate())==(1,2,16000),'Saved production PCM format differs')
            actual=hashlib.sha256(pcm.readframes(pcm.getnframes())).hexdigest()
        expected=hashlib.sha256(transport_pcm(report_path.parent/case['input_wav']['file'])).hexdigest()
        require(actual==expected==raw['saved_audio'].get('pcm_sha256')==raw['saved_audio'].get('expected_pcm_sha256'),'Saved production audio differs from input transport')
        if recipe['expect'].get('no_words') and raw.get('refined',{}).get('canonical_utterances'):
            sys.path.insert(0,str(root/'speakerdesk'))
            from admission_receipt import retained_pcm_digest
            for stage in ('provisional','refined'):
                end=(raw[stage].get('pause_flush',{}).get('speech_observed_sample') if stage=='provisional'
                     else raw[stage].get('canonical_observed_sample'))
                require(raw.get('admission_pcm',{}).get(stage)==retained_pcm_digest(
                    report_path.parent/case['saved_wav']['file'],end),'Admission receipt PCM differs from retained input')
        recomputed=evaluate(recipe,raw)
        require(case['assertions']==recomputed and all(v is True for v in recomputed.values()),'Raw core evidence fails: '+recipe['id'])
    require(report.get('owned_processes_stopped') is True,'Packaged replay processes not cleaned up')
    budget=report.get('budget',{});lease=budget.get('lease',{})
    require(lease.get('admitted') is True and type(lease.get('seconds')) is int and 0<budget.get('wall_seconds',0)<=lease['seconds'],'Bounded resource run unverified')
    require(0<budget.get('peak_owned_rss_bytes',0)<=4*1024**3 and budget.get('minimum_free_memory_percent',0)>=25,'Memory budget failed')
    require(budget.get('minimum_free_disk_bytes',0)>=lease.get('floor_plus_other_reserve_bytes',float('inf')),'Shared disk budget failed')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args();validate(args.report,args.manifest);print('Exact packaged core acceptance passed; publication is a separate authorized action.')
