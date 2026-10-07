"""Versioned verifier corrections; raw transcripts and original v1 stay intact."""
import array,ast,copy,hashlib,json,re,sys,unicodedata
from pathlib import Path
from core_acceptance import evaluate as evaluate_v1,require,models,assignment,ROOT
SPEECH_REVISION=models()['silero-speech']['revision']
SPEECH_POLICY=ast.literal_eval(assignment(ROOT/'speakerdesk/speech_admission.py','INPUT_POLICY'))
CONTRACT='speakerdesk-acoustic-contract-v2'

def normalized_cardinal_text(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c))

def cardinal_tokens(text):
    return re.findall(r'[+-]?\d+(?:[.,]\d+)*(?!\w)|\w+',normalized_cardinal_text(text))

def quiet_cardinal_equivalent(reference,actual):
    # This typed policy is deliberately scoped to the declared cardinal eleven
    # in the entire frozen quiet sentence. It is not lexical error correction.
    normalized=normalized_cardinal_text(actual)
    for amount in re.finditer(r'\b(?:eleven|11)\b',normalized):
        before=normalized[:amount.start()].rstrip();after=normalized[amount.end():].lstrip()
        for decoration in ((before[-1] if before else ''),(after[0] if after else '')):
            if decoration and (unicodedata.category(decoration).startswith('S') or unicodedata.category(decoration)=='Pd'
                    or decoration in '+-−%‰‱#@'):
                return False
    expected=cardinal_tokens(reference);observed=cardinal_tokens(actual)
    if expected.count('eleven')!=1 or len(expected)!=len(observed):return False
    return all(got in ('eleven','11') if wanted=='eleven' else got==wanted
               for wanted,got in zip(expected,observed))

def evaluate(recipe,trace):
    checks=evaluate_v1(recipe,trace)
    if recipe['id']=='quiet-speech':
        reference=recipe['parts'][0]['text']
        for stage in ('provisional','refined'):
            text=' '.join(row['text'] for row in trace[stage]['document']['segments'] if row.get('text','').strip())
            checks[stage+'_coverage']=quiet_cardinal_equivalent(reference,text)
    return checks

def complete_receipt(row,start,end):
    evidence=row.get('acoustic_evidence') or {}
    if (evidence.get('source')!='silero_v6' or evidence.get('complete') is not True
        or evidence.get('decision')!='speech' or evidence.get('uncertain_regions')
        or evidence.get('start_sample')!=start or evidence.get('end_sample')!=end
        or type(evidence.get('observed_until_sample')) is not int
        or evidence['observed_until_sample']<end or not evidence.get('speech_regions')):return False
    spans=evidence['speech_regions']+evidence.get('model_negative_regions',[])
    cursor=start
    try:
        for span in sorted(spans,key=lambda s:s['start_sample']):
            a,b=span['start_sample'],span['end_sample']
            if type(a) is not int or type(b) is not int or not cursor==a<b<=end:return False
            cursor=b
    except (KeyError,TypeError):return False
    return cursor==end

def document_slice(job,boundary,pcm,provisional):
    """Select unchanged whole rows only across bound, verified zero-only context.

    Live and historical Silero calls have different frame origins. Both actual
    ledgers must be complete, not coordinate-identical. No words/times are cut.
    """
    require(job.get('id')==provisional.get('id') and bool(job.get('id')),'Context phase jobs differ')
    require(type(boundary) is int and 0<=boundary<=len(pcm),'Invalid fixture boundary')
    selected=copy.deepcopy(job);rows=[];proofs=[]
    prior={r['id']:r for r in provisional['document']['segments']}
    for row in selected['document']['segments']:
        start,end=round(row['start']*16000),round(row['end']*16000)
        if end<=boundary:continue
        if start<boundary:
            require(0<=start<boundary and not any(pcm[start:boundary]),'Context prefix contains nonzero retained PCM')
            if row.get('text','').strip():
                original=prior.get(row['id'],{});anchor={'start_sample':start,'end_sample':end}
                require(row.get('text_audio_anchor')==original.get('text_audio_anchor')==anchor,'Context anchors differ')
                require(type(row.get('audio_revision')) is int and row['audio_revision']==original.get('audio_revision'),'Context audio revisions differ')
                require(row.get('canonical_utterance_id')==original.get('canonical_utterance_id')==row['id'],'Context utterance bindings differ')
                require(complete_receipt(row,start,end) and complete_receipt(original,start,end),'Context phase receipt missing/incomplete/uncertain')
                current=row['acoustic_evidence'];previous=original['acoustic_evidence']
                require(current.get('model_revision')==previous.get('model_revision')==SPEECH_REVISION
                        and current.get('input_policy')==previous.get('input_policy')==SPEECH_POLICY,'Context model/policy bindings differ')
                require(original.get('speech_regions')==previous['speech_regions'] and row.get('speech_regions')==original['speech_regions'],'Original speech ledger not retained')
            prefix=array.array('h',pcm[start:boundary])
            if sys.byteorder!='little':prefix.byteswap()
            proofs.append({'policy':CONTRACT,'id':row['id'],'audio_revision':row.get('audio_revision'),
                'retained_start_sample':start,'boundary_sample':boundary,'prefix_pcm_sha256':hashlib.sha256(prefix.tobytes()).hexdigest(),
                'prefix_digital_zero':True,'text_and_anchor_unchanged':True})
        rows.append(row)
    selected['document']['segments']=rows;selected['fixture_boundary_evidence']=proofs
    return selected
