"""One selected raw tail, with a narrow acoustic rollover receipt.

This is not recognition verification. Boundary mapping uses only the selected
raw candidate's supplied-text CTC cells; no transcript prefix/suffix matching or
normalization writes text. Unsupported/ambiguous boundaries remain local review.
"""
import copy,hashlib,math,re
from canonical_assembly import part_utterance
from word_alignment import AlignmentRequest,attachment_for,prepare_text,_utf16_offset
from reading_turns import CALIBRATION,MODEL

POLICY='authoritative_tail_last_supported_boundary_v2'
MARGIN=4000
FLOOR=-20


def timing(part,vocabulary=None):
    current=part_utterance(part['request'],part['text'])
    try:
        result=part['alignment']
        if validated_cells(part,vocabulary) is None:return None
        attached=attachment_for(result,AlignmentRequest.from_utterance(current),current)
        if (attached['model_sha256']!=MODEL or attached['frame_calibration_id']!=CALIBRATION
                or attached['score_calibration_id']!=CALIBRATION):return None
        return attached
    except (KeyError,TypeError,ValueError):return None


def validated_cells(part,vocabulary):
    if vocabulary is None:return None
    result=part.get('alignment') or {};characters=result.get('characters')
    try:prepared=prepare_text(part['text'],vocabulary)
    except (ValueError,TypeError):return None
    if prepared.unsupported or not isinstance(characters,list) or len(characters)!=len(prepared.targets):return None
    previous=-1
    for c,t in zip(characters,prepared.targets):
        f=c.get('frame');score=c.get('log_probability')
        if (c.get('token_id')!=t.token_id or c.get('text')!=t.text
                or (c.get('start_char'),c.get('end_char'))!=(t.start_char,t.end_char)
                or c.get('raw_text')!=part['text'][t.start_char:t.end_char]
                or type(f) is not int or f<=previous or c.get('status')!='aligned'
                or not isinstance(score,(int,float)) or isinstance(score,bool) or not math.isfinite(score) or not FLOOR<=score<=0
                or c.get('start_sample')!=part['request']['start_sample']+f*320
                or c.get('end_sample')!=c['start_sample']+320
                or not part['request']['start_sample']<=c['start_sample']<c['end_sample']<=part['request']['end_sample']):return None
        previous=f
    for w in result.get('words',[]):
        selected=[c for c in characters if w['start_char']<=c['start_char']<c['end_char']<=w['end_char']]
        if not selected or (w.get('start_sample'),w.get('end_sample'))!=(selected[0]['start_sample'],selected[-1]['end_sample']):return None
    return characters


def cells(part,a,b,vocabulary):
    characters=validated_cells(part,vocabulary)
    if characters is None:return []
    output=[]
    for c in characters:
        if c.get('text')==' ' or not a<=c.get('start_char',-1)<c.get('end_char',-1)<=b:continue
        x,y=c.get('start_sample'),c.get('end_sample');score=c.get('log_probability')
        if (c.get('status')!='aligned' or type(x) is not int or type(y) is not int
                or not part['request']['start_sample']<=x<y<=part['request']['end_sample']
                or type(c.get('token_id')) is not int or not isinstance(score,(int,float))
                or not math.isfinite(score) or score<FLOOR):return []
        output.append(c)
    return output


def boundary(left,right,frontier,*,vocabulary):
    """Select a raw word first; map only its temporal acoustic identity.

    A match is unique and must consume complete raw display units. Distinct
    repeated events remain distinct; ambiguous repetitions never use a closest
    spelling or nearest-time tie breaker.
    """
    x,y=timing(left,vocabulary),timing(right,vocabulary)
    if x is None or y is None:return None,'unavailable_boundary_timing'
    shared_a=max(left['request']['start_sample'],right['request']['start_sample'])
    shared_b=min(left['request']['end_sample'],right['request']['end_sample'])
    anchors=[w for w in x['words'] if w['start_sample']<frontier+MARGIN
             and shared_a+MARGIN<=w['start_sample']<w['end_sample']<=shared_b-MARGIN]
    if not anchors:return None,'no_complete_boundary_anchor'
    failures=[]
    for word in reversed(anchors):
        receipt,reason=_map_boundary(left,right,frontier,word,y,vocabulary)
        if receipt is not None:
            receipt['anchor_selection']='last_supported_chosen_raw_word_in_fixed_context'
            receipt['later_unmapped_raw_anchors']=failures
            return receipt,None
        failures.append({'raw_start_char':word['start_char'],'raw_end_char':word['end_char'],
            'text':word['text'],'start_sample':word['start_sample'],'end_sample':word['end_sample'],'reason':reason})
    return None,failures[0]['reason']


def _map_boundary(left,right,frontier,word,y,vocabulary):
    source=cells(left,word['start_char'],word['end_char'],vocabulary)
    if not source:return None,'unavailable_boundary_cells'
    target=cells(right,0,len(right['text']),vocabulary);matches=[]
    for i in range(len(target)-len(source)+1):
        group=target[i:i+len(source)]
        if any(a['token_id']!=b['token_id'] or abs(a['start_sample']-b['start_sample'])>MARGIN
               or abs(a['end_sample']-b['end_sample'])>MARGIN for a,b in zip(source,group)):continue
        if not any(max(a['start_sample'],b['start_sample'])<min(a['end_sample'],b['end_sample']) for a,b in zip(source,group)):continue
        units=[w for w in y['words'] if w['start_char']<=group[-1]['start_char'] and w['end_char']>group[0]['start_char']]
        if not units:continue
        complete_cells=cells(right,units[0]['start_char'],units[-1]['end_char'],vocabulary)
        if complete_cells!=group:continue
        matches.append((group,units))
    if len(matches)!=1:return None,'ambiguous_boundary_mapping' if matches else 'boundary_mapping_absent'
    group,units=matches[0]
    # The new boundary must point back to this same selected acoustic event,
    # not merely be the only occurrence left by asymmetric ASR omission.
    reverse=cells(left,0,len(left['text']),vocabulary);reverse_matches=[]
    for i in range(len(reverse)-len(group)+1):
        candidate=reverse[i:i+len(group)]
        if any(a['token_id']!=b['token_id'] or abs(a['start_sample']-b['start_sample'])>MARGIN
               or abs(a['end_sample']-b['end_sample'])>MARGIN for a,b in zip(candidate,group)):continue
        if not any(max(a['start_sample'],b['start_sample'])<min(a['end_sample'],b['end_sample']) for a,b in zip(candidate,group)):continue
        reverse_matches.append(candidate)
    if len(reverse_matches)!=1 or reverse_matches[0]!=source:return None,'ambiguous_bidirectional_boundary'
    return {'policy':POLICY,'nominal_frontier_sample':frontier,
      'left_raw_end_char':word['end_char'],'right_raw_start_char':units[-1]['end_char'],
      'left_boundary_word':copy.deepcopy(word),'right_boundary_units':copy.deepcopy(units),
      'left_text_sha256':hashlib.sha256(left['text'].encode()).hexdigest(),
      'right_text_sha256':hashlib.sha256(right['text'].encode()).hexdigest(),
      'left_request':copy.deepcopy(left['request']),'right_request':copy.deepcopy(right['request']),
      'left_source_request':copy.deepcopy(left.get('source_inference_request',left['request'])),
      'right_source_request':copy.deepcopy(right.get('source_inference_request',right['request'])),
      'left_source_audio_sha256':left.get('source_audio_float32_sha256',left.get('audio_float32_sha256')),
      'right_source_audio_sha256':right.get('source_audio_float32_sha256',right.get('audio_float32_sha256')),
      'model_sha256':MODEL,'calibration_id':CALIBRATION,'uncertainty_samples':MARGIN,
      'bidirectional_unique':True,'intersecting_cell_count':sum(max(a['start_sample'],b['start_sample'])<min(a['end_sample'],b['end_sample']) for a,b in zip(source,group)),
      'cells':[{'token_id':a['token_id'],'left_samples':[a['start_sample'],a['end_sample']],
                'right_samples':[b['start_sample'],b['end_sample']]} for a,b in zip(source,group)]},None


def assemble(row,requests,parts,*,vocabulary=None,committed_receipts=None):
    """Selected prefixes + final authoritative tail; raw history stays outside.

    ASR completeness and timing/rollover quality are separate. An unmapped seam
    returns a finite review record, never an indefinite inference requirement or
    an invented concatenation of overlapping recognitions.
    """
    if not isinstance(parts,list) or not parts or len(parts)!=len(requests):raise ValueError('Missing selected tails.')
    if (any(type(row.get(k)) is not int or row[k]<0 for k in ('start_sample','end_sample','machine_revision','audio_revision','language_epoch')) or not row['start_sample']<row['end_sample']):raise ValueError('Invalid authoritative anchor.')
    cursor=row['start_sample']
    for request,part in zip(requests,parts):
        if (part.get('request')!=request or request['utterance_id']!=row['id']
                or request['machine_revision']!=row['machine_revision']
                or request['audio_revision']!=row['audio_revision'] or request['language_epoch']!=row['language_epoch']
                or any(type(request.get(k)) is not int for k in ('start_sample','end_sample','core_start_sample','core_end_sample'))
                or not row['start_sample']<=request['start_sample']<=request['core_start_sample']<request['core_end_sample']<=request['end_sample']<=row['end_sample']
                or request['end_sample']-request['start_sample']>320000
                or request['core_start_sample']!=cursor):raise ValueError('Stale authoritative tail.')
        cursor=request['core_end_sample']
    if cursor!=row['end_sample']:raise ValueError('Incomplete original audio partition.')
    recognition_complete=all(p.get('complete') is True and isinstance(p.get('text'),str) and bool(p['text'].strip()) for p in parts)
    base={'policy':POLICY,'recognition_complete':recognition_complete,'rollover_receipts':[],
          'local_seams':[],'words':[],'complete':False,'text':None,'reason':None}
    if not recognition_complete:return dict(base,reason='incomplete_selected_cohere_revision')
    # Several diagnostic cores can refer to the exact same whole original
    # audio. Choose the final authoritative revision once, not one per core.
    if all((p['request']['start_sample'],p['request']['end_sample'])==(row['start_sample'],row['end_sample']) for p in parts):
        chosen=copy.deepcopy(parts[-1]);request=dict(chosen['request'],core_start_sample=row['start_sample'],core_end_sample=row['end_sample'])
        chosen['request']=request
        return assemble(row,[request],[chosen],vocabulary=vocabulary) if len(parts)>1 else _whole(row,chosen,base,vocabulary)
    cuts=[0];ends=[]
    for i in range(len(parts)-1):
        receipt,reason=boundary(parts[i],parts[i+1],requests[i]['core_end_sample'],vocabulary=vocabulary)
        if receipt is None:
            base['local_seams'].append({'frontier_sample':requests[i]['core_end_sample'],
                'reason':reason,'recognition_complete':True,'state':'finite_review',
                'left_request':copy.deepcopy(requests[i]),'right_request':copy.deepcopy(requests[i+1])})
            return dict(base,reason=reason)
        if receipt['left_raw_end_char']<cuts[-1]:return dict(base,reason='nonmonotone_boundary_mapping')
        locked=(committed_receipts or {}).get(requests[i]['core_end_sample'])
        binding=('left_raw_end_char','right_raw_start_char','left_text_sha256','right_text_sha256',
                 'left_source_request','right_source_request','left_source_audio_sha256','right_source_audio_sha256')
        if locked is not None and any(locked.get(k)!=receipt.get(k) for k in binding):
            base['local_seams'].append({'frontier_sample':requests[i]['core_end_sample'],
                'reason':'committed_authority_changed','state':'finite_review','recognition_complete':True})
            return dict(base,reason='committed_authority_changed')
        right=requests[i+1]
        receipt['committed']=bool(locked is not None or row.get('state')=='sealed' or (
            right['core_end_sample']-right['core_start_sample']==224000 and
            right['end_sample']==right['core_end_sample']+48000))
        base['rollover_receipts'].append(receipt);ends.append(receipt['left_raw_end_char']);cuts.append(receipt['right_raw_start_char'])
    ends.append(len(parts[-1]['text']));fragments=[]
    for index,(part,a,b) in enumerate(zip(parts,cuts,ends)):
        raw=part['text'];a=a+len(raw[a:b])-len(raw[a:b].lstrip());b=a+len(raw[a:b].rstrip())
        if a>=b:continue # A fully consumed terminal carry is legitimate.
        fragment=raw[a:b];offset=sum(map(len,fragments))+len(fragments);attached=timing(part,vocabulary)
        for unit in re.finditer(r'\S+',fragment):
            start,end=unit.start()+a,unit.end()+a
            candidate=next((w for w in (attached['words'] if attached else []) if (w['start_char'],w['end_char'])==(start,end)),None)
            x,y=(candidate['start_sample'],candidate['end_sample']) if candidate else (None,None)
            if x is not None and any(x<end+MARGIN and y>start-MARGIN for f in base['rollover_receipts'] for start,end in ((f['nominal_frontier_sample'],f['nominal_frontier_sample']),(f['left_boundary_word']['start_sample'],f['left_boundary_word']['end_sample']),(f['right_boundary_units'][-1]['start_sample'],f['right_boundary_units'][-1]['end_sample']))):x=y=None
            base['words'].append({'text':unit.group(),'start_char':offset+unit.start(),'end_char':offset+unit.end(),
                'start_sample':x,'end_sample':y,'raw_start_char':start,'raw_end_char':end,
                'raw_text_sha256':hashlib.sha256(raw.encode()).hexdigest(),'decode_id':part_utterance(part.get('source_inference_request',part['request']),raw)['id'],
                'core_ownership':'owned' if x is not None else 'unknown'})
        fragments.append(fragment)
    text=' '.join(fragments)
    _ordered_word_timings(base['words'])
    for word in base['words']:
        word.update(start_utf16=_utf16_offset(text,word['start_char']),end_utf16=_utf16_offset(text,word['end_char']))
    return dict(base,complete=True,text=text,reason=None,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
       alignment_complete=all(w['start_sample'] is not None for w in base['words']),
       model_sha256=MODEL,frame_calibration_id=CALIBRATION,score_calibration_id=CALIBRATION,timing_kind='ctc_emission_cell_envelope')


def _ordered_word_timings(words):
    previous=-1
    for word in words:
        start,end=word['start_sample'],word['end_sample']
        if start is None:continue
        if start<previous:
            word.update(start_sample=None,end_sample=None,core_ownership='unknown')
        else:previous=end


def _whole(row,part,base,vocabulary):
    # Whole chosen text is accepted even if all optional word times are null.
    raw=part['text'];attached=timing(part,vocabulary);words=[]
    for unit in re.finditer(r'\S+',raw):
        w=next((w for w in (attached['words'] if attached else []) if (w['start_char'],w['end_char'])==unit.span()),None)
        x,y=(w['start_sample'],w['end_sample']) if w else (None,None)
        words.append({'text':unit.group(),'start_char':unit.start(),'end_char':unit.end(),
            'start_utf16':_utf16_offset(raw,unit.start()),'end_utf16':_utf16_offset(raw,unit.end()),
            'start_sample':x,'end_sample':y,'raw_start_char':unit.start(),'raw_end_char':unit.end(),
            'raw_text_sha256':hashlib.sha256(raw.encode()).hexdigest(),'decode_id':part_utterance(part.get('source_inference_request',part['request']),raw)['id'],
            'core_ownership':'owned' if x is not None else 'unknown'})
    _ordered_word_timings(words)
    return dict(base,complete=True,text=raw,reason=None,words=words,
        text_sha256=hashlib.sha256(raw.encode()).hexdigest(),alignment_complete=all(w['start_sample'] is not None for w in words),
        model_sha256=MODEL,frame_calibration_id=CALIBRATION,score_calibration_id=CALIBRATION,timing_kind='ctc_emission_cell_envelope')


def requests(row):
    """Anchored cores keep committed physical context stable as audio grows."""
    a,b=row['start_sample'],row['end_sample']
    if type(a) is not int or type(b) is not int or not 0<=a<b:raise ValueError('Invalid original audio anchor.')
    edges=[a,b] if b-a<=320000 else [*range(a,b,224000),b]
    return [{'utterance_id':row['id'],'machine_revision':row['machine_revision'],'audio_revision':row['audio_revision'],
             'language_epoch':row['language_epoch'],'core_start_sample':x,'core_end_sample':y,
             'start_sample':max(a,x-48000),'end_sample':min(b,y+48000)} for x,y in zip(edges,edges[1:])]
