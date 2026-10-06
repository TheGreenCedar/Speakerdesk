"""Assemble supplied Cohere revisions using complete, revision-bound alignment.

No audio, models or optional native packages are loaded here. A timing envelope
crossing a core boundary remains unresolved; spelling or repeated tokens never
decide ownership. The source contract is not acoustic accuracy evidence.
"""
import copy
import hashlib
import json
from word_alignment import AlignmentRequest, attachment_for

# Frozen AMI coarse-envelope qualification. These cells are useful anchors,
# not phonetic boundaries; interior-core ownership needs the tested 250ms
# uncertainty margin. Outer utterance endpoints do not create a shared seam.
SEAM_UNCERTAINTY_SAMPLES = 4000


def part_utterance(request, text):
    """Give a bounded decode its own identity bound to the whole source revision."""
    identity=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
    return {'id':'decode-'+identity,'machine_revision':request['machine_revision'],
            'audio_revision':request['audio_revision'],'text':text,
            'start_sample':request['start_sample'],'end_sample':request['end_sample'],
            'text_audio_anchor':{'start_sample':request['start_sample'],'end_sample':request['end_sample']}}


def corroborated_seam(left_request,left,right_request,right):
    """Require both contexts to agree on every shared observed display unit.

    This is a rejection guard, not lexical deduplication: it searches for no
    matching prefix/suffix and rewrites no text. Stable interior raw units and
    intersecting envelopes corroborate identity. Only far-from-seam units claim
    a core owner; near-seam units are retained once with unknown timing/owner.
    """
    a=max(left_request['start_sample'],right_request['start_sample'])
    b=min(left_request['end_sample'],right_request['end_sample'])
    seam=left_request['core_end_sample']
    if not a<seam<b or seam!=right_request['core_start_sample']:return False
    def shared(attachment):
        # Decode edges can legitimately cut a word or change its capitalization.
        # Corroboration covers the stable interior, not those context-only edges.
        words=[word for word in attachment['words']
               if a+SEAM_UNCERTAINTY_SAMPLES<=word['start_sample']<word['end_sample']<=b-SEAM_UNCERTAINTY_SAMPLES]
        return words
    x,y=shared(left),shared(right)
    if not x or y is None or len(x)!=len(y):return False
    for first,second in zip(x,y):
        if (first['text']!=second['text']
                or max(first['start_sample'],second['start_sample'])>=min(first['end_sample'],second['end_sample'])
                or (not near_seam(first,seam) and not near_seam(second,seam)
                    and ((first['end_sample']<=seam)!=(second['end_sample']<=seam)
                         or (first['start_sample']>=seam)!=(second['start_sample']>=seam)))):
            return False
    return True


def near_seam(word,seam):
    return (word['start_sample']<seam+SEAM_UNCERTAINTY_SAMPLES
            and word['end_sample']>seam-SEAM_UNCERTAINTY_SAMPLES)


def shared_unknown_words(left,right,seam):
    """One raw copy per corroborated unit; no core/timing ownership claimed."""
    first=[word for word in left['words'] if near_seam(word,seam)]
    second=[word for word in right['words'] if near_seam(word,seam)]
    if len(first)!=len(second):return None
    for x,y in zip(first,second):
        if x['text']!=y['text'] or max(x['start_sample'],y['start_sample'])>=min(x['end_sample'],y['end_sample']):
            return None
    return first


def assemble_parts(row, requests, parts):
    """Assemble exact raw substrings; near-seam word timing remains unknown.

    All source alignments must be complete and revision-bound. Stable interior
    context units corroborate identity, and each near-seam unit must independently
    agree in both contexts. Those units appear once in the canonical text but are
    not assigned to either core and cannot form a complete timing attachment.
    This establishes candidate conservation, not recognition truth.
    """
    def unresolved(reason):return {'complete':False,'reason':reason,'text':None,'words':[]}
    if not isinstance(parts,list) or not parts or len(parts)!=len(requests):
        return unresolved('missing_decode_parts')
    attachments=[];cursor=row['start_sample'];calibration=None
    for request,part in zip(requests,parts):
        if (request['utterance_id']!=row['id'] or request['machine_revision']!=row['machine_revision']
                or request['audio_revision']!=row['audio_revision']
                or request['language_epoch']!=row['language_epoch']
                or request['core_start_sample']!=cursor
                or not request['start_sample']<=cursor<request['core_end_sample']<=request['end_sample']
                or request['end_sample']>row['end_sample']):
            raise ValueError('Invalid current canonical decode requests.')
        cursor=request['core_end_sample']
        if not isinstance(part,dict) or part.get('request')!=request:
            raise ValueError('Decode belongs to a different canonical audio or text revision.')
        text=part.get('text')
        if not isinstance(text,str) or not text.strip() or part.get('complete') is not True:
            return unresolved('incomplete_decode')
        current=part_utterance(request,text)
        try:attachment=attachment_for(part.get('alignment') or {},AlignmentRequest.from_utterance(current),current)
        except (ValueError,KeyError,TypeError):return unresolved('unresolved_alignment')
        provenance=tuple(attachment[k] for k in ('model_sha256','timing_kind','frame_calibration_id','score_calibration_id'))
        if calibration is None:calibration=provenance
        elif provenance!=calibration:return unresolved('inconsistent_alignment_provenance')
        attachments.append(attachment)
    if cursor!=row['end_sample']:raise ValueError('Decode cores do not cover the canonical utterance.')
    seams=[request['core_end_sample'] for request in requests[:-1]]
    if any(sum(near_seam(word,seam) for seam in seams)>1
           for attachment in attachments for word in attachment['words']):
        return unresolved('multiple_seam_emission_unresolved')
    seam_units={}
    for index in range(len(requests)-1):
        left,right=requests[index:index+2];seam=left['core_end_sample']
        if not corroborated_seam(left,attachments[index],right,attachments[index+1]):
            return unresolved('context_ownership_unresolved')
        unknown=shared_unknown_words(attachments[index],attachments[index+1],seam)
        if unknown is None:return unresolved('boundary_uncertainty_unresolved')
        seam_units[index]=unknown
    fragments=[];words=[]
    for index,(request,part,attachment) in enumerate(zip(requests,parts,attachments)):
        a,b=request['core_start_sample'],request['core_end_sample']
        lower=a+(SEAM_UNCERTAINTY_SAMPLES if index else 0)
        upper=b-(SEAM_UNCERTAINTY_SAMPLES if index<len(requests)-1 else 0)
        owned=[word for word in attachment['words'] if lower<=word['start_sample']<word['end_sample']<=upper]
        unknown=seam_units.get(index,[])
        selected=owned+unknown
        if not selected:return unresolved('empty_core_ownership')
        # Never remove an uncorroborated word from inside a retained substring.
        indices=[attachment['words'].index(word) for word in selected]
        if indices!=list(range(indices[0],indices[-1]+1)):
            return unresolved('noncontiguous_raw_ownership')
        first,last=selected[0]['start_char'],selected[-1]['end_char'];text=part['text']
        fragment=text[first:last];offset=sum(map(len,fragments))+len(fragments)
        for word in selected:
            materialized=copy.deepcopy(word)
            materialized.update(start_char=offset+word['start_char']-first,end_char=offset+word['end_char']-first,
                decode_id=part_utterance(request,text)['id'],raw_start_char=word['start_char'],raw_end_char=word['end_char'],
                raw_text_sha256=attachment['text_sha256'],core_ownership='unknown' if word in unknown else 'owned')
            if word in unknown:
                materialized.update(start_sample=None,end_sample=None,status='unresolved_boundary')
            words.append(materialized)
        fragments.append(fragment)
    text=' '.join(fragments)
    for word in words:
        word['start_utf16']=len(text[:word['start_char']].encode('utf-16-le'))//2
        word['end_utf16']=len(text[:word['end_char']].encode('utf-16-le'))//2
    return {'complete':True,'alignment_complete':all(w['start_sample'] is not None for w in words),
            'reason':None,'text':text,'words':words,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'separator_policy':'one_space_between_owned_raw_context_substrings',
            'model_sha256':calibration[0],'timing_kind':calibration[1],
            'frame_calibration_id':calibration[2],'score_calibration_id':calibration[3]}
