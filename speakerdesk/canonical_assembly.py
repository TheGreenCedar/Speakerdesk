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
    matching prefix/suffix and rewrites no text. Exact ordered units, intersecting
    emission envelopes and the same disjoint-core owner must all corroborate.
    A word shifted out of both cores cannot silently disappear as context.
    """
    a=max(left_request['start_sample'],right_request['start_sample'])
    b=min(left_request['end_sample'],right_request['end_sample'])
    seam=left_request['core_end_sample']
    if not a<seam<b or seam!=right_request['core_start_sample']:return False
    def shared(attachment):
        words=[word for word in attachment['words'] if word['start_sample']<b and word['end_sample']>a]
        if any(not a<=word['start_sample']<word['end_sample']<=b for word in words):return None
        return words
    x,y=shared(left),shared(right)
    if not x or y is None or len(x)!=len(y):return False
    for first,second in zip(x,y):
        if (first['text']!=second['text']
                or max(first['start_sample'],second['start_sample'])>=min(first['end_sample'],second['end_sample'])
                or (first['end_sample']<=seam)!=(second['end_sample']<=seam)
                or (first['start_sample']>=seam)!=(second['start_sample']>=seam)):
            return False
    return True


def assemble_parts(row, requests, parts):
    """Return one replacement only when every bounded part has usable evidence.

    Callers supply the book's current requests, rather than trusting requests in
    asynchronous results. Only full emission envelopes inside a disjoint core
    own display units. Raw substrings preserve internal whitespace and Unicode;
    one explicit space separates substrings from different decode contexts.
    Raw full-context texts remain in the durable version journal.
    """
    if not isinstance(parts,list) or not parts or len(parts)!=len(requests):
        return {'complete':False,'reason':'missing_decode_parts','text':None,'words':[]}
    fragments=[];words=[];cursor=row['start_sample'];calibration=None;previous=None
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
            return {'complete':False,'reason':'incomplete_decode','text':None,'words':[]}
        current=part_utterance(request,text)
        try:
            attachment=attachment_for(part.get('alignment') or {},AlignmentRequest.from_utterance(current),current)
        except (ValueError,KeyError,TypeError):
            return {'complete':False,'reason':'unresolved_alignment','text':None,'words':[]}
        provenance=tuple(attachment[k] for k in ('model_sha256','timing_kind','frame_calibration_id','score_calibration_id'))
        if calibration is None:calibration=provenance
        elif provenance!=calibration:
            return {'complete':False,'reason':'inconsistent_alignment_provenance','text':None,'words':[]}
        if previous and not corroborated_seam(previous[0],previous[1],request,attachment):
            return {'complete':False,'reason':'context_ownership_unresolved','text':None,'words':[]}
        previous=(request,attachment)
        owned=[];a,b=request['core_start_sample'],request['core_end_sample']
        for word in attachment['words']:
            x,y=word['start_sample'],word['end_sample']
            if y<=a or x>=b:continue
            if x<a or y>b:
                return {'complete':False,'reason':'boundary_emission_unresolved','text':None,'words':[]}
            if ((a>row['start_sample'] and x<a+SEAM_UNCERTAINTY_SAMPLES)
                    or (b<row['end_sample'] and y>b-SEAM_UNCERTAINTY_SAMPLES)):
                return {'complete':False,'reason':'boundary_uncertainty_unresolved','text':None,'words':[]}
            owned.append(word)
        # An empty text ownership result is not proof that the speech core was
        # silent. Keep the prior words and the entire supplied candidate.
        if not owned:
            return {'complete':False,'reason':'empty_core_ownership','text':None,'words':[]}
        first,last=owned[0]['start_char'],owned[-1]['end_char']
        fragment=text[first:last]
        offset=sum(len(value) for value in fragments)+len(fragments)
        for word in owned:
            materialized=copy.deepcopy(word)
            materialized.update(start_char=offset+word['start_char']-first,end_char=offset+word['end_char']-first,
                decode_id=current['id'],raw_start_char=word['start_char'],raw_end_char=word['end_char'],
                raw_text_sha256=attachment['text_sha256'])
            words.append(materialized)
        fragments.append(fragment)
    if cursor!=row['end_sample']:
        raise ValueError('Decode cores do not cover the canonical utterance.')
    text=' '.join(fragments)
    for word in words:
        word['start_utf16']=len(text[:word['start_char']].encode('utf-16-le'))//2
        word['end_utf16']=len(text[:word['end_char']].encode('utf-16-le'))//2
    return {'complete':True,'reason':None,'text':text,'words':words,
            'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'separator_policy':'one_space_between_owned_raw_context_substrings',
            'model_sha256':calibration[0],'timing_kind':calibration[1],
            'frame_calibration_id':calibration[2],'score_calibration_id':calibration[3]}
