"""Assemble supplied Cohere revisions using complete, revision-bound alignment.

No audio, models or optional native packages are loaded here. A timing envelope
crossing a core boundary remains unresolved; spelling or repeated tokens never
decide ownership. The source contract is not acoustic accuracy evidence.
"""
import copy
import hashlib
import json
from word_alignment import AlignmentRequest, attachment_for


def part_utterance(request, text):
    """Give a bounded decode its own identity bound to the whole source revision."""
    identity=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
    return {'id':'decode-'+identity,'machine_revision':request['machine_revision'],
            'audio_revision':request['audio_revision'],'text':text,
            'start_sample':request['start_sample'],'end_sample':request['end_sample'],
            'text_audio_anchor':{'start_sample':request['start_sample'],'end_sample':request['end_sample']}}


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
    fragments=[];words=[];cursor=row['start_sample'];calibration=None
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
        owned=[];a,b=request['core_start_sample'],request['core_end_sample']
        for word in attachment['words']:
            x,y=word['start_sample'],word['end_sample']
            if y<=a or x>=b:continue
            if x<a or y>b:
                return {'complete':False,'reason':'boundary_emission_unresolved','text':None,'words':[]}
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
