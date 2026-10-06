"""Bounded original-audio partitioning; no text or word-edge inference."""
from speech_admission import INPUT_POLICY, SILERO_SPEC

RATE = 16000
CAP = 20 * RATE
MIN_CORE = 6 * RATE
MIN_GAP = 2048
RADIUS = RATE
POLICY = 'current_silero_negative_gap_bounded20_radius1_v2'


def plan(row, admission):
    a,b=row['start_sample'],row['end_sample']
    count=max(1,(b-a+CAP-1)//CAP)
    gaps=[(left['end_sample'],right['start_sample']) for left,right in
          zip(row['speech_regions'],row['speech_regions'][1:])
          if right['start_sample']-left['end_sample']>=MIN_GAP]
    edges=[a];cuts=[]
    for i in range(1,count):
        remaining=count-i
        low=max(edges[-1]+MIN_CORE,b-remaining*CAP)
        high=min(edges[-1]+CAP,b-remaining*MIN_CORE)
        original_target=a+(b-a)*i//count
        target=max(low,min(high,original_target))
        # At most three independently verified gap reads per core. Raw archive
        # absence/uncertainty never becomes a quiet boundary.
        candidates=sorted(((x,y,(x+y)//2) for x,y in gaps if low<=(x+y)//2<=high
                           and abs((x+y)//2-original_target)<=RADIUS),
                          key=lambda item:(abs(item[2]-target),item[2]))[:3]
        chosen=target;receipt=None
        for x,y,midpoint in candidates:
            evidence=admission(x,y)
            if (evidence.get('complete') is True and evidence.get('decision')=='no_speech'
                    and evidence.get('input_policy')==INPUT_POLICY
                    and evidence.get('model_revision')==SILERO_SPEC['revision']
                    and (evidence.get('start_sample'),evidence.get('end_sample'))==(x,y)
                    and evidence.get('speech_regions')==[] and evidence.get('uncertain_regions')==[]):
                chosen=midpoint;receipt=evidence;break
        edges.append(chosen)
        cuts.append({'sample':chosen,'method':'verified_model_negative_gap' if receipt else 'balanced_bounded_fallback',
                     'negative_receipt':receipt})
    edges.append(b)
    return {'policy':POLICY,'cap_samples':CAP,'radius_samples':RADIUS,'utterance_id':row['id'],
            'machine_revision':row['machine_revision'],'audio_revision':row['audio_revision'],
            'language_epoch':row['language_epoch'],'audio_anchor':{'start_sample':a,'end_sample':b},
            'edges':edges,'cuts':cuts}


def validate(value,row):
    if (not isinstance(value,dict) or value.get('policy')!=POLICY or value.get('cap_samples')!=CAP
            or type(value.get('radius_samples')) is not int or value.get('radius_samples')!=RADIUS
            or any(type(value.get(key)) is not int for key in ('machine_revision','audio_revision','language_epoch','cap_samples'))
            or value.get('utterance_id')!=row['id'] or value.get('machine_revision')!=row['machine_revision']
            or value.get('audio_revision')!=row['audio_revision'] or value.get('language_epoch')!=row['language_epoch']
            or value.get('audio_anchor')!={'start_sample':row['start_sample'],'end_sample':row['end_sample']}):
        raise ValueError('Core plan belongs to another text/audio revision.')
    edges=value.get('edges');cuts=value.get('cuts')
    if (not isinstance(edges,list) or len(edges)<2 or any(type(x) is not int for x in edges)
            or (edges[0],edges[-1])!=(row['start_sample'],row['end_sample'])
            or any(not 0<y-x<=CAP for x,y in zip(edges,edges[1:]))
            or (len(edges)>2 and any(y-x<MIN_CORE for x,y in zip(edges,edges[1:])))
            or not isinstance(cuts,list) or len(cuts)!=len(edges)-2
            or len(edges)-1!=max(1,(row['end_sample']-row['start_sample']+CAP-1)//CAP)):
        raise ValueError('Core plan must cover bounded ordered original audio.')
    a,b=row['start_sample'],row['end_sample'];count=len(edges)-1
    for i,(edge,cut) in enumerate(zip(edges[1:-1],cuts),1):
        original_target=a+(b-a)*i//count
        low=max(edges[i-1]+MIN_CORE,b-(count-i)*CAP)
        high=min(edges[i-1]+CAP,b-(count-i)*MIN_CORE)
        target=max(low,min(high,original_target))
        if not isinstance(cut,dict) or cut.get('sample')!=edge:
            raise ValueError('Core boundary receipt mismatch.')
        receipt=cut.get('negative_receipt')
        if cut.get('method')=='balanced_bounded_fallback':
            if receipt is not None or edge!=target:raise ValueError('Fallback must use the bounded balanced target.')
        elif cut.get('method')=='verified_model_negative_gap':
            if not isinstance(receipt,dict):raise ValueError('Missing core gap evidence.')
            x,y=receipt.get('start_sample'),receipt.get('end_sample')
            if (type(x) is not int or type(y) is not int or y-x<MIN_GAP or edge!=(x+y)//2
                    or abs(edge-original_target)>RADIUS
                    or not row['start_sample']<x<y<row['end_sample']
                    or receipt.get('complete') is not True or receipt.get('decision')!='no_speech'
                    or receipt.get('input_policy')!=INPUT_POLICY
                    or receipt.get('model_revision')!=SILERO_SPEC['revision']
                    or receipt.get('speech_regions')!=[] or receipt.get('uncertain_regions')!=[]
                    or (x,y) not in [(left['end_sample'],right['start_sample']) for left,right in
                        zip(row['speech_regions'],row['speech_regions'][1:])]):
                raise ValueError('Unqualified negative gap cannot anchor a core.')
        else:raise ValueError('Unknown core boundary method.')
    return edges
