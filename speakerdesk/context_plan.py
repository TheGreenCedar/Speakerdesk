"""Frozen physical context bounds; never infer a word boundary from text."""
RATE=16000
CORE=14*RATE
CONTEXT=3*RATE
PHYSICAL_MAX=20*RATE
POLICY='english_bounded14_context3_corroborated_v1'


def requests(row,*,edges=None):
    a,b=row['start_sample'],row['end_sample']
    if type(a) is not int or type(b) is not int or not 0<=a<b:
        raise ValueError('Invalid context audio anchor.')
    if edges is None:
        count=max(1,(b-a+CORE-1)//CORE)
        edges=[a+(b-a)*i//count for i in range(count+1)]
    if (not isinstance(edges,list) or len(edges)<2 or any(type(x) is not int for x in edges)
            or (edges[0],edges[-1])!=(a,b)
            or any(not 0<y-x<=CORE for x,y in zip(edges,edges[1:]))):
        raise ValueError('Context cores must partition bounded original audio.')
    output=[]
    for x,y in zip(edges,edges[1:]):
        start,end=max(a,x-CONTEXT),min(b,y+CONTEXT)
        if end-start>PHYSICAL_MAX:raise ValueError('Oversized physical context decode.')
        output.append({'utterance_id':row['id'],'machine_revision':row['machine_revision'],
            'audio_revision':row['audio_revision'],'language_epoch':row['language_epoch'],
            'core_start_sample':x,'core_end_sample':y,'start_sample':start,'end_sample':end})
    return output
