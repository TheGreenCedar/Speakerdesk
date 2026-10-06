"""Generate only frozen recipes using installed compact voices, without downloads."""
import array
import hashlib
import json
import math
from pathlib import Path
import plistlib
import random
import subprocess
import sys
import wave

RATE=16000
VOICE_ROOT=Path('/System/Library/PrivateFrameworks/TextToSpeechMauiSupport.framework/Versions/A/Resources/TTSResources')

def write_wav(path, samples):
    pcm=array.array('h',samples)
    if sys.byteorder!='little':pcm.byteswap()
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(RATE);stream.writeframes(pcm.tobytes())

def read_wav(path):
    with wave.open(str(path),'rb') as stream:
        if (stream.getnchannels(),stream.getsampwidth(),stream.getframerate())!=(1,2,RATE):raise ValueError('Unexpected synthesis format')
        pcm=array.array('h',stream.readframes(stream.getnframes()))
    if sys.byteorder!='little':pcm.byteswap()
    return list(pcm)

def voice(identifier):
    for info in VOICE_ROOT.glob('*/*/Info.plist'):
        value=plistlib.loads(info.read_bytes())
        if value.get('MobileAssetProperties',{}).get('VoiceId')==identifier:
            # Exact installed resource inventory is recorded; absent compact voices fail closed.
            resource=info.parent
            files={str(p.relative_to(resource)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in resource.rglob('*') if p.is_file()}
            return {'id':identifier,'path':str(resource),'files_sha256':files}
    raise ValueError('Required preinstalled compact voice unavailable: '+identifier)

def generate(recipe, directory, command=None):
    if command is None:
        def command(arguments, timeout):
            return subprocess.run(arguments,check=True,timeout=timeout,capture_output=True)
    directory.mkdir(parents=True,exist_ok=False);metadata=[];counter=0
    def render(part):
        nonlocal counter
        counter+=1;kind=part['kind']
        if kind=='zero':return [0]*round(part['seconds']*RATE)
        if kind=='noise':
            rng=random.Random(part['seed']);last=0.;values=[];smoothing=part.get('smoothing',0)
            for _ in range(round(part['seconds']*RATE)):
                last=smoothing*last+(1-smoothing)*rng.gauss(0,1);values.append(last)
            rms=math.sqrt(sum(v*v for v in values)/len(values))
            return [round(v*part['pcm_rms']/rms) for v in values]
        if kind=='overlap':
            left=render(part['left']);right=render(part['right']);offset=round(part['right_offset_seconds']*RATE)
            mixed=[0]*max(len(left),offset+len(right))
            for i,v in enumerate(left):mixed[i]+=v//2
            for i,v in enumerate(right):mixed[i+offset]+=v//2
            return mixed
        if kind!='speech':raise ValueError('Unknown recipe part')
        identity=voice(part['voice']);stem=directory/f'speech-{counter}'
        command(['/usr/bin/say','-v',part['voice'],'-r',str(part.get('rate',175)),
                 '-o',str(stem.with_suffix('.aiff')),part['text']],timeout=20)
        command(['/usr/bin/afconvert','-f','WAVE','-d','LEI16@16000','-c','1',
                 str(stem.with_suffix('.aiff')),str(stem.with_suffix('.wav'))],timeout=10)
        original=read_wav(stem.with_suffix('.wav'));start=0;end=len(original)
        # Remove digital-zero padding only, never an amplitude threshold or a spoken sample.
        while start<end and original[start]==0:start+=1
        while end>start and original[end-1]==0:end-=1
        gain=part.get('gain',1)
        values=[max(-32768,min(32767,round(v*gain))) for v in original[start:end]]
        metadata.append({'voice':identity,'text':part['text'],'gain':gain,'rate':part.get('rate',175),
                         'original_wav_sha256':hashlib.sha256(stem.with_suffix('.wav').read_bytes()).hexdigest(),
                         'original_frames':len(original),'trim_start':start,'trim_end':end})
        stem.with_suffix('.aiff').unlink();stem.with_suffix('.wav').unlink()
        return values
    samples=[];parts=[]
    for part in recipe['parts']:
        start=len(samples);samples.extend(render(part));parts.append({'kind':part['kind'],'start_sample':start,'end_sample':len(samples)})
    # Tail allows normal production phrase settling without claiming it contains speech.
    samples.extend([0]*RATE)
    write_wav(directory/'replay.wav',samples)
    (directory/'synthesis.json').write_text(json.dumps({'recipe_id':recipe['id'],'sample_rate':RATE,'speech':metadata,'parts':parts,'frames':len(samples)},indent=2)+'\n')
    hold=round(recipe['hold_seconds']*RATE) if recipe.get('hold_seconds') else None
    if 'hold_after_part' in recipe:hold=parts[recipe['hold_after_part']]['end_sample']
    (directory/'replay.json').write_text(json.dumps({'hold_sample':hold})+'\n')
    return len(samples)
