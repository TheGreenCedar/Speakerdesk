"""Retain actual native DSP + bounded startup publication, CPU only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import wave
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'speakerdesk')]
from live_meeting import SourceMixer
from render_startup import StartupSourceSink,StartupReceiptWriter,PARAMETERS,POLICY_SHA256
from capture_sources import catalog
from capture_echo import library_path
from audio import pcm16_bytes
from run_render_copy_cpu_controls import NEW,OLD,PINS,pcm,sha
from compare_public_aec_delay import NATIVE_SHA


def replay(raw,far,folder,*,boundaries=()):
    folder.mkdir(mode=0o700);handles={};waves={};live=[];events=[]
    (folder/'capture-sources.json').write_text(json.dumps(catalog('a'*32,innovation=True,startup=True),sort_keys=True))
    writer=StartupReceiptWriter(folder)
    for name in ('microphone','system','system_reference','microphone_clean','audio'):
        h=(folder/(name+'.wav')).open('x+b');handles[name]=h
        w=wave.open(h,'wb');w.setparams((2 if name=='system_reference' else 1,2,16000,0,'NONE','none'));waves[name]=w
    def raw_sink(tracks):
        for name in ('microphone','system','system_reference'):
            waves[name].writeframes(pcm16_bytes(tracks[name].reshape(-1)));handles[name].flush()
    def publish(mix,tracks):
        start=writer.end;writer.write(start,tracks);data=pcm16_bytes(tracks['microphone_clean'])
        live.append(data);waves['microphone_clean'].writeframes(data);waves['audio'].writeframes(pcm16_bytes(mix))
        handles['microphone_clean'].flush();handles['audio'].flush()
        events.append(dict(start_sample=start,end_sample=writer.end,
            observed_native_sample=m.innovation.selector.end,wall=time.monotonic(),
            selected_pcm_sha256=hashlib.sha256(data).hexdigest()))
    m=SourceMixer(['microphone','system'],publish,innovation_factory=StartupSourceSink,raw_sink=raw_sink)
    scheduled=set()
    try:
        for start in range(0,len(raw),4000):
            end=min(len(raw),start+4000)
            for name,values,step in (('system',far,752),('microphone',raw,496)):
                for a in range(start,end,step):
                    b=min(a+step,end);m.add(name,a/16000,values[a:b].astype('<f4').tobytes())
            for point in boundaries:
                if point not in scheduled and start<=point<=end:m.format_changed('microphone',point/16000);scheduled.add(point)
            m.flush(end/16000)
        m.flush(len(raw)/16000,final=True)
        assert m.innovation.queued==0 and m.innovation.held_count==0 and writer.end==len(raw)
    finally:
        m.innovation.flush();m.echo.close()
        for w in waves.values():w.close()
        for h in handles.values():h.close()
        writer.close()
    stored=pcm(folder/'microphone_clean.wav').tobytes();assert stored==b''.join(live)
    return dict(folder=str(folder),samples=len(raw),first_publication=events[0],
        publications=events,release_log=m.innovation.publication_log,
        published_PCM_equals_saved_PCM=True,
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})


def main():
    os.umask(0o077);assert sha(library_path())==NATIVE_SHA
    for p,h in PINS.items():assert sha(p)==h
    new=json.loads(NEW.read_bytes());old=json.loads(OLD.read_bytes())
    for c in new['cases']:
        for v in c['files'].values():assert sha(v['path'])==v['sha256']
    source=next(c for c in new['cases'] if c['name']=='echo_only')
    public=pcm(Path(source['directory'])/'system.wav').astype(np.float32)[16000:95090]/32768
    far=np.zeros(128000,np.float32);far[:len(public)]=public
    raw=np.zeros_like(far);raw[1232:]=.18*far[:-1232]
    local=next(c for c in old['authenticated_originals'] if c['group_id']=='enrollment-karen-session-1')
    assert sha(local['path'])==local['sha256']
    voice=pcm(local['path']).astype(np.float32)[:83342]/32768
    quiet=raw.copy();quiet[4000:4000+len(voice)]+=.35*voice
    near=np.zeros_like(quiet);near[4000:4000+len(voice)]=.35*voice
    variants=[('immediate_startup',raw,far,np.zeros_like(raw),()),('quiet_overlap',quiet,far,near,())]
    for c in new['cases']:
        f=Path(c['directory']);values=[pcm(f/(s+'.wav')).astype(np.float32)/32768 for s in ('microphone','system','known_near')]
        variants.append((c['name'],*values,()))
    changed=raw.copy();changed[32000:]=.18*far[32000-2400:-2400]
    variants.append(('explicit_route_change',changed,far,np.zeros_like(raw),(32000,)))
    base=Path(tempfile.mkdtemp(prefix='available-startup-native-',dir='/private/tmp/speakerdesk-attribution-private-9yk6lg11'))
    rows=[]
    for name,mic,ref,truth,boundaries in variants:
        row=replay(mic,ref,base/name,boundaries=boundaries);row['case']=name
        row['known_near_PCM_sha256']=hashlib.sha256(pcm16_bytes(truth)).hexdigest()
        if name=='genuine_repeat':
            selected=pcm(Path(row['folder'])/'microphone_clean.wav');native=pcm(Path(row['folder'])/'microphone_native.wav')
            for e in next(c for c in new['cases'] if c['name']==name)['expected']['microphone_clean']:
                a,b=e['start_sample'],e['end_sample'];assert np.array_equal(selected[a:b],native[a:b])
                assert np.array_equal(selected[a:b],np.frombuffer(pcm16_bytes(truth[a:b]),dtype='<i2'))
            row['both_genuine_repeats_exact_native_and_truth']=True
        rows.append(row)
    report=base/'REPORT.json';report.write_text(json.dumps(dict(scope='native_CPU_publication_and_saved_binding',
        private_audio=False,model_calls=0,device_calls=0,parameters=PARAMETERS,policy_sha256=POLICY_SHA256,
        native_sha256=NATIVE_SHA,rows=rows,source_sha256={str(ROOT/p):sha(ROOT/p) for p in
            ('speakerdesk/render_startup.py','speakerdesk/render_innovation.py','speakerdesk/live_meeting.py','speakerdesk/capture_sources.py','experiments/replay_available_startup.py')},
        production_admissible=False),indent=2)+'\n')
    print(json.dumps(dict(report=str(report),sha256=sha(report),cases=len(rows),model_calls=0),indent=2))


if __name__=='__main__':main()
