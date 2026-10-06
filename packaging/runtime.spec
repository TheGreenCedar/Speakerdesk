from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata
from pathlib import Path
import importlib.metadata
import os
root=Path(SPECPATH).parent
data=[]
data.append((str(root/'packaging/THIRD_PARTY_NOTICES.txt'),'.'))
data.append((str(root/'speakerdesk/redimnet2_artifact.json'),'.'))
data.append((str(root/'speakerdesk/voice_calibration.json'),'.'))
data.append((str(root/'packaging/licenses/redimnet2/LICENSE'),'licenses/redimnet2'))
for folder in ['templates','static']:
    data.append((str(root/'speakerdesk'/folder),folder))
metal_distribution=importlib.metadata.distribution('mlx-metal')
# Runtime needs the canonical Metal kernels, not SDK headers or sync-conflict copies.
data += [(str(metal_distribution.locate_file(f)),str(Path(f).parent))
         for f in metal_distribution.files if str(f)=='mlx/lib/mlx.metallib']
for package in ['mlx','mlx-audio','mlx-speech','huggingface-hub','numpy','scipy','soundfile','tokenizers']:
    data+=copy_metadata(package)
hidden=[name for name in collect_submodules('mlx') if all(part.isidentifier() for part in name.split('.'))]
hidden+=collect_submodules('mlx_audio.vad.models.nemotron_diarization')
hidden+=collect_submodules('mlx_audio.vad.models.sortformer')
hidden+=collect_submodules('mlx_audio.vad.models.silero_vad')
hidden+=['speech_admission','utterances','canonical_assembly','word_alignment','language_preferences','job_store']
hidden+=collect_submodules('mlx_audio.stt.models.whisper')
hidden+=collect_submodules('mlx_speech.models.cohere_asr')
hidden+=['mlx_speech.generation.cohere_asr','mlx_audio.vad','inference_worker','live_worker','live_refinement','live_language','meeting_refinement','rolling_refinement','live_meeting','model_setup','language_detection','app','pipeline','audio','transcript','review']
# Desktop ships the voice runtime; users only download model data in Settings.
assert importlib.metadata.version('coremltools')=='9.0', 'Install the pinned bundled voice runtime before packaging.'
data+=copy_metadata('coremltools',recursive=True)
data+=collect_data_files('coremltools',includes=['**/LICENSE*','**/COPYING*'])
hidden+=collect_submodules('coremltools.models')+['coremltools.libcoremlpython']
binary=[(str(metal_distribution.locate_file(f)),str(Path(f).parent))
        for f in metal_distribution.files if str(f).startswith('mlx/') and Path(f).suffix=='.dylib']
binary+=collect_dynamic_libs('soundfile')
binary+=collect_dynamic_libs('coremltools')
a=Analysis([str(root/'packaging/sidecar.py')],pathex=[str(root/'speakerdesk')],binaries=binary,datas=data,
           hiddenimports=hidden,excludes=['torch','transformers','tensorflow','matplotlib','pandas'],noarchive=False,
           module_collection_mode={'mlx_audio':'pyz+py'})
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='speakerdesk-runtime-aarch64-apple-darwin',debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=True,target_arch='arm64',codesign_identity=os.getenv("SPEAKERDESK_SIGNING_IDENTITY"))
