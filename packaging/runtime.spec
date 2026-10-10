from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata
from pathlib import Path
import importlib.metadata
from importlib.machinery import EXTENSION_SUFFIXES
import os
import json
root=Path(SPECPATH).parent
def installed_submodules(package, distribution, *, namespace=False):
    # Inventory sources and extensions without initializing Metal or executing
    # eager model-family imports in the build process.
    directory=Path(importlib.metadata.distribution(distribution).locate_file(package.replace('.','/')))
    if not directory.is_dir():
        raise RuntimeError(f'Installed package source unavailable: {package}')
    names={package} if namespace else set()
    for source in directory.rglob('*'):
        if not source.is_file():continue
        suffix=next((s for s in ['.py',*EXTENSION_SUFFIXES] if source.name.endswith(s)),None)
        if suffix is None:continue
        parts=list(source.relative_to(directory).parts)
        parts[-1]=parts[-1][:-len(suffix)]
        if parts[-1]=='__init__':parts.pop()
        if all(part.isidentifier() for part in parts):
            names.add('.'.join([package,*parts]))
    if package not in names:
        raise RuntimeError(f'Installed package initializer unavailable: {package}')
    return sorted(names)
data=[]
data.append((str(root/'packaging/THIRD_PARTY_NOTICES.txt'),'.'))
data.append((str(root/'packaging/licenses/capture-echo'),'licenses/capture-echo'))
data.append((str(root/'speakerdesk/redimnet2_artifact.json'),'.'))
data.append((str(root/'speakerdesk/voice_calibration.json'),'.'))
# Qualified GPU voice policy and exact source identities; no model weights.
for name in ['voice_gpu_calibration.json','voice_gpu_checkpoint.json','voice_gpu_qualification.json']:
    data.append((str(root/'speakerdesk'/name),'.'))
voice_source_modes={name:'pyz+py' for name in ['voice_mlx_backend','mlx_voice_mil',
                    'voice_waveform','thread_bound_backend','voice_profiles']}
source_modes={name:'pyz+py' for name in ['capture_sources','source_runtime','source_voice','voice_source_audio','render_innovation','render_startup','capture_processing','speech_conditioning']}
hidden_source_modules=list(source_modes)
data.append((str(root/'packaging/licenses/redimnet2/LICENSE'),'licenses/redimnet2'))
for folder in ['templates','static']:
    data.append((str(root/'speakerdesk'/folder),folder))
data.append((str(root/'packaging/licenses/alignment-runtime'),'licenses/alignment-runtime'))
data.append((str(root/'packaging/licenses/mlx-ctc-components'),'licenses/mlx-ctc-components'))
data.append((str(root/'speakerdesk/alignment-mlx'),'alignment-mlx'))
# Retain the installed Python sources whose exact bytes identify timing behavior.
timing_identity=json.loads((root/'speakerdesk/alignment-mlx/provider-identity.json').read_text())
timing_source_modes={Path(name).stem:'pyz+py' for name in timing_identity['implementation_source_sha256']}
# Exact policy/decoder file checks also work in the frozen runtime. The profile
# does not hash itself or its loader, avoiding recursive digest dependencies.
data.append((str(root/'speakerdesk/asr-reuse-profile.json'),'.'))
data.append((str(root/'speakerdesk/asr-source-reuse-profile.json'),'.'))
data.append((str(root/'speakerdesk/utterances.py'),'.'))
for name in ['canonical_runtime','final_asr_reuse','language_detection','language_probe_cache',
             'live_refinement','meeting_refinement']:
    data.append((str(root/'speakerdesk'/(name+'.py')),'.'))
# These exact components produced the frozen English GPU calibration.
for package,version in [('mlx','0.32.2'),('mlx-metal','0.32.2'),('mlx-audio','0.5.7'),('numpy','2.5.3')]:
    assert importlib.metadata.version(package)==version, f'GPU alignment runtime differs: {package}'
metal_distribution=importlib.metadata.distribution('mlx-metal')
# Runtime needs the canonical Metal kernels, not SDK headers or sync-conflict copies.
metal_kernel_files=[f for f in metal_distribution.files if str(f)=='mlx/lib/mlx.metallib']
assert len(metal_kernel_files)==1, 'Pinned MLX Metal kernel library missing.'
metal_kernel_path=str(metal_distribution.locate_file(metal_kernel_files[0]))
# PyInstaller resolves mlx.core's @rpath dependency to root/libmlx.dylib.
# MLX 0.32.2 searches beside that loaded library; retain its canonical package
# placement too, since library dependency placement can differ across builds.
data += [(metal_kernel_path,'.'),(metal_kernel_path,'mlx/lib')]
for package in ['mlx','mlx-metal','mlx-audio','mlx-speech','huggingface-hub','numpy','scipy','soundfile','tokenizers']:
    data+=copy_metadata(package)
hidden=installed_submodules('mlx','mlx',namespace=True)
hidden+=installed_submodules('mlx_audio.vad.models.nemotron_diarization','mlx-audio')
hidden+=installed_submodules('mlx_audio.vad.models.sortformer','mlx-audio')
hidden+=installed_submodules('mlx_audio.vad.models.silero_vad','mlx-audio')
hidden+=['speech_admission','admission_receipt','utterances','canonical_assembly','canonical_runtime','reading_turns','core_plan','context_plan','authoritative_tail','authority_store','word_alignment','coarse_alignment','alignment_cache','alignment_artifact','language_preferences','job_store']
for package,version in [('onnxruntime','1.30.0'),('ctc-segmentation','1.7.4'),('flatbuffers','25.12.19')]:
    assert importlib.metadata.version(package)==version, f'Install pinned alignment runtime: {package}'
    data+=copy_metadata(package)
hidden+=['onnxruntime','onnxruntime.capi.onnxruntime_pybind11_state',
         'ctc_segmentation','ctc_segmentation.ctc_segmentation_dyn','flatbuffers']
hidden+=installed_submodules('mlx_audio.stt.models.whisper','mlx-audio')
hidden+=['alignment_model','mlx_ctc_forward','asr_reuse_profile','alignment_setup','alignment_provider']
hidden+=hidden_source_modules
hidden+=['voice_gpu_runtime','voice_gpu_process','voice_gpu_worker','voice_worker_registry',
         'voice_profile_status',*voice_source_modes]
hidden+=['mlx_ctc_components','mlx_ctc_components.models.base',
         'mlx_ctc_components.models.mms.mms','mlx_ctc_components.models.wav2vec.wav2vec']
hidden+=installed_submodules('mlx_speech.models.cohere_asr','mlx-speech')
hidden+=['mlx_speech.generation.cohere_asr','mlx_audio.vad','inference_worker','live_worker','live_refinement','live_language','meeting_refinement','rolling_refinement','live_meeting','model_setup','language_detection','app','pipeline','audio','transcript','review']
# Desktop ships the voice runtime; users only download model data in Settings.
assert importlib.metadata.version('coremltools')=='9.0', 'Install the pinned bundled voice runtime before packaging.'
data+=copy_metadata('coremltools',recursive=True)
data+=collect_data_files('coremltools',includes=['**/LICENSE*','**/COPYING*'])
hidden+=installed_submodules('coremltools.models','coremltools')+['coremltools.libcoremlpython']
binary=[(str(metal_distribution.locate_file(f)),str(Path(f).parent))
        for f in metal_distribution.files if str(f).startswith('mlx/') and Path(f).suffix=='.dylib']
binary.append((str(root/'desktop/capture/echo/libspeakerdesk_echo.dylib'),'.'))
binary+=collect_dynamic_libs('soundfile')
binary+=collect_dynamic_libs('coremltools')
binary+=collect_dynamic_libs('onnxruntime')
a=Analysis([str(root/'packaging/sidecar.py')],pathex=[str(root/'speakerdesk')],binaries=binary,datas=data,
           hiddenimports=hidden,excludes=['torch','transformers','tensorflow','matplotlib','pandas','Cython','pyximport'],noarchive=False,
           module_collection_mode={'mlx_audio':'pyz+py','mlx_speech':'pyz+py',
                                   'mlx_ctc_components':'pyz+py',**timing_source_modes,**voice_source_modes,**source_modes})
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='speakerdesk-runtime-aarch64-apple-darwin',debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=True,target_arch='arm64',codesign_identity=os.getenv("SPEAKERDESK_SIGNING_IDENTITY"))
