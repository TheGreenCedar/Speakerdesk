"""Source-owned GPU voice policy; verification never initializes a neural model."""
import importlib.metadata
import json
from pathlib import Path
import platform

from voice_coreml import approved_calibration, file_digest, read_voice_config
from voice_profiles import VoiceModel
from voice_waveform import PIN, verify_artifact

ROOT = Path(__file__).parent
RELEASE_CALIBRATION = ROOT/'voice_gpu_calibration.json'
CALIBRATION_SHA256 = 'fa4eba93aa5611f16dab48a11afb741e3cb23985bfc19c345df6c647157c5927'
QUALIFICATION_SHA256 = '43291bace1ee457f8487c29fe35c904daf598109684effd27366d3252eae6862'
CHECKPOINT_SHA256 = 'f2e20efab39ad0d3dc63bb4f48c7d69dbb4666691e59786cdab7e0be0b69e36d'


def require_runtime():
    version = platform.mac_ver()[0]
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64'
            or not version or int(version.split('.')[0]) < 15):
        raise ValueError('Voice recognition requires Apple Silicon and macOS 15 or later.')
    try:
        if importlib.metadata.version('mlx') != '0.32.2':
            raise ValueError('Voice recognition requires the pinned MLX GPU runtime.')
    except importlib.metadata.PackageNotFoundError:
        raise ValueError('The qualified local voice runtime is not installed.') from None


def release_policy(config_path=RELEASE_CALIBRATION):
    """Bind the separately accepted candidate to the unchanged sealed evidence."""
    config_path = Path(config_path)
    if file_digest(config_path) != CALIBRATION_SHA256:
        raise ValueError('Voice policy differs from the qualified GPU candidate.')
    config = read_voice_config(config_path)
    qualification_path, checkpoint_path = ROOT/'voice_gpu_qualification.json', ROOT/'voice_gpu_checkpoint.json'
    if (file_digest(qualification_path) != QUALIFICATION_SHA256
            or file_digest(checkpoint_path) != CHECKPOINT_SHA256):
        raise ValueError('Voice qualification evidence is missing or changed.')
    qualification = json.loads(qualification_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    for name, digest in {
        'redimnet2_artifact.json':'01d69f47a1ec8af7c84a254ed836c421657c6933ca36d9430a314e811e716ecd',
        'voice_profiles.py':'9651006e9b851275136f6b4db1c94c10a3dbf110814da57cd9c0914073eeaeeb',
    }.items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('The qualified voice artifact or scoring contract has changed.')
    for name, digest in qualification['neural_source_sha256'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('The qualified voice implementation has changed.')
    if file_digest(ROOT/'thread_bound_backend.py') != qualification['thread_wrapper_sha256']:
        raise ValueError('The qualified inference owner has changed.')
    model = VoiceModel(**qualification['model'])
    if (config['qualification_sha256'] != QUALIFICATION_SHA256
            or config['sealed_checkpoint_sha256'] != CHECKPOINT_SHA256
            or config['implementation_identity_sha256'] != qualification['implementation_identity_sha256']
            or config['model'] != checkpoint['model']
            or any(config['calibration'][key] != checkpoint[key] for key in config['calibration'])):
        raise ValueError('Voice policy does not match its sealed GPU evidence.')
    return model, approved_calibration(config, model)


def load_approved_runtime(config_path=RELEASE_CALIBRATION, *, model_dir=None, updates, audio_root):
    model, calibration = release_policy(config_path)
    require_runtime()
    root = Path(model_dir) if model_dir is not None else Path(config_path).parent/'redimnet2-b6'
    verify_artifact(root)
    from voice_gpu_process import OwnedVoiceBackend
    return OwnedVoiceBackend(root, config_path, model, updates, audio_root), calibration
