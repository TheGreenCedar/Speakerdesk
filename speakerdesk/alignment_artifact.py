"""Pinned supplied-text CTC artifacts; functional coverage and accuracy are separate."""
ALIGNMENT_SPEC = {
    'name': 'Transcript timing', 'optional': False,
    'provider_id': 'omnilingual-ctc-mlx-supplied-text-v2',
    'supported_languages': ['en','de','fr','it','es','pt','el','nl','pl','vi','zh','ar','ja','ko'],
    'timing_accuracy_calibrated_languages': ['en'],
    'provider_identity_sha256': '7a285f556e32b8b1fb3da3ad5b7988bae5c3d7315f4f9498c6f9591cb2328829',
    'repo': 'csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12',
    'revision': '6fc542a3b0661c8278cca1230c34deb989f31202',
    'directory': 'coarse-alignment', 'weight_file': 'model.int8.onnx',
    'bytes': 365352120, 'download_bytes': 365453052,
    'file_sources': {'model.int8.onnx': {'bytes':365352120}, 'tokens.txt': {'bytes':86423},
                     'LICENSE': {'bytes':581}, 'README.md': {'bytes':13928}},
    'prepared_files': {'mlx-f32-v1-38ff6a225e75/weights.npz': {
        'bytes': 1302120358, 'sha256': '38ff6a225e75caa6e19c35a9b5823015e2550451bd0a394a10f6cda87f42058d'}},
    'sha256': 'e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c',
    'files': ['model.int8.onnx', 'tokens.txt', 'LICENSE', 'README.md'],
    'file_sha256': {
        'model.int8.onnx': 'e7c4e54ee4c4c47829cc6667d5d00ed8ea7bef1dcfeef0fce766f77752a2726c',
        'tokens.txt': 'a7a044c52cb29cbe8b0dc1953e92cefd4ca16b0ed968177b6beab21f9a7d0b31',
        'LICENSE': 'a70a523bafbb595c2844104feb313d204904dac91c3d186c05f22a10a71c7a94',
        'README.md': '8462bbca4935ffab8745b047fe6baab9b0329805818e58a926f0f7306af410fd',
    },
}
