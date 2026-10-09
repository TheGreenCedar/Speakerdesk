"""Managed transcript timing: pinned artifacts, restart discovery and explicit activation."""
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

import certifi
from alignment_provider import (ALIGNMENT_SPEC,
                                digest_file, manifest, prepare_model, require_runtime,
                                verify, gpu_check, validate_receipt)

RESERVE = 512 * 1024**2


def available(directory):
    if not directory or not ALIGNMENT_SPEC['supported_languages']:return False
    try:
        metadata=verify(directory)
        validate_receipt(json.loads((Path(directory)/'gpu-ready.json').read_text()),metadata)
        return True
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError):
        return False


def enabled(root):
    # Timing is part of the required transcription pipeline. Historical off
    # preferences cannot silently choose an incomplete pipeline after upgrade.
    return True


def save_enabled(root, value):
    if value is not True:
        raise ValueError('Transcript timing is required. Finish setup or retry timing setup.')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    temporary = root/'alignment-settings.json.part'
    temporary.write_text(json.dumps({'enabled': value})+'\n')
    temporary.replace(root/'alignment-settings.json')


def probe(directory):
    from pipeline import run_worker
    folder=Path(directory)/'.setup-check';folder.mkdir(exist_ok=True)
    python=os.getenv('ASR_PYTHON',sys.executable)
    try:
        # Owned by the same child registry used by inference and app shutdown.
        receipt=run_worker(python,'alignment-check',{'directory':str(directory)},folder,timeout=120)
        validate_receipt(receipt,manifest())
        return receipt
    except (RuntimeError,ValueError,KeyError,TypeError) as error:
        raise RuntimeError('Transcript timing could not run on the GPU. Update Speakerdesk or retry setup. No CPU fallback is used.') from error


class AlignmentSetup:
    def __init__(self, root, lock, logger):
        self.root, self.lock, self.logger = Path(root), lock, logger
        self.folder = self.root/ALIGNMENT_SPEC['directory']
        self.state = {'status': 'idle', 'phase': '', 'error': None, 'downloaded_bytes': 0,
                      'total_bytes': ALIGNMENT_SPEC.get('download_bytes',ALIGNMENT_SPEC['bytes'])}
        try:
            if not ALIGNMENT_SPEC['supported_languages']:
                raise ValueError('Timing provider has no functional language coverage')
            self.metadata=manifest()
        except (OSError,ValueError,KeyError,TypeError):
            self.metadata=None
            self.state.update(status='failed',error='Transcript timing is unavailable in this build. Update Speakerdesk.')

    def update(self, **changes):
        with self.lock:
            self.state.update(changes)

    def ready(self):
        if self.metadata is None:return False
        try:
            metadata = verify(self.folder)
            validate_receipt(json.loads((self.folder/'gpu-ready.json').read_text()), metadata)
            return True
        except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError):
            return False

    def status(self, *, busy=False, supported=True):
        with self.lock:
            result = self.state.copy()
        ready = self.ready()
        on = enabled(self.root)
        if result['status']=='idle' and not ready:
            try:
                error=json.loads((self.folder/'setup-error.json').read_text())['error']
                if isinstance(error,str):result.update(status='failed',error=error[:2048])
            except (OSError,ValueError,KeyError,TypeError):pass
        if result['status'] == 'idle' and any(self.folder.glob('*.part')):
            result.update(status='interrupted', error='Download interrupted. Resume Transcript timing setup.')
        if result['status'] == 'idle' and ready:
            result['status'] = 'ready'
        message = ('Required transcript timing is ready for new meetings and imports.' if ready else
                   'Finish required transcript timing setup before transcription. Unsupported text is marked for review.')
        if not ready and (self.folder/ALIGNMENT_SPEC['weight_file']).is_file():
            message = 'Downloaded timing data needs preparation and a GPU check. Retry setup.'
        return {**result, 'id': ALIGNMENT_SPEC['directory'], 'provider_id': ALIGNMENT_SPEC['provider_id'],
                'provider_identity_sha256': self.metadata.get('provider_identity_sha256') if self.metadata else None,
                'supported_languages': ALIGNMENT_SPEC['supported_languages'],
                'timing_accuracy_calibrated_languages': ALIGNMENT_SPEC.get('timing_accuracy_calibrated_languages',
                    ['en'] if ALIGNMENT_SPEC['provider_id']=='omnilingual-ctc-mlx-english-v1' else []), 'ready': ready, 'enabled': on, 'active': ready and on,
                'stored_bytes': sum(item['bytes'] for item in ALIGNMENT_SPEC['prepared_files'].values())+self.state['total_bytes'],
                'required_free_bytes': sum(item['bytes'] for item in ALIGNMENT_SPEC['prepared_files'].values())+self.state['total_bytes']+RESERVE,
                'can_download': not busy and supported and self.metadata is not None, 'message': message}

    def _download_file(self, name, digest, finished=0):
        source=ALIGNMENT_SPEC.get('file_sources',{}).get(name,{})
        size=source.get('bytes',ALIGNMENT_SPEC['bytes'] if name==ALIGNMENT_SPEC['weight_file'] else None)
        destination = self.folder/name
        if destination.is_file() and digest_file(destination) == digest:
            self.update(downloaded_bytes=finished+(size or 0))
            return
        partial = self.folder/(name+'.part')
        offset = partial.stat().st_size if partial.exists() else 0
        if size is not None and offset > size:
            partial.unlink(); offset = 0
        self.update(phase='Downloading Transcript timing', downloaded_bytes=finished+offset if size else finished)
        if size is None or offset < size:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            repo=source.get('repo',ALIGNMENT_SPEC['repo']);revision=source.get('revision',ALIGNMENT_SPEC['revision'])
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true'
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60,
                                        context=ssl.create_default_context(cafile=certifi.where())) as response:
                if response.status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        partial.unlink(missing_ok=True)
                        raise ValueError('Download could not resume safely. Retry transcript timing setup.')
                else:
                    offset = 0
                with partial.open('ab' if offset else 'wb') as output:
                    while block := response.read(1024**2):
                        offset += len(block)
                        if size is not None and offset > size:
                            partial.unlink(missing_ok=True)
                            raise ValueError('Transcript timing download is invalid. Retry setup.')
                        if shutil.disk_usage(self.folder).free < RESERVE+len(block):
                            raise ValueError('Transcript timing download paused. Free disk space, then resume.')
                        output.write(block)
                        if size:self.update(downloaded_bytes=finished+offset)
        self.update(phase='Verifying Transcript timing')
        if (size is not None and partial.stat().st_size != size) or digest_file(partial) != digest:
            partial.unlink(missing_ok=True)
            raise ValueError('Transcript timing download verification failed. Retry setup.')
        partial.replace(destination)

    def download(self, *, activate=True):
        try:
            if self.metadata is None:raise RuntimeError('Transcript timing is unavailable in this build. Update Speakerdesk.')
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder/'gpu-ready.json').unlink(missing_ok=True)
            (self.folder/'setup-error.json').unlink(missing_ok=True)
            # Runtime drift must be actionable before downloading hundreds of MB.
            try:require_runtime(self.metadata)
            except (ImportError,RuntimeError) as error:
                raise RuntimeError('Timing requires the packaged GPU runtime. Update Speakerdesk, then retry setup.') from error
            missing=0
            for name,digest in ALIGNMENT_SPEC['file_sha256'].items():
                path=self.folder/name
                if not path.is_file() or digest_file(path)!=digest:
                    missing+=ALIGNMENT_SPEC.get('file_sources',{}).get(name,{}).get('bytes',
                        ALIGNMENT_SPEC['bytes'] if name==ALIGNMENT_SPEC['weight_file'] else 0)
            prepared=[(self.folder/name,item) for name,item in ALIGNMENT_SPEC['prepared_files'].items()]
            invalid_prepared=[(path,item) for path,item in prepared if not path.is_file() or digest_file(path)!=item['sha256']]
            needed = missing+sum(item['bytes'] for path,item in invalid_prepared)+RESERVE
            if shutil.disk_usage(self.folder).free < needed:
                raise ValueError(f'Free at least {needed/1e9:.1f} GB, then retry transcript timing setup.')
            self.update(status='downloading', error=None)
            finished=0
            for name, digest in ALIGNMENT_SPEC['file_sha256'].items():
                self._download_file(name, digest,finished)
                finished+=ALIGNMENT_SPEC.get('file_sources',{}).get(name,{}).get('bytes',
                    ALIGNMENT_SPEC['bytes'] if name==ALIGNMENT_SPEC['weight_file'] else 0)
            for path,item in invalid_prepared:
                path.unlink(missing_ok=True)  # Explicit repair; loader still refuses corruption.
            self.update(phase='Preparing transcript timing for GPU')
            prepare_model(self.folder)
            verify(self.folder)
            self.update(phase='Checking transcript timing on GPU')
            receipt = probe(self.folder)
            validate_receipt(receipt,self.metadata)
            temporary = self.folder/'gpu-ready.json.part'
            temporary.write_text(json.dumps(receipt)+'\n')
            temporary.replace(self.folder/'gpu-ready.json')
            save_enabled(self.root, True)
            self.update(status='ready', phase='Transcript timing enabled' if activate else 'Transcript timing ready', error=None,
                        downloaded_bytes=self.state['total_bytes'])
        except Exception as error:
            self.logger.warning('Transcript timing setup failed: %s', error)
            message = ('Download interrupted. Check your connection, then resume transcript timing setup.'
                       if isinstance(error, urllib.error.URLError) else
                       'Transcript timing check timed out. Retry setup.' if isinstance(error, subprocess.TimeoutExpired) else
                       str(error) if isinstance(error, (ValueError, RuntimeError)) else
                       'Transcript timing could not be saved. Check disk space and retry setup.')
            self.update(status='failed', phase='Transcript timing needs attention', error=message)
            try:(self.folder/'setup-error.json').write_text(json.dumps({'error':message})+'\n')
            except OSError:pass


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--alignment-check':
    print(json.dumps(gpu_check(sys.argv[2])), flush=True)
