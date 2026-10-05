"""App-managed standard voice-model cache. Downloads bytes; never enrolls or predicts."""
from pathlib import Path
import shutil
import ssl
import urllib.error
import urllib.request
import certifi
from voice_coreml import (PIN, approved_calibration, file_digest, model_identity,
                          read_voice_config, require_runtime, verify_artifact)

MODEL_DIRECTORY = 'redimnet2-b6'
RELEASE_CALIBRATION = Path(__file__).with_name('voice_calibration.json')
DISK_RESERVE = 512*1024**2


class VoiceSetup:
    def __init__(self, root, lock, activate, logger):
        self.root = Path(root)/MODEL_DIRECTORY
        self.calibration_path = RELEASE_CALIBRATION
        self.lock, self.activate, self.logger = lock, activate, logger
        self.state = {'status': 'idle', 'phase': '', 'downloaded_bytes': 0,
                      'total_bytes': PIN['total_bytes'], 'error': None}
        self._verified_signature = None

    def released(self):
        try:
            config = read_voice_config(self.calibration_path)
            approved_calibration(config, model_identity(config.get('compute_units', 'ALL')))
            return True
        except (ValueError, OSError, KeyError, TypeError):
            return False

    def supported(self):
        try:
            require_runtime()
            return True
        except ValueError:
            return False

    def installed(self):
        try:
            if self.root.is_symlink() or not self.root.is_dir():
                return False
            # Cache verification only while every file/directory signature remains unchanged.
            signature = tuple((str(p.relative_to(self.root)), p.lstat().st_mode, p.stat().st_size,
                               p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                              for p in sorted(self.root.rglob('*')))
            if signature == self._verified_signature:
                return True
            verify_artifact(self.root)
            self._verified_signature = signature
            return True
        except (ValueError, OSError):
            self._verified_signature = None
            return False

    def status(self, *, core_busy=False, available=False):
        with self.lock:
            result = self.state.copy()
        released, supported, installed = self.released(), self.supported(), self.installed()
        message = ('Saved voices are recognized once when a new speaker appears. You can correct any name.' if available else
                   'Voice recognition requires the current Speakerdesk on an Apple Silicon Mac with macOS 15 or later.' if not supported else
                   'Voice recognition is not ready in this build. Saved names are available.' if installed and not released else
                   'Included in local model setup. Save a voice in People to recognize it in future meetings.')
        if installed and result['status'] not in ('downloading','failed'):
            result['status'] = 'ready'
        return {**result, 'released': released, 'supported': supported, 'installed': installed,
                'available': available, 'can_download': released and supported and not core_busy,
                'message': message, 'license': 'MIT', 'name': 'Voice recognition'}

    def update(self, **changes):
        with self.lock:
            self.state.update(changes)

    def _safe_path(self, relative):
        path = self.root/relative
        if self.root.is_symlink() or any((self.root/Path(*Path(relative).parts[:i])).is_symlink()
                                       for i in range(1, len(Path(relative).parts)+1)):
            raise ValueError('Voice model storage is unavailable. Update Speakerdesk or contact support.')
        return path

    def _valid_file(self, path, expected):
        return path.is_file() and path.stat().st_size == expected['size'] and file_digest(path) == expected['sha256']

    def download(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.is_symlink():
                raise ValueError('Voice model storage is unavailable. Update Speakerdesk or contact support.')
            missing = sum(v['size'] for name, v in PIN['files'].items() if not self._valid_file(self._safe_path(name), v))
            if shutil.disk_usage(self.root).free < DISK_RESERVE+missing:
                raise ValueError('Free at least 0.6 GB of disk space, then retry voice setup.')
            finished = 0
            for relative, expected in PIN['files'].items():
                destination = self._safe_path(relative)
                partial = self._safe_path(relative+'.part')
                destination.parent.mkdir(parents=True, exist_ok=True)
                if self._valid_file(destination, expected):
                    partial.unlink(missing_ok=True)
                    finished += expected['size']; self.update(downloaded_bytes=finished)
                    continue
                if destination.exists():
                    destination.unlink()
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > expected['size']:
                    partial.unlink(); offset = 0
                self.update(phase='Downloading voice recognition', downloaded_bytes=finished+offset)
                if offset < expected['size']:
                    headers = {'Range': f'bytes={offset}-'} if offset else {}
                    url = f"https://huggingface.co/{PIN['model_id']}/resolve/{PIN['revision']}/{relative}"
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(request, timeout=30, context=ssl.create_default_context(cafile=certifi.where())) as response:
                        if response.status == 206:
                            expected_range = f'bytes {offset}-'
                            if not response.headers.get('Content-Range', '').startswith(expected_range):
                                partial.unlink(missing_ok=True)
                                raise ValueError('Download could not resume safely. Retry the download.')
                        else:
                            offset = 0
                        with partial.open('ab' if offset else 'wb') as output:
                            while chunk := response.read(min(512*1024, expected['size']-offset+1)):
                                offset += len(chunk)
                                if offset > expected['size']:
                                    raise ValueError('Voice model verification failed. Retry the download.')
                                if shutil.disk_usage(self.root).free < DISK_RESERVE+len(chunk):
                                    raise ValueError('Download paused because disk space is low. Free space, then resume.')
                                output.write(chunk)
                                self.update(downloaded_bytes=finished+offset)
                self.update(phase='Checking voice recognition')
                if not self._valid_file(partial, expected):
                    partial.unlink(missing_ok=True)
                    raise ValueError('Voice model verification failed. Retry the download.')
                partial.replace(destination)
                finished += expected['size']
            try:verify_artifact(self.root)
            except (ValueError,OSError,KeyError,TypeError) as error:
                raise ValueError('Voice model verification failed. Retry the download.') from error
            if self.released():
                try:self.activate(self.root, self.calibration_path)
                except (ValueError,OSError,KeyError,TypeError) as error:
                    raise ValueError('Voice recognition could not start. Update Speakerdesk or retry setup.') from error
            self.update(status='ready', phase='Ready', downloaded_bytes=PIN['total_bytes'], error=None)
        except Exception as error:
            self.logger.warning('Voice setup failed: %s', error)
            message = (str(error) if isinstance(error, ValueError) else
                       'Download interrupted. Check your connection, then resume.' if isinstance(error, urllib.error.URLError) else
                       'Voice model could not be saved. Check free disk space, then retry.' if isinstance(error, OSError) else
                       'Voice setup could not finish. Update Speakerdesk or retry.')
            self.update(status='failed', phase='Voice setup needs attention', error=message)
