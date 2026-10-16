"""Execute the shipped NSIS hooks on disposable destinations (Windows + NSIS only).

Run with a non-elevated Python to include the Program Files permission regression.
The harness never installs Speakerdesk or uses its registry keys.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'nt', 'native NSIS checks require Windows')
class WindowsInstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.makensis = shutil.which('makensis') or str(
            Path(os.environ['LOCALAPPDATA']) / 'tauri/NSIS/makensis.exe')
        if not Path(cls.makensis).is_file():
            raise unittest.SkipTest('Install the pinned Tauri NSIS build tools first')
        cls.build = tempfile.TemporaryDirectory(prefix='speakerdesk-installer-test-')
        cls.root = Path(cls.build.name)
        cls.exe = cls.root / 'hooks.exe'
        cls.directory_check = cls.root / 'directory-check.exe'
        script = cls.root / 'hooks.nsi'
        hook = ROOT / 'desktop/src-tauri/windows-installer.nsh'
        script.write_text(f'''Unicode true
RequestExecutionLevel user
!include MUI2.nsh
!include FileFunc.nsh
!include "{hook}"
Var MultiUser.InstallMode
OutFile "{cls.exe}"
Section
  StrCpy $MultiUser.InstallMode "CurrentUser"
  SetOutPath $INSTDIR
  !insertmacro NSIS_HOOK_PREINSTALL
  FileOpen $0 "$INSTDIR\\payload.txt" w
  FileWrite $0 "installed"
  FileClose $0
SectionEnd
''', encoding='utf-8')
        subprocess.run([cls.makensis, '/V2', str(script)], check=True,
                       capture_output=True, text=True, timeout=60)
        # Invoke the same callback that enables Next, with no filesystem mutations.
        script.write_text(f'''Unicode true
RequestExecutionLevel user
!include MUI2.nsh
!include FileFunc.nsh
!include "{hook}"
OutFile "{cls.directory_check}"
Section
  SetErrorLevel 5
  Call .onVerifyInstDir
  SetErrorLevel 0
SectionEnd
''', encoding='utf-8')
        subprocess.run([cls.makensis, '/V2', str(script)], check=True,
                       capture_output=True, text=True, timeout=60)


    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def install(self, destination):
        # /D must be last and unquoted, as required by NSIS.
        return subprocess.run(f'"{self.exe}" /S /D={destination}',
                              timeout=30, capture_output=True).returncode

    def directory_is_accepted(self, destination):
        result = subprocess.run(f'"{self.directory_check}" /S /D={destination}',
                                timeout=30, capture_output=True)
        self.assertIn(result.returncode, (0, 5))
        return result.returncode == 0

    def test_directory_page_accepts_writable_existing_and_nested_folders(self):
        for target in (self.root, self.root / 'new' / 'nested'):
            with self.subTest(target=target):
                self.assertTrue(self.directory_is_accepted(target))
        self.assertFalse((self.root / 'new').exists())

    def test_directory_page_rejects_an_existing_file(self):
        target = self.root / 'directory-page-file'
        target.write_text('retain this file')
        self.assertFalse(self.directory_is_accepted(target))
        self.assertEqual(target.read_text(), 'retain this file')

    def test_directory_page_accepts_program_files_only_when_elevated(self):
        import ctypes
        target = Path(os.environ['ProgramW6432']) / self.root.name
        self.assertFalse(target.exists())
        self.assertEqual(self.directory_is_accepted(target),
                         bool(ctypes.windll.shell32.IsUserAnAdmin()))
        self.assertFalse(target.exists())

    def test_writable_custom_folder_installs_and_removes_probe(self):
        target = self.root / 'custom installation'
        self.assertEqual(self.install(target), 0)
        self.assertEqual((target / 'payload.txt').read_text(), 'installed')
        self.assertEqual([p.name for p in target.iterdir()], ['payload.txt'])

    def test_invalid_destination_aborts_before_payload(self):
        target = self.root / 'existing-file'
        target.write_text('retain this file')
        self.assertEqual(self.install(target), 5)
        self.assertEqual(target.read_text(), 'retain this file')
        self.assertFalse((self.root / 'payload.txt').exists())

    def test_program_files_cannot_silently_succeed_without_permission(self):
        import ctypes
        if ctypes.windll.shell32.IsUserAnAdmin():
            self.skipTest('permission denial needs a non-elevated process')
        target = Path(os.environ['ProgramW6432']) / self.root.name
        self.assertFalse(target.exists())
        self.assertEqual(self.install(target), 5)
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
