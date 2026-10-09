"""Compiler launcher: inspect only its own newly created compiler PID."""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

class Usage(ctypes.Structure):
    _fields_=[('uuid',ctypes.c_ubyte*16)]+[(name,ctypes.c_uint64) for name in
        'user system wake interrupts pageins wired rss footprint start exit childuser childsystem childwake childinterrupts childpageins childelapsed read write'.split()]

lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
lib.proc_pid_rusage.restype=ctypes.c_int
p=subprocess.Popen(sys.argv[1:])
peak=0;reason=None;start=time.monotonic()
while p.poll() is None:
    usage=Usage()
    if lib.proc_pid_rusage(p.pid,2,ctypes.byref(usage)):
        if p.poll() is None:reason='owned compiler RSS unavailable';break
    peak=max(peak,usage.rss)
    if peak>900*1024**2:reason='compiler RSS cap';break
    if time.monotonic()-start>600:reason='compiler wall cap';break
    time.sleep(.05)
if reason:p.kill()
p.wait()
folder=Path(__file__).resolve().parents[1]/'.cache/echo-build/compiler-receipts'
folder.mkdir(exist_ok=True)
(folder/f'{p.pid}.json').write_text(json.dumps({'owned_pid':p.pid,'peak_rss_bytes':peak,'exit_code':p.returncode,'stop_reason':reason})+'\n')
if reason:print(reason,file=sys.stderr)
raise SystemExit(p.returncode if not reason else 1)
