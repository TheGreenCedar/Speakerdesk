"""Voice ownership in the existing registry, preserving timing-pinned pipeline bytes."""
import os
import signal
import subprocess

import pipeline

_owned_workers = set()


def launch_owned_worker(command, *, process_group=True, **options):
    if not process_group:
        raise ValueError('Voice inference requires its own process group.')
    with pipeline._worker_lock:
        if pipeline._stopping:
            raise RuntimeError('Speakerdesk is closing.')
        worker = subprocess.Popen(command, start_new_session=True, **options)
        pipeline._workers.add(worker)
        _owned_workers.add(worker)
        return worker


def release_owned_worker(worker):
    with pipeline._worker_lock:
        if worker.poll() is None:
            raise RuntimeError('An inference child has not exited.')
        pipeline._workers.discard(worker)
        _owned_workers.discard(worker)


def cancel_owned_workers():
    """Cancel voice groups after pipeline admission is sealed, before app locks."""
    with pipeline._worker_lock:
        workers = list(_owned_workers)
    for worker in workers:
        stop_owned_worker(worker)


def stop_owned_worker(worker):
    def send(sig):
        try:os.killpg(worker.pid, sig)
        except ProcessLookupError:pass
    send(signal.SIGTERM)
    try:worker.wait(timeout=3)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL);worker.wait(timeout=3)
    # Reap a frozen bootloader descendant even after its service parent exits.
    send(signal.SIGKILL)
