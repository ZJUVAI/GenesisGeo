"""Forkserver setup and lifecycle for isolated worker processes."""

import multiprocessing as mp
from multiprocessing.connection import wait as wait_for_process_exit
import os
import signal
import threading
import time

_START_LOCK = threading.Lock()


def configure_runtime(*preload):
    os.environ["PYTHONHASHSEED"] = "0"
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    mp.set_forkserver_preload(list(preload))


def _serve(connection, function):
    os.setsid()
    with connection:
        while True:
            try:
                argument = connection.recv()
            except EOFError:
                return
            try:
                result = {"status": "ok", "output": function(argument)}
            except Exception as exc:
                result = {"status": "error", "error": repr(exc)}
            connection.send(result)


class Workers:
    """Each coordinator owns a child and recycles it after a bounded job count."""

    def __init__(self, limit):
        self.limit = limit
        self.local = threading.local()
        self.children = {}

    def _start(self, function):
        context = mp.get_context("forkserver")
        connection, endpoint = context.Pipe()
        child = context.Process(target=_serve, args=(endpoint, function))
        try:
            with _START_LOCK:
                child.start()
                self.children[child] = connection
        except BaseException:
            connection.close()
            raise
        finally:
            endpoint.close()
        self.local.child = child
        self.local.count = 0
        return child

    def _stop(self, child):
        with _START_LOCK:
            self.children.pop(child).close()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                if child.exitcode is None:
                    child.kill()
        # Waiting for the forkserver outside the lock lets other coordinators
        # start or terminate their own workers. Reap and close under the lock
        # because Process.start() also polls the global child inventory.
        wait_for_process_exit([child.sentinel])
        with _START_LOCK:
            child.join()
            child.close()

    def close(self):
        for child in list(self.children):
            self._stop(child)

    def call(self, function, argument, timeout):
        started = time.monotonic()
        child = getattr(self.local, "child", None)
        try:
            if child is None:
                child = self._start(function)
            connection = self.children[child]
            connection.send(argument)
            result = (
                connection.recv()
                if connection.poll(timeout)
                else {"status": "wall_timeout"}
            )
        except Exception as exc:
            result = {"status": "error", "error": repr(exc)}
        if child is not None:
            self.local.count += 1
            if result["status"] != "ok" or self.local.count >= self.limit:
                self._stop(child)
                self.local.child = None
        return {**result, "wall": time.monotonic() - started}
