"""Bounded CPU decode/encode queues surrounding the ordered GPU worker.

No frames are skipped or reordered. Each PyAV container/codec has a single
owning thread, and cancellation/errors wake both bounded queues.
"""
from __future__ import annotations

import queue
import threading


END = object()


class VideoPipeline:
    def __init__(self, decode, encode, finish, check, depth=2):
        self.decode, self.encode, self.finish, self.check = decode, encode, finish, check
        self.inputs = queue.Queue(maxsize=depth)
        self.outputs = queue.Queue(maxsize=depth)
        self.abort = threading.Event()
        self.failure = None
        self.threads = []

    def _fail(self, exc):
        if self.failure is None:
            self.failure = exc
        self.abort.set()

    def _check(self):
        if self.failure is not None:
            raise self.failure
        if self.abort.is_set():
            raise RuntimeError('视频管线已停止')
        self.check()

    def _put(self, target, value):
        while True:
            self._check()
            try:
                target.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _get(self, source):
        while True:
            self._check()
            try:
                return source.get(timeout=.1)
            except queue.Empty:
                pass

    def _reader(self):
        try:
            for event in self.decode():
                self._put(self.inputs, event)
            self._put(self.inputs, END)
        except BaseException as exc:
            self._fail(exc)

    def _writer(self):
        try:
            while True:
                event = self._get(self.outputs)
                if event is END:
                    self.finish()
                    return
                self.encode(event)
        except BaseException as exc:
            self._fail(exc)

    def __enter__(self):
        self.threads = [threading.Thread(target=self._reader, name='video-decode', daemon=True),
                        threading.Thread(target=self._writer, name='video-encode', daemon=True)]
        for thread in self.threads:
            thread.start()
        return self

    def __iter__(self):
        while True:
            event = self._get(self.inputs)
            if event is END:
                return
            yield event

    def submit(self, event):
        self._put(self.outputs, event)

    def __exit__(self, kind, exc, traceback):
        if exc is not None:
            self.abort.set()
        else:
            try:
                self._put(self.outputs, END)
            except BaseException as error:
                self._fail(error)
        for thread in self.threads:
            thread.join(timeout=30)
        if any(t.is_alive() for t in self.threads):
            self.abort.set()
            raise RuntimeError('视频管线线程未及时停止')
        if exc is None and self.failure is not None:
            raise self.failure
        return False
