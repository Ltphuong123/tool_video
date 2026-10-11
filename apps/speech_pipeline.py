"""Keep CPU audio processing busy while the next CUDA batch is generated."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import closing
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterator

import numpy as np

from apps.video_audio_parallel import ordered_parallel_map


@dataclass
class ProcessedAudioBatch:
    start: int
    model_seconds: float
    results: Iterator
    processing_seconds: float = 0.0
    _completion: list[float] = field(default_factory=list, repr=False)


def processed_audio_batches(batches, prepare: Callable, workers: Callable, *,
                            overlap=False, check_stop=lambda: None):
    """Yield batches in order, with ordered results and bounded CUDA lookahead.

    ``batches`` yields (offset, validated waveforms, generation duration) on the
    calling thread. CPU workers never generate audio or update the interface.
    In overlap mode only the current processing batch and one generated batch
    are retained. Inputs are copied before the SDK is called again, so reused
    SDK buffers cannot change an earlier sentence. Close the iterator on error
    or cancellation to wait for all native processing calls before returning.
    """
    source = iter(batches)
    executor = None
    active = None

    def sequential(start, wavs, model_seconds):
        batch = ProcessedAudioBatch(start, model_seconds, iter(()))

        def results():
            started = perf_counter()
            try:
                with closing(ordered_parallel_map(
                        lambda item: prepare(start + item[0], item[1]), enumerate(wavs),
                        workers=workers(start, wavs), check_stop=check_stop)) as processed:
                    yield from processed
            finally:
                batch.processing_seconds = perf_counter() - started

        batch.results = results()
        return batch

    def concurrent(start, wavs, model_seconds):
        nonlocal executor
        count = workers(start, wavs)
        # One CPU worker still overlaps with CUDA generation. At most four
        # independent native operations run at once for a desktop request.
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=count, thread_name_prefix="speech-tempo")
        batch = ProcessedAudioBatch(start, model_seconds, iter(()))
        started = perf_counter()

        def process(index, wav):
            try:
                check_stop()
                return prepare(index, wav)
            finally:
                batch._completion.append(perf_counter())

        futures = [executor.submit(process, start + index, np.array(wav, dtype=np.float32, copy=True))
                   for index, wav in enumerate(wavs)]

        def results():
            try:
                for future in futures:
                    while True:
                        check_stop()
                        try:
                            value = future.result(timeout=0.05)
                            break
                        except TimeoutError:
                            if future.done():
                                raise
                    check_stop()
                    yield value
            finally:
                if batch._completion:
                    batch.processing_seconds = max(batch._completion) - started
                for future in futures:
                    future.cancel()

        batch.results = results()
        return batch

    try:
        if not overlap:
            for start, wavs, model_seconds in source:
                check_stop()
                active = sequential(start, wavs, model_seconds)
                yield active
                active.results.close()
            return

        pending = None
        for start, wavs, model_seconds in source:
            check_stop()
            if pending is not None:
                active = pending
                yield active
                active.results.close()
            pending = concurrent(start, wavs, model_seconds)
        if pending is not None:
            active = pending
            yield active
            active.results.close()
    finally:
        if active is not None:
            active.results.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(source, "close", None)
        if close is not None:
            close()
