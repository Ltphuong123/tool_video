"""Bounded, ordered native audio work with sequential source decoding."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys


def ordered_parallel_map(function, jobs, *, check_stop=lambda: None,
                         workers=4, prefetch=None):
    """Yield independent results in input order, keeping only a few jobs alive.

    ``jobs`` is advanced exclusively on the calling thread. A MoviePy audio
    reader can therefore decode inputs there while native Rubber Band calls
    run concurrently. Input order and each function's arguments are unchanged.

    Closing the generator, cancellation, or an error cancels queued jobs and
    waits for native calls that have already started. No worker can keep using
    its audio input after this generator returns. Zero/one job and ``workers=1``
    do not create a thread pool.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if prefetch is None:
        prefetch = workers
    if not isinstance(prefetch, int) or isinstance(prefetch, bool) or prefetch < 1:
        raise ValueError("prefetch must be a positive integer")
    workers = min(workers, prefetch)
    iterator = iter(jobs)
    executor = None
    pending = deque()

    def next_job():
        check_stop()
        return next(iterator)

    try:
        if workers == 1:
            while True:
                try:
                    job = next_job()
                except StopIteration:
                    return
                result = function(job)
                check_stop()
                yield result

        try:
            first = next_job()
        except StopIteration:
            return
        try:
            second = next_job()
        except StopIteration:
            check_stop()
            result = function(first)
            check_stop()
            yield result
            return

        executor = ThreadPoolExecutor(max_workers=workers,
                                      thread_name_prefix="video-audio")
        pending.append(executor.submit(function, first))
        pending.append(executor.submit(function, second))
        del first, second
        exhausted = False
        while len(pending) < prefetch:
            try:
                job = next_job()
            except StopIteration:
                exhausted = True
                break
            pending.append(executor.submit(function, job))

        while pending:
            future = pending.popleft()
            while True:
                check_stop()
                try:
                    result = future.result(timeout=0.05)
                    break
                except TimeoutError:
                    # A worker may itself raise TimeoutError. Once done, that
                    # is its original failure rather than a polling timeout.
                    if future.done():
                        raise
            check_stop()
            yield result
            if not exhausted:
                try:
                    job = next_job()
                except StopIteration:
                    exhausted = True
                else:
                    pending.append(executor.submit(function, job))
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            for future in pending:
                future.cancel()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException:
                    if not failed:
                        raise
