import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pedalboard import time_stretch

from apps.video_editor import _ramped_audio_chunks, build_multi_speed_time_map, SpeedSegment
from apps.video_markers import TimeMarker, build_marker_time_map


def stretch(item):
    data, factor = item
    return time_stretch(data, 48000, stretch_factor=factor,
                        pitch_shift_in_semitones=0.0, high_quality=True,
                        retain_phase_continuity=True, preserve_formants=True)


if __name__ == '__main__':
    print('CPUs', os.cpu_count(), flush=True)
    sample_count = 99840
    t = np.arange(sample_count, dtype=np.float32) / 48000
    base = np.stack((0.12 * np.sin(2*np.pi*220*t) + .04*np.sin(2*np.pi*2000*t),
                     .09 * np.sin(2*np.pi*390*t) + .04*np.sin(2*np.pi*1210*t)))
    jobs = [(np.ascontiguousarray(base), .8 + (i % 6)*.12) for i in range(30)]
    outputs = None
    for workers in (1, 2, 4, 8):
        first = time.perf_counter()
        if workers == 1:
            result = list(map(stretch, jobs))
            outputs = result
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                result = list(pool.map(stretch, jobs))
        print('Native', workers, round(time.perf_counter()-first, 3),
              'exact', all(np.array_equal(a, b) for a, b in zip(outputs, result)), flush=True)

    class ToneAudio:
        duration = 60.
        nchannels = 2
        calls = 0
        seconds = 0.
        def get_frame(self, times):
            now = time.perf_counter()
            self.calls += 1
            out = np.stack((.12*np.sin(2*np.pi*220*times)+.04*np.sin(2*np.pi*2000*times),
                            .09*np.sin(2*np.pi*390*times)+.04*np.sin(2*np.pi*1210*times)), axis=-1)
            self.seconds += time.perf_counter()-now
            return out

    for count in (4, 100):
        old = tuple(TimeMarker(i+1, round(i*59000/(count-1))) for i in range(count))
        new = tuple(TimeMarker(i+1, round(i*59000/(count-1)*.87+(200 if i%2 else 0))) for i in range(count))
        mapping = build_marker_time_map(60., old, new)
        audio = ToneAudio()
        now = time.perf_counter()
        chunks = list(_ramped_audio_chunks(audio, mapping))
        print('Mapping', count, 'seconds', round(time.perf_counter()-now, 3),
              'read_calls', audio.calls, 'read_seconds', round(audio.seconds, 3),
              'samples', sum(len(c) for c in chunks), flush=True)

    from moviepy import AudioFileClip
    from apps.video_editor import _guard_reader_cleanup
    from unittest.mock import patch
    from apps.video_audio_parallel import ordered_parallel_map

    source = AudioFileClip(r'C:\Users\Admin\Desktop\New folder\video_139cec222941.mp4', fps=48000)
    _guard_reader_cleanup(source)
    print('Real audio', source.duration, source.nchannels, flush=True)
    duration = min(30, source.duration)
    source.duration = duration
    source.end = duration
    mapping = build_multi_speed_time_map(duration, [SpeedSegment(0,duration,1.1,0)])
    real_jobs = []
    timings = [0.]
    def measured(data, rate, **kwargs):
        real_jobs.append((data.copy(), kwargs['stretch_factor']))
        now = time.perf_counter()
        result = time_stretch(data, rate, **kwargs)
        timings[0] += time.perf_counter()-now
        return result
    now = time.perf_counter()
    try:
        with patch('apps.speech_speed.require_rubberband', return_value=measured):
            for chunk in _ramped_audio_chunks(source, mapping):
                pass
        print('Real baseline', round(time.perf_counter()-now,3), 'RubberBand', round(timings[0],3),
              'windows', len(real_jobs), flush=True)
    finally:
        source.close()
    reference = None
    for workers in (1, 4):
        now = time.perf_counter()
        result = list(ordered_parallel_map(stretch, real_jobs, workers=workers))
        if reference is None:
            reference = result
        print('Real native',workers, round(time.perf_counter()-now,3),
              'exact',all(np.array_equal(a,b) for a,b in zip(reference,result)),flush=True)
