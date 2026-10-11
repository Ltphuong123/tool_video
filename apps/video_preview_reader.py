"""A bounded-rate preview pipe whose frame index follows the FFmpeg clock."""
from __future__ import annotations

import subprocess

from moviepy.config import FFMPEG_BINARY
from moviepy.tools import cross_platform_popen_params
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader


class PreviewVideoReader(FFMPEG_VideoReader):
    # MoviePy indexes a raw frame pipe by FPS. For a VFR source, FFmpeg's
    # default pipe rate can differ from the metadata average MoviePy uses.
    # Explicitly resample only this preview pipe, leaving exports untouched.
    @property
    def reader(self):
        return self

    @property
    def w(self):
        return self.size[0]

    @property
    def h(self):
        return self.size[1]

    def initialize(self, start_time=0):
        self.close(delete_lastread=False)
        self.fps = min(30.0, max(1.0, self.fps))
        self.pos = self.get_frame_number(start_time)
        start_time = self.pos / self.fps
        width, height = self.size
        command = [FFMPEG_BINARY, '-hide_banner', '-loglevel', 'error', '-nostdin',
                   '-ss', format(start_time, '.9f'), '-i', self.filename,
                   '-map', '0:v:0', '-an', '-sn', '-dn',
                   '-vf', (f'scale={width}:{height},tpad=stop_mode=clone:stop_duration=1,'
                           f'fps={self.fps:.9f}:start_time=0:eof_action=pass'),
                   '-sws_flags', self.resize_algo, '-pix_fmt', self.pixel_format,
                   '-fps_mode', 'passthrough', '-f', 'image2pipe', '-vcodec', 'rawvideo', 'pipe:1']
        params = cross_platform_popen_params(dict(bufsize=self.bufsize,
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE,
                                                  stdin=subprocess.DEVNULL))
        self.proc = subprocess.Popen(command, **params)
        self.last_read = self.read_frame()

    def close(self, delete_lastread=True):
        process = getattr(self, 'proc', None)
        try:
            super().close(delete_lastread=delete_lastread)
        finally:
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
