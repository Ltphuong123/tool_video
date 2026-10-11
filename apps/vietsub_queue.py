"""Video/SRT queue snapshots; one model and cancellation scope per queue run."""
from dataclasses import dataclass, field
from pathlib import Path
import uuid

from apps.v3turbo_tool import Cancelled, Sampling


@dataclass(frozen=True)
class VietsubTask:
    video: Path
    subtitles: Path
    voice: str | None
    sampling: Sampling
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)


def run_vietsub_queue(tool, tasks, *, model_settings=None, cancelled=lambda: False,
                      notify=lambda identifier, state, value: None):
    from apps.desktop_queue import run_queue
    from apps.vietsub_video import _create_vietsub_video

    with tool.operation(require_model=False):
        def stopped():
            return cancelled() or tool.stop_event.is_set()

        def process(task, progress):
            if stopped():
                raise Cancelled("Đã dừng hàng đợi.")
            return _create_vietsub_video(tool, task.video, task.subtitles, task.voice,
                                         task.sampling, model_settings=model_settings, progress=progress)

        return run_queue(tasks, process, stopped, notify)
