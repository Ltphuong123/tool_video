"""Publish finished result folders without overwriting existing outputs."""
from pathlib import Path
import os
import re


def bundle_folder_name(source, fallback='Video'):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(source).stem).rstrip(' .') or fallback
    if re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]', name.split('.')[0], re.I):
        name = '_' + name
    return name


def publish_output_folder(ready, output_dir, name, check_stop):
    # A same-volume rename exposes the complete bundle without another copy.
    output_dir = output_dir.resolve()
    if ready.is_symlink() or ready.resolve().parent.parent != output_dir:
        raise ValueError('Thư mục tạm không nằm trong nơi lưu kết quả.')
    index = 1
    while True:
        check_stop()
        destination = output_dir / (name if index == 1 else f'{name}_{index}')
        if destination.exists() or destination.is_symlink():
            index += 1
            continue
        if destination.resolve().parent != output_dir:
            raise ValueError('Thư mục kết quả không nằm trong nơi lưu đã chọn.')
        try:
            os.rename(ready, destination)
        except FileExistsError:
            index += 1
            continue
        return destination
