from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache'
RUNTIME = ROOT / '.runtime'
LOGS = ROOT / 'logs'
MAX_FILES = 32
MAX_UPLOAD_BYTES = int(os.environ.get('VIDEO_POLISH_MAX_UPLOAD_GB', '20')) * 1024**3
MAX_CACHE_BYTES = int(os.environ.get('VIDEO_POLISH_MAX_CACHE_GB', '80')) * 1024**3
TTL_HOURS = float(os.environ.get('VIDEO_POLISH_TTL_HOURS', '24'))
VERSION = '1.2.0'


def ensure_dirs():
    for path in (CACHE, RUNTIME, LOGS):
        if path.is_symlink():
            raise RuntimeError(f'程序目录不能是符号链接：{path}')
        path.mkdir(exist_ok=True)
        if path.resolve() != ROOT / path.name:
            raise RuntimeError(f'目录超出项目范围：{path}')
    for name in ('uploads', 'results', 'tmp'):
        path = CACHE / name
        if path.is_symlink() or (path.exists() and path.resolve() != path):
            raise RuntimeError(f'缓存目录不能重定向到其他位置：{path}')
        path.mkdir(exist_ok=True)


def remove_cache_path(path: Path):
    """Delete only a direct job child of the three project cache directories."""
    ensure_dirs()
    if path.parent not in [CACHE / n for n in ('uploads', 'results', 'tmp')]:
        raise RuntimeError('拒绝清理项目缓存范围之外的路径')
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def clear_cache():
    ensure_dirs()
    for name in ('uploads', 'results', 'tmp'):
        for path in (CACHE / name).iterdir():
            remove_cache_path(path)


def cache_bytes() -> int:
    return sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file() and not p.is_symlink())
