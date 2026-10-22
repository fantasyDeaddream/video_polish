"""Run a module/script with project-only dependency overlays; never change conda."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')
DEPS = ROOT / '.deps' / f'{sys.platform}-{sys.version_info.major}{sys.version_info.minor}'
ACCEL_DEPS = ROOT / '.deps' / f'tensorrt-{sys.platform}-{sys.version_info.major}{sys.version_info.minor}'
sys.path.insert(0, str(ROOT))
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
if ACCEL_DEPS.exists():
    sys.path.insert(0, str(ACCEL_DEPS))
# Keep directory handles alive for Windows' DLL search lifetime.
DLL_HANDLES = []
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    for directory in (ACCEL_DEPS / 'tensorrt_libs', ACCEL_DEPS / 'nvidia' / 'cuda_runtime' / 'bin'):
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONUNBUFFERED', '1')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/runner.py MODULE [args...]')
    module = sys.argv.pop(1)
    if module == '-c':
        exec(sys.argv.pop(1), {'__name__': '__main__'})
    elif module.endswith('.py'):
        sys.argv[0] = module
        runpy.run_path(module, run_name='__main__')
    else:
        sys.argv[0] = module
        runpy.run_module(module, run_name='__main__', alter_sys=True)
