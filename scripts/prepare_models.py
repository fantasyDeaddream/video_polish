"""Prepare all release models; verified existing models are reused read-only."""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.downloads import fetch, sha256

ROOT = Path(__file__).resolve().parents[1]


def existing_model(relative):
    candidates = [ROOT / 'models' / relative]
    if not relative.startswith('rife/'):
        if os.environ.get('VIDEO_POLISH_MODELS'):
            candidates.insert(0, Path(os.environ['VIDEO_POLISH_MODELS']) / relative)
        candidates.append(ROOT.parent / 'pic_polish' / 'models' / relative)
    return [p for p in candidates if p.is_file()]


def main():
    catalog = json.loads((ROOT / 'models' / 'catalog.json').read_text(encoding='utf-8'))
    missing_rife = False
    for record in catalog:
        matches = [p for p in existing_model(record['path'])
                   if p.stat().st_size == record['bytes'] and sha256(p) == record['sha256']]
        if matches:
            print('Reuse verified model:', record['path'], flush=True)
        elif record['family'] == 'rife':
            missing_rife = True
        else:
            fetch([record['source']], ROOT / 'models' / record['path'], record['sha256'], record['bytes'])
    # These reviewed sources are only needed for the official-architecture tests.
    manifest = json.loads((ROOT / 'models/rife/manifest.json').read_text(encoding='utf-8'))
    for record in manifest:
        for name, info in record['members'].items():
            path = ROOT / 'models/rife' / record['model'] / name
            if not path.is_file() or sha256(path) != info['sha256']:
                missing_rife = True
    if missing_rife:
        from scripts.fetch_interpolation import main as prepare_rife
        prepare_rife()
    print('All eight super-resolution/interpolation models are ready.', flush=True)


if __name__ == '__main__':
    main()
