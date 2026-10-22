"""Prepare the two official Practical-RIFE archives, pinned after source review."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.downloads import fetch, sha256
from scripts.network import open_url

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    '4.25': {'id': '1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg',
             'archive_sha256': 'e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4',
             'sha256': '6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2', 'bytes': 24636301},
    '4.25.lite': {'id': '1zlKblGuKNatulJNFf5jdB-emp9AqGK05',
                  'archive_sha256': 'e84366c23e0eb1a637a55aad2e4f3442ec2bf4c388043b4aabbf590be289e6e8',
                  'sha256': '81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1', 'bytes': 24467405},
}


def main():
    folder = ROOT / 'models' / 'rife'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for version, meta in MODELS.items():
        url = f'https://drive.usercontent.google.com/download?id={meta["id"]}&export=download&confirm=t'
        archive = folder / f'{version}.zip'
        fetch([url], archive, meta['archive_sha256'])
        output = folder / version
        output.mkdir(exist_ok=True)
        members = {}
        with zipfile.ZipFile(archive) as source:
            # Whitelisted members only. No paths or Python bytecode are extracted.
            for name in ('flownet.pkl', 'IFNet_HDv3.py', 'RIFE_HDv3.py'):
                content = source.read('train_log/' + name)
                digest = hashlib.sha256(content).hexdigest()
                if name == 'flownet.pkl' and (digest != meta['sha256'] or len(content) != meta['bytes']):
                    raise RuntimeError('RIFE checkpoint does not match the pinned official archive')
                (output / name).write_bytes(content)
                members[name] = {'sha256': digest, 'bytes': len(content)}
        records.append({'model': version, 'source': url, 'archive_sha256': sha256(archive),
                        'members': members, 'license': 'MIT', 'pin_origin': 'official archive retrieved 2026-10-21'})
    (folder / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    license_path = ROOT / 'docs' / 'licenses' / 'Practical-RIFE-MIT.txt'
    if not license_path.exists():
        with open_url('https://raw.githubusercontent.com/hzwer/Practical-RIFE/main/LICENSE') as response:
            license_path.write_bytes(response.read())
    print('Both official RIFE checkpoints verified. Archived source is reference-only, never imported by the app.')


if __name__ == '__main__':
    main()
