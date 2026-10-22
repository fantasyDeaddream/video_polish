"""Fetch official lightweight checkpoints; existing RRDB files remain read-only."""
import json
import urllib.request
from pathlib import Path

from scripts.downloads import fetch, sha256
from scripts.network import open_url

ROOT = Path(__file__).resolve().parents[1]
FILENAMES = ['realesr-general-x4v3.pth', 'realesr-general-wdn-x4v3.pth', 'realesr-animevideov3.pth']
PINNED_SHA256 = {
    'realesr-general-x4v3.pth':'8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
    'realesr-general-wdn-x4v3.pth':'1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca',
    'realesr-animevideov3.pth':'b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d',
}


def main():
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    with open_url('https://api.github.com/repos/xinntao/Real-ESRGAN/releases/tags/v0.2.5.0') as response:
        release = json.load(response)
    records = []
    for filename in FILENAMES:
        asset = next(a for a in release['assets'] if a['name'] == filename)
        published = asset.get('digest')
        published = published.split(':',1)[1] if published and published.startswith('sha256:') else None
        expected = published or PINNED_SHA256[filename]
        path = folder / filename
        fetch([asset['browser_download_url']], path, expected, asset['size'])
        records.append({'file':filename,'source':asset['browser_download_url'],'bytes':path.stat().st_size,
                        'sha256':sha256(path),'release':'v0.2.5.0','official_sha256_available':bool(published),
                        'verified_against_pinned_sha256':True})
    (folder / 'lightweight-manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
