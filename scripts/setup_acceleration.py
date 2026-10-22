"""Install pinned optional TensorRT/ONNX into a project-only dependency overlay."""
from __future__ import annotations

import json
import importlib.metadata
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from scripts.downloads import fetch
from scripts.network import open_url

ROOT = Path(__file__).resolve().parents[1]
TRT_VERSION = '10.10.0.31'


def pypi_archive(package, version):
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    with open_url(f'https://pypi.org/pypi/{package}/{version}/json') as response:
        metadata = json.load(response)
    tags = set(sys_tags())
    wheels = [f for f in metadata['urls'] if f['filename'].endswith('.whl') and parse_wheel_filename(f['filename'])[3] & tags]
    archive = wheels[0] if wheels else next(f for f in metadata['urls'] if f['packagetype'] == 'sdist')
    destination = ROOT / '.runtime' / 'downloads' / archive['filename']
    url = archive['url']
    fetch([url, url.replace('https://files.pythonhosted.org/packages/', 'https://pypi.tuna.tsinghua.edu.cn/packages/'),
           url.replace('https://files.pythonhosted.org/packages/', 'https://mirrors.aliyun.com/pypi/packages/')],
          destination, archive['digests']['sha256'], archive['size'])
    return destination


def main():
    if sys.platform != 'win32':
        raise SystemExit('This pinned installer targets Windows; WSL default mode reuses Windows. For native Linux install compatible NVIDIA packages into a project environment.')
    downloads = ROOT / '.runtime' / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    target = ROOT / '.deps' / f'tensorrt-{sys.platform}-{sys.version_info.major}{sys.version_info.minor}'
    target.mkdir(parents=True, exist_ok=True)
    manifest = {'tensorrt':TRT_VERSION, 'onnx':'1.17.0', 'protobuf':'5.29.5', 'cuda_runtime':'12.8.90', 'python':sys.version, 'target':str(target)}
    installed = {d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions(path=[str(target)])}
    archives = []
    # The large NVIDIA library wheel is published on NVIDIA's own index.
    index = 'https://pypi.nvidia.com/tensorrt-cu12-libs/'
    if installed.get('tensorrt-cu12-libs') != TRT_VERSION:
        with open_url(index) as response:
            listing = response.read().decode('utf-8')
        pattern = rf'href="(tensorrt_cu12_libs-{re.escape(TRT_VERSION)}-py2\.py3-none-win_amd64\.whl)#sha256=([a-f0-9]+)"'
        match = re.search(pattern, listing)
        if not match:
            raise RuntimeError('Pinned NVIDIA wheel not found')
        url = urllib.parse.urljoin(index, match[1])
        with open_url(url, method='HEAD') as response:
            size = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
        library = downloads / match[1]
        print('NVIDIA library: one equivalent verified official source; no mirror switching for this file.', flush=True)
        fetch([url], library, match[2], size)
        archives.append(library)
    for package, version in [('tensorrt-cu12-bindings',TRT_VERSION), ('tensorrt-cu12',TRT_VERSION),
                             ('onnx','1.17.0'), ('nvidia-cuda-runtime-cu12','12.8.90'), ('protobuf','5.29.5')]:
        if installed.get(package) != version:
            archives.append(pypi_archive(package, version))
    env = os.environ.copy()
    env['PIP_CACHE_DIR'] = str(ROOT / '.runtime' / 'pip-cache')
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation', '--upgrade', '--target', str(target), *map(str, archives)]
    if archives:
        subprocess.run(command, env=env, check=True)
    subprocess.run(['icacls.exe', str(target), '/reset', '/T', '/C'], check=True, stdout=subprocess.DEVNULL)
    (target / 'videopolish-install.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Optional acceleration installed locally. Existing Conda/PicPolish environments unchanged.', flush=True)


if __name__ == '__main__':
    main()
