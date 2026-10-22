"""Install only the missing PyAV wheel into a local overlay, with verified downloads."""
from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.network import open_url

ROOT = Path(__file__).resolve().parents[1]


def download(urls, destination: Path, expected_sha: str, expected_size: int):
    """Measure actual received bytes; switch after a slow 30s window; finite fallback."""
    results = []
    for index in list(range(len(urls))) + [-1]:
        if index == -1:
            available = [r for r in results if r['bytes'] > 0 and not r.get('failed')]
            if not available:
                raise RuntimeError(f'所有下载源失败：{results}')
            winner = max(available, key=lambda r: r['bytes'] / max(r['seconds'], .001))
            url = winner['url']
            print(f'Fallback to fastest measured source: {url}', flush=True)
        else:
            url = urls[index]
        partial = destination.with_suffix(destination.suffix + '.part')
        started = time.monotonic()
        received = 0
        window = collections.deque([(started, 0)])
        last_log = started
        entry = {'url': url, 'bytes': 0, 'seconds': 0, 'failed': False}
        try:
            with open_url(url, timeout=30) as response, partial.open('wb') as output:
                entry['proxy'] = response.project_proxy
                started = time.monotonic()
                window = collections.deque([(started, 0)])
                digest = hashlib.sha256()
                while chunk := response.read(256 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    now = time.monotonic()
                    window.append((now, received))
                    while len(window) > 2 and window[1][0] < now - 30:
                        window.popleft()
                    average = (received - window[0][1]) / max(now - window[0][0], .001)
                    if now - last_log >= 5:
                        print(f'{url}: {received / 1e6:.1f} MB, window {average / 1e6:.2f} MB/s', flush=True)
                        last_log = now
                    if now - started >= 30 and average < 3_000_000 and index != -1:
                        print('30s average below 3 MB/s; switching equivalent source.', flush=True)
                        break
                else:
                    if received != expected_size or digest.hexdigest() != expected_sha:
                        raise RuntimeError('下载大小或 SHA256 校验失败')
                    output.close()  # Windows cannot rename a file with an open handle.
                    partial.replace(destination)
                    entry.update(bytes=received, seconds=time.monotonic() - started, complete=True, sha256=expected_sha)
                    results.append(entry)
                    print(f'Verified {destination.name}: {received / 1e6:.1f} MB, {received / max(entry["seconds"], .001) / 1e6:.2f} MB/s', flush=True)
                    (destination.parent / 'download-report.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                    return
        except Exception as exc:
            entry.update(failed=True, error=str(exc))
            print(f'Download failed: {exc}', flush=True)
        entry.update(bytes=received, seconds=time.monotonic() - started)
        results.append(entry)
    raise RuntimeError('下载未完成')


def main():
    try:
        import av
        if getattr(av, '__version__', '') == '16.1.0' and hasattr(av, 'open'):
            print('Reusing installed PyAV 16.1.0; no package changes.')
            return
    except ImportError:
        pass
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    with open_url('https://pypi.org/pypi/av/16.1.0/json') as response:
        metadata = json.load(response)
    compatible = set(sys_tags())
    wheel = next(f for f in metadata['urls'] if f['filename'].endswith('.whl') and parse_wheel_filename(f['filename'])[3] & compatible)
    folder = ROOT / '.runtime' / 'downloads'
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / wheel['filename']
    sha = wheel['digests']['sha256']
    partial = archive.with_suffix(archive.suffix + '.part')
    if partial.exists() and partial.stat().st_size == wheel['size'] and hashlib.sha256(partial.read_bytes()).hexdigest() == sha:
        partial.replace(archive)
        print('Reused fully downloaded wheel after SHA256 verification; no repeat transfer.')
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != sha:
        url = wheel['url']
        urls = [url, url.replace('https://files.pythonhosted.org/packages/', 'https://mirrors.aliyun.com/pypi/packages/'),
                url.replace('https://files.pythonhosted.org/packages/', 'https://pypi.tuna.tsinghua.edu.cn/packages/')]
        download(urls, archive, sha, wheel['size'])
    target = ROOT / '.deps' / f'{sys.platform}-{sys.version_info.major}{sys.version_info.minor}'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--upgrade', '--target', str(target), str(archive)], check=True)
    if sys.platform == 'win32':
        # pip moves private temporary directories; restore workspace inheritance.
        subprocess.run(['icacls.exe', str(target), '/reset', '/T', '/C'], check=True, stdout=subprocess.DEVNULL)
    print(f'Local dependencies: {target}; shared environments untouched.')


if __name__ == '__main__':
    main()
