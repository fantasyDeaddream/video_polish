"""Measured, hash-verified project downloads; no global configuration changes."""
from __future__ import annotations

import collections
import hashlib
import json
import threading
import time
import urllib.request
from pathlib import Path

from scripts.network import opener, routes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(urls, destination, expected_sha=None, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size) and (expected_sha is None or sha256(destination) == expected_sha):
        print(f'Reuse verified file: {destination.name}', flush=True)
        return
    records = []
    candidates = [(url, proxy) for url in dict.fromkeys(urls) for proxy in routes(url)]
    for attempt in range(len(candidates) + 1):
        fallback = attempt == len(candidates)
        if fallback:
            good = [r for r in records if r['received_bytes'] and not r.get('error')]
            if not good:
                raise RuntimeError(f'No usable download source: {records}')
            fastest = max(good, key=lambda r: r['average_mb_s'])
            selected, proxy = fastest['url'], fastest['proxy']
            print(f'All candidates measured; continue fastest usable route: {selected} via {proxy or "direct"}', flush=True)
        else:
            selected, proxy = candidates[attempt]
        print(f'Download route: {selected} via {proxy or "direct"}', flush=True)
        partial = destination.with_name(destination.name + '.part')
        status = {'received_bytes':0, 'error':None, 'done':False, 'started':None, 'digest':None}
        abort = threading.Event()
        def transfer():
            digest = hashlib.sha256()
            try:
                request = urllib.request.Request(selected, headers={'User-Agent':'VideoPolish/1.1'})
                with opener(proxy).open(request, timeout=30) as response, partial.open('wb') as output:
                    status['started'] = time.monotonic()
                    while not abort.is_set():
                        chunk = response.read1(256 * 1024) if hasattr(response, 'read1') else response.read(256 * 1024)
                        if not chunk:
                            status['digest'] = digest.hexdigest()
                            status['done'] = True
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        status['received_bytes'] += len(chunk)
            except Exception as exc:
                status['error'] = str(exc)
        thread = threading.Thread(target=transfer, daemon=True)
        thread.start()
        connected = time.monotonic()
        samples = collections.deque()
        switched = False
        last_log = connected
        while thread.is_alive():
            thread.join(timeout=1)
            now = time.monotonic()
            started = status['started']
            if started is None:
                if now - connected > 35:
                    abort.set()
                continue
            if not samples:
                samples.append((started, 0))
            samples.append((now, status['received_bytes']))
            while len(samples) > 2 and samples[1][0] < now - 30:
                samples.popleft()
            window = (status['received_bytes'] - samples[0][1]) / max(now-samples[0][0], .001) / 1e6
            if now - last_log >= 5:
                print(f'{selected}: {status["received_bytes"]/1e6:.1f} MB; window {window:.2f} MB/s', flush=True)
                last_log = now
            if now - started >= 30 and window < 3 and not fallback and len(candidates) > 1:
                print('30s average < 3 MB/s; switch equivalent source.', flush=True)
                switched = True
                abort.set()
                thread.join(timeout=35)
                if thread.is_alive():
                    raise RuntimeError('Previous transfer could not be stopped safely')
                break
        elapsed = time.monotonic() - (status['started'] or connected)
        record = {'url':selected, 'proxy':proxy, 'received_bytes':status['received_bytes'], 'seconds':elapsed,
                  'average_mb_s':status['received_bytes']/max(elapsed,.001)/1e6,
                  'error':status['error'], 'switched_low_speed':switched}
        records.append(record)
        if status['done'] and not status['error']:
            if expected_size is not None and status['received_bytes'] != expected_size:
                record['error'] = 'Size mismatch'
            elif expected_sha and status['digest'] != expected_sha:
                record['error'] = 'SHA256 mismatch'
            else:
                partial.replace(destination)
                record.update(complete=True, sha256=status['digest'])
                destination.with_name(destination.name + '.download.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
                print(f'Verified {destination.name}: {record["average_mb_s"]:.2f} MB/s, SHA256 {status["digest"]}', flush=True)
                return
        if record['error']:
            print(f'Source failed: {record["error"]}', flush=True)
    raise RuntimeError(f'Download incomplete: {records}')
