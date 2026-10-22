"""Project-scoped routing: known CN sources direct, other sources via local VPN."""
from __future__ import annotations

import os
import urllib.request
from urllib.parse import urlsplit

CN_HOSTS = ('hf-mirror.com', 'mirrors.aliyun.com', 'mirrors.cloud.tencent.com')
DEFAULT_PROXIES = ('http://127.0.0.1:7900', 'http://127.0.0.1:7897')


def routes(url):
    host = (urlsplit(url).hostname or '').lower()
    local_or_cn = host in {'localhost', '127.0.0.1', '::1'} or host.endswith('.cn') or any(
        host == name or host.endswith('.' + name) for name in CN_HOSTS)
    if local_or_cn:
        return [None]
    override = os.environ.get('VIDEO_POLISH_DOWNLOAD_PROXY')
    if override == 'direct':
        return [None]
    proxies = tuple(p.strip() for p in override.split(',') if p.strip()) if override else DEFAULT_PROXIES
    return list(dict.fromkeys([*proxies, None]))


def opener(proxy):
    # Explicitly bypass environment/system proxy configuration for a direct route.
    return urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': proxy, 'https': proxy} if proxy else {}))


def open_url(url, timeout=20, headers=None, method='GET'):
    errors = []
    for proxy in routes(url):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'VideoPolish/1.2', **(headers or {})}, method=method)
            response = opener(proxy).open(request, timeout=timeout)
            response.project_proxy = proxy
            return response
        except Exception as exc:
            errors.append(f'{proxy or "direct"}: {exc}')
    raise OSError('All project network routes failed: ' + '; '.join(errors))
