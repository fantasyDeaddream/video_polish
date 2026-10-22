"""Cross-platform service lifecycle, identity-checked termination and safe cleanup."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from app import config

STATE = config.RUNTIME / 'server.json'
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def identity(pid):
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            handle = kernel.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return None
            try:
                created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
                if not kernel.GetProcessTimes(wintypes.HANDLE(handle), ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                    return None
                code = wintypes.DWORD()
                kernel.GetExitCodeProcess(wintypes.HANDLE(handle), ctypes.byref(code))
                if code.value != 259:
                    return None
                return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
            finally:
                kernel.CloseHandle(wintypes.HANDLE(handle))
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def lock():
    config.ensure_dirs()
    path = config.RUNTIME / 'service.lock'
    with path.open('a+b') as handle:
        handle.seek(0)
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_state():
    if not STATE.exists():
        return None
    state = json.loads(STATE.read_text(encoding='utf-8'))
    if Path(state.get('root', '')).resolve() != config.ROOT:
        raise RuntimeError('服务记录不属于当前项目，拒绝操作')
    return state


def health(state):
    try:
        with HTTP.open(state['url'] + '/api/health', timeout=2) as response:
            value = json.load(response)
        return value if value.get('app') == 'VideoPolish' and value.get('instance') == state['instance'] else None
    except (OSError, ValueError):
        return None


def alive(state):
    return state and state.get('platform') == sys.platform and identity(state['pid']) == state.get('identity') and state.get('identity') is not None


def start(args):
    previous = read_state()
    if previous and health(previous):
        print(f'Already running: {previous["url"]}')
        if not args.no_browser:
            webbrowser.open(previous['url'])
        return
    if alive(previous):
        raise RuntimeError('已有本项目进程尚未就绪，请先运行 stop 脚本')
    try:
        import av
        av_ready = getattr(av, '__version__', '') == '16.1.0' and hasattr(av, 'open')
    except ImportError:
        av_ready = False
    if not av_ready:
        from scripts.setup import main as setup
        setup()
        # Include the newly created overlay in this process as well as the server.
        overlay = config.ROOT / '.deps' / f'{sys.platform}-{sys.version_info.major}{sys.version_info.minor}'
        sys.path.insert(0, str(overlay))
    chosen = None
    for port in range(args.port, min(args.port + 20, 65536)):
        with socket.socket() as probe:
            try:
                probe.bind((args.host, port))
            except OSError:
                continue
            chosen = port
            break
    if chosen is None:
        raise RuntimeError('没有可用端口')
    instance, token = secrets.token_hex(16), secrets.token_hex(32)
    env = os.environ.copy()
    env.update(VIDEO_POLISH_INSTANCE=instance, VIDEO_POLISH_SHUTDOWN_TOKEN=token, PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, str(config.ROOT / 'scripts' / 'runner.py'), str(config.ROOT / 'scripts' / 'run_server.py'), '--host', args.host, '--port', str(chosen)]
    kwargs = {'cwd': str(config.ROOT), 'env': env, 'stdin': subprocess.DEVNULL}
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['start_new_session'] = True
    with (config.LOGS / 'server.log').open('ab') as stdout, (config.LOGS / 'server.err.log').open('ab') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **kwargs)
    state = {'pid': process.pid, 'identity': identity(process.pid), 'root': str(config.ROOT), 'platform': sys.platform,
             'url': f'http://{"127.0.0.1" if args.host == "0.0.0.0" else args.host}:{chosen}', 'port': chosen,
             'instance': instance, 'token': token, 'python': sys.executable, 'started_at': time.time()}
    STATE.write_text(json.dumps(state, indent=2), encoding='utf-8')
    (config.RUNTIME / 'python.json').write_text(json.dumps({'python': sys.executable, 'platform': sys.platform}), encoding='utf-8')
    for _ in range(90):
        if health(state):
            print(f'VideoPolish ready: {state["url"]}\nPython: {sys.executable}\nStop and clear cached videos with stop.bat / stop.sh.')
            if not args.no_browser:
                webbrowser.open(state['url'])
            return
        if process.poll() is not None:
            raise RuntimeError(f'服务器启动失败，退出码 {process.returncode}；查看 logs/server.err.log')
        time.sleep(.5)
    # Do not leave an orphan server after a failed start.
    if alive(state):
        process.terminate()
        process.wait(timeout=10)
    STATE.unlink(missing_ok=True)
    raise RuntimeError('服务器启动超时；查看 logs/server.err.log')


def stop():
    state = read_state()
    if state and health(state):
        request = urllib.request.Request(state['url'] + '/api/admin/shutdown', data=b'', method='POST', headers={'X-VideoPolish-Token': state['token']})
        try:
            HTTP.open(request, timeout=3).close()
        except OSError:
            pass
        deadline = time.monotonic() + 45
        while alive(state) and time.monotonic() < deadline:
            time.sleep(.25)
    if alive(state):
        # Match process creation time before termination, preventing PID-reuse kills.
        os.kill(state['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while alive(state) and time.monotonic() < deadline:
            time.sleep(.2)
        if alive(state):
            raise RuntimeError('本项目进程未能终止，缓存暂不清理以避免写入竞争')
    elif state and state.get('platform') != sys.platform and health(state):
        raise RuntimeError('服务运行在另一系统，请使用该系统的 stop 脚本结束后清理')
    config.clear_cache()
    STATE.unlink(missing_ok=True)
    print('VideoPolish stopped. Uploaded, output and temporary cached videos removed. Models and saved copies kept.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    parser.add_argument('--port', type=int, default=int(os.environ.get('VIDEO_POLISH_PORT', '8800')))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be 1024..65535')
    with lock():
        if args.action == 'start':
            start(args)
        elif args.action == 'stop':
            stop()
        else:
            state = read_state()
            print(json.dumps({'running': bool(state and health(state)), 'url': state['url'] if state else None}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
