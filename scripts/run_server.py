from __future__ import annotations

import argparse
import logging

import uvicorn
import av

if not hasattr(av, 'open'):
    raise RuntimeError('PyAV 安装不完整或目录不可读取，请重新运行 scripts/setup.py')

from app import main

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=8800)
args = parser.parse_args()
server = uvicorn.Server(uvicorn.Config(main.app, host=args.host, port=args.port, log_level='info', timeout_graceful_shutdown=35))
main.shutdown_callback = lambda: setattr(server, 'should_exit', True)
server.run()
