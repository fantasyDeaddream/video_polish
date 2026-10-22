from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import config
from .engine import model_info
from .jobs import Manager
from .media import probe
from .target import Options, Target, InterpolationOptions, spatial_plan
from .framerate import interpolation_plan
from .interpolation import interpolation_status
from .tensorrt_backend import acceleration_status

manager = None
shutdown_callback = None
upload_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    global manager
    config.ensure_dirs()
    config.clear_cache()
    manager = Manager()
    yield
    await asyncio.to_thread(manager.shutdown)
    config.clear_cache()


app = FastAPI(title='VideoPolish', version=config.VERSION, lifespan=lifespan,
              description='本地视频超分与 RIFE 插帧 API。先上传探测，再预览尺寸/帧率并提交批量任务。')


@app.middleware('http')
async def local_origin(request: Request, call_next):
    if request.url.path == '/api/uploads':
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            return JSONResponse({'detail': '无效的 Content-Length'}, status_code=400)
        if length > config.MAX_UPLOAD_BYTES + 1024 * 1024:
            return JSONResponse({'detail': '本批次上传超过容量上限'}, status_code=413)
    # Browser writes must originate at the local app; CLI requests have no Origin.
    origin = request.headers.get('origin')
    if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin:
        from urllib.parse import urlsplit
        if urlsplit(origin).netloc != request.headers.get('host'):
            return JSONResponse({'detail': '拒绝跨站请求'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.exception_handler(KeyError)
async def not_found(request, exc):
    return JSONResponse({'detail': str(exc).strip("'")}, status_code=404)


@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse({'detail': str(exc)}, status_code=400)


@app.get('/api/health')
def health():
    import sys
    return {'status': 'ok', 'app': 'VideoPolish', 'version': config.VERSION,
            'instance': os.environ.get('VIDEO_POLISH_INSTANCE', 'direct'),
            'python': sys.executable, 'models': model_info(),
            'cache_bytes': config.cache_bytes(), 'cache_limit_bytes': config.MAX_CACHE_BYTES,
            'max_files': config.MAX_FILES, 'ttl_hours': config.TTL_HOURS, 'acceleration':acceleration_status(),
            'interpolation': interpolation_status()}


@app.get('/api/acceleration')
def acceleration():
    return acceleration_status()


@app.get('/api/interpolation')
def interpolation_capabilities():
    return interpolation_status()


@app.post('/api/uploads', status_code=201)
async def upload(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= config.MAX_FILES:
        raise HTTPException(400, f'每批次支持 1–{config.MAX_FILES} 份视频')
    async with upload_lock:
        baseline_bytes = config.cache_bytes()
        job_id = uuid.uuid4().hex
        folder = config.CACHE / 'uploads' / job_id
        folder.mkdir()
        items, total = [], 0
        try:
            for index, file in enumerate(files):
                name = (file.filename or f'video-{index}').replace('\\', '/').split('/')[-1]
                name = ''.join(c for c in name if ord(c) >= 32)[:240] or f'video-{index}'
                path = folder / f'{index}.video'
                item = {'index': index, 'name': name, 'status': 'uploaded', 'progress': 0}
                try:
                    with path.open('wb') as handle:
                        while chunk := await file.read(1024 * 1024):
                            total += len(chunk)
                            if total > config.MAX_UPLOAD_BYTES:
                                raise HTTPException(413, '本批次上传超过容量上限')
                            if shutil.disk_usage(config.CACHE).free < 512 * 1024**2:
                                raise HTTPException(507, '磁盘空间不足')
                            if total + baseline_bytes > config.MAX_CACHE_BYTES:
                                raise HTTPException(507, '缓存容量不足，请删除旧任务')
                            handle.write(chunk)
                    item.update(path=str(path), metadata=await asyncio.to_thread(probe, path))
                except HTTPException:
                    raise
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    item.update(status='failed', error=f'视频探测失败：{exc}', progress=1)
                finally:
                    await file.close()
                items.append(item)
            if not any(item.get('metadata') for item in items):
                raise ValueError('没有有效视频：' + '; '.join(f'{i["name"]}: {i["error"]}' for i in items))
            return manager.add(job_id, items)
        except Exception:
            config.remove_cache_path(folder)
            raise


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    upload_id: str
    target: Target = Field(default_factory=Target)
    workflow: Literal['upscale', 'interpolate', 'combined'] = 'upscale'
    interpolation: InterpolationOptions = Field(default_factory=InterpolationOptions)


@app.post('/api/plan')
def preview_plan(request: PlanRequest):
    record = manager.get(request.upload_id)
    options = Options(target=request.target, workflow=request.workflow, interpolation=request.interpolation)
    return {'id': record['id'], 'files': [{'index': item['index'], 'name': item['name'],
             'source': item['metadata'], 'target': spatial_plan(item['metadata'], options),
             'interpolation': interpolation_plan(item['metadata'], options.interpolation)}
             for item in record['files'] if item.get('metadata')]}


class JobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    upload_id: str
    options: Options = Field(default_factory=Options)


@app.post('/api/jobs', status_code=202)
def create_job(request: JobRequest):
    return manager.start(request.upload_id, request.options)


@app.get('/api/jobs')
def list_jobs():
    with manager.lock:
        ids = list(manager.records)
        return {'jobs': [manager.get(key) for key in reversed(ids)]}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    return manager.get(job_id)


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    return manager.cancel(job_id)


@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str):
    return manager.cancel(job_id, delete=True)


@app.get('/api/jobs/{job_id}/files/{index}/download')
def download(job_id: str, index: int):
    path, name = manager.result(job_id, index)
    return FileResponse(path, media_type='video/mp4', filename=name, headers={'Cache-Control': 'no-store'})


@app.get('/api/jobs/{job_id}/files/{index}/preview')
def preview(job_id: str, index: int):
    path, _ = manager.result(job_id, index)
    return FileResponse(path, media_type='video/mp4', headers={'Cache-Control': 'no-store'})


@app.post('/api/admin/shutdown', include_in_schema=False)
async def shutdown(request: Request):
    token = os.environ.get('VIDEO_POLISH_SHUTDOWN_TOKEN', '')
    provided = request.headers.get('X-VideoPolish-Token', '')
    if not token or not secrets.compare_digest(token, provided) or request.client.host not in {'127.0.0.1', '::1'}:
        raise HTTPException(403)
    if shutdown_callback:
        asyncio.get_running_loop().call_later(.2, shutdown_callback)
    return {'status': 'stopping'}


app.mount('/', StaticFiles(directory=str(config.ROOT / 'web'), html=True), name='web')
