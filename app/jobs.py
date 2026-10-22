from __future__ import annotations

import copy
import json
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

from . import config
from .engine import Cancelled, Upscaler
from .media import process_video
from .target import Options, spatial_plan
from .framerate import interpolation_plan

TERMINAL = {'done', 'partial', 'failed', 'cancelled'}


class Manager:
    def __init__(self):
        self.lock = threading.RLock()
        self.records = {}
        self.pending = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.engine = Upscaler()
        self.thread = threading.Thread(target=self._work, daemon=True, name='video-worker')
        self.thread.start()

    def add(self, job_id, files):
        with self.lock:
            if len(self.records) >= 100:
                raise ValueError('任务已达上限，请删除旧任务后重试')
            self.records[job_id] = {'id': job_id, 'status': 'uploaded', 'created_at': time.time(),
                                    'updated_at': time.time(), 'files': files, 'cancel': False,
                                    'delete': False, 'options': None}
        return self.get(job_id)

    def get(self, job_id):
        with self.lock:
            record = self.records.get(job_id)
            if record is None:
                raise KeyError('任务不存在或缓存已过期')
            result = copy.deepcopy(record)
        result.pop('cancel', None)
        result.pop('delete', None)
        for item in result['files']:
            item.pop('path', None)
            item.pop('result_path', None)
            if item['status'] == 'done':
                item['download_url'] = f'/api/jobs/{job_id}/files/{item["index"]}/download'
                item['preview_url'] = f'/api/jobs/{job_id}/files/{item["index"]}/preview'
        valid = [f for f in result['files'] if f.get('metadata')]
        result['progress'] = round(sum(f.get('progress', 0) for f in valid) / max(1, len(valid)), 5)
        return result

    def start(self, job_id, options: Options):
        with self.lock:
            record = self.records[job_id]
            if record['status'] != 'uploaded':
                raise ValueError('此批次已启动；要使用新参数，请重新上传')
            plans = [(item, spatial_plan(item['metadata'], options), interpolation_plan(item['metadata'], options.interpolation))
                     for item in record['files'] if item.get('metadata')]
            if not plans:
                raise ValueError('没有有效视频')
            if self.pending.full():
                raise ValueError('处理队列已满')
            if options.interpolation.enabled:
                from .interpolation import weight_path
                if not weight_path(options.interpolation.model).is_file():
                    raise ValueError('缺少 RIFE 插帧权重，请运行 scripts/fetch_interpolation.py')
            for item, target, timing in plans:
                item.update(target=target, interpolation_plan=timing, status='queued', progress=0)
            record.update(status='queued', options=options.model_dump(), updated_at=time.time())
            self.pending.put_nowait(job_id)
        return self.get(job_id)

    def cancel(self, job_id, delete=False):
        with self.lock:
            record = self.records[job_id]
            record['cancel'] = True
            record['delete'] = delete
            if record['status'] not in {'running'}:
                record.update(status='cancelled', updated_at=time.time())
                for item in record['files']:
                    item.update(status='cancelled', progress=0)
                self._clean(job_id)
                if delete:
                    del self.records[job_id]
                    return {'id': job_id, 'status': 'deleted'}
        return self.get(job_id)

    def _clean(self, job_id):
        for name in ('uploads', 'results', 'tmp'):
            config.remove_cache_path(config.CACHE / name / job_id)

    def cleanup_expired(self):
        cutoff = time.time() - config.TTL_HOURS * 3600
        with self.lock:
            expired = [key for key, r in self.records.items() if r['status'] in TERMINAL | {'uploaded'} and r['updated_at'] < cutoff]
            for key in expired:
                self._clean(key)
                del self.records[key]

    def result(self, job_id, index):
        with self.lock:
            record = self.records[job_id]
            item = next((f for f in record['files'] if f['index'] == index), None)
            if item is None:
                raise KeyError('文件不存在')
            if item['status'] != 'done' or not Path(item.get('result_path', '')).is_file():
                raise ValueError('结果尚未生成或已清除')
            return Path(item['result_path']), item['output_name']

    def _work(self):
        while not self.stop_event.is_set():
            try:
                job_id = self.pending.get(timeout=1)
            except queue.Empty:
                self.cleanup_expired()
                continue
            try:
                with self.lock:
                    record = self.records.get(job_id)
                    if not record or record['cancel']:
                        continue
                    record['status'] = 'running'
                    options = Options.model_validate(record['options'])
                output_dir = config.CACHE / 'results' / job_id
                output_dir.mkdir(exist_ok=True)
                last_disk_check = 0

                def check():
                    nonlocal last_disk_check
                    if self.stop_event.is_set() or record['cancel']:
                        raise Cancelled('已取消')
                    if time.monotonic() - last_disk_check > 5:
                        last_disk_check = time.monotonic()
                        if shutil.disk_usage(config.CACHE).free < 512 * 1024**2:
                            raise RuntimeError('剩余磁盘空间不足 512MB，处理已停止')
                        if config.cache_bytes() > config.MAX_CACHE_BYTES:
                            raise RuntimeError('缓存达到容量上限，请下载并删除旧任务')

                for item in record['files']:
                    check()
                    if not item.get('metadata'):
                        continue
                    with self.lock:
                        item.update(status='running', stage='正在加载模型 / 处理视频')
                    def stage(message):
                        with self.lock:
                            item['stage'] = message
                    self.engine.stage_cb = stage
                    output = output_dir / f'{item["index"]}.mp4'

                    def progress(count, timestamp, elapsed, backend):
                        with self.lock:
                            duration = item['metadata']['duration']
                            ratio = timestamp / duration if duration else count / (item['metadata'].get('frames') or (count + 1))
                            fps = count / max(elapsed, .001)
                            total = item['metadata'].get('frames') or round(duration * item['metadata']['fps'])
                            timing = backend.get('interpolation')
                            stage_text = ('RIFE 插帧' if options.workflow == 'interpolate' else '超分 → RIFE 插帧') if timing else ('超分与时序稳定' if backend['backend'] == 'realesrgan' else '高质量缩放')
                            item.update(progress=min(.995, max(item.get('progress', 0), ratio)),
                                        processed_frames=count, processing_fps=round(fps, 3),
                                        elapsed_seconds=round(elapsed, 2), eta_seconds=round(max(0, total-count) / max(fps, .001), 1),
                                        engine=backend, stage=stage_text)
                            if timing:
                                item.update(output_frames=timing['output_frames'], generated_frames=timing['generated_frames'], output_rate=timing['rate'])

                    try:
                        evidence = process_video(Path(item['path']), output, item['metadata'], item['target'], options, self.engine, check, progress)
                        name = Path(item['name']).stem[:120]
                        suffix = ('_' + item['interpolation_plan']['rate'].replace('/', '-') + 'fps') if options.interpolation.enabled else ''
                        output_name = f'{name}_{item["target"]["width"]}x{item["target"]["height"]}{suffix}.mp4'
                        with self.lock:
                            item.update(status='done', progress=1, stage='完成', result_path=str(output), output_name=output_name,
                                        result_bytes=output.stat().st_size, evidence=evidence)
                        # Reclaim the uploaded source as soon as each file completes.
                        Path(item['path']).unlink(missing_ok=True)
                    except Cancelled:
                        raise
                    except Exception as exc:
                        output.unlink(missing_ok=True)
                        with self.lock:
                            item.update(status='failed', error=str(exc), stage='处理失败', progress=1)
                        Path(item['path']).unlink(missing_ok=True)
                with self.lock:
                    if record['cancel'] or self.stop_event.is_set():
                        raise Cancelled('已取消')
                    good = sum(f['status'] == 'done' for f in record['files'])
                    bad = sum(f['status'] == 'failed' for f in record['files'])
                    record.update(status=('partial' if bad else 'done') if good else 'failed', updated_at=time.time())
            except Cancelled:
                with self.lock:
                    if record:
                        record.update(status='cancelled', updated_at=time.time())
                        for item in record['files']:
                            item.update(status='cancelled', progress=0)
                        self._clean(job_id)
            except Exception as exc:
                with self.lock:
                    if record:
                        record.update(status='failed', error=str(exc), updated_at=time.time())
                        for item in record['files']:
                            if item['status'] in {'queued', 'running'}:
                                item.update(status='failed', error=str(exc), progress=1)
                        config.remove_cache_path(config.CACHE / 'uploads' / job_id)
            finally:
                with self.lock:
                    record = self.records.get(job_id)
                    if record and record.get('delete'):
                        self._clean(job_id)
                        del self.records[job_id]
                self.engine.close()
                self.pending.task_done()

    def shutdown(self):
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('视频工作线程尚未停止')
        self.engine.close()
