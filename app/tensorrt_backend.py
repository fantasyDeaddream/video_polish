"""Optional, cached TensorRT 10 backend with real CUDA execution and validation."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import time

import numpy as np

from .config import ROOT


def acceleration_status():
    available = importlib.util.find_spec('tensorrt') is not None and importlib.util.find_spec('onnx') is not None
    try:
        version = importlib.metadata.version('tensorrt-cu12') if available else None
    except importlib.metadata.PackageNotFoundError:
        version = None
    folder = ROOT / '.runtime' / 'tensorrt'
    return {'available':available, 'version':version, 'default_enabled':False,
            'precisions':['fp16','fp32'], 'cached_engines':len(list(folder.glob('*.engine'))) if folder.exists() else 0,
            'note':'首次启用会编译并缓存；实际数值差异见任务 evidence 与验证报告。' if available else '运行 scripts/setup_acceleration.py 安装项目内可选组件。'}


class TensorRTNetwork:
    def __init__(self, reference, digest, shape, precision, stage, check):
        import torch
        try:
            import tensorrt as trt
            import onnx
        except ImportError as exc:
            raise RuntimeError('TensorRT/ONNX 尚未安装。请运行 scripts/setup_acceleration.py，或关闭 TensorRT。') from exc
        self.identity = digest, precision, shape
        self.reference = reference
        self.precision = precision
        self.validations = {}
        self._trt = trt
        self._torch = torch
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._output = None
        self._output_shape = None
        self.version = trt.__version__
        self.build_seconds = 0
        gpu = torch.cuda.get_device_properties(0)
        spec = {'model_sha256':digest,'precision':precision,'max_shape':list(shape),'trt':self.version,
                'gpu':gpu.name,'compute':[gpu.major,gpu.minor],'torch_cuda':torch.version.cuda,'export':2}
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
        folder = ROOT / '.runtime' / 'tensorrt'
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f'{key}.engine'
        self.from_cache = self.path.exists()
        self._runtime = trt.Runtime(self._logger)
        if self.from_cache:
            stage('载入 TensorRT 缓存')
            self._engine = self._runtime.deserialize_cuda_engine(self.path.read_bytes())
        else:
            self._engine = None
        if self._engine is None:
            check()
            started = time.monotonic()
            onnx_file = folder / f'{digest[:16]}-{precision}-op17.onnx'
            if not onnx_file.exists():
                stage('首次导出 ONNX 模型')
                export_model = copy.deepcopy(reference).float() if precision == 'fp32' else copy.deepcopy(reference)
                dtype = torch.float32 if precision == 'fp32' else torch.float16
                sample = torch.zeros((1,3,64,64), dtype=dtype, device='cuda')
                temporary = onnx_file.with_suffix('.partial.onnx')
                with torch.inference_mode():
                    torch.onnx.export(export_model, sample, str(temporary), input_names=['input'], output_names=['output'],
                                      dynamic_axes={'input':{2:'height',3:'width'},'output':{2:'out_height',3:'out_width'}},
                                      opset_version=17, do_constant_folding=True)
                onnx.checker.check_model(str(temporary))
                temporary.replace(onnx_file)
                del export_model, sample
                torch.cuda.empty_cache()
            stage(f'首次编译 TensorRT {precision.upper()}（完成后复用缓存）')
            builder = trt.Builder(self._logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self._logger)
            if not parser.parse(onnx_file.read_bytes()):
                errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f'TensorRT ONNX 解析失败：{errors}')
            settings = builder.create_builder_config()
            settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024**2)
            settings.builder_optimization_level = 3
            settings.max_aux_streams = 0
            settings.clear_flag(trt.BuilderFlag.TF32)
            if precision == 'fp16':
                settings.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile()
            maximum = (1,3,*shape)
            profile.set_shape('input', (1,3,2,2), maximum, maximum)
            settings.add_optimization_profile(profile)
            check()
            serialized = builder.build_serialized_network(network, settings)
            if serialized is None:
                raise RuntimeError('TensorRT 编译失败；请尝试较小分块，或关闭 TensorRT。详情见 logs/server.err.log')
            check()
            temporary = self.path.with_suffix('.partial.engine')
            temporary.write_bytes(bytes(serialized))
            temporary.replace(self.path)
            self.path.with_suffix('.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
            self._engine = self._runtime.deserialize_cuda_engine(bytes(serialized))
            self.build_seconds = round(time.monotonic()-started, 3)
        if self._engine is None:
            raise RuntimeError('TensorRT 引擎加载失败')
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError('TensorRT 无法创建执行上下文，请尝试较小分块')
        self._check = check
        self._stage = stage
        self.stream = torch.cuda.Stream()

    def _run(self, x):
        torch, trt = self._torch, self._trt
        dtype = torch.float32 if self.precision == 'fp32' else torch.float16
        x = x.to(dtype=dtype).contiguous()
        if not self._context.set_input_shape('input', tuple(x.shape)):
            raise RuntimeError('TensorRT 输入尺寸不在当前引擎范围内')
        shape = tuple(self._context.get_tensor_shape('output'))
        if self._output is None or self._output_shape != shape:
            self._output = torch.empty(shape, dtype=dtype, device=x.device)
            self._output_shape = shape
        self._context.set_tensor_address('input', x.data_ptr())
        self._context.set_tensor_address('output', self._output.data_ptr())
        if not self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT 执行失败')
        return self._output

    def __call__(self, x):
        torch = self._torch
        output = self._run(x)
        shape = tuple(x.shape)
        if shape not in self.validations:
            self._check()
            self._stage('校验 TensorRT 与 PyTorch 的数值差异')
            # A new engine/shape is checked against the actual input tile, not
            # synthetic zeros. Validation never changes the super-resolution path.
            with torch.inference_mode():
                reference = self.reference(x)
                if not torch.isfinite(output).all().item():
                    raise RuntimeError('TensorRT 产生非有限数值，已停止处理；请关闭加速或尝试 FP32')
                a = reference[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
                b = output[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
                from .quality import compare
                metrics = compare(a[:, :, ::-1], b[:, :, ::-1])
                metrics['input_shape'] = list(shape)
                metrics['passed'] = metrics['mae'] <= .5 and (metrics['psnr_db'] is None or metrics['psnr_db'] >= 45)
                self.validations[shape] = metrics
                if not metrics['passed']:
                    raise RuntimeError(f'TensorRT 精度校验未通过（MAE {metrics["mae"]:.3f}，PSNR {metrics["psnr_db"]}）；请尝试 FP32 或关闭加速')
            self._stage('超分与时序稳定')
        return output

    def info(self):
        return {'version':self.version,'precision':self.precision,'engine_file':self.path.name,
                'loaded_from_cache':self.from_cache,'build_seconds':self.build_seconds,
                'validations':list(self.validations.values()),'quality_checked':bool(self.validations)}
