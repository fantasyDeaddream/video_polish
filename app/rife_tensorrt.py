"""RIFE-specific static spatial engine, dynamic scalar interpolation time."""
from __future__ import annotations

import hashlib
import json
import time

from .config import ROOT


class RifeTensorRT:
    NAMES = ('image0', 'image1', 'timestep', 'features0', 'features1')

    def __init__(self, reference, digest, inputs, scale, precision, stage, check):
        import torch
        try:
            import onnx
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError('插帧 TensorRT 组件未安装，请运行 scripts/setup_acceleration.py 或选 PyTorch') from exc
        self.reference, self.stage, self.check = reference, stage, check
        self.validations, self.build_seconds = {}, 0
        self.version, self.precision = trt.__version__, precision
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        gpu = torch.cuda.get_device_properties(0)
        arch_hash = hashlib.sha256((ROOT / 'app' / 'rife_arch.py').read_bytes()).hexdigest()
        backend_hash = hashlib.sha256((ROOT / 'app' / 'rife_tensorrt.py').read_bytes()).hexdigest()
        spec = {'model_sha256': digest, 'arch_sha256': arch_hash, 'shape': list(inputs[0].shape),
                'backend_sha256': backend_hash,
                'scale': scale, 'precision': precision, 'trt': trt.__version__, 'gpu': gpu.name,
                'compute': [gpu.major, gpu.minor], 'torch_cuda': torch.version.cuda, 'export': 1}
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
        folder = ROOT / '.runtime' / 'rife-tensorrt'
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f'{key}.engine'
        self.from_cache = self.path.is_file()
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes()) if self.from_cache else None
        if self.engine is None:
            check()
            started = time.monotonic()
            onnx_path = folder / f'{key}.onnx'
            if not onnx_path.exists():
                stage('首次导出 RIFE ONNX（含任意插帧时刻）')
                partial = onnx_path.with_suffix('.partial.onnx')
                with torch.inference_mode():
                    torch.onnx.export(reference, inputs, str(partial), input_names=list(self.NAMES),
                                      output_names=['output'], opset_version=17, do_constant_folding=True)
                onnx.checker.check_model(str(partial))
                partial.replace(onnx_path)
            stage('首次编译 RIFE TensorRT（编译后复用缓存）')
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(onnx_path.read_bytes()):
                raise RuntimeError('RIFE ONNX 解析失败：' + '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            settings = builder.create_builder_config()
            settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024**2)
            settings.builder_optimization_level = 3
            settings.max_aux_streams = 0
            settings.clear_flag(trt.BuilderFlag.TF32)
            if precision == 'fp16':
                settings.set_flag(trt.BuilderFlag.FP16)
                settings.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                # Half coordinates visibly lose subpixel accuracy at HD resolutions.
                for i in range(network.num_layers):
                    layer = network.get_layer(i)
                    if layer.type == trt.LayerType.GRID_SAMPLE:
                        layer.precision = trt.float32
                        layer.set_output_type(0, trt.float32)
            check()
            serialized = builder.build_serialized_network(network, settings)
            check()
            if serialized is None:
                raise RuntimeError('RIFE TensorRT 编译失败，请尝试 FP32 或 PyTorch；详情见服务日志')
            partial = self.path.with_suffix('.partial.engine')
            partial.write_bytes(bytes(serialized))
            partial.replace(self.path)
            self.path.with_suffix('.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
            self.engine = self.runtime.deserialize_cuda_engine(bytes(serialized))
            self.build_seconds = round(time.monotonic() - started, 3)
        if self.engine is None:
            raise RuntimeError('RIFE TensorRT 引擎载入失败')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('RIFE TensorRT 显存不足，无法创建上下文')
        self.output = torch.empty(tuple(self.context.get_tensor_shape('output')), device='cuda', dtype=inputs[0].dtype)
        self.stream = torch.cuda.Stream()

    def __call__(self, *inputs):
        import torch
        self.check()
        tensors = [x.contiguous() for x in inputs]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for name, tensor in zip(self.NAMES, tensors):
                self.context.set_tensor_address(name, tensor.data_ptr())
            self.context.set_tensor_address('output', self.output.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('RIFE TensorRT 执行失败')
        torch.cuda.current_stream().wait_stream(self.stream)
        # Check three different actual times, not just a fixed midpoint engine.
        bucket = min(2, int(float(inputs[2].item()) * 3))
        if bucket not in self.validations:
            from .quality import compare
            self.stage('校验 RIFE TensorRT 与 PyTorch')
            with torch.inference_mode():
                reference = self.reference(*inputs)
                if not torch.isfinite(self.output).all().item():
                    raise RuntimeError('RIFE TensorRT 产生非有限数值')
                def pixels(x):
                    return x[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                metrics = compare(pixels(reference), pixels(self.output))
                metrics.update(timestep=float(inputs[2].item()), passed=False)
                metrics['passed'] = metrics['mae'] <= .5 and (metrics['psnr_db'] is None or metrics['psnr_db'] >= 45)
                self.validations[bucket] = metrics
                if not metrics['passed']:
                    raise RuntimeError(f'RIFE TensorRT 精度校验未通过（MAE {metrics["mae"]:.3f}），请尝试 FP32 / PyTorch')
            self.stage('RIFE 插帧')
        return self.output

    def info(self):
        return {'version': self.version, 'precision': self.precision, 'engine_file': self.path.name,
                'loaded_from_cache': self.from_cache, 'build_seconds': self.build_seconds,
                'validations': list(self.validations.values()), 'quality_checked': bool(self.validations),
                'timestep_input': 'runtime scalar; not frozen at 0.5', 'grid_sample_precision': 'fp32'}
