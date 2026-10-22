# 方案调研与选择

调研日期：2026-10-21。以官方仓库、论文及文档为依据。

| 方案 | 能力 | 本项目判断 |
| --- | --- | --- |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 官方图像/视频恢复项目，RRDB x2/x4 和动漫权重；支持分块与任意输出尺寸 | 本机已有权重与兼容 CUDA PyTorch，适合作为可直接复用的神经恢复基底 |
| [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) | 传播与对齐式多帧视频超分，官方实现依赖 MMEditing / mmcv-full | 具备真正的多帧特征传播，但部署需要额外兼容性工作，未作为本次默认依赖 |
| [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) | 面向真实退化视频的视频恢复研究与实现 | 可供后续多帧模型扩展；本项目没有声称已经实现其架构或效果 |
| [Video2X](https://github.com/k4yt3x/video2x) | 基于多种恢复/插帧模型的视频处理框架 | 参考流式视频处理和编码分工；本项目未复制其代码，避免额外可执行文件和另一套 GPU 推理运行时 |
| [PyAV](https://pyav.org/docs/stable/) | FFmpeg 的 Python 绑定，帧时间戳、音轨、解复用、编码和封装 | 本项目选用 16.1.0 wheel，以少量新依赖完成跨平台编解码，无须系统安装 ffmpeg.exe |

实现采用 Real-ESRGAN 的神经单帧恢复，加本项目自己的光流残差稳定。它不是 BasicVSR++ 等端到端多帧超分模型，也不宣称具有相同重建能力。

光流使用 [OpenCV Farnebäck 官方说明](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)中的稠密光流；通过 `remap` 将相邻帧的增强残差对齐到当前帧，限制融合到高置信度区域。场景切换检测和遮挡检查是工程启发式，不能保证消除所有伪影。

网络结构从相邻图片项目的 `app/rrdbnet.py` 复用，保持 BasicSR 层名，使用官方 checkpoint 严格加载，不安装旧版 BasicSR / torchvision 组合；模型加载使用 `weights_only=True`（当前 PyTorch 支持），避免对任意 checkpoint 进行普通 pickle 回退。

官方权重下载入口：

- [RealESRGAN x2plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)
- [RealESRGAN x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
- [RealESRGAN Anime 6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

1.0 版本只读使用图片项目已有权重。1.1 保留该路径，另从同一官方 v0.2.5.0 release 下载三份轻量 SRVGG 权重，记录于 `models/lightweight-manifest.json`。每次实际加载的 SHA256 会出现在任务的 `evidence.engine.sha256` 中，方便复现。

1.1 同时增加 NVIDIA TensorRT 10.10.0.31 的项目独立可选组件。以现有 PyTorch 官方 checkpoint 导出 ONNX，使用实际 CUDA 引擎；按模型、精度与形状缓存，运行前以真实输入验证数值一致性。官方说明见 [TensorRT 性能优化](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html) 和 [精度注意事项](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)。实现没有切换到 INT8、降低输入尺寸或变更原生超分后缩放逻辑。本机实测见 [ACCELERATION.md](ACCELERATION.md)。
