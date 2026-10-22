# VideoPolish v1.2.0

首次公开 Release，2026-10-22。

本地视频超分与插帧 Web 工作台，支持批量输入、统一尺寸与帧率、任务队列、预览、单份和批量另存，以及同一处理管线的 HTTP API。

- 三种工作流：仅超分、仅插帧、超分后插帧。
- Real-ESRGAN 原生 x2/x4、动漫 RRDB，以及三种轻量 SRVGG 模型。
- RIFE 4.25 标准与 4.25.lite，倍数、自定义与分数帧率。
- 独立可选的超分 / 插帧 TensorRT FP16、FP32 后端，编译缓存和数值检查。
- 保留原生 x2 后缩至 1080P 的超分逻辑；组合流程只超分原始帧。
- PTS 调度、音轨保留、切镜检测、有界流水线、缓存过期清理。
- Windows PowerShell 7 和 WSL Ubuntu 互操作启动/停止脚本；停止时清理视频缓存。
- 下载器优先尝试 7900 / 7897 本地代理，已知国内镜像直连。

## 发布附件

**VideoPolish-v1.2.0-with-models.zip**：源码、文档、六份超分模型与两份插帧模型，附逐文件散列清单。运行需要合适的 Python 环境。TensorRT SDK、Python 环境、视频缓存、日志及本机编译引擎由运行环境独立准备。

**VideoPolish-v1.2.0-with-models.zip.sha256**：完整压缩包 SHA256。

模型与参考实现的许可见 THIRD_PARTY_NOTICES.md 和 docs/licenses；各模型来源与固定散列见 models/catalog.json。应用代码采用 MIT。

## 验证与适用范围

核心功能已在 RTX 4060 Laptop 8GB、Python 3.12.13、PyTorch 2.5.1+cu121 上验证，TensorRT 为 10.10.0.31。覆盖 44 项自动化测试、真实浏览器、音轨/时间戳和 Windows/WSL 启停清理。

一分钟 1080P30 动画工程样片使用 RIFE lite + TensorRT FP16，仅插帧至 60fps 用时约 2 分 33 秒；这是特定输入、缓存引擎下的整段测试，不是超分加插帧或任意视频的速度承诺。详细画质与速度证据见 docs/INTERPOLATION.md。

当前支持 SDR 逐行视频。HDR/PQ/HLG、隔行和中途改变尺寸的输入会报错。AI 超分与插帧可能改变纹理、遮挡和字幕；TensorRT 精度检查不通过时可选择 FP32 或 PyTorch。原生 Linux GPU 尚未完成实测，WSL 验证使用 Windows Python/CUDA 互操作模式。
