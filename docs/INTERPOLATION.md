# VideoPolish 1.2：插帧使用与验证

更新日期：2026-10-21，Asia/Shanghai。GPU 为 RTX 4060 Laptop 8GB；继续复用图片项目的 Python 3.12.13 / PyTorch 2.5.1+cu121。没有安装或降级共享环境中的包，新增模型在本项目内。

## 使用

运行 start.bat，访问 http://127.0.0.1:8800。WSL Ubuntu 可使用原有 start.sh / stop.sh，默认调用同一套 Windows Python / CUDA。

处理流程有三种：

| 模式 | 行为 |
| --- | --- |
| 仅超分（默认） | 执行原来的视频处理分支，保持原始帧序列与 PTS |
| 仅插帧 | 保持显示尺寸，提高帧率；非方形像素和奇数尺寸按原有编码约束归一化 |
| 超分并插帧 | 只对原始帧超分及稳定，再在最终尺寸插帧，最后统一编码 |

组合模式中，720P→1080P 继续是原生 x2 生成 2560×1440，然后 Lanczos4 缩到 1920×1080。没有给每个插值帧再次运行超分。两个模型由同一个 GPU 工作线程按顺序执行，解码与编码仍通过有界队列并行，不导出整段图片缓存。相邻帧的输入张量与编码特征被复用。

插帧模型可选 RIFE 4.25 标准版或 4.25.lite 快速版。目标帧率支持 2x / 4x、48 / 50 / 60 / 120，以及自定义 60000/1001 等分数，最高 240fps。目标不能低于源视频的探测帧率。普通升帧率保持时长与语速，不是慢动作。

默认插帧后端为 PyTorch FP16，CPU 实际使用 FP32。TensorRT 是独立开关，不会随超分 TensorRT 一起开启；可分别选择 FP16 / FP32。首次按模型、尺寸、尺度、精度和硬件编译，之后复用 .runtime/rife-tensorrt 中的缓存。停止脚本保留模型和编译缓存。

内部尺度默认为 1.0；0.5 / 0.25 用于降低计算与显存需求，可能改变运动细节。它只作用于插帧，不改变超分输入与输出逻辑。显存不足时明确报错，用户自行选择尺度或模型，不自动降低画质设置。

时间调度使用真实输入 PTS 和有理数帧率，覆盖 VFR、分数帧率、音视频起点偏移及末帧持续时间。切镜处保持原镜头到下个镜头的时间边界，避免跨镜头变形；完全重复的帧可复用。严格 60fps 下，24fps 原片的全部时间点无法同时保留。任意时长与目标帧率不能整除时，编码尾段允许不到一个目标帧区间的量化误差。音轨沿用 AAC 压缩包复制、其他格式转 AAC 的策略。

切镜识别是启发式；插帧可能改变遮挡、快速运动和字幕的形状。本轮测试没有建立真人、所有分辨率及所有片源的通用画质保证。

## 模型与环境

官方模型包来自 Practical-RIFE README 所列 Google Drive 链接。通过项目代理策略下载，并固定归档、权重和配套源码的 SHA256。清单位于 models/rife/manifest.json。哈希由本次官方包实取建立，不冒充上游发布的独立签名。

| 模型 | 权重 SHA256 |
| --- | --- |
| RIFE 4.25 | 6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2 |
| RIFE 4.25.lite | 81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1 |

补齐模型：

    ../pic_polish/.venv/Scripts/python.exe scripts/runner.py scripts/fetch_interpolation.py

没有原样安装上游旧版 requirements。推理结构适配到 app/rife_arch.py，移除训练依赖、复用端点编码；权重中的 teacher/caltime 是上游未参与推理的训练头，明确排除后，对所有推理参数严格加载。使用 weights_only=True，不做普通 pickle 回退。应用不导入下载包中的 Python 文件；这些文件仅用于独立对照测试。

MIT 许可保留于 docs/licenses/Practical-RIFE-MIT.txt。来源见 [官方仓库](https://github.com/hzwer/Practical-RIFE)。

## 精度检查

FP32 推理结构在受控 cuDNN 设置下与官方配套 IFNet 源码对照，两种模型的 t=0.2 / 0.5 / 0.8 均通过。初次对照发现坐标除法不同实现会放大浮点差异，现已保持官方标量运算顺序；RIFE 的 FP32 卷积关闭 TF32。FP16 卷积仍使用 FP32 光流坐标与双线性采样，避免高分辨率半精度坐标的量化误差。

TensorRT 的 timestep 是运行时输入，不是固定 0.5 的图。按三个插值时刻区间检查真实输入，并在长片中每生成约 180 帧重新检查后续内容，避免片头黑场成为唯一数值样本。与同精度 PyTorch 比较；MAE 超过 0.5/255、PSNR 低于 45dB 或非有限值会明确失败，不保留该任务成片，不静默更换后端。检查不改变生成像素，覆盖范围及实际误差写入 evidence.interpolation.tensorrt。

随机高频噪点压力测试中，标准版 TensorRT FP16 在 t=0.5 出现 MAE 约 0.663/255，被保护机制拒绝。这是保留的负面结果，说明 FP16 不适合承诺所有输入数值一致；可以选择 PyTorch，或已验证的 TensorRT FP32。

实际运动画面使用 Blender 官方 Sintel 预告片的三组相邻三帧（中心索引 301 / 601 / 901）。端点与被抽掉的中间帧统一用 Lanczos4 放到 1080P；此项没有运行超分。共检查九个插值点，只有中点有保留的真实帧可对照。图像含原片黑边，指标是整帧统计。

| 后端 | 相对各自 PyTorch FP32 的 MAE 范围 /255 | 最低 PSNR |
| --- | ---: | ---: |
| 标准 PyTorch FP16 | 0.00535–0.02796 | 63.60dB |
| 轻量 PyTorch FP16 | 0.00579–0.02945 | 63.44dB |
| 标准 TensorRT FP16 | 0.06037–0.18538 | 51.92dB |
| 轻量 TensorRT FP16 | 0.06255–0.18423 | 51.19dB |
| 标准 TensorRT FP32 | 0.0000053–0.0000238 | 94.37dB |

这些是后端一致性指标，不是生成帧与真实摄影完全相同的证明。标准版 FP32 三个中点对保留真值的 PSNR 分别约 40.04 / 31.46 / 45.10dB；线性混帧对照约 31.78 / 26.06 / 36.50dB。该样本中 RIFE 优于线性混帧，但不能推广成所有片源或两个模型的统一排名。

首次 1080P 编译约为：标准 FP16 228 秒、轻量 FP16 228 秒、标准 FP32 131 秒。具体值和 PyTorch 分配器显存见 tests/artifacts/v12/quality.json；该显存不包含 TensorRT 私有分配及桌面占用，不能当作整卡峰值。少量单帧时延样本只供诊断，完整视频速度以整段测试为准。

## 一分钟实测与组合流程

完整测试输入为 1920×1080、30fps、60 秒、1800 帧。画面来自上述 24fps 的 Sintel 预告片，经采样到 30fps、Lanczos4 放大至 1080P，并循环尾段补足 60 秒；音轨为独立生成的 60 秒 AAC 测试音。因此这是一段真实运动画面的工程负载，不是原生 1080P30 摄影的画质基准。

RIFE 4.25.lite、TensorRT FP16、内部尺度 1.0、流水线开启，使用已有引擎缓存：

| 检查 | 结果 |
| --- | --- |
| 完整处理耗时（解码、插帧、编码与数值检查） | 152.625 秒，约 2 分 33 秒 |
| 输出 | 1920×1080、60fps、3600 帧、60.000 秒 |
| AAC 压缩包 SHA256 | 与输入完全一致 |
| 神经生成 / 原始锚点 / 保持帧 | 1640 / 1800 / 160 |
| 切镜 / 完全重复帧对 | 2 / 157 |
| 片头及中段定期数值检查 | 10 次全部通过，最大 MAE 约 0.346/255，最低 PSNR 约 48.80dB |
| 增加中段检查前后的成片 SHA256 | 完全相同，检查不改变输出像素 |

该用时不含下载、测试输入准备及首次编译，也不是“720P 超分并插帧”用时。较长的编码、不同片源、模型、目标分辨率、切镜和静态画面比例会改变吞吐。记录见 tests/artifacts/v12/minute-rife-4.25-lite-tensorrt-fp16.json。

另外完成了真实 HTTP 720P→1080P 任务：12 张原始帧，超分证据确认 native_scale=2、native_size=[2560,1440]、output_size=[1920,1080]。仅超分仍输出 12 帧/24fps；组合模式输出 30 帧/60fps。两者 AAC 包均与输入一致，组合模式复用 RIFE TensorRT 缓存。该短样片仅超分 8.75 秒，组合模式 14.75 秒，不据此外推所有长片。

对应成片和完整请求/响应证据位于 tests/artifacts/v12/http-e2e.json、http-upscale-1080p.mp4、http-combined-1080p.mp4。

## 自动化与网页验证

44 项 pytest 检查全部通过，包含原有 27 项回归与新增 17 项：官方模型结构、精确时钟、24→60、30→60、分数帧率、VFR、单帧/两帧尾段、多音轨、非零视频起点、切镜、取消、仅对原始帧超分、串行/流水线一致性、API 计划与下载，以及代理策略。

网页通过真实浏览器执行：横竖屏批量读取、分数帧率预览、组合处理、仅插帧保持尺寸、预览、单份另存和三份批量下载。没有捕获到页面脚本错误，390px 移动视口未出现横向溢出。截图保存于 tests/artifacts/v12/web-desktop.png 与 web-mobile.png。本轮另存自动化覆盖浏览器下载回退路径；系统文件选择对话框未重新自动化。

两个本地 HTTP/CONNECT 代理端口 7900 / 7897 均通过 GitHub API 请求，国内来源按规则直连。两份官方模型包实际使用 7900，分别约 1.93 / 1.21 MB/s，均在完整 30 秒测速窗口形成前结束；没有虚构低速换源或最快源结论。下载记录在 models/rife/*.download.json，代理测试见 tests/artifacts/v12/proxy.json。

在 Windows PowerShell 7 和 WSL Ubuntu-24.04 分别执行启动/停止脚本，并在正在处理的 RIFE TensorRT 任务中停止服务。两次均确认服务退出、cache/uploads / results / tmp 中视频全部清空；待处理上传和中间视频一并移除，模型散列、编译引擎和明确另存的样片均保留。证据为 tests/artifacts/v12/lifecycle-windows.json 与 lifecycle-wsl.json。WSL 测试覆盖默认 Windows 互操作模式，未声称验证原生 Linux GPU 环境。

## 复现

PowerShell 7 中，以项目根目录为当前目录：

    ../pic_polish/.venv/Scripts/python.exe scripts/runner.py pytest tests/test_core.py tests/test_api.py tests/test_acceleration.py tests/test_interpolation.py -q --basetemp=.runtime/pytest-v12-local
    ../pic_polish/.venv/Scripts/python.exe -u scripts/runner.py tests/benchmark_interpolation.py quality --backend pytorch --precision fp16
    ../pic_polish/.venv/Scripts/python.exe -u scripts/runner.py tests/benchmark_interpolation.py quality --backend tensorrt --precision fp16
    ../pic_polish/.venv/Scripts/python.exe -u scripts/runner.py tests/benchmark_interpolation.py video --backend tensorrt --models rife-4.25-lite
    ../pic_polish/.venv/Scripts/python.exe -u scripts/runner.py tests/http_interpolation.py proxy
    ../pic_polish/.venv/Scripts/python.exe -u scripts/runner.py tests/http_interpolation.py e2e

质量与长片脚本使用 tests/artifacts/v12/sintel-trailer-720p.mp4；本机已从 Blender 官方下载服务器保存并记录下载散列。输入/输出视频和证据明确另存于 tests/artifacts，不属于服务视频缓存，停止脚本保留。

API 配置与客户端参数见 [API.md](API.md)。
