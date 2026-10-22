# 视频插帧调研与接入建议

调研日期：2026-10-21（Asia/Shanghai）。本轮仅阅读工程和公开资料、保存本文件；没有修改应用代码、安装依赖、下载权重或操作现有服务。

## 结论

建议首先验证 **Practical-RIFE 4.25（标准）和 4.25.lite（快速）**，为网页增加“仅超分 / 仅插帧 / 超分并插帧”三种处理模式。插帧先建立 PyTorch 正确性基线，再评估独立的 TensorRT 引擎。官方当前仍推荐 4.25 用于多数场景；仓库同时列出 4.26，不能仅按版本号判断画质优劣。标准与 lite 均有官方权重入口，README 明确说明这些模型与项目同为 MIT 许可。[Practical-RIFE 官方模型说明](https://github.com/hzwer/Practical-RIFE#trained-model)

结合本项目目前超分开销，优先测试“原始帧超分 → 现有时序稳定 → 在最终尺寸插帧”。这样增加帧率时不会同时增加超分推理次数。这是基于工程成本的建议，尚未验证它与另一种顺序的画质差异。

本轮没有运行插帧模型，因此没有本机插帧速度、峰值显存或画质实测结果，也不承诺 1080P/4K 实时处理。

## 当前工程的接入条件

本机只读查询确认 GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU，显存 8188 MiB。现有部署文档记录的环境为 Python 3.12、PyTorch 2.5.1+cu121、PyAV 16.1.0，项目独立 TensorRT 组件为 10.10.0.31；沿用图片项目的 Python 环境。

从当前代码确认：

- 解码、GPU 超分、CPU 时序融合与编码通过有界队列衔接；编解码上下文分别由固定线程持有。
- 当前“帧间稳定”对增强残差做光流对齐与融合，没有生成新帧，也没有提高帧率。
- 视频使用原始 PTS 和公共音视频起点，现有输出时基为 1/90000；AAC 音轨复制，其他音轨转 AAC。
- 当前处理 SDR 逐行视频。插帧首版宜沿用这个输入范围。
- 720P→1080P 的默认超分路径继续是 **1280×720 → 原生 x2 的 2560×1440 → Lanczos4 缩至 1920×1080**。

本次核对文件为 app/media.py、app/pipeline.py，以及 docs/ACCELERATION.md。文档中已有的超分速度不能当作插帧速度。

## 候选方案比较

表中部署判断是结合本工程的分析，速度和效果没有经过本机横向测试。

| 方案 | 已核实的特点 | 本项目判断 |
| --- | --- | --- |
| [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) | 标准与 lite 模型；官方推理程序对较新模型直接传入中间时刻；MIT | 首选。可把模型推理接到现有 PyAV 管线，不需要采用其整套演示程序 |
| [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | Windows/Linux/macOS 便携程序，支持 NVIDIA/AMD/Intel，无须 CUDA/PyTorch；仓库为 MIT | 可作为替代后端。现成 CLI 接收图片或图片目录，原样套用会引入大量磁盘帧缓存；应另做内存接口适配 |
| [AMT](https://github.com/MCG-NKU/AMT) | AMT-S/L/G；Vimeo90K 权重标为固定时刻，另有支持任意时刻的 GoPro AMT-S；代码为 CC BY-NC 4.0 | 可列为后续画质对照，不能把所有 AMT 权重都当成任意帧率模型。首版集成优先级低于 RIFE |
| [FILM](https://github.com/google-research/frame-interpolation) | Google 官方 TensorFlow 2 实现，面向大运动插帧；代码为 Apache 2.0 | 可做大运动片段的对照模型，需要另一套推理依赖，与当前 PyTorch 管线的直接复用程度较低 |
| [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI) | 任意时刻插帧；RAFT / FlowFormer 变体及感知增强权重；官方环境涉及 PyTorch 1.13、CUDA 11.6、CuPy；S-Lab 非商业许可 | 作为研究候选。其官方 2K/4K 示例的显存开销值得注意，不适合作为本机 8GB 的首个默认方案 |
| [FFmpeg minterpolate](https://ffmpeg.org/ffmpeg-filters.html#minterpolate) | 传统运动补偿插帧，支持目标分数帧率，也有复制帧和混帧模式 | 可作为非神经网络对照。提高帧率的复制帧/混帧模式不应包装成 AI 插帧；当前 PyAV 构建是否包含该滤镜另需核对 |

许可依据分别见 [Practical-RIFE](https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE)、[ncnn 包装项目](https://github.com/nihui/rife-ncnn-vulkan/blob/master/LICENSE)、[AMT](https://github.com/MCG-NKU/AMT#license)、[FILM](https://github.com/google-research/frame-interpolation/blob/main/LICENSE)、[GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE)。代码和权重的具体版本仍需在实际接入时分别记录来源与散列值。

GIMM 官方公布的 V100、8 倍插帧示例中，2K 在 DS_SCALE=0.5 时约用 7932 MiB，4K 在 0.25 时约用 10922 MiB。这些是特定硬件和参数的上游数据，不是 RTX 4060 的测量值；前者已接近本机整张显卡容量，还没有计算本工程同时驻留的超分资源。[GIMM 官方高分辨率说明](https://github.com/GSeanCDAT/GIMM-VFI#demo)

ncnn 项目的 latest 发布链接本次指向 20221029。不能把这个便携包直接当成支持 Practical-RIFE 4.25/4.26 的同版本实现；必须核对包内模型和后端实际支持范围。[本次核对的官方 release](https://github.com/nihui/rife-ncnn-vulkan/releases/tag/20221029)

## RIFE 的环境与 TensorRT 路线

Practical-RIFE 的 README 写明 Python ≤3.11，依赖文件还限制 NumPy ≤1.23.5。因此，**不能直接把上游 requirements 安装进当前 Python 3.12 环境**。更合适的验证路线是固定 4.25 对应模型代码与权重，仅适配张量推理接口，复用当前编解码；若现代环境适配不成立，再评估项目独立环境，避免改动共享环境。[官方安装说明](https://github.com/hzwer/Practical-RIFE#installation)、[依赖文件](https://github.com/hzwer/Practical-RIFE/blob/main/requirements.txt)

上游要求同时取得模型对应的 Python 文件与 flownet.pkl；模型代码、权重、版本需要配套。其视频演示用 OpenCV/sk-video 和外部 ffmpeg 处理媒体，手动指定 fps 时会关闭音频合并。本项目应复用模型，而继续自行管理 PTS、音轨、取消和缓存。[官方推理程序](https://github.com/hzwer/Practical-RIFE/blob/main/inference_video.py)

RIFE 使用 TensorRT 有现成实现可参考：[vs-mlrt](https://github.com/AmusementClub/vs-mlrt) 提供 TensorRT 后端并包含 [RIFE 模型接口](https://github.com/AmusementClub/vs-mlrt/wiki/RIFE)；[HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) 也提供 TensorRT 模式。但后者当前要求 PyTorch ≥2.10、TensorRT ≥10.14.1 及 Torch-TensorRT ≥2.10，不能直接套用到本工程 PyTorch 2.5.1 / TensorRT 10.10 的组合。

建议先在既有版本上尝试专用 ONNX→TensorRT 导出，不把升级整套共享环境当成前提。若算子不兼容，再决定采用独立后端还是独立环境。vs-mlrt 自身为 GPL-3.0，适合用作部署参考；实际引入代码应单独处理其许可证。[vs-mlrt 仓库](https://github.com/AmusementClub/vs-mlrt)

需要单独验证的项目：

1. 光流变形的坐标归一化、边界填充和 align_corners 必须与参考实现一致。官方 warplayer 使用双线性 grid_sample、border 填充和 align_corners=True；更换实现不能只比较输出尺寸。[官方变形算子](https://github.com/hzwer/Practical-RIFE/blob/main/model/warplayer.py)
2. 任意时刻参数必须真正进入模型。只构建固定 t=0.5 的引擎不足以完成一般的 24→60 转换。
3. 新引擎应分别比较 PyTorch FP32、PyTorch FP16 与 TensorRT 输出，覆盖多个时刻、运动幅度及尺寸；现有超分模型的数值验证结论不能移用于插帧。
4. 缓存需要区分模型代码/权重散列、输入形状、精度和编译环境。已有 Real-ESRGAN 引擎不能执行 RIFE。

FP16 / TensorRT 的差异应分别测量，观察误差图和连续播放结果；不承诺数学零差异。关闭插帧时，应以回归测试确保现有超分分支仍产生原来的结果。

## 超分和插帧的处理顺序

以一分钟、30fps、720P→1080P→60fps 为例：

| 顺序 | 大致推理工作量 | 主要取舍 |
| --- | --- | --- |
| **先超分，再插帧（优先测试）** | 约 1800 帧超分；在 1080P 相邻帧之间插帧 | 保留原始帧的完整超分路径，避免给生成帧再做昂贵超分；插帧本身在更大尺寸运行 |
| 先插帧，再超分 | 在 720P 插帧；约 3600 帧进入超分 | 插帧输入较小，但超分推理次数约翻倍；生成细节与时序一致性的变化需要对照验证 |

两种顺序的画质并不等价。先超分可能使不稳定纹理参与运动估计；先插帧可能把插帧伪影送入超分增强，不能只凭架构断言哪种一定更好。

建议的首版数据流是：

    解码原始帧与时间戳
      → 原始帧超分（保留既有原生 x2 / x4 后缩放逻辑）
      → 对原始帧序列做现有时序稳定
      → 缓存相邻的两帧，按目标时间戳生成中间帧
      → 编码与音频封装

当前时序稳定和编码同在写入线程；加入 GPU 插帧后需要重新划分阶段。宜由统一 GPU 调度控制超分与插帧，避免两个线程无节制地同时占用显存。队列应有界，仅缓存必要的相邻帧和少量输出帧，不导出整段 PNG 序列。

8GB 显存预算必须包含两个模型、TensorRT 工作空间及帧缓冲。现有超分的 768～2048 分块选项不能直接移用于插帧：运动估计依赖跨区域位移，简单切块可能产生接缝或运动错误。4K 应独立验证整帧、模型支持的内部尺度和峰值显存。

官方 RIFE 提供高分辨率时使用 scale=0.5 的选项，但这是需要单独评价的插帧设置，不能默认声称与 scale=1 画质相同，也不能借此改变本工程的超分尺寸路径。[Practical-RIFE 使用说明](https://github.com/hzwer/Practical-RIFE#run)

## 帧率、时长、切镜和音轨

首版建议定位为“增加流畅度，保持播放时长”，慢动作另作功能设计。30→60 的一分钟视频仍应播放一分钟，音调和语速保持原样。

目标帧应按有理数时钟生成，结合相邻输入帧的真实 PTS 计算插值时刻，而不是只读平均 fps 后逐对补固定张数。输出时基应能准确表达目标帧率，或进行受控的有理数重采样，避免浮点累加漂移。

| 场景 | 建议处理规则 |
| --- | --- |
| 30→60，时间网格对齐 | 保留原始帧作为锚点，在中间插入新帧 |
| 24→60 | 使用任意时刻插帧。严格 60fps 的均匀时间网格无法同时包含全部原始 24fps 时间点，不能承诺每张原帧都原位保留 |
| 30000/1001→60000/1001 | 用分数帧率，不将 29.97 / 59.94 当作精确值 |
| 可变帧率输入 | 按真实 PTS 夹取相邻帧，再采样到目标时间网格 |
| 最后一帧 | 结合末帧持续时间处理尾段；无未来参考时保持末帧，不虚构运动外推 |
| 硬切镜 | 禁止生成跨场景混合帧；按切镜边界保持上一镜头或切到下一镜头，继续输出合法时间戳 |
| 完全相同的相邻帧 | 可复用图像省去推理，但保留输出时刻与持续时间 |
| 多音轨 / 音视频起点不同 | 保留公共时钟与轨道偏移；沿用 AAC 复制和其他格式转 AAC 的策略 |

只对 N 帧之间插一个中间帧会得到 2N−1 张图，直接按两倍 fps 编码可能缩短尾段。比如 60 秒的标准 30fps 输入有 1800 帧，60fps 输出应按完整 60 秒时间区间规划约 3600 帧，包含最后原帧之后的合法保持区间。

RIFE 的较新模型推理接口支持直接传入插值时刻，适合实现这种调度；但是本工程的非整数转换、VFR 和音频处理仍需自己实现和测试。[官方任意时刻调用](https://github.com/hzwer/Practical-RIFE/blob/main/inference_video.py)

画质上需重点观察快速运动、遮挡、新显露区域、人物手指、细线、字幕、闪光、淡入淡出和水面。插帧是在估计未被拍摄的时刻，可以改善流畅度，也可能出现扭曲或重影，不能保证恢复真实高帧率摄影的全部信息。

## 网页与 API 的建议范围

以下是待实现设计，不是当前接口已有能力：

- 模式：仅超分、仅插帧、超分并插帧；默认关闭插帧。
- 目标帧率：2倍、4倍、48/50/60/120、自定义；自定义允许 60000/1001 等分数输入。
- 模型：RIFE 4.25 标准、4.25.lite 快速；4.26 作为后续对照，不因版本较新自动替代默认值。
- 独立的插帧后端选择及 TensorRT 状态，明确区分它与超分 TensorRT。
- 上传反馈同时显示尺寸、帧率、时长；计划页显示目标尺寸、目标帧率与预计输出帧数。低于源帧率属于降帧转换，需单独明确语义。
- 进度区分已完成原始帧、已生成插值帧和编码进度；处理速度 fps 与成片播放帧率分别标注。
- 继续使用现有批量任务、下载/另存为和取消操作；终止时清理视频缓存，保留模型与编译缓存。

API 可在现有任务配置中增加独立对象，例如：

    "interpolation": {
      "enabled": true,
      "model": "rife-4.25",
      "target_fps": "60",
      "backend": "pytorch",
      "scene_cut_detection": true,
      "order": "sr_then_vfi"
    }

该片段只表达设计意图，字段命名、错误返回和兼容规则在实现阶段确定。

## 性能判断与验收方案

RIFE 官方早期实现曾报告 RTX 2080Ti 上 720P、2倍插帧达到 30+fps。这只能说明该算法有高吞吐实现，不能直接预测本机 RTX 4060 Laptop 上 4.25/4.25.lite、1080P、双模型流水线的速度。[官方 RIFE 基线说明](https://github.com/hzwer/ECCV2022-RIFE#introduction)

粗略串行预算可写为：总耗时 ≈ 原始帧数 × 超分单帧耗时 + 新生成帧数 × 插帧单帧耗时 + 稳定/编解码耗时。实际流水线重叠、同一帧对的特征复用和多时刻批处理会改变结果，必须测完整流程。4K 的像素数是 1080P 的四倍，也不能因此断言耗时或显存恰好四倍。

建议实际实现时按以下顺序验收：

1. **兼容与单独插帧**：固定官方模型代码和权重散列，验证当前 Python/PyTorch；先 FP32，再 FP16，再 TensorRT。记录冷启动、编译时间、预热后生成帧吞吐和整段处理吞吐。
2. **时间与声音**：覆盖 30→60、24→60、分数帧率、VFR、单帧/两帧、非零起点、末帧持续时间、多音轨。检查 PTS 单调、计划帧数、完整时长；允许的时基舍入单独记录，视频尾段误差不超过一个目标帧区间，音轨无新增漂移。
3. **锚点与画质**：网格对齐的原帧在编码前保持一致；不要要求不同帧率下有损 H.264 的解码锚点逐像素相同。用真实高帧率素材抽帧再重建，比较生成帧 PSNR/SSIM，并连续播放检查重影和闪烁。
4. **顺序对照**：同片源、同模型、同编码设置，比较先超分与先插帧；分别统计标准/lite 在真人、运动、动漫和字幕片段的表现。TensorRT 与 PyTorch 的误差另行归因。
5. **显存与长片**：统计整张显卡的占用，包含 TensorRT 在 PyTorch 分配器之外的内存。先短片排错，再至少测一段真实一分钟 30fps 视频；不能仅凭静态图平移或几个帧对外推结论。
6. **工程回归**：关闭插帧时，既有超分尺寸逻辑、输出像素与时间戳按同配置回归；测试取消、停止脚本、异常退出和缓存清理。有界队列不得因等待下一参考帧而死锁。

建议首轮验证范围为 RIFE 4.25 / 4.25.lite 的 1080P 2倍插帧和 24→60，随后才确定默认模型、TensorRT 参数及 4K 可用档位。本文件中的候选和设计均未在本轮转化为代码变更。
