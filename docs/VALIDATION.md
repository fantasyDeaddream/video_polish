# 本机验证记录

此页保留 1.0 版本的原始验证记录；1.1 的大分块、流水线、轻量模型、TensorRT 及当前 27 项测试结果见 [加速与画质报告](ACCELERATION.md)。

验证日期：2026-10-21（Asia/Shanghai）。全部由当前主代理在本机执行。

## 环境

- Windows、PowerShell 7、WSL Ubuntu-24.04。
- NVIDIA GeForce RTX 4060 Laptop GPU，8GB 显存。
- 复用 PicPolish 的 Python 3.12.13 / PyTorch 2.5.1+cu121 / OpenCV 5.0.0；CUDA 实际 forward 成功。
- PyAV 16.1.0 只安装在 `video_polish/.deps/win32-312`。
- 模型只读复用 PicPolish 文件。x2plus SHA256：`49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`。

PyAV wheel 31,752,045 字节。下载中的 Windows 打开句柄重命名问题已修复，依次尝试官方、阿里、清华的等价文件；最后一次清华源实收约 23.1 MB/s，随后复用完整临时文件并核对 PyPI SHA256 完成安装。文件均在 30 秒测速窗口形成前完成，因此此次没有触发低速窗口换源或最快源回退；换源由已修复的重命名错误造成。依赖权限已恢复为项目目录继承。

## 自动化测试

`tests/test_core.py` + `tests/test_api.py`：**15 passed**。复用环境中的 Starlette TestClient 有一条 httpx 弃用提示，不影响本次功能验证；未为消除提示修改共享环境。

覆盖内容：

- 720P / 1080P / 2K / 4K 横竖屏、方形、自定义奇数、非方形像素和过大输出限制。
- 合成增强残差的时序抖动降低、硬切换不混入前镜头细节。
- 真实官方 checkpoint 严格加载、实际神经推理、输出与插值基底不同。
- 可变帧率输入和输出时间戳误差不超过 1/90000 秒；中文文件名。
- AAC 压缩包内容 SHA256 一致；PCM 双音轨全部转为 AAC，语言标记保留、音频样本数量误差控制在 AAC 尾部填充范围。
- 取消后的半成品删除；批量中一份无效时其余可处理。
- UTF-8 下载文件名、HTTP Range 206 / 416、源上传完成即回收、删除、过期清理、退出清理。
- 无效参数、全部无效媒体、跨站写请求和未授权 shutdown 拒绝。

## 实际视频流程

`tests/e2e.py` 对脚本启动的真实 HTTP 服务测试上传、探测、全部预设尺寸预览、队列、CUDA 超分和结果下载。

| 输入 | 输出 | 帧数 | 音轨 | 本次耗时 |
| --- | --- | --- | --- | --- |
| 1280×720 / 24fps / 0.5s | 1920×1080 / 24fps / 0.5s | 12 → 12 | 1 → 1 | 约 22.2s |
| 90×160 / 24fps / 0.25s | 1080×1920 / 24fps / 0.25s | 6 → 6 | 0 → 0 | 约 4s |

第一份输入是从图片项目的真实照片生成的逐帧水平平移短片，并非自然拍摄的长视频；第二份为合成图形。它们验证管线与模型确实工作，不足以证明各种真实视频的重建质量、长视频性能或无伪影。输出全部时间戳逐帧核对，下载 SHA256 和实际模型/设备证据保存在报告中。

- [实际 HTTP 流程报告](../tests/artifacts/e2e-report.json)
- [720P 输入短片](../tests/artifacts/sample_720p.mp4)
- [1080P 输出短片](../tests/artifacts/result_1080p.mp4)
- [插值与神经增强对比裁切](../tests/artifacts/quality-comparison.png)

对比图来自首帧，首帧尚无前帧时序融合；展示的是实际神经恢复与编码后的区别。没有以锐度指标冒充真实细节恢复，也没有声称已经评估自然视频的画质得分。

## 网页和脚本

浏览器实测：上传、横竖屏预览、2K/4K 预设、自定义长边 241px 自动得到 136×242、提交、结果预览、普通浏览器下载落盘。390px 手机视口无横向溢出。系统另存为和文件夹选择的 JS 流式写入路径使用模拟文件句柄验证；没有声称自动点击过操作系统的原生保存对话框。

- [桌面截图](../tests/artifacts/web-desktop.png)
- [移动端截图](../tests/artifacts/web-mobile.png)
- 普通下载实际保存到 `tests/artifacts/browser-saved.mp4`。
- API 客户端已实际下载到 `tests/artifacts/api-saved/`。

Windows `start.bat` / `stop.bat` 在 PowerShell 7 下实际执行。WSL 中对三个 shell 文件执行 `bash -n`，并实测 Bash 从停止状态启动、重复启动、处理中停止、重复停止。WSL 通过 Windows Python/CUDA 运行，没有重新安装 Linux GPU 栈。

`tests/lifecycle.py` 准备进行中的 GPU 任务、尚未处理的上传和临时视频，然后由实际停止脚本结束服务，验证上传/输出/临时缓存为空、服务退出、另存的结果仍在。报告保存在 [lifecycle-report.json](../tests/artifacts/lifecycle-report.json)。

仍未实测：自然长视频、HDR 或隔行预处理、原生 Linux GPU 环境、所有相机旋转矩阵及全部音频布局。程序当前对 HDR / 隔行明确拒绝。上述限制不影响已验证的常见逐行 SDR 视频流程。
