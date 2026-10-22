# VideoPolish API

默认根地址 `http://127.0.0.1:8800`；真实端口以启动输出为准。`GET /docs` 提供交互式文档，`GET /openapi.json` 是完整 schema。API 采用异步任务：上传 → 探测 → 尺寸预览 → 提交 → 轮询 → 下载。

## 1. 上传

```bash
curl -X POST http://127.0.0.1:8800/api/uploads \
  -F "files=@/path/video-a.mp4" \
  -F "files=@/path/video-b.mov"
```

PowerShell 请用 `curl.exe`，避免别名混淆。每份视频使用一个同名 `files` 字段。返回 HTTP 201：

```json
{
  "id": "批次ID",
  "status": "uploaded",
  "files": [{
    "index": 0,
    "name": "video-a.mp4",
    "status": "uploaded",
    "metadata": {
      "width": 1280, "height": 720,
      "coded_width": 1280, "coded_height": 720,
      "sar": 1.0, "rotation": 0,
      "fps": 24.0, "rate": "24", "duration": 12.5,
      "frames": 300, "audio_tracks": 1, "audio_codecs": ["aac"]
    }
  }]
}
```

`width/height` 已考虑旋转，`coded_width/coded_height` 为原始编码尺寸。`frames` 未知时为 null。批次中单份损坏会记录 `status=failed` 与 `error`，其他视频可继续处理；全部损坏返回 400。内部磁盘路径不出现在任务 JSON。

## 2. 尺寸预览

`POST /api/plan`，JSON：

```json
{"upload_id":"批次ID","target":{"preset":"1080p"}}
```

`preset`：`720p`、`1080p`、`2k`、`4k`、`custom`。自定义示例：

```json
{"upload_id":"批次ID","target":{"preset":"custom","edge":"long","pixels":3000}}
```

`edge` 为 `long` / `short`，`pixels` 是 16–7680 的整数。预览响应按文件返回 `source` 与 `target`，包含实际宽高、倍率、比例误差和 `mode=upscale/same/downscale`。

## 3. 提交异步任务

`POST /api/jobs`，返回 202：

```json
{
  "upload_id":"批次ID",
  "options":{
    "target":{"preset":"1080p"},
    "model":"auto",
    "strength":0.75,
    "temporal":0.4,
    "crf":18,
    "tile":1024,
    "tensorrt":false,
    "tensorrt_precision":"fp16",
    "pipeline":true
  }
}
```

| 字段 | 默认值 | 范围 / 含义 |
| --- | --- | --- |
| model | auto | auto / quality：按倍率选 RRDB x2/x4；anime：Anime 6B；general-fast / general-clean / anime-fast：三种轻量 SRVGG；lanczos：非 AI 缩放 |
| strength | 0.75 | 0.1–1，神经重建残差占比 |
| temporal | 0.4 | 0–0.65，前帧可信细节的最大融合比例；0 关闭 |
| crf | 18 | 14–28 的整数，越低质量越高、文件通常越大 |
| tile | 256 | 64–2048 的整数；保留原默认值；网页包含 128 / 256 / 384 / 512 / 768 / 1024 / 1536 / 2048，显存不足时减半并记住本次视频的可用值 |
| tensorrt | false | 是否使用可选 TensorRT 引擎；需要本地安装且 CUDA 可用 |
| tensorrt_precision | fp16 | fp16 / fp32；两种精度均与 PyTorch 做实际输入的数值检查 |
| pipeline | true | 将 CPU 解码、光流稳定与编码和有序 GPU 超分重叠执行；有界队列，不丢帧 |
| workflow | upscale | upscale：原有仅超分；interpolate：仅插帧并保留显示尺寸；combined：原始帧超分后插帧 |
| interpolation | enabled=false | 独立插帧配置，见下文；旧请求不变 |

目标不大于原片时跳过网络，证据中标记 `backend=lanczos, reason=target_not_larger`。不能对同一个批次重复提交；更换参数请重新上传。单个工作线程按队列处理，避免多个任务争抢显存。

原生超分逻辑保持：默认 720P→1080P 仍是 x2plus 先生成 2560×1440，再用原有 Lanczos4 缩至 1920×1080。轻量模型原生 4 倍，也先生成完整原生输出，再缩至目标尺寸。没有预先降低输入尺寸或跳帧。

`general-fast` 使用通用轻量权重，`general-clean` 使用 WDN 弱降噪权重（保留更多噪点），`anime-fast` 适合动画。各模型画质表现不同；可参考本机 [速度和一致性验证](ACCELERATION.md)。

首次 TensorRT 请求可能需要数分钟导出/编译，`stage` 会显示当前阶段。引擎按模型 SHA256、GPU、TensorRT 版本、精度和输入形状上限缓存于 `.runtime/tensorrt`，停止服务保留这些非视频编译成果。后续同配置复用缓存。首次实际输入的每种分块形状会与 PyTorch 对比；非有限结果或超过验收阈值（MAE > 0.5/255 或 PSNR < 45dB）会明确失败，可切换 FP32 或关闭 TensorRT，不会静默更换后端。

`evidence.engine` 新增 `inference_engine`、`requested_tile`、`native_scale`、`native_size`、`output_size`、`transfers` 和 GPU 显存峰值；TensorRT 还包含 `precision`、`build_seconds`、`loaded_from_cache`、逐形状 `validations`。PSNR 为 null 且 identical=true 表示像素完全相同。显存数值是 PyTorch 分配器统计，不包含所有 TensorRT 私有分配和桌面占用，不能等同于整卡显存。

### 插帧与组合流程

预览请求的 workflow 与 interpolation 和任务配置一致，例如：

```json
{
  "upload_id":"批次ID",
  "workflow":"combined",
  "target":{"preset":"1080p"},
  "interpolation":{"enabled":true,"model":"rife-4.25","target_fps":"60"}
}
```

返回每份文件的 target 和 interpolation，后者包含源/目标分数帧率、预计帧数与处理顺序。预计帧数依赖探测时长，精确输出以实际 PTS 和末帧结束时刻为准。

对应的 POST /api/jobs 示例：

```json
{
  "upload_id":"批次ID",
  "options":{
    "workflow":"combined",
    "target":{"preset":"1080p"},
    "model":"auto",
    "tile":512,
    "tensorrt":true,
    "interpolation":{
      "enabled":true,
      "model":"rife-4.25",
      "target_fps":"60000/1001",
      "backend":"pytorch",
      "precision":"fp16",
      "scale":1.0,
      "scene_cut_detection":true,
      "order":"sr_then_vfi"
    }
  }
}
```

- 插帧 model：rife-4.25 / rife-4.25-lite；backend：pytorch / tensorrt；precision：fp16 / fp32，CPU 实际使用 FP32。
- target_fps：2x / 4x，或 1–240 之间的十进制/分数字符串；必须不低于每份视频探测到的源帧率。
- scale：1 / 0.5 / 0.25，控制插帧内部尺度。减小可能改变画质，不影响最终尺寸，也不改变超分的原生倍率路径。
- workflow=interpolate 时忽略超分模型、超分 TensorRT 与目标放大设置；按显示宽高比保留尺寸，必要时转方形像素并取偶数。
- workflow=interpolate/combined 自动启用插帧；只传 interpolation.enabled=true 则自动选择 combined，方便增量扩展旧请求。
- 正常升帧率保持时长和音轨，尾段保持末帧；时长不能整除目标帧间隔时，编码尾段量化误差小于一个目标帧区间。24→60 按目标均匀时间网格重采样。切镜检测为启发式，不保证识别所有切镜。
- evidence.interpolation 包含实际模型散列、精度、设备、生成/锚点/保持帧数、切镜数、尾段策略和 TensorRT 验证。evidence.frames 是原始帧数，output_frames 才是成片帧数。
- 插帧 TensorRT 缓存与超分缓存分开；不同尺寸/精度首次编译，数值校验失败时不输出成片、不静默替换后端。

## 4. 状态和下载

- `GET /api/health`：服务、依赖路径、权重可用性、缓存容量。
- `GET /api/acceleration`：TensorRT 安装状态、版本、可用精度及已缓存引擎数量。
- `GET /api/interpolation`：RIFE 权重可用性、后端、精度与帧率上限；health 同时包含该信息。
- `GET /api/jobs`：当前服务中的全部批次。
- `GET /api/jobs/{id}`：状态、每份文件进度、处理 fps、ETA、结果和证据。
- `GET /api/jobs/{id}/files/{index}/download`：附件下载，UTF-8 文件名。
- `GET /api/jobs/{id}/files/{index}/preview`：内联 MP4，支持 Range / 206。
- `POST /api/jobs/{id}/cancel`：取消并清除该批次视频，已开始的工作在下一次检查点停止。
- `DELETE /api/jobs/{id}`：删除记录和缓存；运行中先取消，工作线程结束后移除。

批次状态：`uploaded → queued → running → done / partial / failed / cancelled`。`partial` 表示有结果可下载，但也有单份失败。逐文件 `evidence` 包含实际后端、模型 SHA256、设备、帧数、时间戳范围、耗时和场景切换计数。状态只保存在当前服务内存中；停止/重启后无旧任务。清理后下载返回 404 或 400。

常见错误：400 业务参数或媒体错误；404 不存在；413 上传过大；422 schema 校验失败；507 磁盘或缓存容量不足。错误 JSON 的 `detail` 含说明。

## 5. Python 客户端

本项目自带流式下载客户端，依赖现有环境中的 httpx：

```powershell
& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py scripts\api_client.py `
  --url http://127.0.0.1:8800 --preset 1080p --output D:\Videos\out `
  "D:\Videos\a.mp4" "D:\Videos\b.mov"

# 自定义长边
& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py scripts\api_client.py `
  --preset custom --edge long --pixels 3000 --output D:\Videos\out "D:\Videos\a.mp4"

# TensorRT 与大分块
& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py scripts\api_client.py `
  --preset 1080p --tile 1024 --tensorrt --trt-precision fp16 `
  --output D:\Videos\out "D:\Videos\a.mp4"

# 通用轻量模型，保留默认异步管线
& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py scripts\api_client.py `
  --preset 1080p --model general-fast --tile 1024 --output D:\Videos\out "D:\Videos\a.mp4"

# 仅插帧到 60fps，保留显示尺寸
& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py scripts\api_client.py `
  --workflow interpolate --fps 60 --rife-model rife-4.25-lite `
  --output D:\Videos\out "D:\Videos\a.mp4"

# 先超分到 1080P，再插帧；两个 TensorRT 后端分别启用
& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py scripts\api_client.py `
  --workflow combined --preset 1080p --tensorrt --tile 512 `
  --fps 60000/1001 --rife-backend tensorrt --rife-precision fp32 `
  --output D:\Videos\out "D:\Videos\a.mp4"
```

结果按数据流写入目标目录，不将整份视频读入内存；同名文件自动改名。下载完成后可在网页中删除缓存。需要取消任务时，先访问上述 cancel 路由，然后再运行停止脚本。
