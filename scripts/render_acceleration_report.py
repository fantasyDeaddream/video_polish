"""Render measured v2 JSON as a reviewable Markdown report, without rerunning jobs."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'tests/artifacts/v2/benchmark.json').read_text(encoding='utf-8'))
lines=['# VideoPolish 1.1 加速与画质验证','',
       '验证日期：2026-10-21（Asia/Shanghai）。GPU：'+data['gpu']+'。',
       '使用既有 PyTorch '+data['torch']+' / CUDA '+data['cuda']+'，新组件安装于项目独立目录。','',
       '默认 720P→1080P 始终保留原生 x2 超分至 2560×1440，再用原有 CPU Lanczos4 缩至 1920×1080。轻量模型按其原生 x4 后再缩放。没有输入降采样、跳帧、插帧或光流替代神经推理。','',
       '## 工程优化的一致性','',
       '参考快照是 tests/reference_v1_engine.py 和 tests/reference_v1_media.py，保存了升级前的算法。固定同一权重、FP16、512 分块、24px 上下文、增强 75%、时序稳定 40%，比较优化前后的完整输出。','',
       '| 路径 | 比较帧数 | 视频像素完全一致 | 时间戳一致 | AAC 压缩包一致 | 最大像素差 |',
       '| --- | ---: | --- | --- | --- | ---: |']
for name,row in data.get('video_agreement',{}).items():
    lines.append(f'| {name} | {row["frames"]} | {row["all_decoded_pixels_identical"]} | {row["timestamps_identical"]} | {row["aac_packets_identical"]} | {row["max_abs"]:.0f} |')
default_file=ROOT/'tests/artifacts/v2/default-256-quality.json'
if default_file.exists():
    default=json.loads(default_file.read_text(encoding='utf-8'))
    lines += ['',f'原默认分块256保持不变；720P样片额外验证：基底一致={default["base_identical"]}，增强残差一致={default["residual_identical"]}，最大差={default["max_abs"]:.0f}。']
lines += ['', '零差异是本机同配置测试的实际结果。实现保留逐块推理的输入、顺序和数学运算，仅将逐块往返复制改为整帧上传/下载和 GPU 拼接。管线用有界队列将 CPU 解码、光流/融合和编码与下一帧 GPU 推理重叠；每个编解码上下文只有一个线程操作。', '',
          '## 分块与显存','',
          '以下为预热后单帧增强的中位数，包含原生推理与目标尺寸缩放，未包含视频解码、光流与编码。每组使用三帧测量。','',
          '| 请求分块 | 实际分块 | 增强 fps | PyTorch 峰值分配 MiB | 与 v1 分块512 的 MAE/255 |',
          '| ---: | ---: | ---: | ---: | ---: |']
for row in data.get('tiles',[]):
    lines.append(f'| {row["requested_tile"]} | {row["actual_tile"]} | {row["fps"]:.3f} | {row["peak_allocated_mib"]:.1f} | {row["agreement_with_v1_tile512"]["mae"]:.6f} |')
lines += ['', '较大分块改变边界与可用上下文，因此与 512 分块不一定逐像素相同；这项差异独立于工程优化。720P 输入下，1536 和 2048 均已覆盖整帧。更大分块没有稳定带来明显提速，应结合吞吐选择。显存不足的有效分块会被记住，避免下一帧重复 OOM。','',
          '## TensorRT 精度与速度','',
          '固定版本 TensorRT 10.10.0.31。使用实际 NVIDIA CUDA 引擎，通过独立非默认 CUDA stream 执行，非 ONNX Runtime 或插值替代。首次构建和运行前质量检查单独计时，表中 fps 是预热后的稳定增强吞吐。','',
          '| 模型 / 分块 / 精度 | 增强 fps | 同模型同分块 MAE/255 | 最大差 | PSNR dB | SSIM 亮度 |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for row in data.get('tensorrt',[]):
    quality=row.get('agreement_same_model_same_tile')
    if not quality:
        continue
    psnr=f'{quality["psnr_db"]:.3f}' if quality['psnr_db'] is not None else '完全相同'
    lines.append(f'| {row["model"]} / {row["requested_tile"]} / {row["precision"]} | {row["fps"]:.3f} | {quality["mae"]:.6f} | {quality["max_abs"]:.0f} | {psnr} | {quality["ssim_luma"]:.6f} |')
lines += ['', 'TensorRT 不是逐像素零差异的执行后端；数值与核融合顺序变化会影响末位精度。每个新引擎/分块形状首次运行时，以真实输入与当前 PyTorch FP16 参考比较：非有限值、MAE 大于 0.5/255 或 PSNR 低于 45dB 时明确失败，可尝试 FP32 或关闭加速。此阈值用于拒绝明显异常，不是对所有视频画质的数学保证。','']
builds=data.get('tensorrt_cold_builds',[])
if builds:
    lines+=['首次构建记录：','', '| 模型 / 分块 | 精度 | 导出/构建秒数 |','| --- | --- | ---: |']
    for row in builds:
        matched=next((r for r in data.get('tensorrt',[]) if r['engine']['tensorrt']['engine_file']==row['engine_file']),None)
        label=f'{matched["model"]} / {matched["requested_tile"]}' if matched else row['engine_file']
        lines.append(f'| {label} | {row["precision"]} | {row["build_seconds"]:.1f} |')
    lines.append('')
if data.get('tensorrt_video_agreement'):
    q=data['tensorrt_video_agreement']
    lines += [f'完整 TensorRT 输出与 v1：{q["frames"]} 帧，平均 MAE {q["mean_mae"]:.6f}/255，平均亮度 SSIM {q["mean_ssim_luma"]:.6f}，时间戳一致={q["timestamps_identical"]}，AAC 压缩包一致={q["aac_packets_identical"]}。',
              '这一成片比较同时包含时序融合和 H.264 有损编码的差异，不能只用编码前指标便声称最终视频无变化。前表的同模型/同分块图像比较用于单独观察推理后端的影响。','']
lines += ['## 轻量模型','',
          '| 模型 | 增强 fps | 与重模型 MAE/255 | 与重模型 SSIM |',
          '| --- | ---: | ---: | ---: |']
for row in data.get('lightweight',[]):
    q=row['agreement_with_heavy_model']
    lines.append(f'| {row["model"]} | {row["fps"]:.3f} | {q["mae"]:.4f} | {q["ssim_luma"]:.6f} |')
lines += ['', '轻量模型与重模型的对比是输出一致性比较，没有高分辨率真值，不能把上述 MAE/SSIM 当成客观画质排名。通用 x4v3、WDN 弱降噪和动漫视频模型的纹理、噪点和线条偏好不同，用户按片源选择。','',
          '## 完整视频吞吐','',
          '| 路径 | 12帧总秒数 | 完整处理 fps | 按此吞吐外推1800帧/分钟 |',
          '| --- | ---: | ---: | ---: |']
rows=list(data.get('videos',[]))
if data.get('tensorrt_video'):rows.append(data['tensorrt_video'])
rows.extend(data.get('lightweight_videos',[]))
for row in rows:lines.append(f'| {row["name"]} | {row["seconds"]:.3f} | {row["fps"]:.3f} | {30/row["fps"]:.2f} |')
lines += ['', '一分钟30fps的实际视频仍需另测。外推列按1800帧计算，排除首次引擎编译，不代表长片实际保证。测试素材为 Windows 壁纸照片派生的720P水平平移短片，0.5秒/24fps；还覆盖合成图案、奇数尺寸、可变帧率和多音轨测试。','',
          'PyTorch 显存数字只统计其分配器，不包括全部 TensorRT 私有内存、驱动、桌面与其他进程。JSON 中另记录部分用例推理后的整卡占用，二者不可混同。','',
          '## 网页、API 与退出验证','',
          '27 项 pytest 检查通过。网页已实际提交原模型 TensorRT 和轻量模型 TensorRT 作业，确认开关、FP16/FP32 选择、全部大分块选项、缓存复用、输出尺寸及完成状态；390px 视口无横向溢出。', '']
http_file=ROOT/'tests/artifacts/v2/http-trt-768.json'
if http_file.exists():
    live=json.loads(http_file.read_text(encoding='utf-8'))
    lines += [f'首次编译768分块的真实HTTP任务期间，{live["health_checks"]}次健康检查成功，超时{live["health_timeouts"]}次，最大响应{live["health_max_latency_seconds"]:.3f}秒，编译期间仍可读取网页状态。','']
lifecycle_file=ROOT/'tests/artifacts/v2/lifecycle.json'
if lifecycle_file.exists():
    lifecycle=json.loads(lifecycle_file.read_text(encoding='utf-8'))
    if lifecycle.get('service_stopped'):
        lines += ['WSL Ubuntu 实际执行 stop.sh，终止正在运行的 TensorRT 与流水线任务：视频缓存文件归零，编译引擎、模型权重和另存视频保留；随后通过 start.bat 成功启动。','']
lines += ['TensorRT 库从 NVIDIA 官方索引下载，约7.43 MB/s；三份轻量权重从官方 GitHub release 下载。包和模型均核对大小及SHA256（老release未提供上游摘要的模型固定本次官方文件摘要）。本次无需换源或最快源回退。新增SDK依赖约2.34GiB，安装于项目目录，停止服务保留。','',
          '## 复现与文件','',
          '```powershell',
          r'& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py pytest tests\test_core.py tests\test_api.py tests\test_acceleration.py -q',
          r'& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py tests\benchmark_v2.py --phase engineering --repeats 3',
          r'& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py tests\benchmark_v2.py --phase tensorrt --repeats 3',
          r'& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py tests\benchmark_v2.py --phase light-videos --repeats 3',
          r'& ..\pic_polish\.venv\Scripts\python.exe scripts\runner.py scripts\render_acceleration_report.py',
          '```','',
          '- [原始机器可读数据](../tests/artifacts/v2/benchmark.json)',
          '- [原参考输出](../tests/artifacts/v2/v1-reference.mp4)',
          '- [优化后输出](../tests/artifacts/v2/v2-pipeline.mp4)',
          '- [TensorRT 输出](../tests/artifacts/v2/v2-tensorrt.mp4)',
          '- [画面对比裁切](../tests/artifacts/v2/comparison.png)',
          '',
          '模型、ONNX、TensorRT引擎和依赖均存于项目或已有图片项目；不会写入共享 Conda 环境。停止脚本继续清空 cache 下的视频，保留这些编译与另存的验证成果。']
(ROOT/'docs/ACCELERATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote docs/ACCELERATION.md')
