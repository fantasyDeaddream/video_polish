# 发布与复现

发布包使用 scripts/build_release.py 的明确文件清单构建。只读收集源码和已验证模型；不打包 .git、.runtime、.deps、Conda/venv、视频缓存、日志、测试视频或下载临时文件。

## 构建

在选定 Python 环境执行：

```powershell
python scripts/runner.py scripts/prepare_models.py
# 提交源码后构建，确保归档记录准确的 source_commit。
python scripts/runner.py scripts/build_release.py
```

输出为 dist/VideoPolish-v<版本>-with-models.zip 和同名 .sha256。默认包包含六个 Real-ESRGAN 系列权重、两个 RIFE 权重及其官方架构对照文件。已有相邻 PicPolish 模型或 VIDEO_POLISH_MODELS 可只读复用，归档内统一放到 models 下。

只构建源码或核对已有归档：

```powershell
python scripts/runner.py scripts/build_release.py --source-only
python scripts/runner.py scripts/build_release.py --verify dist/VideoPolish-v1.2.0-with-models.zip
```

归档内 RELEASE-MANIFEST.json 记录版本、源提交、文件大小与 SHA256。构建器按固定顺序和提交时间写 ZIP，检查每个归档成员的完整内容和 CRC，并生成压缩包散列；.sh 文件保留 Unix 可执行标志。

模型固定散列来自本机已验证的上游文件，不冒充上游独立签名。来源与许可在 models/catalog.json、models/rife/manifest.json 和 THIRD_PARTY_NOTICES.md 中。TensorRT/CUDA 组件需由用户环境按其原始许可独立准备。

## 发布检查

1. 在项目目录运行既有 pytest；临时目录放在 .runtime 下，测试输出保持在被忽略的 tests/artifacts 中。
2. 验证 ZIP 的文件清单、散列、八份模型和许可证，确认没有运行状态、令牌、日志及视频缓存。
3. 解压到独立目录，使用独立端口启动，以归档内模型运行超分/插帧 API，然后停止该独立服务并核对缓存清理。
4. 推送公开源码和版本标签；创建 GitHub Release，上传 ZIP 与 .sha256，核对远端附件大小和 digest 后发布。

压缩包可以复用合适的既有 Python 环境。启动器不会自动安装/替换共享环境中的基础 PyTorch、OpenCV、FastAPI 等依赖；这些依赖需由用户在选定环境准备。
