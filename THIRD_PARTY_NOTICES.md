# Third-party notices / 第三方许可

## Practical-RIFE 4.25 / 4.25.lite (VideoPolish 1.2)

Upstream: https://github.com/hzwer/Practical-RIFE, Copyright (c) 2021 hzwer, MIT.
The complete notice is retained in docs/licenses/Practical-RIFE-MIT.txt. app/rife_arch.py
is an inference-only adaptation of the matching official archives: training code
is omitted, adjacent endpoint encodings are cached, and half-precision inference
uses FP32 warp coordinates. The architecture names and inference weights remain
compatible. Training-only teacher/caltime checkpoint entries are explicitly
excluded; all inference parameters are loaded strictly with weights_only=True.
Official archive and member SHA256 values and download sources are recorded in
models/rife/manifest.json. Downloads are excluded from source distribution.

The Sintel trailer used by local tests is from Blender Foundation's official
download server: https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4
and https://durian.blender.org/. It is a test asset, not part of the application's
MIT-licensed source. Preserve the original film's attribution and licensing when
redistributing any derived test media. Local generated samples identify their
resampling and synthetic audio in their evidence reports.

VideoPolish 复用相邻 PicPolish 的 RRDBNet 代码，并只读使用已有模型权重。原作者许可保留如下。

## 1. Real-ESRGAN（模型权重 + 参考实现）

- 上游：https://github.com/xinntao/Real-ESRGAN
- 使用方式：运行时下载官方权重 `RealESRGAN_x4plus.pth` / `RealESRGAN_x2plus.pth` /
  `RealESRGAN_x4plus_anime_6B.pth`；RRDBNet 网络结构参考其与 BasicSR 的实现。
- 许可：BSD 3-Clause License
- 版权：Copyright (c) 2021, Xintao Wang

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

---

## 2. BasicSR（RRDBNet 结构来源，已 vendor 到 `app/rrdbnet.py`）

- 上游：https://github.com/XPixelGroup/BasicSR
- 使用方式：`app/rrdbnet.py` 是 BasicSR 1.4.2 `basicsr/archs/rrdbnet_arch.py` 与
  `arch_util.py` 中三个辅助函数的**原样移植**（去掉 basicsr 包依赖，保留层名以兼容官方权重）。
- 许可：Apache License 2.0
- 版权：Copyright (c) 2018-2022, XPixel Group

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2018-2022 BasicSR Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```

---


## PicPolish

RRDBNet 文件来自本机相邻图片项目，MIT，Copyright (c) 2026 deaddream and PicPolish contributors。完整原许可见 [PicPolish-MIT.txt](docs/licenses/PicPolish-MIT.txt)。本项目未复制 Real-CUGAN 代码，未使用 Real-CUGAN 权重。

## Runtime dependencies

依赖通过已有环境或项目本地 .deps 使用，不随源码打包。各依赖的完整许可保存在其原始发行版中。

- PyTorch: BSD-3-Clause, https://github.com/pytorch/pytorch
- OpenCV: Apache-2.0, https://github.com/opencv/opencv
- NumPy: BSD-3-Clause, https://github.com/numpy/numpy
- FastAPI / Starlette / Uvicorn / python-multipart: their respective MIT/BSD/Apache licenses in the installed distributions.
- PyAV: BSD-3-Clause, https://github.com/PyAV-Org/PyAV
- PyAV Windows wheel includes FFmpeg libraries and libx264. FFmpeg license/build details and the associated GPL/LGPL obligations depend on the exact linked components; consult av-16.1.0.dist-info/licenses in the local installed wheel before redistributing binaries. The source project does not relicense these dependencies.

## Test media

Local smoke-test clips derive from PicPolish sample photos sourced from Windows wallpaper, plus generated gradients/shapes. They are local validation artifacts, excluded from source distribution; no license to redistribute Microsoft wallpaper is implied.

## SRVGGNetCompact / lightweight Real-ESRGAN

`app/srvgg.py` is a dependency-light adaptation of the upstream architecture from Real-ESRGAN, Copyright (c) 2021 Xintao Wang, under the BSD-3-Clause terms reproduced above. The BasicSR registry decoration is omitted; tensor shapes and state-dict names are retained. Three official v0.2.5.0 checkpoint files are downloaded at runtime, excluded from source distribution; their sources and SHA256 values are recorded in `models/lightweight-manifest.json`.

## Optional NVIDIA TensorRT / CUDA runtime and ONNX

TensorRT 10.10.0.31 and CUDA Runtime 12.8.90 are NVIDIA proprietary runtime components, installed only into `.deps/tensorrt-win32-312`; their original NVIDIA license files remain in the installed distributions. They are not relicensed under this project's MIT license. ONNX 1.17.0 uses Apache-2.0; protobuf 5.29.5 uses its upstream BSD license. Source distribution excludes the downloaded packages, generated ONNX files, and serialized engine caches. Consult the actual upstream distribution licenses before redistributing those binaries.
