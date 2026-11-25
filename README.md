# 艾莲 Blender 动捕 V2

适用于 Windows、NVIDIA CUDA 和 Blender 5.2.2。右侧「动捕」面板从单人视频生成艾莲骨架动作、镜头和可选音轨。固定机位保持镜头固定，移动机位估计镜头轨迹。手部结合逐帧检测、HaMeR 指形、CoTracker 边缘追踪和最终曲线稳定处理。声音口型和表情分别编辑。

- [所需模型、下载入口与放置位置](docs/MODELS.md)
- [FlowHMR 2 MMR v1.3.1 安装包](addons/FlowHMR_2_MMR_v1.3.1.zip) · [插件源码和许可证](addons/mmr_motion_bridge/)
- [动捕面板](tools/mocap_v2.py) · [后台任务与各阶段调用顺序](新逻辑/v2/jobs/blender_job.py) · [艾莲重定向](新逻辑/v2/blender_bridge/calibrated_bridge.py)

## 恢复运行环境

本仓库是现用插件的源码快照。推理需要外部源码、模型和 Python 环境。迁移时保留以下目录结构，或在对应入口调整路径。

1. 在仓库的 `新逻辑/source/flowhmr-main` 放置 [FlowHMR b79fc11](https://github.com/flowhmr/flowhmr/tree/b79fc11fc23d477d8d8cda2cea8635d15c26a9f0)，在其 `third_party` 下准备 YOLOX（6ddff48）、sam-3d-body（b5c765a）、vggt-omega（48b23c8）。将仓库 `overlays/flowhmr/` 的内容按相对路径覆盖到 FlowHMR 根目录。这些文件保留当前使用的 Windows 视频处理、模型路径和显存优化。
2. 在 `新逻辑/source/FlowHMR-Windows` 放置 [Windows 项目](https://github.com/xiaoerofficial021-collab/FlowHMR-Windows)，应用 `overlays/FlowHMR-Windows/`，将 [HaMeR 源码](https://github.com/geopavlakos/hamer/tree/3a01849f4148352e9260b69bf28b65d1671a4905) 放到其 `third_party/hamer_inference/`，使内部有 `hamer/` 包。在 `新逻辑/source/HaWoR` 放置 [HaWoR](https://github.com/ThunderVVV/HaWoR) 供调用 MANO 封装；将 [CoTracker](https://github.com/facebookresearch/co-tracker) 的 Python 包复制为 `新逻辑/source/cotracker/predictor.py` 所在的包目录。
3. 按模型表准备权重。为 FlowHMR 的 `ckpts/yolox` 和 `ckpts/vggt-omega` 建立指向相应模型目录的目录联接，或把文件放到两处。创建 `新逻辑/outputs` 后运行 `python 新逻辑/v2/setup_ascii_model_paths.py`；确认 `C:/mocap_v2_models` 下的 SMPL-H 和 SAM 联接指向本次准备的模型。
4. 身体推理使用仓库同级 `sam-3d-body-main/.venv/Scripts/python.exe`，本机为 Python 3.11、PyTorch 2.8.0+cu128、torchvision 0.23.0。按 FlowHMR 和 SAM 的推理依赖安装，并准备 supervision、torchdiffeq、imageio-ffmpeg 等包。手部环境默认在用户下载目录 `WiLoR-main/.venv/Scripts/python.exe`，本机为 Python 3.11、PyTorch 2.11.0+cu128、torchvision 0.26.0、ultralytics 8.1.34、smplx 0.1.28，另需 HaMeR 和 CoTracker 的推理依赖。安装 FFmpeg 和 ffprobe 并加入 PATH。
5. 当前手脸提取脚本从仓库内 `vtuber_tools/apps/Blender52/blender-5.2.2-windows-x64/portable/extensions` 读取 MediaPipe 模型及 `.local/lib/python*/site-packages` 中的 mediapipe、OpenCV、NumPy。使用已有便携版时可将该目录联接到实际扩展目录。
6. 执行 `python install_panel.py "你的 Blender scripts/addons 目录"`，重新启动 Blender，在偏好设置启用「动捕」。选择艾莲骨架和视频后点击「生成 V2 动捕」。插件入口引用本仓库的源文件，迁移后需更新入口。项目在 `C:/mocap_v2_outputs/ui/` 下创建每次任务的日志和结果。

## FlowHMR 2 MMR 动作微调

在 Blender 的「偏好设置 → 插件 → 从磁盘安装」选择上面的 ZIP 并启用。已有艾莲 MMR 工程可在「FlowHMR 2 MMR」面板使用「接管现有动作」和「建立实时预览」。从普通角色骨架创建 MMR 控制骨架还需 MikuMikuRig 和 Rigify。MMR 调整在动捕生成后单独进行。

## 验证与许可

发布时用 Blender 5.2.2 加载了仓库副本里的动捕、旧版兼容模块、动作微调及 MMR 插件，并通过 `新逻辑/v2/validation/test_joint_motion.py` 的摄像机与人物抖动分离和快速动作时序检查。新机器的 GPU 推理环境需要单独配置。

FlowHMR 源码适用其 Apache-2.0 许可证，模型另有[权重使用条款](docs/FlowHMR_WEIGHTS_LICENSE.txt)。FlowHMR 2 MMR 保留包内 Apache-2.0 许可证。口型适配和 Wiggle 的上游说明与许可见 [tools/vendor](tools/vendor/)。角色工程、视频及模型权重由使用者自行准备。
