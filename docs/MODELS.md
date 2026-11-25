# 当前动捕 V2 模型清单

以下路径以仓库的 `新逻辑/` 为起点。每行列出同一组件需要配套放置的文件。下载受许可限制的模型时，请在作者网站接受相应条款。

| 使用阶段 | 组件与文件 | 来源 |
|---|---|---|
| 两种机位 · 身体动作 | `models/flowhmr_latest/{flowhmr_latest.ckpt,config.yml}` | [FlowHMR 0.46B 指定版本](https://huggingface.co/fafsaf1/flowhmr-0.46B/tree/1eb105e554e0cfc1846ea5feac8c2d516a36d319/flowhmr_latest) |
| 两种机位 · 人体检测 | `models/yolox/yolox_l.pth`；同时供 FlowHMR 的 `ckpts/yolox/` 读取 | [YOLOX-L 权重](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
| 两种机位 · 身体观测 | `models/sam-3d-body-dinov3/model.ckpt`、同目录 `model_config.yaml`、`assets/mhr_model.pt` | [SAM 3D Body DINOv3](https://huggingface.co/facebook/sam-3d-body-dinov3) |
| 两种机位 · 身体骨架 | `models/body_models/smplh/neutral/model.npz`；下载 Extended SMPL+H（AMASS）的 `smplh.tar.xz`，取中性版 | [MANO / SMPL+H 官网](https://mano.is.tue.mpg.de/download.php) |
| 两种机位 · 关节回归 | `models/body_models/smpl_neutral_J_regressor.pt` | [GVHMR 文件](https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt) |
| 移动机位 · 相机轨迹 | `models/vggt-omega/vggt_omega_1b_512.pt`；同时供 FlowHMR 的 `ckpts/vggt-omega/` 读取 | [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) |
| 两种机位 · 手框 | `source/HaWoR/weights/external/detector.pt`，HaMeR 裁手使用 WiLoR 的检测器 | [WiLoR detector.pt](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt) |
| 两种机位 · 指形 | `source/FlowHMR-Windows/models/hamer/{hamer.ckpt,model_config.yaml,mano_mean_params.npz}`；三者来自 HaMeR demo 数据包 | [HaMeR 官方下载说明](https://github.com/geopavlakos/hamer#demo) |
| 两种机位 · MANO 左右手 | 官网 Models & Code 包中的 `MANO_LEFT.pkl` 放入 `source/HaWoR/_DATA/data_left/mano_left/`；`MANO_RIGHT.pkl` 放入 `source/HaWoR/_DATA/data/mano/`，并供 `source/FlowHMR-Windows/models/mano_v1_2/models/MANO_RIGHT.pkl` 读取 | [MANO 官网](https://mano.is.tue.mpg.de/download.php) |
| 手部出入镜 · 局部追踪 | `models/cotracker3_offline.pth`，检测到满足条件的边缘轨迹时加载离线模型 | [CoTracker3 离线权重](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth) |
| 两种机位 · 手部关键点 | `hand_landmarker.task` | [Google MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models) |
| 两种机位 · 头部与眼神 | `face_landmarker.task` | [Google MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models) |

两个 MediaPipe 文件位于 README 第 5 步的 Blender 扩展目录下，具体位置是 `user_default/media_to_pose/assets/models/`。固定机位跳过 VGGT。声音口型由音频包络和频谱驱动。FlowHMR 2 MMR 操作 Blender 骨架，这两个功能均无需额外神经网络权重。

早期准备过 GeoCalib 和 WiLoR 的整手重建权重。当前面板用默认内参或 VGGT 估计相机，用 HaMeR 重建指形。HaWoR 项目在这条路径中提供 MANO 封装，完整的 HaWoR、DROID-SLAM、Metric3D 和 infiller 权重均未被本任务入口加载。
