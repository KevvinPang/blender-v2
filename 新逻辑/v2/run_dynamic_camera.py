"""Run the dynamic-camera FlowHMR baseline and export a Blender camera track.

This stage deliberately stops at source SMPL-H plus camera data.  Character
retargeting and Blender Action creation consume these artifacts later, so a
camera failure cannot silently turn into a hand or body pose failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from runtime import ASCII_OUTPUT_ROOT, PROJECT_ROOT, REPO_ROOT, configure_imports, write_runtime_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mocap V2 dynamic-camera baseline")
    parser.add_argument("video", help="continuous single-shot input video")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "dynamic_camera"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--vggt-interval", type=int, default=2)
    parser.add_argument("--sam-chunk-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--static-camera", action="store_true",
                        help="skip VGGT when the source camera is fixed")
    parser.add_argument("--no-smooth-camera", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_imports()
    from setup_ascii_model_paths import main as setup_ascii_model_paths
    setup_ascii_model_paths()
    configure_imports()
    video = Path(args.video).resolve()
    requested_output = Path(args.output)
    if not requested_output.is_absolute():
        requested_output = PROJECT_ROOT / requested_output
    requested_output = requested_output.resolve()
    outputs_root = (PROJECT_ROOT / "outputs").resolve()
    try:
        output = ASCII_OUTPUT_ROOT / requested_output.relative_to(outputs_root)
    except ValueError:
        output = requested_output
    output.mkdir(parents=True, exist_ok=True)
    if not video.is_file():
        raise FileNotFoundError(video)

    job = output / video.stem
    job.mkdir(parents=True, exist_ok=True)
    write_runtime_info(job / "runtime.json")
    (job / "request.json").write_text(json.dumps(vars(args), ensure_ascii=True, indent=2), encoding="utf-8")

    import torch
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine
    from v2.camera.blender_export import export_blender_camera_npz

    ckpt = PROJECT_ROOT / "models" / "flowhmr_latest" / "flowhmr_latest.ckpt"
    cfg = PROJECT_ROOT / "models" / "flowhmr_latest" / "config.yml"
    smpl_model = Path(r"C:\mocap_v2_models\smplh\neutral\model.npz")
    j_regressor = PROJECT_ROOT / "models" / "body_models" / "smpl_neutral_J_regressor.pt"
    if not ckpt.is_file() or not cfg.is_file():
        raise FileNotFoundError(f"FlowHMR model files missing: {ckpt}, {cfg}")
    if args.device.startswith("cuda:"):
        torch.cuda.set_device(args.device)
    os.chdir(REPO_ROOT)

    print(f"[v2] video={video}")
    print(f"[v2] output={job}")
    print("[v2] loading FlowHMR with dynamic camera")
    t0 = time.time()
    engine = V2MInferenceEngine.build(
        ckpt=str(ckpt), model_cfg_path=str(cfg), device=args.device,
        validation_steps=args.steps, postprocess=False,
        smpl_model_path=str(smpl_model), j_regressor_path=str(j_regressor),
    )

    def progress(stage, value, description):
        print(f"[v2] {stage}: {value}% {description}", flush=True)

    engine.process_video(
        str(video), str(output), seed=args.seed, cfg_scale=1.0,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        vggt_frame_interval=max(1, args.vggt_interval), static_camera=args.static_camera,
        progress_callback=progress, sam_chunk_size=max(1, args.sam_chunk_size),
    )

    raw_camera = job / f"{video.stem}_{'static' if args.static_camera else 'vggt'}_camera.npz"
    if not raw_camera.is_file():
        raise FileNotFoundError(f"FlowHMR did not produce camera track: {raw_camera}")
    blender_camera = job / "camera_blender.npz"
    export_result = export_blender_camera_npz(raw_camera, blender_camera, smooth=not args.no_smooth_camera)

    metadata = {
        "schema_version": "mocap-v2-dynamic-baseline-1",
        "video": str(video),
        "raw_camera": str(raw_camera),
        "blender_camera": str(blender_camera),
        "motion_npz": str(job / f"{video.stem}_seed{args.seed}_origin.npz"),
        "camera_mode": "static" if args.static_camera else "dynamic",
        "scale_mode": export_result["scale_mode"],
        "frame_count": export_result["frame_count"],
        "elapsed_seconds": time.time() - t0,
        "notes": [
            "VGGT translation is relative-scale until a scene anchor is supplied.",
            "Camera is exported separately from character retargeting.",
        ],
    }
    (job / "motion_take.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
