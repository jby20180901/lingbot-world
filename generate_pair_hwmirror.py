import argparse
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.cam_utils import interpolate_camera_poses
from wan.utils.utils import save_video


def _build_parser():
    parser = argparse.ArgumentParser(description="Pair-wise LingBot + HunyuanWorld-Mirror complement filtering generation")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--action_path", type=str, required=True)
    parser.add_argument("--size", type=str, default="1280*720")
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--save_file", type=str, default="pair_hwmirror_output.mp4")
    parser.add_argument("--hwm_repo", type=str, default="../HunyuanWorld-Mirror")
    parser.add_argument("--hwm_conda_env", type=str, default=None, help="Conda env name for HunyuanWorld-Mirror rendering subprocess.")
    parser.add_argument("--hwm_python", type=str, default=None, help="Python executable path for HunyuanWorld-Mirror rendering subprocess.")
    parser.add_argument("--hwm_use_subprocess", action="store_true", help="Use subprocess bridge to run HunyuanWorld-Mirror rendering in a separate environment.")
    parser.add_argument("--guidance_fft_radius", type=int, default=10)
    return parser


def _load_hwm_app(hwm_repo: str):
    repo_path = str(Path(hwm_repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import app as hwm_app
    return hwm_app


def _save_frame_for_hwm(image_pil: Image.Image):
    temp_root = tempfile.mkdtemp(prefix="hwm_pair_")
    image_dir = Path(temp_root) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "image_0001.png"
    image_pil.save(image_path)
    return temp_root


def _render_with_hwm(hwm_app, first_frame_pil: Image.Image, pose_b: np.ndarray, intr_b: np.ndarray, target_h: int, target_w: int):
    workdir = _save_frame_for_hwm(first_frame_pil)
    try:
        outputs, _ = hwm_app.run_model(
            target_dir=workdir,
            confidence_percentile=10,
            edge_normal_threshold=5.0,
            edge_depth_threshold=0.03,
            apply_confidence_mask=True,
            apply_edge_mask=True,
        )

        splats = outputs.get("splats", None)
        if splats is None:
            raise RuntimeError("HunyuanWorld-Mirror app did not return splats")

        model = hwm_app.model
        if model is None:
            raise RuntimeError("HunyuanWorld-Mirror app model not initialized")

        device = next(model.parameters()).device
        viewmats = torch.from_numpy(pose_b.astype(np.float32)).to(device)[None, None]
        Ks = torch.from_numpy(intr_b.astype(np.float32)).to(device)[None, None]

        means = splats["means"].to(device)
        quats = splats["quats"].to(device)
        scales = splats["scales"].to(device)
        opacities = splats["opacities"].to(device)
        colors = splats["sh"].to(device) if "sh" in splats else splats["colors"].to(device)

        with torch.no_grad():
            rendered_colors, _, _ = model.gs_renderer.rasterizer.rasterize_batches(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                Ks,
                width=target_w,
                height=target_h,
                sh_degree=min(model.gs_renderer.sh_degree, 0) if "sh" in splats else None,
            )

        rendered = rendered_colors[0, 0].permute(2, 0, 1).detach().cpu().clamp(0, 1)
        return rendered.mul(2.0).sub(1.0)
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def _render_with_hwm_subprocess(first_frame_pil, pose_b, intr_b, target_h, target_w, args):
    temp_root = tempfile.mkdtemp(prefix="hwm_subproc_")
    try:
        input_image = Path(temp_root) / "input.png"
        pose_file = Path(temp_root) / "pose.npy"
        intr_file = Path(temp_root) / "intrinsics.npy"
        output_png = Path(temp_root) / "rendered.png"

        first_frame_pil.save(input_image)
        np.save(pose_file, pose_b.astype(np.float32))
        np.save(intr_file, intr_b.astype(np.float32))

        bridge_script = Path(__file__).with_name("hwm_render_bridge.py")

        if args.hwm_python is not None:
            cmd = [
                args.hwm_python,
                str(bridge_script),
            ]
        elif args.hwm_conda_env is not None:
            cmd = [
                "conda",
                "run",
                "-n",
                args.hwm_conda_env,
                "python",
                str(bridge_script),
            ]
        else:
            raise ValueError("hwm_use_subprocess enabled but neither hwm_conda_env nor hwm_python is provided")

        cmd.extend([
            "--hwm_repo", str(args.hwm_repo),
            "--input_image", str(input_image),
            "--pose_file", str(pose_file),
            "--intrinsics_file", str(intr_file),
            "--target_h", str(int(target_h)),
            "--target_w", str(int(target_w)),
            "--output_png", str(output_png),
        ])

        subprocess.run(cmd, check=True)

        rendered = Image.open(output_png).convert("RGB")
        rendered_np = np.asarray(rendered, dtype=np.float32) / 255.0
        rendered_tensor = torch.from_numpy(rendered_np).permute(2, 0, 1)
        return rendered_tensor.mul(2.0).sub(1.0)
    finally:
        import shutil
        shutil.rmtree(temp_root, ignore_errors=True)


def _make_pair_action_dir(poses_pair: np.ndarray, intr_pair: np.ndarray):
    src_indices = np.array([0.0, 1.0], dtype=np.float32)
    tgt_indices = np.linspace(0.0, 1.0, 5, dtype=np.float32)

    interp_poses = interpolate_camera_poses(
        src_indices=src_indices,
        src_rot_mat=poses_pair[:, :3, :3],
        src_trans_vec=poses_pair[:, :3, 3],
        tgt_indices=tgt_indices,
    ).numpy().astype(np.float32)

    if intr_pair.ndim == 2:
        intr_pair = np.stack([intr_pair, intr_pair], axis=0)

    alpha = tgt_indices[:, None, None]
    intr_interp = (1.0 - alpha) * intr_pair[0:1] + alpha * intr_pair[1:2]
    intr_interp = intr_interp.astype(np.float32)

    temp_dir = tempfile.mkdtemp(prefix="pair_action_")
    np.save(Path(temp_dir) / "poses.npy", interp_poses)
    np.save(Path(temp_dir) / "intrinsics.npy", intr_interp)
    return temp_dir


def generate_pair_hwmirror_video(pipeline, cfg, args):
    sample_steps = args.sample_steps if args.sample_steps is not None else cfg.sample_steps
    sample_shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift
    guide_scale = args.sample_guide_scale if args.sample_guide_scale is not None else cfg.sample_guide_scale

    image = Image.open(args.image).convert("RGB")

    poses = np.load(os.path.join(args.action_path, "poses.npy")).astype(np.float32)
    intr = np.load(os.path.join(args.action_path, "intrinsics.npy")).astype(np.float32)

    if intr.ndim == 2:
        intr = np.repeat(intr[None, ...], repeats=poses.shape[0], axis=0)

    pair_count = poses.shape[0] // 2
    if pair_count <= 0:
        raise ValueError("Need at least 2 poses for pair-wise process")

    hwm_app = None
    if not args.hwm_use_subprocess:
        hwm_app = _load_hwm_app(args.hwm_repo)
    output_frames = []

    for pair_idx in range(pair_count):
        idx_a = pair_idx * 2
        idx_b = idx_a + 1

        poses_pair = poses[[idx_a, idx_b]]
        intr_pair = intr[[idx_a, idx_b]]

        pair_action_dir = _make_pair_action_dir(poses_pair, intr_pair)
        try:
            base_video = pipeline.generate(
                input_prompt=args.prompt,
                img=image,
                action_path=pair_action_dir,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=5,
                shift=sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=sample_steps,
                guide_scale=guide_scale,
                seed=args.base_seed + pair_idx,
                offload_model=args.offload_model if hasattr(args, "offload_model") else True,
                save_intermediate_dir=args.save_intermediate_dir if hasattr(args, "save_intermediate_dir") else None,
                save_latents=args.save_latents if hasattr(args, "save_latents") else True,
                save_decoded=args.save_decoded if hasattr(args, "save_decoded") else False,
            )

            first_frame = base_video[:, 0].detach().cpu().clamp(-1, 1)
            first_frame_pil = to_pil_image((first_frame + 1.0) / 2.0)

            if args.hwm_use_subprocess:
                rendered_guidance = _render_with_hwm_subprocess(
                    first_frame_pil=first_frame_pil,
                    pose_b=poses_pair[1],
                    intr_b=intr_pair[1],
                    target_h=int(base_video.shape[2]),
                    target_w=int(base_video.shape[3]),
                    args=args,
                )
            else:
                rendered_guidance = _render_with_hwm(
                    hwm_app=hwm_app,
                    first_frame_pil=first_frame_pil,
                    pose_b=poses_pair[1],
                    intr_b=intr_pair[1],
                    target_h=int(base_video.shape[2]),
                    target_w=int(base_video.shape[3]),
                )

            guided_video = pipeline.generate(
                input_prompt=args.prompt,
                img=image,
                action_path=pair_action_dir,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=5,
                shift=sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=sample_steps,
                guide_scale=guide_scale,
                seed=args.base_seed + pair_idx,
                offload_model=args.offload_model if hasattr(args, "offload_model") else True,
                save_intermediate_dir=args.save_intermediate_dir if hasattr(args, "save_intermediate_dir") else None,
                save_latents=args.save_latents if hasattr(args, "save_latents") else True,
                save_decoded=args.save_decoded if hasattr(args, "save_decoded") else False,
                guidance_frame_tensors={4: rendered_guidance},
                guidance_fft_radius=getattr(args, "guidance_fft_radius", 10),
                guidance_early_step_end=15,
                guidance_mid_step_end=30,
                guidance_lambda_early=0.7,
                guidance_lambda_mid=0.2,
            )

            output_frames.append(base_video[:, 0].detach().cpu())
            output_frames.append(guided_video[:, 4].detach().cpu())

            image = to_pil_image(((guided_video[:, 4].detach().cpu().clamp(-1, 1) + 1.0) / 2.0))
        finally:
            import shutil
            shutil.rmtree(pair_action_dir, ignore_errors=True)

    final_video = torch.stack(output_frames, dim=1)
    return final_video


def main():
    args = _build_parser().parse_args()

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipeline = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
    )

    final_video = generate_pair_hwmirror_video(pipeline=pipeline, cfg=cfg, args=args)
    save_video(
        tensor=final_video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Done. Saved to {args.save_file}")


if __name__ == "__main__":
    main()
