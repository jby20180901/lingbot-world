import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .utils.worldmirror_service_client import WorldMirrorServiceClient


def generate_pairwise_with_worldmirror(
    wan_i2v,
    input_prompt: str,
    init_image: Image.Image,
    action_path: str,
    frame_num: int,
    max_area: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale,
    seed: int,
    offload_model: bool,
    save_intermediate_dir: Optional[str],
    save_latents: bool,
    save_decoded: bool,
    service_url: str,
    complementary_alpha: float,
):
    poses = np.load(os.path.join(action_path, "poses.npy"))
    intrinsics_path = os.path.join(action_path, "intrinsics.npy")
    intrinsics = np.load(intrinsics_path) if os.path.exists(intrinsics_path) else None

    total_frames = min(frame_num, poses.shape[0])
    if total_frames <= 0:
        raise ValueError("No frames available for pairwise generation")

    if intrinsics is not None and intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None, ...], repeats=total_frames, axis=0)
    if intrinsics is not None and intrinsics.ndim == 3:
        intrinsics = intrinsics[:total_frames]

    poses = poses[:total_frames]
    client = WorldMirrorServiceClient(service_url=service_url)

    frames = []
    current_ref = init_image

    pair_count = total_frames // 2
    for pair_idx in range(pair_count):
        first_idx = pair_idx * 2
        second_idx = first_idx + 1

        first_action = _create_single_pose_action_dir(poses[first_idx], None if intrinsics is None else intrinsics[first_idx])
        second_action = _create_single_pose_action_dir(poses[second_idx], None if intrinsics is None else intrinsics[second_idx])
        try:
            first_save_dir = None
            second_save_dir = None
            if save_intermediate_dir is not None:
                first_save_dir = os.path.join(save_intermediate_dir, f"pair_{pair_idx:03d}", "first")
                second_save_dir = os.path.join(save_intermediate_dir, f"pair_{pair_idx:03d}", "second")

            first_video = wan_i2v.generate(
                input_prompt=input_prompt,
                img=current_ref,
                action_path=first_action,
                max_area=max_area,
                frame_num=1,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed + first_idx if seed >= 0 else -1,
                offload_model=offload_model,
                save_intermediate_dir=first_save_dir,
                save_latents=save_latents,
                save_decoded=save_decoded,
            )
            first_frame = first_video[:, 0].detach().cpu()
            first_image = to_pil_image(first_frame.clamp(-1, 1).add(1).div(2))
            frames.append(first_frame)

            scene_id = client.build_scene(first_image)
            render_img = client.render_pose(
                scene_id=scene_id,
                pose=poses[second_idx],
                intrinsics=None if intrinsics is None else intrinsics[second_idx],
                width=first_image.width,
                height=first_image.height,
            )

            second_video = wan_i2v.generate(
                input_prompt=input_prompt,
                img=first_image,
                action_path=second_action,
                max_area=max_area,
                frame_num=1,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed + second_idx if seed >= 0 else -1,
                offload_model=offload_model,
                save_intermediate_dir=second_save_dir,
                save_latents=save_latents,
                save_decoded=save_decoded,
                injection_image=render_img,
                complementary_alpha=complementary_alpha,
            )
            second_frame = second_video[:, 0].detach().cpu()
            frames.append(second_frame)
            current_ref = to_pil_image(second_frame.clamp(-1, 1).add(1).div(2))

        finally:
            shutil.rmtree(first_action, ignore_errors=True)
            shutil.rmtree(second_action, ignore_errors=True)

    if total_frames % 2 == 1:
        last_idx = total_frames - 1
        last_action = _create_single_pose_action_dir(poses[last_idx], None if intrinsics is None else intrinsics[last_idx])
        try:
            last_save_dir = None
            if save_intermediate_dir is not None:
                last_save_dir = os.path.join(save_intermediate_dir, "last_single")
            last_video = wan_i2v.generate(
                input_prompt=input_prompt,
                img=current_ref,
                action_path=last_action,
                max_area=max_area,
                frame_num=1,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed + last_idx if seed >= 0 else -1,
                offload_model=offload_model,
                save_intermediate_dir=last_save_dir,
                save_latents=save_latents,
                save_decoded=save_decoded,
            )
            frames.append(last_video[:, 0].detach().cpu())
        finally:
            shutil.rmtree(last_action, ignore_errors=True)

    stacked = torch.stack(frames, dim=1)
    return stacked


def _create_single_pose_action_dir(pose: np.ndarray, intrinsic: Optional[np.ndarray]) -> str:
    temp_dir = tempfile.mkdtemp(prefix="pair_action_")
    np.save(os.path.join(temp_dir, "poses.npy"), pose[None, ...].astype(np.float32))
    if intrinsic is not None:
        intr = np.asarray(intrinsic)
        if intr.ndim == 1 and intr.shape[0] == 4:
            intr_save = intr[None, :]
        elif intr.ndim == 2 and intr.shape == (3, 3):
            intr_save = intr[None, :, :]
        elif intr.ndim == 2 and intr.shape[-1] == 4:
            intr_save = intr[:1, :]
        elif intr.ndim == 3 and intr.shape[-2:] == (3, 3):
            intr_save = intr[:1, :, :]
        elif intr.ndim == 3 and intr.shape[-1] == 4:
            intr_save = intr.reshape(-1, 4)[:1, :]
        else:
            raise ValueError(f"Unsupported intrinsic shape for pairwise action: {intr.shape}")

        np.save(os.path.join(temp_dir, "intrinsics.npy"), intr_save.astype(np.float32))
    return temp_dir
