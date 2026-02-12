"""
Sliding window implementation for long video generation in LingBot World.
This module enables generation of long videos with constant memory usage by
processing video in overlapping segments instead of all at once.
"""

import gc
import logging
import math
import os
import random
import sys
from typing import Optional, List, Tuple
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from .image2video import WanI2V


class SlidingWindowI2V:
    """
    Extended I2V model with sliding window mechanism for long video generation.
    
    This class wraps the WanI2V model and implements sliding window processing
    to generate long videos with bounded memory usage. Instead of processing
    the entire video sequence at once, it processes overlapping segments and
    stitches them together.
    
    Args:
        wan_i2v (WanI2V): The base WanI2V model instance
        window_size (int): Number of frames to process in each window.
                          Must be of form 4n+1 (e.g., 49, 53, 57, 81)
        overlap_size (int): Number of overlapping frames between consecutive windows.
                           Default: window_size // 2
        blend_mode (str): How to blend overlapping regions. Options: 'lerp' or 'crossfade'
    """
    
    def __init__(
        self,
        wan_i2v: WanI2V,
        window_size: int = 49,
        overlap_size: Optional[int] = None,
        blend_mode: str = "lerp",
    ):
        self.wan_i2v = wan_i2v
        self.window_size = window_size
        self.overlap_size = overlap_size or window_size // 2
        self.blend_mode = blend_mode
        self.device = wan_i2v.device
        
        # Validate window size (must be 4n+1)
        if (window_size - 1) % 4 != 0:
            raise ValueError(f"Window size must be of form 4n+1, got {window_size}")
        
        # Overlap should be less than window size
        if self.overlap_size >= window_size:
            raise ValueError(f"Overlap size ({self.overlap_size}) must be < window size ({window_size})")
        
        self.logger = logging.getLogger(__name__)
    
    def _blend_latents(
        self,
        prev_latents: torch.Tensor,
        curr_latents: torch.Tensor,
        overlap: int,
    ) -> torch.Tensor:
        """
        Blend overlapping latent representations between consecutive windows.
        
        Args:
            prev_latents: Latents from previous window [C, F, H, W]
            curr_latents: Latents from current window [C, F, H, W]
            overlap: Number of overlapping frames
            
        Returns:
            Blended latents for the overlap region
        """
        if self.blend_mode == "lerp":
            # Linear interpolation based on frame position in overlap
            weights = torch.linspace(0, 1, overlap + 2)[1:-1].to(self.device)
            weights = weights.view(1, -1, 1, 1)
            
            # Blend last 'overlap' frames of prev with first 'overlap' frames of curr
            blended = (1 - weights) * prev_latents[:, -overlap:, :, :] + \
                     weights * curr_latents[:, :overlap, :, :]
            return blended
        
        elif self.blend_mode == "crossfade":
            # Smooth crossfade using raised cosine
            alpha = torch.arange(overlap).float().to(self.device) / (overlap - 1)
            alpha = (1 - torch.cos(alpha * math.pi)) / 2
            alpha = alpha.view(1, -1, 1, 1)
            
            blended = (1 - alpha) * prev_latents[:, -overlap:, :, :] + \
                     alpha * curr_latents[:, :overlap, :, :]
            return blended
        
        else:
            raise ValueError(f"Unknown blend mode: {self.blend_mode}")
    
    def _stitch_latents(
        self,
        all_latents: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Stitch together latent representations from multiple windows.
        
        Args:
            all_latents: List of latent tensors from each window [C, F, H, W]
            
        Returns:
            Stitched latent tensor for entire video
        """
        if len(all_latents) == 1:
            return all_latents[0]
        
        stitched_frames = []
        
        for idx, latents in enumerate(all_latents):
            if idx == 0:
                # First window: use all frames except the last overlap
                stitched_frames.append(latents[:, :-self.overlap_size, :, :])
            elif idx == len(all_latents) - 1:
                # Last window: use all frames after the overlap region
                stitched_frames.append(latents[:, self.overlap_size:, :, :])
            else:
                # Middle windows: blend overlap and use unique frames
                blended = self._blend_latents(
                    all_latents[idx - 1],
                    latents,
                    self.overlap_size,
                )
                stitched_frames.append(blended)
                stitched_frames.append(latents[:, self.overlap_size:, :, :])
        
        # Handle first overlap with proper blending if needed
        if len(all_latents) > 1:
            stitched_frames = [
                stitched_frames[0],
                self._blend_latents(all_latents[0], all_latents[1], self.overlap_size),
            ]
            if len(all_latents) > 2:
                for idx in range(1, len(all_latents) - 1):
                    blended = self._blend_latents(
                        all_latents[idx],
                        all_latents[idx + 1],
                        self.overlap_size,
                    )
                    stitched_frames.append(blended)
                    stitched_frames.append(all_latents[idx + 1][:, self.overlap_size:, :, :])
            
            stitched_frames.append(all_latents[-1][:, self.overlap_size:, :, :])
        
        # Concatenate along frame dimension
        stitched_latent = torch.cat(stitched_frames, dim=1)
        return stitched_latent
    
    def _get_window_frames(
        self,
        total_frames: int,
        window_idx: int,
    ) -> Tuple[int, int]:
        """
        Calculate the frame range for a given window.
        
        Args:
            total_frames: Total number of frames to generate
            window_idx: Index of current window
            
        Returns:
            Tuple of (start_frame, end_frame) indices
        """
        stride = self.window_size - self.overlap_size
        start_frame = window_idx * stride
        end_frame = min(start_frame + self.window_size, total_frames)
        
        return start_frame, end_frame
    
    def _extract_reference_frame(
        self,
        img: torch.Tensor,
        frame_idx: int,
        total_frames: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Extract reference frame information for a window.
        Currently uses the input image for all windows (could be extended).
        
        Args:
            img: Input reference image tensor
            frame_idx: Frame index for this window
            total_frames: Total number of frames
            h: Target height
            w: Target width
            
        Returns:
            Reference frame tensor
        """
        import torchvision.transforms.functional as TF
        
        # Always use the input image as reference
        # In future, could interpolate between keyframes
        ref_frame = TF.to_tensor(img).sub_(0.5).div_(0.5)
        ref_frame = TF.resize(ref_frame, size=(h, w), antialias=True)
        
        return ref_frame
    
    def generate_sliding_window(
        self,
        img,
        input_prompt: str,
        frame_num: int,
        max_area: int = 768 * 512,
        guide_scale: Tuple[float, float] = (3.0, 7.5),
        seed: int = -1,
        n_prompt: str = "",
        action_path: Optional[str] = None,
        sample_steps: int = 30,
        sample_shift: float = 1.0,
        sample_guide_scale: Optional[Tuple[float, float]] = None,
        sample_solver: str = "unipc",
        offload_model: bool = True,
    ) -> torch.Tensor:
        """
        Generate long video using sliding window mechanism.
        
        Args:
            img: Input reference image (PIL Image or tensor)
            input_prompt: Text prompt for video generation
            frame_num: Total number of frames to generate
            max_area: Maximum area for video frames
            guide_scale: Guidance scale for the base model
            seed: Random seed for reproducibility
            n_prompt: Negative prompt
            action_path: Path to camera trajectory
            sample_steps: Number of diffusion steps
            sample_shift: Shift for flow matching scheduler
            sample_guide_scale: Override guide_scale for sampling
            sample_solver: Solver for diffusion ('unipc' or 'dpm++')
            offload_model: Whether to offload models to CPU
            
        Returns:
            Generated video tensor [C, N, H, W]
        """
        # Calculate number of windows needed
        stride = self.window_size - self.overlap_size
        num_windows = (frame_num - 1 + stride - 1) // stride + 1
        
        self.logger.info(f"Generating {frame_num} frames using sliding window")
        self.logger.info(f"  Window size: {self.window_size}")
        self.logger.info(f"  Overlap: {self.overlap_size}")
        self.logger.info(f"  Number of windows: {num_windows}")
        
        all_latents = []
        all_videos = []
        camera_poses = None
        
        # Load camera poses if provided
        if action_path is not None:
            camera_poses = np.load(os.path.join(action_path, "poses.npy"))
        
        # Process each window
        for window_idx in tqdm(range(num_windows), desc="Processing windows"):
            window_start, window_end = self._get_window_frames(frame_num, window_idx)
            window_frames = window_end - window_start
            
            # Adjust window size if this is the last window and it's smaller
            current_window_size = min(self.window_size, window_frames)
            
            # Ensure window_size is still 4n+1
            if (current_window_size - 1) % 4 != 0:
                current_window_size = ((current_window_size - 1) // 4) * 4 + 1
            
            self.logger.debug(f"Window {window_idx}: frames {window_start}-{window_end} "
                            f"(size={current_window_size})")
            
            # Create a temporary action path for this window if needed
            window_action_path = None
            if action_path is not None and camera_poses is not None:
                window_action_path = self._create_window_action_path(
                    action_path, window_start, window_end, camera_poses
                )
            
            # Generate video for this window
            window_video = self.wan_i2v.generate(
                img=img,
                input_prompt=input_prompt,
                frame_num=current_window_size,
                max_area=max_area,
                guide_scale=guide_scale,
                seed=seed + window_idx if seed >= 0 else -1,
                n_prompt=n_prompt,
                action_path=window_action_path,
                sampling_steps=sample_steps,
                shift=sample_shift,
                sample_solver=sample_solver,
                offload_model=offload_model,
            )
            
            if window_video is not None:
                all_videos.append(window_video)
            
            # Clean up
            if window_action_path is not None and os.path.exists(window_action_path):
                import shutil
                shutil.rmtree(window_action_path)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Stitch videos together
        if len(all_videos) > 0:
            final_video = self._stitch_videos(all_videos, frame_num)
            return final_video
        else:
            return None
    
    def _create_window_action_path(
        self,
        original_action_path: str,
        start_frame: int,
        end_frame: int,
        camera_poses: np.ndarray,
    ) -> str:
        """
        Create a temporary action path for a window with interpolated camera poses.
        
        Args:
            original_action_path: Original action path directory
            start_frame: Start frame index
            end_frame: End frame index
            camera_poses: Original camera poses array
            
        Returns:
            Path to temporary window action directory
        """
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="window_action_")
        
        # Extract poses for this window
        window_poses = camera_poses[start_frame:end_frame]
        
        # Save poses
        poses_path = os.path.join(temp_dir, "poses.npy")
        np.save(poses_path, window_poses)
        
        # Copy intrinsics if they exist
        intrinsics_src = os.path.join(original_action_path, "intrinsics.npy")
        if os.path.exists(intrinsics_src):
            intrinsics_dst = os.path.join(temp_dir, "intrinsics.npy")
            shutil.copy(intrinsics_src, intrinsics_dst)
        
        return temp_dir
    
    def _stitch_videos(
        self,
        video_segments: List[torch.Tensor],
        total_frames: int,
    ) -> torch.Tensor:
        """
        Stitch video segments together with blending in overlap regions.
        
        Args:
            video_segments: List of video tensors [C, F, H, W]
            total_frames: Target total number of frames
            
        Returns:
            Stitched video tensor [C, N, H, W]
        """
        if len(video_segments) == 1:
            # Single segment: crop to target frames
            video = video_segments[0]
            if video.shape[1] > total_frames:
                video = video[:, :total_frames, :, :]
            return video
        
        stitched_frames = []
        
        for idx, video in enumerate(video_segments):
            if idx == 0:
                # First segment: use all frames except overlap region at the end
                stitched_frames.append(video[:, :-self.overlap_size, :, :])
                
                # Blend the overlap region with the next segment
                if len(video_segments) > 1:
                    next_video = video_segments[idx + 1]
                    blended = self._blend_video_frames(
                        video[:, -self.overlap_size:, :, :],
                        next_video[:, :self.overlap_size, :, :],
                    )
                    stitched_frames.append(blended)
                    
            elif idx == len(video_segments) - 1:
                # Last segment: use all frames after the overlap region
                # (overlap was already handled in previous iteration)
                stitched_frames.append(video[:, self.overlap_size:, :, :])
                
            else:
                # Middle segments: skip overlap region at start (already blended)
                # and blend with next segment
                stitched_frames.append(video[:, self.overlap_size:, :, :])
                
                # Blend the overlap region with the next segment
                next_video = video_segments[idx + 1]
                blended = self._blend_video_frames(
                    video[:, -self.overlap_size:, :, :],
                    next_video[:, :self.overlap_size, :, :],
                )
                stitched_frames.append(blended)
        
        # Concatenate all frames
        stitched_video = torch.cat(stitched_frames, dim=1)
        
        # Ensure total frames matches target
        if stitched_video.shape[1] > total_frames:
            stitched_video = stitched_video[:, :total_frames, :, :]
        elif stitched_video.shape[1] < total_frames:
            self.logger.warning(f"Generated {stitched_video.shape[1]} frames, "
                              f"expected {total_frames}")
        
        return stitched_video
    
    def _blend_video_frames(
        self,
        prev_frames: torch.Tensor,
        curr_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend overlapping frames from consecutive segments.
        
        Args:
            prev_frames: Last N frames from previous segment [C, N, H, W]
            curr_frames: First N frames from current segment [C, N, H, W]
            
        Returns:
            Blended frames [C, N, H, W]
        """
        overlap = min(prev_frames.shape[1], curr_frames.shape[1])
        
        if self.blend_mode == "lerp":
            # Linear interpolation
            weights = torch.linspace(0, 1, overlap).to(self.device)
            weights = weights.view(1, -1, 1, 1)
            
            blended = (1 - weights) * prev_frames[:, -overlap:, :, :] + \
                     weights * curr_frames[:, :overlap, :, :]
        
        elif self.blend_mode == "crossfade":
            # Smooth crossfade
            alpha = torch.arange(overlap).float().to(self.device) / max(overlap - 1, 1)
            alpha = (1 - torch.cos(alpha * math.pi)) / 2
            alpha = alpha.view(1, -1, 1, 1)
            
            blended = (1 - alpha) * prev_frames[:, -overlap:, :, :] + \
                     alpha * curr_frames[:, :overlap, :, :]
        
        else:
            blended = curr_frames[:, :overlap, :, :]
        
        return blended
