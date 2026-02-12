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
    
    ## Key Improvements:
    1. **Autoregressive Mode**: Uses previous window's last frame as next window's input
       for better temporal consistency (recommended for long videos)
    2. **Latent Space Stitching**: Blends in latent space before decoding for smoother
       transitions (requires WanI2V support for latent output)
    3. **Multiple Seed Strategies**: Offers 'uniform', 'incremental', and 'position_based'
       seed modes to balance consistency and diversity
    4. **Camera Trajectory Support**: Correctly distributes camera poses across windows
    
    ## Usage Examples:
    
    ### Basic Usage (Non-Autoregressive):
    ```python
    sliding_i2v = SlidingWindowI2V(wan_i2v_model, window_size=49, overlap_size=24)
    video = sliding_i2v.generate_sliding_window(
        img=image,
        input_prompt="A beautiful landscape",
        frame_num=200
    )
    ```
    
    ### Recommended for Long Videos (Autoregressive + Latent Stitching):
    ```python
    video = sliding_i2v.generate_sliding_window_with_latent_stitching(
        img=image,
        input_prompt="A beautiful landscape",
        frame_num=200,
        use_autoregressive=True,
        seed_mode="uniform"
    )
    ```
    
    ### With Camera Trajectory:
    ```python
    video = sliding_i2v.generate_sliding_window(
        img=image,
        input_prompt="A beautiful landscape",
        frame_num=200,
        action_path="path/to/trajectory",
        use_autoregressive=True,
        seed_mode="position_based"
    )
    ```
    
    Args:
        wan_i2v (WanI2V): The base WanI2V model instance
        window_size (int): Number of frames to process in each window.
                          Must be of form 4n+1 (e.g., 49, 53, 57, 81)
                          Larger window = higher quality but more memory
        overlap_size (int): Number of overlapping frames between consecutive windows.
                           Default: window_size // 2
                           Larger overlap = smoother transitions but slower generation
        blend_mode (str): How to blend overlapping regions. Options: 'lerp', 'crossfade', 'sigmoid'
                         'sigmoid' provides the smoothest transitions
    """
    
    def __init__(
        self,
        wan_i2v: WanI2V,
        window_size: int = 49,
        overlap_size: Optional[int] = None,
        blend_mode: str = "crossfade",
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
        self.logger.info(f"Initialized SlidingWindowI2V with window_size={window_size}, "
                        f"overlap_size={self.overlap_size}, blend_mode={blend_mode}")
    
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
        Stitch together latent representations from multiple windows with improved blending.
        LATENT SPACE STITCHING produces better quality than pixel-space stitching.
        
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
                
                # Blend the overlap region with the next latent
                if len(all_latents) > 1:
                    next_latents = all_latents[idx + 1]
                    blended = self._blend_latents(
                        latents[:, -self.overlap_size:, :, :],
                        next_latents[:, :self.overlap_size, :, :],
                        self.overlap_size,
                    )
                    stitched_frames.append(blended)
                    
            elif idx == len(all_latents) - 1:
                # Last window: use all frames after the overlap region
                # (overlap was already handled in previous iteration)
                stitched_frames.append(latents[:, self.overlap_size:, :, :])
                
            else:
                # Middle windows: skip overlap region at start (already blended)
                # and blend with next latent
                stitched_frames.append(latents[:, self.overlap_size:, :, :])
                
                # Blend the overlap region with the next latent
                next_latents = all_latents[idx + 1]
                blended = self._blend_latents(
                    latents[:, -self.overlap_size:, :, :],
                    next_latents[:, :self.overlap_size, :, :],
                    self.overlap_size,
                )
                stitched_frames.append(blended)
        
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
        use_autoregressive: bool = True,
        seed_mode: str = "uniform",
    ) -> torch.Tensor:
        """
        Generate long video using sliding window mechanism with autoregressive conditioning.
        
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
            use_autoregressive: If True, use previous window's last frame as next window's input (Autoregressive).
                               If False, use original image for all windows (Non-Autoregressive).
            seed_mode: How to handle random seeds. Options: 
                      'uniform' (same seed for all windows, best for consistency),
                      'incremental' (seed + window_idx, may cause ghosting),
                      'position_based' (seed based on frame position, compromise)
            
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
        self.logger.info(f"  Autoregressive mode: {use_autoregressive}")
        self.logger.info(f"  Seed mode: {seed_mode}")
        
        all_videos = []
        camera_poses = None
        current_input_img = img  # Track the input image for autoregressive mode
        
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
            
            # Calculate seed for this window based on mode
            if seed >= 0:
                if seed_mode == "uniform":
                    window_seed = seed
                elif seed_mode == "incremental":
                    window_seed = seed + window_idx
                elif seed_mode == "position_based":
                    # Use seed based on frame position instead of window index
                    window_seed = seed + window_start // max(self.window_size, 1)
                else:
                    window_seed = seed
            else:
                window_seed = -1
            
            # Generate video for this window
            window_video = self.wan_i2v.generate(
                img=current_input_img,
                input_prompt=input_prompt,
                frame_num=current_window_size,
                max_area=max_area,
                guide_scale=guide_scale,
                seed=window_seed,
                n_prompt=n_prompt,
                action_path=window_action_path,
                sampling_steps=sample_steps,
                shift=sample_shift,
                sample_solver=sample_solver,
                offload_model=offload_model,
            )
            
            if window_video is not None:
                all_videos.append(window_video)
                
                # For autoregressive mode: use the last frame of this window as input for next window
                if use_autoregressive and window_idx < num_windows - 1:
                    # Extract last frame and convert to PIL Image if needed
                    last_frame = window_video[:, -1, :, :]  # [C, H, W]
                    # Convert tensor to PIL Image for next iteration
                    from torchvision.transforms.functional import to_pil_image
                    current_input_img = to_pil_image(last_frame)
            
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
        Corrected logic to avoid frame duplication in middle segments.
        
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
                # Middle segments: 
                # Skip overlap region at start (already blended in previous iteration)
                # and blend with next segment at the end
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
        Supports both linear and smooth crossfade blending modes.
        
        Args:
            prev_frames: Last N frames from previous segment [C, N, H, W]
            curr_frames: First N frames from current segment [C, N, H, W]
            
        Returns:
            Blended frames [C, N, H, W]
        """
        overlap = min(prev_frames.shape[1], curr_frames.shape[1])
        
        if overlap == 0:
            return curr_frames
        
        if self.blend_mode == "lerp":
            # Linear interpolation with smooth start and end
            weights = torch.linspace(0, 1, overlap + 2)[1:-1].to(self.device)
            weights = weights.view(1, -1, 1, 1)
            
            blended = (1 - weights) * prev_frames[:, -overlap:, :, :] + \
                     weights * curr_frames[:, :overlap, :, :]
        
        elif self.blend_mode == "crossfade":
            # Smooth crossfade using raised cosine (Hann window style)
            alpha = torch.arange(overlap).float().to(self.device) / max(overlap - 1, 1)
            alpha = (1 - torch.cos(alpha * math.pi)) / 2
            alpha = alpha.view(1, -1, 1, 1)
            
            blended = (1 - alpha) * prev_frames[:, -overlap:, :, :] + \
                     alpha * curr_frames[:, :overlap, :, :]
        
        elif self.blend_mode == "sigmoid":
            # Sigmoid-based smooth blending for more natural transitions
            alpha = torch.arange(overlap).float().to(self.device)
            # Scale sigmoid to cover -6 to 6 range for smooth transition
            alpha = 6 * (2 * alpha / (overlap - 1) - 1) if overlap > 1 else torch.zeros(overlap)
            alpha = torch.sigmoid(alpha)
            alpha = alpha.view(1, -1, 1, 1)
            
            blended = (1 - alpha) * prev_frames[:, -overlap:, :, :] + \
                     alpha * curr_frames[:, :overlap, :, :]
        
        else:
            blended = curr_frames[:, :overlap, :, :]
        
        return blended
    
    def generate_sliding_window_with_latent_stitching(
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
        sample_solver: str = "unipc",
        use_autoregressive: bool = True,
        seed_mode: str = "uniform",
    ) -> torch.Tensor:
        """
        Generate long video using sliding window with LATENT SPACE stitching.
        This method produces better quality by blending in latent space before decoding.
        
        NOTE: This requires WanI2V to support latent output. Falls back to pixel stitching
        if not available.
        
        Args:
            Same as generate_sliding_window
            
        Returns:
            Generated video tensor [C, N, H, W]
        """
        # Check if WanI2V supports latent output
        if not hasattr(self.wan_i2v, 'generate_with_latent'):
            self.logger.warning(
                "WanI2V does not support latent output. "
                "Falling back to pixel-space stitching."
            )
            return self.generate_sliding_window(
                img=img,
                input_prompt=input_prompt,
                frame_num=frame_num,
                max_area=max_area,
                guide_scale=guide_scale,
                seed=seed,
                n_prompt=n_prompt,
                action_path=action_path,
                sample_steps=sample_steps,
                sample_shift=sample_shift,
                sample_solver=sample_solver,
                use_autoregressive=use_autoregressive,
                seed_mode=seed_mode,
            )
        
        # Calculate number of windows
        stride = self.window_size - self.overlap_size
        num_windows = (frame_num - 1 + stride - 1) // stride + 1
        
        self.logger.info(f"Generating {frame_num} frames using sliding window WITH LATENT STITCHING")
        self.logger.info(f"  Window size: {self.window_size}")
        self.logger.info(f"  Overlap: {self.overlap_size}")
        self.logger.info(f"  Number of windows: {num_windows}")
        self.logger.info(f"  Autoregressive mode: {use_autoregressive}")
        self.logger.info(f"  Seed mode: {seed_mode}")
        
        all_latents = []
        camera_poses = None
        current_input_img = img
        
        if action_path is not None:
            camera_poses = np.load(os.path.join(action_path, "poses.npy"))
        
        # Process each window and collect latents
        for window_idx in tqdm(range(num_windows), desc="Processing windows (Latent)"):
            window_start, window_end = self._get_window_frames(frame_num, window_idx)
            window_frames = window_end - window_start
            
            current_window_size = min(self.window_size, window_frames)
            if (current_window_size - 1) % 4 != 0:
                current_window_size = ((current_window_size - 1) // 4) * 4 + 1
            
            window_action_path = None
            if action_path is not None and camera_poses is not None:
                window_action_path = self._create_window_action_path(
                    action_path, window_start, window_end, camera_poses
                )
            
            # Calculate seed
            if seed >= 0:
                if seed_mode == "uniform":
                    window_seed = seed
                elif seed_mode == "incremental":
                    window_seed = seed + window_idx
                elif seed_mode == "position_based":
                    window_seed = seed + window_start // max(self.window_size, 1)
                else:
                    window_seed = seed
            else:
                window_seed = -1
            
            # Generate latents for this window
            window_latent = self.wan_i2v.generate_with_latent(
                img=current_input_img,
                input_prompt=input_prompt,
                frame_num=current_window_size,
                max_area=max_area,
                guide_scale=guide_scale,
                seed=window_seed,
                n_prompt=n_prompt,
                action_path=window_action_path,
                sampling_steps=sample_steps,
                shift=sample_shift,
                sample_solver=sample_solver,
            )
            
            if window_latent is not None:
                all_latents.append(window_latent)
                
                if use_autoregressive and window_idx < num_windows - 1:
                    # Decode last latent frame to get pixel image for next window
                    last_latent = window_latent[:, -1:, :, :]  # [C, 1, H, W]
                    # Attempt to decode (this may vary by implementation)
                    if hasattr(self.wan_i2v, 'vae_decode'):
                        last_frame = self.wan_i2v.vae_decode(last_latent)
                    else:
                        # Fallback: use pixel stitching instead
                        self.logger.warning("Cannot decode latent for autoregressive continuation")
                        use_autoregressive = False
                    
                    if use_autoregressive:
                        from torchvision.transforms.functional import to_pil_image
                        current_input_img = to_pil_image(last_frame[0])
            
            if window_action_path is not None and os.path.exists(window_action_path):
                import shutil
                shutil.rmtree(window_action_path)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # Stitch latents and decode
        if len(all_latents) > 0:
            stitched_latent = self._stitch_latents(all_latents)
            # Decode stitched latent to pixel space
            final_video = self.wan_i2v.vae_decode(stitched_latent)
            
            if final_video.shape[1] > frame_num:
                final_video = final_video[:, :frame_num, :, :]
            
            return final_video
        else:
            return None
    
    def get_recommended_config(self, total_frames: int, available_memory_gb: float = 8.0) -> dict:
        """
        Get recommended sliding window configuration based on video length and available memory.
        
        Args:
            total_frames: Total number of frames to generate
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Dictionary with recommended settings
        """
        config = {}
        
        # Recommend window size based on available memory
        if available_memory_gb >= 24:
            config['window_size'] = 81
            config['overlap_size'] = 40
            config['blend_mode'] = 'sigmoid'
            config['seed_mode'] = 'uniform'
            config['use_autoregressive'] = True
        elif available_memory_gb >= 16:
            config['window_size'] = 65
            config['overlap_size'] = 32
            config['blend_mode'] = 'crossfade'
            config['seed_mode'] = 'uniform'
            config['use_autoregressive'] = True
        elif available_memory_gb >= 12:
            config['window_size'] = 49
            config['overlap_size'] = 24
            config['blend_mode'] = 'crossfade'
            config['seed_mode'] = 'position_based'
            config['use_autoregressive'] = True
        else:
            config['window_size'] = 49
            config['overlap_size'] = 16
            config['blend_mode'] = 'lerp'
            config['seed_mode'] = 'position_based'
            config['use_autoregressive'] = False
        
        # Calculate estimated number of windows
        stride = config['window_size'] - config['overlap_size']
        num_windows = (total_frames - 1 + stride - 1) // stride + 1
        
        config['num_windows'] = num_windows
        config['estimated_unique_frames'] = total_frames
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"RECOMMENDED CONFIG for {total_frames} frames, {available_memory_gb}GB memory:")
        self.logger.info(f"  Window size: {config['window_size']} frames")
        self.logger.info(f"  Overlap size: {config['overlap_size']} frames")
        self.logger.info(f"  Blend mode: {config['blend_mode']}")
        self.logger.info(f"  Seed mode: {config['seed_mode']}")
        self.logger.info(f"  Autoregressive: {config['use_autoregressive']}")
        self.logger.info(f"  Number of windows: {num_windows}")
        self.logger.info(f"{'='*60}\n")
        
        return config
    
    def set_blend_mode(self, blend_mode: str):
        """
        Change the blending mode for overlap regions.
        
        Args:
            blend_mode: 'lerp' (linear, fastest),
                       'crossfade' (smooth, recommended),
                       'sigmoid' (smoothest, slightly slower)
        """
        if blend_mode not in ['lerp', 'crossfade', 'sigmoid']:
            raise ValueError(f"Invalid blend_mode: {blend_mode}")
        self.blend_mode = blend_mode
        self.logger.info(f"Blend mode changed to: {blend_mode}")
    
    def get_config_info(self) -> str:
        """Get information about current configuration."""
        return f"""
SlidingWindowI2V Configuration:
  - Window size: {self.window_size} frames (form 4n+1)
  - Overlap size: {self.overlap_size} frames ({self.overlap_size/self.window_size*100:.1f}% of window)
  - Stride: {self.window_size - self.overlap_size} frames
  - Blend mode: {self.blend_mode}
  - Device: {self.device}

Key Parameters for Long Video Generation:
  1. Use autoregressive=True for better temporal consistency
  2. For very long videos (>500 frames), use latent_stitching_mode=True if available
  3. Increase overlap_size for smoother transitions (but slower generation)
  4. Use seed_mode='uniform' for consistency across windows
  5. Use seed_mode='position_based' for more natural diversity
"""
