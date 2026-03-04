import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import merge_video_audio, save_video, str2bool


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "The video presents a cinematic, first-person wandering experience through a hyper-realistic urban environment rendered in a video game engine. It begins with a static, sun-drenched alley framed by graffiti-laden industrial walls and overhead power lines, immediately establishing a gritty, lived-in atmosphere. As the camera pans right and tilts upward, it reveals a sprawling cityscape dominated by towering skyscrapers and industrial infrastructure, all bathed in warm, late-afternoon light that casts long shadows and produces dramatic lens flares. The perspective then transitions into a smooth forward tracking shot along a cracked sidewalk, passing weathered fences, palm trees, and distant pedestrians, creating a sense of immersion and exploration. Midway, the camera briefly follows a walking figure before refocusing on the broader streetscape, culminating in a stabilized view of a small blue van parked at an intersection surrounded by urban elements like parking garages and traffic lights. The entire sequence is characterized by its photorealistic detail, dynamic lighting, and deliberate pacing, evoking the feel of a quiet, sunlit afternoon in a futuristic metropolis.",
        "image":
            "examples/02/image.jpg",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--use_sliding_window",
        action="store_true",
        default=False,
        help="Whether to use sliding window mechanism for long video generation. This allows generation of long videos with bounded memory usage.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=49,
        help="Size of sliding window in frames (must be 4n+1). Default: 49")
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=None,
        help="Size of overlap between windows in frames. Default: window_size // 2")
    parser.add_argument(
        "--blend_mode",
        type=str,
        default="lerp",
        choices=["lerp", "crossfade"],
        help="Blending mode for overlapping regions: 'lerp' (linear interpolation) or 'crossfade' (smooth crossfade). Default: lerp")
    parser.add_argument(
        "--save_intermediate_dir",
        type=str,
        default=None,
        help="Directory for saving intermediate diffusion results.")
    parser.add_argument(
        "--save_latents",
        type=str2bool,
        default=True,
        help="Whether to save latent tensors for each diffusion step when --save_intermediate_dir is set.")
    parser.add_argument(
        "--save_decoded",
        type=str2bool,
        default=False,
        help="Whether to decode and save RGB frames for each diffusion step when --save_intermediate_dir is set.")
    parser.add_argument(
        "--save_camera_matrices",
        type=str2bool,
        default=False,
        help="Whether to save per-frame camera matrices to an npz file.")
    parser.add_argument(
        "--camera_matrices_file",
        type=str,
        default=None,
        help="Optional output npz path for per-frame camera matrices. Default: <save_file>.camera_matrices.npz")
    parser.add_argument(
        "--use_pair_hwmirror",
        type=str2bool,
        default=False,
        help="Whether to use pair-wise (2 poses per group) HunyuanWorld-Mirror complementary guidance flow.")
    parser.add_argument(
        "--hwm_repo",
        type=str,
        default="../HunyuanWorld-Mirror",
        help="Path to HunyuanWorld-Mirror repository for pair-wise guidance flow.")
    parser.add_argument(
        "--hwm_use_subprocess",
        type=str2bool,
        default=True,
        help="Run HunyuanWorld-Mirror rendering via subprocess so Lingbot and HWM can use separate conda envs.")
    parser.add_argument(
        "--hwm_conda_env",
        type=str,
        default=None,
        help="Conda env name used when --hwm_use_subprocess is true (example: hunyuanworld-mirror).")
    parser.add_argument(
        "--hwm_python",
        type=str,
        default=None,
        help="Optional Python executable path for HWM subprocess. If set, it overrides --hwm_conda_env.")
    parser.add_argument(
        "--guidance_fft_radius",
        type=int,
        default=10,
        help="FFT radius used for low/high frequency complementary filtering when --use_pair_hwmirror is enabled.")
    
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _save_frame_camera_matrices(action_path, save_npz_path, frame_count):
    if action_path is None:
        raise ValueError("action_path is required when saving camera matrices")

    poses_path = os.path.join(action_path, "poses.npy")
    intrinsics_path = os.path.join(action_path, "intrinsics.npy")

    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"poses.npy not found: {poses_path}")
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"intrinsics.npy not found: {intrinsics_path}")

    c2ws = np.load(poses_path)
    Ks = np.load(intrinsics_path)

    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape [N,4,4], got {c2ws.shape}")

    if Ks.ndim == 2:
        Ks = np.repeat(Ks[None, ...], repeats=c2ws.shape[0], axis=0)
    elif Ks.ndim != 3 or Ks.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape [3,3] or [N,3,3], got {Ks.shape}")

    min_len = min(c2ws.shape[0], Ks.shape[0])
    c2ws = c2ws[:min_len]
    Ks = Ks[:min_len]

    if min_len == 0:
        raise ValueError("No valid camera poses/intrinsics found")

    if min_len < frame_count:
        pad_num = frame_count - min_len
        c2ws = np.concatenate([c2ws, np.repeat(c2ws[-1:], repeats=pad_num, axis=0)], axis=0)
        Ks = np.concatenate([Ks, np.repeat(Ks[-1:], repeats=pad_num, axis=0)], axis=0)
    else:
        c2ws = c2ws[:frame_count]
        Ks = Ks[:frame_count]

    np.savez(
        save_npz_path,
        frame_idx=np.arange(frame_count, dtype=np.int32),
        c2w=c2ws.astype(np.float32),
        K=Ks.astype(np.float32),
    )


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    if args.use_pair_hwmirror and args.action_path is None:
        raise ValueError("action_path is required when use_pair_hwmirror is enabled")
    if args.use_pair_hwmirror and args.hwm_use_subprocess and args.hwm_conda_env is None and args.hwm_python is None:
        raise ValueError("When use_pair_hwmirror and hwm_use_subprocess are enabled, provide hwm_conda_env or hwm_python")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            input_prompt = args.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")
    
    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info("Generating video ...")
    
    if args.use_pair_hwmirror:
        from generate_pair_hwmirror import generate_pair_hwmirror_video
        video = generate_pair_hwmirror_video(
            pipeline=wan_i2v,
            cfg=cfg,
            args=args,
        )
    elif args.use_sliding_window:
        # Use sliding window for long video generation
        logging.info(f"Using sliding window mechanism for long video generation")
        logging.info(f"  Window size: {args.window_size}")
        if args.overlap_size is None:
            args.overlap_size = args.window_size // 2
        logging.info(f"  Overlap size: {args.overlap_size}")
        logging.info(f"  Blend mode: {args.blend_mode}")
        
        from wan.sliding_window_i2v import SlidingWindowI2V
        sliding_window_i2v = SlidingWindowI2V(
            wan_i2v=wan_i2v,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            blend_mode=args.blend_mode,
        )
        
        video = sliding_window_i2v.generate_sliding_window(
            img=img,
            input_prompt=args.prompt,
            frame_num=args.frame_num,
            max_area=MAX_AREA_CONFIGS[args.size],
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            n_prompt="",
            action_path=args.action_path,
            sample_steps=args.sample_steps,
            sample_shift=args.sample_shift,
            sample_guide_scale=args.sample_guide_scale,
            sample_solver=args.sample_solver,
            offload_model=args.offload_model,
            save_intermediate_dir=args.save_intermediate_dir,
            save_latents=args.save_latents,
            save_decoded=args.save_decoded)
    else:
        # Standard generation
        video = wan_i2v.generate(
            args.prompt,
            img,
            action_path=args.action_path,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            save_intermediate_dir=args.save_intermediate_dir,
            save_latents=args.save_latents,
            save_decoded=args.save_decoded)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

        if args.save_camera_matrices:
            if args.action_path is None:
                logging.warning("save_camera_matrices is enabled but action_path is None, skip camera export")
            else:
                frame_count = int(video.shape[1])
                camera_file = args.camera_matrices_file
                if camera_file is None:
                    camera_file = f"{args.save_file}.camera_matrices.npz"
                _save_frame_camera_matrices(
                    action_path=args.action_path,
                    save_npz_path=camera_file,
                    frame_count=frame_count)
                logging.info(f"Saved per-frame camera matrices to {camera_file}")

        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
