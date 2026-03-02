import torch
import torchvision
from pathlib import Path


class IntermediateResultSaver:
    """Save intermediate diffusion results per step."""

    def __init__(self, save_dir, save_latents=True, save_decoded=False, vae=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_latents = save_latents
        self.save_decoded = save_decoded
        self.vae = vae

        if save_decoded and vae is None:
            raise ValueError("VAE decoder is required when save_decoded=True")

    def save_step_results(self, latents, step, step_idx, frame_num, vae_stride, decoded_vae_input=None):
        step_dir = self.save_dir / f"step_{step_idx:03d}_t{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if self.save_latents:
            torch.save(latents.detach().cpu(), step_dir / "latents.pt")

        if self.save_decoded:
            self._save_decoded_frames(
                latents=latents,
                step_dir=step_dir,
                frame_num=frame_num,
                vae_stride=vae_stride,
                decoded_vae_input=decoded_vae_input,
            )

    def _save_decoded_frames(self, latents, step_dir, frame_num, vae_stride, decoded_vae_input=None):
        if decoded_vae_input is not None:
            video = decoded_vae_input
        else:
            with torch.no_grad():
                video = self.vae.decode([latents])[0]

        video = video.detach().cpu()
        valid_frame_num = min(frame_num, video.shape[1])
        for frame_idx in range(valid_frame_num):
            frame = video[:, frame_idx]
            frame_path = step_dir / f"frame_{frame_idx:03d}.png"
            torchvision.utils.save_image(
                frame,
                str(frame_path),
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

    def save_final_result(self, video, filename="final_video.pt"):
        torch.save(video.detach().cpu(), self.save_dir / filename)

    def get_summary(self):
        steps = [d for d in self.save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
        steps.sort(key=lambda x: int(x.name.split("_")[1]))
        print(f"\nIntermediate Results Summary")
        print(f"Total steps saved: {len(steps)}")
        print(f"Save directory: {self.save_dir}")
        print(f"Latents saved: {self.save_latents}")
        print(f"Decoded frames saved: {self.save_decoded}")
