import argparse
import os
import sys
import tempfile
import json
import contextlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _load_hwm_app(hwm_repo: str):
    repo_path = str(Path(hwm_repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    import app as hwm_app
    return hwm_app


def _render_single(hwm_app, input_image, pose, intr_mat, target_h, target_w, output_png):
    tmp_root = tempfile.mkdtemp(prefix="hwm_bridge_")
    try:
        image_dir = Path(tmp_root) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        src_img = Image.open(input_image).convert("RGB")
        src_path = image_dir / "image_0001.png"
        src_img.save(src_path)

        with contextlib.redirect_stdout(sys.stderr):
            outputs, _ = hwm_app.run_model(
                target_dir=tmp_root,
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
        viewmats = torch.from_numpy(pose).to(device)[None, None]
        Ks = torch.from_numpy(intr_mat).to(device)[None, None]

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
                width=int(target_w),
                height=int(target_h),
                sh_degree=min(model.gs_renderer.sh_degree, 0) if "sh" in splats else None,
            )

        rendered = rendered_colors[0, 0].permute(2, 0, 1).detach().cpu().clamp(0, 1)
        arr = (rendered.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(arr).save(output_png)
    finally:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)


def _canonical_intrinsics(intr):
    if intr.shape == (4,):
        fx, fy, cx, cy = intr.tolist()
        intr_mat = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    elif intr.shape == (3, 3):
        intr_mat = intr
    else:
        raise ValueError(f"intrinsics must be [4] or [3,3], got {intr.shape}")
    return intr_mat.astype(np.float32)


def _run_server(hwm_repo: str):
    hwm_app = _load_hwm_app(hwm_repo)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "shutdown":
                print(json.dumps({"ok": True, "msg": "shutdown"}), flush=True)
                break

            pose = np.load(req["pose_file"]).astype(np.float32)
            intr = np.load(req["intrinsics_file"]).astype(np.float32)
            if pose.shape != (4, 4):
                raise ValueError(f"pose must be [4,4], got {pose.shape}")
            intr_mat = _canonical_intrinsics(intr)

            _render_single(
                hwm_app=hwm_app,
                input_image=req["input_image"],
                pose=pose,
                intr_mat=intr_mat,
                target_h=int(req["target_h"]),
                target_w=int(req["target_w"]),
                output_png=req["output_png"],
            )
            print(json.dumps({"ok": True, "output_png": req["output_png"]}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Render one pose image from HunyuanWorld-Mirror 3DGS via app.py")
    parser.add_argument("--server", action="store_true", help="Run persistent JSON-line request server mode.")
    parser.add_argument("--hwm_repo", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--pose_file", type=str, required=True)
    parser.add_argument("--intrinsics_file", type=str, required=True)
    parser.add_argument("--target_h", type=int, required=True)
    parser.add_argument("--target_w", type=int, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    args = parser.parse_args()

    if args.server:
        _run_server(args.hwm_repo)
        return

    pose = np.load(args.pose_file).astype(np.float32)
    intr = np.load(args.intrinsics_file).astype(np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be [4,4], got {pose.shape}")
    intr_mat = _canonical_intrinsics(intr)
    hwm_app = _load_hwm_app(args.hwm_repo)
    _render_single(
        hwm_app=hwm_app,
        input_image=args.input_image,
        pose=pose,
        intr_mat=intr_mat,
        target_h=args.target_h,
        target_w=args.target_w,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()
