import base64
import io
import json
import urllib.request
from typing import Optional

import numpy as np
from PIL import Image


class WorldMirrorServiceClient:
    def __init__(self, service_url: str, timeout: int = 120):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout

    def build_scene(self, image: Image.Image) -> str:
        payload = {
            "image_base64": _pil_to_base64(image),
        }
        result = self._post_json("/build_3dgs", payload)
        scene_id = result.get("scene_id")
        if not scene_id:
            raise RuntimeError(f"invalid build_3dgs response: {result}")
        return scene_id

    def render_pose(
        self,
        scene_id: str,
        pose: np.ndarray,
        intrinsics: Optional[np.ndarray],
        width: int,
        height: int,
    ) -> Image.Image:
        payload = {
            "scene_id": scene_id,
            "pose": pose.tolist(),
            "intrinsics": None if intrinsics is None else intrinsics.tolist(),
            "width": int(width),
            "height": int(height),
        }
        result = self._post_json("/render_pose", payload)
        image_b64 = result.get("image_base64")
        if image_b64 is None:
            raise RuntimeError(f"invalid render_pose response: {result}")
        return _base64_to_pil(image_b64)

    def predict_pose(self, image: Image.Image):
        payload = {
            "image_base64": _pil_to_base64(image),
        }
        return self._post_json("/predict_pose", payload)

    def _post_json(self, endpoint: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.service_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)


def _pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _base64_to_pil(text: str) -> Image.Image:
    raw = base64.b64decode(text.encode("utf-8"))
    return Image.open(io.BytesIO(raw)).convert("RGB")
