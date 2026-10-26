"""Visual Trace overlay and camera-to-world constraint conversion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import mujoco


@dataclass(frozen=True)
class WorldTraceConstraint:
    start_uv: list[float]
    end_uv: list[float]
    endpoint_world_m: list[float]
    projection_plane_z_m: float
    camera: str


@dataclass(frozen=True)
class CropTransform:
    """Pixel crop in the original camera frame."""
    left: int
    top: int
    right: int
    bottom: int
    full_width: int
    full_height: int

    def crop_uv_to_full_uv(self, uv: list[float]) -> list[float]:
        x = self.left + float(uv[0]) * max(1, self.right - self.left - 1)
        y = self.top + float(uv[1]) * max(1, self.bottom - self.top - 1)
        return [x / (self.full_width - 1), y / (self.full_height - 1)]

    def full_uv_to_crop_uv(self, uv: list[float]) -> list[float]:
        x = float(uv[0]) * (self.full_width - 1)
        y = float(uv[1]) * (self.full_height - 1)
        return [float(np.clip((x - self.left) / max(1, self.right-self.left-1), 0, 1)),
                float(np.clip((y - self.top) / max(1, self.bottom-self.top-1), 0, 1))]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def task_crop(hand_uv: list[float], target_uv: list[float],
              full_size: tuple[int, int], padding_px: int = 90) -> CropTransform:
    """Build a square ROI containing both the measured hand and target."""
    width, height = full_size
    points = np.array([hand_uv, target_uv]) * np.array([width - 1, height - 1])
    # Target-centred attention crop: the live hand stays visible, while the
    # manipulation target has a stable visual location near the ROI centre.
    centre = points[1]
    side = int(max(2 * abs(points[0] - points[1]).max() + 2 * padding_px, 256))
    side = min(side, width, height)
    left = int(np.clip(round(centre[0] - side / 2), 0, width - side))
    top = int(np.clip(round(centre[1] - side / 2), 0, height - side))
    return CropTransform(left, top, left + side, top + side, width, height)


def crop_for_vlm(full_image: Path, crop: CropTransform, output: Path,
                 model_size: tuple[int, int] = (384, 384)) -> Path:
    image = Image.open(full_image).convert("RGB")
    image = image.crop((crop.left, crop.top, crop.right, crop.bottom))
    image = image.resize(model_size, Image.Resampling.BILINEAR)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def world_point_to_uv(model: Any, data: Any, camera_name: str,
                      point_world: np.ndarray,
                      image_size: tuple[int, int]) -> list[float]:
    """Project a trusted MuJoCo world point to normalized image coordinates."""
    camera_id = model.camera(camera_name).id
    width, height = image_size
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    point_camera = rotation.T @ (np.asarray(point_world) - data.cam_xpos[camera_id])
    depth = -point_camera[2]
    if depth <= 1e-7:
        raise ValueError("world point is behind the camera")
    focal = .5 * height / np.tan(np.deg2rad(float(model.cam_fovy[camera_id])) / 2.)
    u = .5 * (width - 1) + focal * point_camera[0] / depth
    v = .5 * (height - 1) - focal * point_camera[1] / depth
    return (np.array([u / (width - 1), v / (height - 1)]).clip(0, 1)).tolist()


def overlay_trace(image_path: Path, start_uv: list[float], end_uv: list[float],
                  output_path: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    point = lambda uv: (int(np.clip(uv[0], 0, 1) * (width - 1)),
                        int(np.clip(uv[1], 0, 1) * (height - 1)))
    start, end = point(start_uv), point(end_uv)
    draw.line([start, end], fill=(0, 255, 80), width=max(4, width // 120))
    radius = max(7, width // 75)
    draw.ellipse([end[0]-radius, end[1]-radius, end[0]+radius, end[1]+radius],
                 outline=(255, 40, 40), fill=(255, 220, 0), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def trace_endpoint_to_world(model: Any, data: Any,
                            camera_name: str, end_uv: list[float],
                            image_size: tuple[int, int], plane_z_m: float,
                            start_uv: list[float] | None = None) -> WorldTraceConstraint:
    """Back-project a normalized pixel ray onto a known horizontal task plane.

    The plane height is supplied by trusted MuJoCo state (normally the top/centre
    height of the current target), never guessed by the VLM.
    """
    camera = model.camera(camera_name)
    camera_id = camera.id
    width, height = image_size
    u, v = np.asarray(end_uv, dtype=float) * np.array([width - 1, height - 1])
    focal = .5 * height / np.tan(np.deg2rad(float(model.cam_fovy[camera_id])) / 2.)
    ray_camera = np.array([(u - .5 * (width - 1)) / focal,
                           -((v - .5 * (height - 1)) / focal), -1.])
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    ray_world = rotation @ ray_camera
    origin = data.cam_xpos[camera_id].copy()
    if abs(ray_world[2]) < 1e-7:
        raise ValueError("visual trace ray is parallel to the projection plane")
    distance = (float(plane_z_m) - origin[2]) / ray_world[2]
    if distance <= 0:
        raise ValueError("visual trace endpoint projects behind the camera")
    endpoint = origin + distance * ray_world
    return WorldTraceConstraint(
        start_uv=list(start_uv or end_uv), end_uv=list(end_uv),
        endpoint_world_m=endpoint.tolist(), projection_plane_z_m=float(plane_z_m),
        camera=camera_name)
