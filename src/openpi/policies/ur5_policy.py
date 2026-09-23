"""UR5 policy inputs/outputs for openpi.

Adapted for the A1astor/ur5_gamepad dataset: 6-DoF UR5 (absolute joint actions,
no gripper dimension), 2 cameras (external "global" + "wrist").
"""

import dataclasses

import einops
import numpy as np

from openpi.models import model as _model
from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """Maps the ur5_gamepad dataset / inference observations to the model format.

    State: 6 joint positions (rad). Actions: 6 absolute joint targets (rad).
    Cameras: base_0_rgb = external "global", left_wrist_0_rgb = "wrist".
    The unused right_wrist_0_rgb slot is zero-filled and masked out.
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["joints"], dtype=np.float32)

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ClampActions(transforms.DataTransformFn):
    """Clips normalized action targets to ±limit, sanitizing episode-boundary artifacts."""

    limit: float = 6.0

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            data["actions"] = np.clip(np.asarray(data["actions"], dtype=np.float32), -self.limit, self.limit)
        return data


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """Returns the first `action_dim` action dimensions (absolute joint targets in rad, optionally gripper)."""

    action_dim: int = 6

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
