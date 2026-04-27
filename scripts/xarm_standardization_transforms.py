# xarm_standardization_transforms.py: RLDS standardization transform for xarm datasets.
# xarm_standardization_transforms.py: Maps xarm bag-extracted keys to Octo's internal format.

"""
Standardization transform for the xarm_place_toolbox RLDS dataset.

This maps raw dataset keys to Octo's expected format:
- action: (T, 7) float32 delta EEF [dx, dy, dz, droll, dpitch, dyaw, dgripper]
- observation/proprio: (T, 7) float32 EEF state [x, y, z, roll, pitch, yaw, gripper]
- language_instruction: str

Reference: octo/data/oxe/oxe_standardization_transforms.py
"""

from typing import Any, Dict

import tensorflow as tf


def xarm_place_toolbox_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardization transform for xarm place-object-in-toolbox dataset.

    The dataset already has actions and state in the correct format from
    mcap_to_rlds.py extraction, so this transform just maps the proprio key.
    """
    # Map 'state' to 'proprio' for Octo's observation tokenizer
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory
