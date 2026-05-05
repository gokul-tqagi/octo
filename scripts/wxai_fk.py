# wxai_fk.py: Forward kinematics for WidowX AI / Mobile AI right arm.
# wxai_fk.py: Computes EEF pose from 6 joint angles using URDF-derived transforms.

"""
Forward kinematics for the WidowX AI (wxai) 6-DOF arm.
Derived from TrossenRobotics/trossen_arm_description mobile_ai.urdf.

Kinematic chain (right arm): base_link -> link_1 -> ... -> link_6 -> ee_gripper
Joint axes: Z, Y, -Y, -Y, -Z, X

Returns EEF pose as [x, y, z, roll, pitch, yaw] in the arm base frame.
Optionally includes the mobile base mount offset to get world-frame pose.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# Joint definitions from mobile_ai.urdf (follower_right_*)
# Each entry: (origin_xyz, origin_rpy, axis, joint_type)
# All origins are in parent link frame.
WXAI_JOINTS = [
    # joint_0: base_link -> link_1, rotation about Z
    {"origin_xyz": [0.0, 0.0, 0.05725], "axis": [0, 0, 1]},
    # joint_1: link_1 -> link_2, rotation about Y
    {"origin_xyz": [0.02, 0.0, 0.04625], "axis": [0, 1, 0]},
    # joint_2: link_2 -> link_3, rotation about -Y
    {"origin_xyz": [-0.264, 0.0, 0.0], "axis": [0, -1, 0]},
    # joint_3: link_3 -> link_4, rotation about -Y
    {"origin_xyz": [0.245, 0.0, 0.06], "axis": [0, -1, 0]},
    # joint_4: link_4 -> link_5, rotation about -Z
    {"origin_xyz": [0.06775, 0.0, 0.0455], "axis": [0, 0, -1]},
    # joint_5: link_5 -> link_6, rotation about X
    {"origin_xyz": [0.02895, 0.0, -0.0455], "axis": [1, 0, 0]},
]

# Fixed transform from link_6 to ee_gripper
EE_OFFSET = np.array([0.156062, 0.0, 0.0])

# Mobile AI right arm mount offset (base_link -> follower_right_base_link)
RIGHT_ARM_MOUNT = np.array([0.331, -0.3, 0.831])


def _rotation_matrix(axis, angle):
    """Rotation matrix for rotation about an arbitrary axis by angle (radians)."""
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def _homogeneous(R, t):
    """Build 4x4 homogeneous transform from 3x3 rotation and 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def forward_kinematics(joint_angles, include_mount=False):
    """Compute EEF pose from 6 joint angles.

    Args:
        joint_angles: array of 6 joint angles in radians [j0, j1, j2, j3, j4, j5]
        include_mount: if True, include the mobile base mount offset for world-frame pose

    Returns:
        eef_pos: (3,) xyz position in meters
        eef_rot: (3, 3) rotation matrix
    """
    assert len(joint_angles) == 6, f"Expected 6 joint angles, got {len(joint_angles)}"

    # Start from arm base frame
    T = np.eye(4)

    if include_mount:
        T[:3, 3] = RIGHT_ARM_MOUNT

    # Chain through each joint
    for i, jdef in enumerate(WXAI_JOINTS):
        # Fixed transform to joint origin
        t_origin = np.array(jdef["origin_xyz"])
        T_origin = _homogeneous(np.eye(3), t_origin)

        # Joint rotation
        R_joint = _rotation_matrix(jdef["axis"], joint_angles[i])
        T_joint = _homogeneous(R_joint, np.zeros(3))

        T = T @ T_origin @ T_joint

    # Add fixed EE offset
    T_ee = _homogeneous(np.eye(3), EE_OFFSET)
    T = T @ T_ee

    eef_pos = T[:3, 3]
    eef_rot = T[:3, :3]
    return eef_pos, eef_rot


def forward_kinematics_euler(joint_angles, include_mount=False):
    """Compute EEF pose as [x, y, z, roll, pitch, yaw].

    Args:
        joint_angles: array of 6 joint angles in radians
        include_mount: include mobile base mount offset

    Returns:
        (6,) array: [x, y, z, roll, pitch, yaw] in meters and radians
    """
    pos, rot = forward_kinematics(joint_angles, include_mount)
    euler = Rotation.from_matrix(rot).as_euler("xyz")
    return np.concatenate([pos, euler]).astype(np.float32)


def forward_kinematics_quat(joint_angles, include_mount=False):
    """Compute EEF pose as [x, y, z, qx, qy, qz, qw].

    Args:
        joint_angles: array of 6 joint angles in radians
        include_mount: include mobile base mount offset

    Returns:
        (7,) array: [x, y, z, qx, qy, qz, qw]
    """
    pos, rot = forward_kinematics(joint_angles, include_mount)
    quat = Rotation.from_matrix(rot).as_quat()  # [x, y, z, w]
    return np.concatenate([pos, quat]).astype(np.float32)


def batch_fk_euler(joint_angles_batch, include_mount=False):
    """Batch FK: (N, 6) joint angles -> (N, 6) EEF poses [x,y,z,r,p,y]."""
    return np.stack([
        forward_kinematics_euler(ja, include_mount) for ja in joint_angles_batch
    ])


def compute_delta_eef(eef_poses):
    """Compute delta EEF actions from consecutive EEF poses.

    Uses quaternion-space rotation deltas to avoid euler wrapping.

    Args:
        eef_poses: (T, 6) array of [x, y, z, roll, pitch, yaw]

    Returns:
        (T-1, 6) array of [dx, dy, dz, droll, dpitch, dyaw]
    """
    deltas = []
    for i in range(len(eef_poses) - 1):
        # Position delta
        dpos = eef_poses[i + 1, :3] - eef_poses[i, :3]

        # Rotation delta via quaternion relative rotation
        r_curr = Rotation.from_euler("xyz", eef_poses[i, 3:6])
        r_next = Rotation.from_euler("xyz", eef_poses[i + 1, 3:6])
        r_delta = r_next * r_curr.inv()
        drot = r_delta.as_euler("xyz").astype(np.float32)

        deltas.append(np.concatenate([dpos, drot]))

    return np.stack(deltas).astype(np.float32)


if __name__ == "__main__":
    # Quick sanity check: home position and a few test poses
    home = [0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0]
    pos, rot = forward_kinematics(home)
    euler = forward_kinematics_euler(home)

    print("=== WidowX AI FK Sanity Check ===")
    print(f"Home joints (rad): {[f'{j:.3f}' for j in home]}")
    print(f"EEF position (m):  {pos}")
    print(f"EEF euler (rad):   {euler[3:]}")
    print(f"EEF euler (deg):   {np.degrees(euler[3:])}")
    print()

    # Zero position
    zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    euler_zero = forward_kinematics_euler(zero)
    print(f"Zero joints -> EEF: {euler_zero[:3]} (should be straight up)")
    print()

    # With mount offset
    euler_mount = forward_kinematics_euler(home, include_mount=True)
    print(f"Home + mount -> EEF: {euler_mount[:3]}")
    print(f"Mount offset applied: {RIGHT_ARM_MOUNT}")
