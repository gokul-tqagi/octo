# mcap_dataloader.py: Loads xarm mcap ROS2 bag trajectories and extracts
# images, EEF poses, gripper positions, and computes delta actions for Octo inference.

import os
import glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def quat_to_euler(qx, qy, qz, qw):
    """Convert quaternion (x, y, z, w) to euler angles (roll, pitch, yaw)."""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_euler("xyz")


def compute_delta_action(pose_curr, pose_next, gripper_next, gripper_open=0.0, gripper_closed=0.85):
    """Compute 7-dim delta action: [dx, dy, dz, droll, dpitch, dyaw, gripper_normalized].

    Gripper is normalized to [-1, 1] where -1 = open, 1 = closed.
    """
    pos_curr = np.array([
        pose_curr.pose.position.x,
        pose_curr.pose.position.y,
        pose_curr.pose.position.z,
    ])
    pos_next = np.array([
        pose_next.pose.position.x,
        pose_next.pose.position.y,
        pose_next.pose.position.z,
    ])
    delta_pos = pos_next - pos_curr

    euler_curr = quat_to_euler(
        pose_curr.pose.orientation.x, pose_curr.pose.orientation.y,
        pose_curr.pose.orientation.z, pose_curr.pose.orientation.w,
    )
    euler_next = quat_to_euler(
        pose_next.pose.orientation.x, pose_next.pose.orientation.y,
        pose_next.pose.orientation.z, pose_next.pose.orientation.w,
    )
    delta_rot = euler_next - euler_curr
    # Wrap to [-pi, pi]
    delta_rot = (delta_rot + np.pi) % (2 * np.pi) - np.pi

    # Normalize gripper: 0 (open) -> -1, 0.85 (closed) -> 1
    grip_norm = 2.0 * (gripper_next - gripper_open) / (gripper_closed - gripper_open) - 1.0
    grip_norm = np.clip(grip_norm, -1.0, 1.0)

    return np.concatenate([delta_pos, delta_rot, [grip_norm]])


def ros_image_to_numpy(img_msg):
    """Convert a ROS2 sensor_msgs/Image to numpy array (H, W, 3) uint8."""
    h, w = img_msg.height, img_msg.width
    if img_msg.encoding == "rgb8":
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(h, w, 3)
    elif img_msg.encoding == "bgr8":
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(h, w, 3)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image encoding: {img_msg.encoding}")
    return arr


def load_episode(mcap_path, front_size=(256, 256), wrist_size=(128, 128)):
    """Load a single episode from an mcap file.

    Returns:
        dict with keys:
            - front_images: (T, H, W, 3) uint8
            - wrist_images: (T, H, W, 3) uint8
            - eef_poses: list of PoseStamped messages
            - gripper_positions: (T,) float
            - actions: (T-1, 7) float delta actions
    """
    from mcap_ros2.reader import read_ros2_messages

    # Collect messages by topic, keyed by timestamp for sync
    front_imgs = []
    wrist_imgs = []
    eef_poses = []
    gripper_pos = []

    topic_data = {
        "/front/color/image_raw": [],
        "/wrist/color/image_raw": [],
        "/eef_pose": [],
        "/gripper_position": [],
    }

    for msg in read_ros2_messages(mcap_path):
        topic = msg.channel.topic
        if topic in topic_data:
            topic_data[topic].append((msg.log_time, msg.ros_msg))

    # Sort by timestamp
    for topic in topic_data:
        topic_data[topic].sort(key=lambda x: x[0])

    # All topics have 226 synced messages, just take in order
    n = min(len(topic_data[t]) for t in topic_data)

    for i in range(n):
        # Front image
        img = ros_image_to_numpy(topic_data["/front/color/image_raw"][i][1])
        img = cv2.resize(img, front_size)
        front_imgs.append(img)

        # Wrist image
        img = ros_image_to_numpy(topic_data["/wrist/color/image_raw"][i][1])
        img = cv2.resize(img, wrist_size)
        wrist_imgs.append(img)

        # EEF pose
        eef_poses.append(topic_data["/eef_pose"][i][1])

        # Gripper
        gripper_pos.append(topic_data["/gripper_position"][i][1].data)

    # Compute delta actions
    actions = []
    for i in range(n - 1):
        action = compute_delta_action(eef_poses[i], eef_poses[i + 1], gripper_pos[i + 1])
        actions.append(action)

    return {
        "front_images": np.array(front_imgs, dtype=np.uint8),
        "wrist_images": np.array(wrist_imgs, dtype=np.uint8),
        "eef_poses": eef_poses,
        "gripper_positions": np.array(gripper_pos, dtype=np.float64),
        "actions": np.array(actions, dtype=np.float32),
    }


def load_dataset(data_dir, max_episodes=None, front_size=(256, 256), wrist_size=(128, 128)):
    """Load all episodes from a directory of mcap bags.

    Args:
        data_dir: Path to directory containing episode_*_bag/ folders.
        max_episodes: Maximum number of episodes to load (None for all).

    Returns:
        List of episode dicts from load_episode().
    """
    episode_dirs = sorted(glob.glob(os.path.join(data_dir, "episode_*_bag")))
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    episodes = []
    for ep_dir in episode_dirs:
        mcap_files = glob.glob(os.path.join(ep_dir, "*.mcap"))
        if not mcap_files:
            print(f"Warning: No mcap file found in {ep_dir}, skipping")
            continue
        mcap_path = mcap_files[0]
        ep_name = os.path.basename(ep_dir)
        print(f"Loading {ep_name}...")
        episode = load_episode(mcap_path, front_size=front_size, wrist_size=wrist_size)
        episode["name"] = ep_name
        episodes.append(episode)

    print(f"Loaded {len(episodes)} episodes")
    return episodes


if __name__ == "__main__":
    data_dir = "/home/gokul/tqagi/3d_flowmatch_actor/data/xarm/default_task"
    episodes = load_dataset(data_dir, max_episodes=1)
    ep = episodes[0]
    print(f"Front images: {ep['front_images'].shape}")
    print(f"Wrist images: {ep['wrist_images'].shape}")
    print(f"Actions: {ep['actions'].shape}")
    print(f"Action stats - mean: {ep['actions'].mean(0)}, std: {ep['actions'].std(0)}")
