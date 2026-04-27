# mcap_to_rlds.py: Convert xarm ROS2 mcap bags to RLDS TFRecord format for Octo finetuning.
# mcap_to_rlds.py: Uses time-synchronized resampling at target_hz with configurable sync slop.

"""
Convert xarm ROS2 mcap bag files into RLDS-format TFRecord dataset for Octo finetuning.

Uses time-synchronized resampling (not naive message ordering) to align
front camera, wrist camera, EEF pose, and gripper position at a fixed Hz.

Input:
    <bag_dir>/
        episode_0_bag/episode_0_bag_0.mcap
        episode_1_bag/episode_1_bag_0.mcap
        ...

Output:
    <output_dir>/xarm_place_toolbox/1.0.0/
        xarm_place_toolbox-train.tfrecord-00000-of-00001
        xarm_place_toolbox-val.tfrecord-00000-of-00001
        features.json
        dataset_info.json

Usage:
    python scripts/mcap_to_rlds.py \
        --bag_dir /path/to/episode_bags \
        --output_dir /path/to/rlds_output \
        --language_instruction "place object in toolbox" \
        --target_hz 10.0 \
        --val_ratio 0.1
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError:
    raise ImportError(
        "mcap-ros2-support is required. Install with: pip install mcap mcap-ros2-support"
    )

import tensorflow as tf


# ── Topic configuration (matches xarm bag structure) ──────────────────────────

TOPICS = {
    "front_rgb": "/front/color/image_raw",
    "wrist_rgb": "/wrist/color/image_raw",
    "eef_pose": "/eef_pose",
    "gripper": "/gripper_position",
}

FRONT_SIZE = (256, 256)
WRIST_SIZE = (128, 128)


# ── ROS message decoding ─────────────────────────────────────────────────────

def _find_closest_msg(target_time, messages, slop):
    """Find message closest to target_time within slop tolerance (seconds)."""
    best = None
    best_diff = float("inf")
    for msg_time, msg in messages:
        diff = abs(msg_time - target_time)
        if diff < best_diff and diff <= slop:
            best_diff = diff
            best = msg
    return best


def _decode_image(msg):
    """Decode sensor_msgs/Image to numpy (H, W, 3) uint8 RGB."""
    h, w = msg.height, msg.width
    encoding = msg.encoding
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "RGB8"):
        return data.reshape(h, w, 3)
    elif encoding in ("bgr8", "BGR8"):
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    elif encoding in ("rgba8", "RGBA8"):
        return data.reshape(h, w, 4)[:, :, :3]
    elif encoding in ("bgra8", "BGRA8"):
        return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")


def _decode_pose_quat(msg):
    """Extract [x, y, z, qx, qy, qz, qw] from PoseStamped (raw quaternion)."""
    p = msg.pose.position
    q = msg.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64)


def _quat_to_euler(quat_xyzw):
    """Convert quaternion [qx, qy, qz, qw] to euler [roll, pitch, yaw]."""
    return Rotation.from_quat(quat_xyzw).as_euler("xyz").astype(np.float32)


def _compute_rotation_delta(quat_curr, quat_next):
    """Compute rotation delta as euler angles using quaternion relative rotation.

    This avoids the euler wrapping problem (e.g. 179° → -179° = fake 358° jump)
    by computing q_delta = q_next * q_curr^{-1} in quaternion space first,
    then converting to euler. The resulting euler angles are guaranteed bounded.
    """
    r_curr = Rotation.from_quat(quat_curr)
    r_next = Rotation.from_quat(quat_next)
    r_delta = r_next * r_curr.inv()
    return r_delta.as_euler("xyz").astype(np.float32)


def _binarize_gripper(val, threshold=0.5):
    """Binarize gripper to 0 (open) / 1 (closed), matching bridge dataset format.

    Bridge dataset uses: 0 = closed, 1 = open. We match that convention.
    Raw xarm gripper: 0.0 = open, 0.85 = closed.
    """
    normalized = val / 0.85  # normalize to [0, 1] range
    return np.float32(1.0 if normalized > threshold else 0.0)


def _msg_time_to_sec(log_time):
    """Convert mcap log_time to seconds. Handles both int (nanoseconds) and datetime."""
    if isinstance(log_time, (int, float)):
        return log_time / 1e9
    # datetime.datetime (newer mcap_ros2 versions)
    return log_time.timestamp()


def _collect_messages_by_topic(mcap_path, topics):
    """Read all messages from given topics using mcap_ros2, return {topic: [(time_sec, msg)]}."""
    result = {t: [] for t in topics}
    for msg in read_ros2_messages(mcap_path):
        topic = msg.channel.topic
        if topic in result:
            t_sec = _msg_time_to_sec(msg.log_time)
            result[topic].append((t_sec, msg.ros_msg))
    return result


# ── Episode extraction with time synchronization ─────────────────────────────

def extract_episode(mcap_path, target_hz=2.0, sync_slop=0.05):
    """
    Extract a single episode from an mcap bag with time-synchronized resampling.

    Uses quaternion-space rotation deltas (not euler subtraction) to avoid
    wrapping artifacts. Gripper is binarized to 0/1 matching bridge dataset.
    Default 2Hz resampling produces action deltas comparable in scale to bridge
    dataset (~5-10mm per step vs bridge's ~10mm).

    Returns dict with:
        front_images: list of (H, W, 3) uint8 arrays
        wrist_images: list of (H, W, 3) uint8 arrays
        states: (T, 7) float32 [x, y, z, roll, pitch, yaw, gripper_binary]
        actions: (T-1, 7) float32 delta actions [dx, dy, dz, droll, dpitch, dyaw, dgripper]
    Or None if extraction fails.
    """
    all_topics = list(TOPICS.values())
    msgs = _collect_messages_by_topic(mcap_path, all_topics)

    eef_msgs = msgs[TOPICS["eef_pose"]]
    gripper_msgs = msgs[TOPICS["gripper"]]
    front_msgs = msgs[TOPICS["front_rgb"]]
    wrist_msgs = msgs[TOPICS["wrist_rgb"]]

    if len(eef_msgs) < 2:
        print(f"  Skipping {mcap_path}: too few EEF messages ({len(eef_msgs)})")
        return None

    # Resample at target_hz using EEF trajectory time range
    t_start = eef_msgs[0][0]
    t_end = eef_msgs[-1][0]
    dt = 1.0 / target_hz
    sample_times = np.arange(t_start, t_end, dt)

    if len(sample_times) < 2:
        print(f"  Skipping {mcap_path}: duration too short")
        return None

    front_images = []
    wrist_images = []
    states = []
    raw_quats = []  # keep raw quaternions for rotation delta computation

    for t_sample in sample_times:
        # Find closest messages within sync slop
        eef_msg = _find_closest_msg(t_sample, eef_msgs, sync_slop * 2)
        grip_msg = _find_closest_msg(t_sample, gripper_msgs, sync_slop * 2)
        front_msg = _find_closest_msg(t_sample, front_msgs, sync_slop)
        wrist_msg = _find_closest_msg(t_sample, wrist_msgs, sync_slop)

        if any(m is None for m in [eef_msg, grip_msg, front_msg, wrist_msg]):
            continue

        # Decode and resize images
        front_img = _decode_image(front_msg)
        front_img = cv2.resize(front_img, FRONT_SIZE)
        front_images.append(front_img)

        wrist_img = _decode_image(wrist_msg)
        wrist_img = cv2.resize(wrist_img, WRIST_SIZE)
        wrist_images.append(wrist_img)

        # Decode raw pose (position + quaternion)
        pose_quat = _decode_pose_quat(eef_msg)
        pos = pose_quat[:3].astype(np.float32)
        quat = pose_quat[3:]  # [qx, qy, qz, qw]
        raw_quats.append(quat)

        # Convert to euler for state representation
        euler = _quat_to_euler(quat)

        # Binarize gripper (matching bridge dataset convention)
        gripper = _binarize_gripper(float(grip_msg.data))

        state = np.concatenate([pos, euler, [gripper]])
        states.append(state)

    if len(states) < 2:
        print(f"  Skipping {mcap_path}: too few synchronized frames ({len(states)})")
        return None

    states = np.stack(states)  # (T, 7)

    # Check for constant gripper and warn
    unique_grippers = np.unique(states[:, 6])
    if len(unique_grippers) == 1:
        print(f"  Warning: gripper is constant ({unique_grippers[0]}) for this episode")

    # Compute delta actions using quaternion-space rotation deltas
    actions = []
    for i in range(len(states) - 1):
        # Position delta: simple subtraction
        delta_pos = states[i + 1, :3] - states[i, :3]

        # Rotation delta: quaternion relative rotation (avoids euler wrapping)
        delta_rot = _compute_rotation_delta(raw_quats[i], raw_quats[i + 1])

        # Gripper delta
        delta_grip = np.array([states[i + 1, 6] - states[i, 6]], dtype=np.float32)

        action = np.concatenate([delta_pos, delta_rot, delta_grip]).astype(np.float32)
        actions.append(action)

    return {
        "front_images": front_images,
        "wrist_images": wrist_images,
        "states": states,
        "actions": np.stack(actions),
    }


# ── TFRecord serialization ───────────────────────────────────────────────────

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bool_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _encode_image_jpeg(img):
    """Encode (H, W, 3) uint8 array as JPEG bytes."""
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def episode_to_tfrecord_example(episode_data, episode_id, language_instruction):
    """
    Convert extracted episode data to a single tf.train.Example in RLDS format.

    RLDS stores entire trajectories as a single Example with repeated step features.
    """
    T = len(episode_data["front_images"])
    actions = episode_data["actions"]
    states = episode_data["states"]

    # Build the step-level features as serialized SequenceExamples
    # RLDS uses a nested structure: trajectory -> steps -> features
    # We serialize steps as a sequence within the example
    steps_bytes_list = []

    for t in range(T):
        step_features = {}

        # Images encoded as JPEG
        step_features["observation/image_0"] = _bytes_feature(
            _encode_image_jpeg(episode_data["front_images"][t])
        )
        step_features["observation/image_1"] = _bytes_feature(
            _encode_image_jpeg(episode_data["wrist_images"][t])
        )

        # State (proprio): [x, y, z, roll, pitch, yaw, gripper]
        step_features["observation/state"] = _float_feature(
            states[t].tolist()
        )

        # Action: delta [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        # Last step gets zero action (no next state to compute delta from)
        if t < T - 1:
            step_features["action"] = _float_feature(actions[t].tolist())
        else:
            step_features["action"] = _float_feature([0.0] * 7)

        # Language instruction
        step_features["language_instruction"] = _bytes_feature(
            language_instruction.encode("utf-8")
        )

        # Episode structure flags
        step_features["is_first"] = _bool_feature(t == 0)
        step_features["is_last"] = _bool_feature(t == T - 1)
        step_features["is_terminal"] = _bool_feature(t == T - 1)
        step_features["discount"] = _float_feature([1.0])
        step_features["reward"] = _float_feature([1.0 if t == T - 1 else 0.0])

        step_example = tf.train.Example(
            features=tf.train.Features(feature=step_features)
        )
        steps_bytes_list.append(step_example.SerializeToString())

    # Wrap steps into trajectory-level example
    traj_features = {
        "steps": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=steps_bytes_list)
        ),
        "episode_metadata/episode_id": _int64_feature(episode_id),
        "episode_metadata/file_path": _bytes_feature(
            f"episode_{episode_id}".encode("utf-8")
        ),
        "episode_metadata/has_image_0": _bool_feature(True),
        "episode_metadata/has_image_1": _bool_feature(True),
        "episode_metadata/has_image_2": _bool_feature(False),
        "episode_metadata/has_image_3": _bool_feature(False),
        "episode_metadata/has_language": _bool_feature(True),
    }

    return tf.train.Example(
        features=tf.train.Features(feature=traj_features)
    )


def write_tfrecords(episodes, output_path):
    """Write list of tf.train.Example to a TFRecord file."""
    with tf.io.TFRecordWriter(output_path) as writer:
        for example in episodes:
            writer.write(example.SerializeToString())
    print(f"  Wrote {len(episodes)} episodes to {output_path}")


# ── RLDS metadata generation ─────────────────────────────────────────────────

def generate_features_json(front_size, wrist_size):
    """Generate features.json matching RLDS/TFDS schema for this dataset."""
    return {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "steps": {
                    "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                    "sequence": {
                        "feature": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {
                                    "action": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {
                                            "shape": {"dimensions": ["7"]},
                                            "dtype": "float32",
                                            "encoding": "none",
                                        },
                                        "description": "Delta EEF action: [dx, dy, dz, droll, dpitch, dyaw, dgripper].",
                                    },
                                    "is_terminal": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step of the episode.",
                                    },
                                    "is_last": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step of the episode.",
                                    },
                                    "is_first": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on first step of the episode.",
                                    },
                                    "language_instruction": {
                                        "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                        "text": {},
                                        "description": "Language instruction for the task.",
                                    },
                                    "observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "image_0": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {
                                                        "shape": {
                                                            "dimensions": [
                                                                str(front_size[1]),
                                                                str(front_size[0]),
                                                                "3",
                                                            ]
                                                        },
                                                        "dtype": "uint8",
                                                        "encodingFormat": "jpeg",
                                                    },
                                                    "description": "Front camera RGB observation.",
                                                },
                                                "image_1": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {
                                                        "shape": {
                                                            "dimensions": [
                                                                str(wrist_size[1]),
                                                                str(wrist_size[0]),
                                                                "3",
                                                            ]
                                                        },
                                                        "dtype": "uint8",
                                                        "encodingFormat": "jpeg",
                                                    },
                                                    "description": "Wrist camera RGB observation.",
                                                },
                                                "state": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {
                                                        "shape": {"dimensions": ["7"]},
                                                        "dtype": "float32",
                                                        "encoding": "none",
                                                    },
                                                    "description": "EEF state: [x, y, z, roll, pitch, yaw, gripper].",
                                                },
                                            }
                                        },
                                    },
                                    "discount": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Discount, default 1.",
                                    },
                                    "reward": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Reward, 1 on final step.",
                                    },
                                }
                            },
                        },
                        "length": "-1",
                    },
                },
                "episode_metadata": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "file_path": {
                                "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                "text": {},
                                "description": "Path to the original data file.",
                            },
                            "has_language": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                "description": "True if language instruction exists.",
                            },
                            "has_image_0": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                "description": "True if front camera image exists.",
                            },
                            "has_image_1": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                "description": "True if wrist camera image exists.",
                            },
                            "has_image_2": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                "description": "Unused, always False.",
                            },
                            "has_image_3": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                "description": "Unused, always False.",
                            },
                            "episode_id": {
                                "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"},
                                "description": "Episode index.",
                            },
                        }
                    },
                },
            }
        },
    }


def generate_dataset_info(dataset_name, num_train, num_val, train_bytes, val_bytes):
    """Generate dataset_info.json for TFDS compatibility."""
    return {
        "citation": "",
        "description": (
            "xarm place-object-in-toolbox dataset. "
            "Extracted from ROS2 mcap bags with time-synchronized resampling. "
            "Front camera (256x256) + wrist camera (128x128), "
            "7-dim delta EEF actions, 7-dim proprio state."
        ),
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name}.{dataset_name}_dataset_builder",
        "name": dataset_name,
        "releaseNotes": {"1.0.0": "Initial extraction from mcap bags."},
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": str(train_bytes),
                "shardLengths": [str(num_train)],
            },
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "val",
                "numBytes": str(val_bytes),
                "shardLengths": [str(num_val)],
            },
        ],
        "version": "1.0.0",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def find_mcap_files(bag_dir):
    """Find all mcap files in episode_*_bag/ directories."""
    episode_dirs = sorted(glob.glob(os.path.join(bag_dir, "episode_*_bag")))
    mcap_files = []
    for ep_dir in episode_dirs:
        mcaps = glob.glob(os.path.join(ep_dir, "*.mcap"))
        if mcaps:
            mcap_files.append(mcaps[0])
        else:
            print(f"Warning: no mcap file in {ep_dir}, skipping")
    return mcap_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert xarm mcap bags to RLDS TFRecord for Octo finetuning"
    )
    parser.add_argument(
        "--bag_dir", type=str, required=True,
        help="Directory containing episode_*_bag/ folders with mcap files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for RLDS dataset",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="xarm_place_toolbox",
        help="Name for the RLDS dataset",
    )
    parser.add_argument(
        "--language_instruction", type=str,
        default="place object in toolbox",
        help="Language instruction for all episodes",
    )
    parser.add_argument("--target_hz", type=float, default=2.0,
                        help="Resampling frequency in Hz (2Hz matches bridge action scale)")
    parser.add_argument("--sync_slop", type=float, default=0.05,
                        help="Time sync tolerance in seconds")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of episodes for validation")
    args = parser.parse_args()

    mcap_files = find_mcap_files(args.bag_dir)
    print(f"Found {len(mcap_files)} mcap files in {args.bag_dir}")

    if not mcap_files:
        print("No mcap files found, exiting.")
        return

    # Extract all episodes
    all_episodes = []
    for i, mcap_path in enumerate(tqdm(mcap_files, desc="Extracting episodes")):
        ep_name = os.path.basename(os.path.dirname(mcap_path))
        data = extract_episode(mcap_path, args.target_hz, args.sync_slop)
        if data is None:
            continue
        n_frames = len(data["front_images"])
        print(f"  {ep_name}: {n_frames} synced frames, "
              f"actions shape {data['actions'].shape}")
        all_episodes.append((i, data))

    print(f"\nSuccessfully extracted {len(all_episodes)} / {len(mcap_files)} episodes")

    if not all_episodes:
        print("No valid episodes, exiting.")
        return

    # Print dataset statistics
    all_actions = np.concatenate([d["actions"] for _, d in all_episodes])
    all_states = np.concatenate([d["states"] for _, d in all_episodes])
    print(f"\nDataset statistics:")
    print(f"  Total frames: {sum(len(d['front_images']) for _, d in all_episodes)}")
    print(f"  Action mean: {all_actions.mean(0)}")
    print(f"  Action std:  {all_actions.std(0)}")
    print(f"  State mean:  {all_states.mean(0)}")
    print(f"  State std:   {all_states.std(0)}")
    print(f"  State range: min={all_states.min(0)}, max={all_states.max(0)}")

    # Split into train/val
    n_val = max(1, int(len(all_episodes) * args.val_ratio))
    n_train = len(all_episodes) - n_val
    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train:]
    print(f"\nSplit: {n_train} train, {n_val} val")

    # Convert to TFRecord examples
    train_examples = [
        episode_to_tfrecord_example(data, ep_id, args.language_instruction)
        for ep_id, data in tqdm(train_episodes, desc="Serializing train")
    ]
    val_examples = [
        episode_to_tfrecord_example(data, ep_id, args.language_instruction)
        for ep_id, data in tqdm(val_episodes, desc="Serializing val")
    ]

    # Write output
    ds_dir = os.path.join(args.output_dir, args.dataset_name, "1.0.0")
    os.makedirs(ds_dir, exist_ok=True)

    train_path = os.path.join(
        ds_dir, f"{args.dataset_name}-train.tfrecord-00000-of-00001"
    )
    val_path = os.path.join(
        ds_dir, f"{args.dataset_name}-val.tfrecord-00000-of-00001"
    )
    write_tfrecords(train_examples, train_path)
    write_tfrecords(val_examples, val_path)

    train_bytes = os.path.getsize(train_path)
    val_bytes = os.path.getsize(val_path)

    # Write metadata
    features = generate_features_json(FRONT_SIZE, WRIST_SIZE)
    with open(os.path.join(ds_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=4)

    dataset_info = generate_dataset_info(
        args.dataset_name, n_train, n_val, train_bytes, val_bytes
    )
    with open(os.path.join(ds_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nRLDS dataset written to: {ds_dir}")
    print(f"  Train: {n_train} episodes ({train_bytes / 1e6:.1f} MB)")
    print(f"  Val:   {n_val} episodes ({val_bytes / 1e6:.1f} MB)")
    print(f"\nTo finetune Octo, use:")
    print(f'  --config.dataset_kwargs.name="{args.dataset_name}"')
    print(f'  --config.dataset_kwargs.data_dir="{args.output_dir}"')


if __name__ == "__main__":
    main()
