# lerobot_to_rlds.py: Convert LeRobot v3 datasets to RLDS TFRecord format for Octo finetuning.
# lerobot_to_rlds.py: Modular reader supporting configurable arm/camera/action selection and resampling.

"""
Convert LeRobot v3 format (parquet + mp4 videos) to RLDS TFRecord for Octo.

Supports:
- Configurable arm selection (left/right/both) for bimanual robots
- Configurable camera selection (primary, wrist, etc.)
- Action space selection (joint positions, subset of dims)
- Temporal resampling to target Hz
- Train/val split

Input (LeRobot v3):
    <record_dir>/
        data/chunk-000/file-000.parquet
        meta/info.json, stats.json, tasks.parquet, episodes/...
        videos/observation.images.<cam>/chunk-000/file-000.mp4

Output (RLDS TFRecord):
    <output_dir>/<dataset_name>/1.0.0/
        <dataset_name>-train.tfrecord-00000-of-00001
        <dataset_name>-val.tfrecord-00000-of-00001
        features.json, dataset_info.json

Usage:
    python scripts/lerobot_to_rlds.py \
        --config scripts/configs/lerobot_aloha.yaml

    Or with CLI overrides:
    python scripts/lerobot_to_rlds.py \
        --record_dirs /path/to/record1 /path/to/record2 \
        --output_dir /path/to/rlds_output \
        --dataset_name aloha_pack_red_bag \
        --active_arm right \
        --primary_camera cam_high \
        --wrist_camera cam_right_wrist \
        --target_hz 5.0
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow required. Install with: pip install pyarrow")

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("tensorflow required for TFRecord writing.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ── LeRobot v3 Reader ────────────────────────────────────────────────────────

@dataclass
class LeRobotConfig:
    """Configuration for LeRobot data extraction."""
    record_dirs: List[str] = field(default_factory=list)
    output_dir: str = "./data/rlds_output"
    dataset_name: str = "lerobot_dataset"
    language_instruction: str = ""

    # Arm selection: "left", "right", "both"
    active_arm: str = "right"
    # Indices into the action/state vector for each arm
    left_arm_indices: List[int] = field(default_factory=lambda: list(range(0, 7)))
    right_arm_indices: List[int] = field(default_factory=lambda: list(range(7, 14)))
    # Base velocity indices (excluded from actions by default)
    base_vel_indices: List[int] = field(default_factory=lambda: [14, 15])
    include_base_vel: bool = False

    # Camera mapping: LeRobot camera name -> Octo role
    primary_camera: str = "cam_high"
    wrist_camera: Optional[str] = "cam_right_wrist"
    primary_size: Tuple[int, int] = (256, 256)
    wrist_size: Tuple[int, int] = (128, 128)

    # Resampling
    target_hz: float = 5.0
    original_fps: int = 30

    # FK conversion: convert joint positions to delta EEF actions
    # "joint" = keep raw joint positions, "delta_eef" = FK + delta computation
    action_mode: str = "delta_eef"
    # Joint indices for FK (6 revolute joints, excluding gripper)
    fk_joint_indices: List[int] = field(default_factory=lambda: list(range(0, 6)))
    # Gripper index (appended as 7th dim after FK conversion)
    gripper_index: int = 6
    # Include mobile base mount offset in FK
    fk_include_mount: bool = False

    # Train/val split
    val_ratio: float = 0.1


class LeRobotReader:
    """Reads LeRobot v3 datasets (parquet + mp4) and extracts episodes."""

    def __init__(self, config: LeRobotConfig):
        self.config = config

    def read_all_records(self) -> List[dict]:
        """Read and merge episodes from all record directories."""
        all_episodes = []
        for record_dir in self.config.record_dirs:
            episodes = self._read_record(record_dir)
            all_episodes.extend(episodes)
        return all_episodes

    def _read_record(self, record_dir: str) -> List[dict]:
        """Read all episodes from a single LeRobot record directory."""
        info_path = os.path.join(record_dir, "meta", "info.json")
        with open(info_path) as f:
            info = json.load(f)

        # Read task text
        tasks_table = pq.read_table(os.path.join(record_dir, "meta", "tasks.parquet"))
        tasks_dict = tasks_table.to_pydict()
        task_text = self.config.language_instruction
        if not task_text and "__index_level_0__" in tasks_dict:
            task_text = tasks_dict["__index_level_0__"][0]

        # Read all data
        data_files = sorted(
            glob.glob(os.path.join(record_dir, "data", "chunk-*", "*.parquet"))
        )
        all_data = pq.read_table(data_files[0]) if len(data_files) == 1 else \
            pq.concat_tables([pq.read_table(f) for f in data_files])

        # Group by episode
        n_rows = len(all_data)
        episode_groups = {}
        for i in range(n_rows):
            ep_idx = all_data.column("episode_index")[i].as_py()
            if ep_idx not in episode_groups:
                episode_groups[ep_idx] = []
            episode_groups[ep_idx].append(i)

        # Detect image source: videos/ (mp4) or images/ (png frames)
        image_source = self._detect_image_source(record_dir, info)

        # Open video readers if using mp4
        video_readers = {}
        if image_source == "video":
            video_readers = self._open_videos(record_dir, info)

        episodes = []
        record_name = os.path.basename(record_dir)
        for ep_idx in sorted(episode_groups.keys()):
            row_indices = episode_groups[ep_idx]
            ep = self._extract_episode(
                all_data, row_indices, video_readers,
                record_dir, image_source, ep_idx, task_text,
            )
            if ep is not None:
                ep["source"] = f"{record_name}/ep{ep_idx}"
                episodes.append(ep)

        for cap in video_readers.values():
            cap.release()

        print(f"  {record_name}: {len(episodes)} episodes ({image_source} images)")
        return episodes

    def _detect_image_source(self, record_dir: str, info: dict) -> str:
        """Detect whether images are stored as mp4 videos or png frames."""
        videos_dir = os.path.join(record_dir, "videos")
        images_dir = os.path.join(record_dir, "images")
        if os.path.isdir(videos_dir) and glob.glob(os.path.join(videos_dir, "*", "chunk-*", "*.mp4")):
            return "video"
        elif os.path.isdir(images_dir):
            return "frames"
        else:
            return "none"

    def _open_videos(self, record_dir: str, info: dict) -> Dict[str, cv2.VideoCapture]:
        """Open video captures for configured cameras."""
        readers = {}
        cam_map = {"primary": self.config.primary_camera}
        if self.config.wrist_camera:
            cam_map["wrist"] = self.config.wrist_camera

        for role, cam_name in cam_map.items():
            lerobot_key = f"observation.images.{cam_name}"
            video_dir = os.path.join(record_dir, "videos", lerobot_key)
            video_files = sorted(glob.glob(os.path.join(video_dir, "chunk-*", "*.mp4")))
            if not video_files:
                continue

            cap = cv2.VideoCapture(video_files[0])
            if cap.isOpened():
                readers[role] = cap

        return readers

    def _read_frame_png(self, record_dir, cam_name, ep_idx, frame_idx, size):
        """Read a single PNG frame from the images/ directory."""
        path = os.path.join(
            record_dir, "images", f"observation.images.{cam_name}",
            f"episode-{ep_idx:06d}", f"frame-{frame_idx:06d}.png",
        )
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return cv2.resize(img, size)
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def _extract_episode(
        self, data_table, row_indices, video_readers,
        record_dir, image_source, ep_idx, task_text,
    ) -> Optional[dict]:
        """Extract a single episode with arm selection and resampling."""
        cfg = self.config
        stride = max(1, int(round(cfg.original_fps / cfg.target_hz)))

        # Select arm indices
        if cfg.active_arm == "right":
            arm_indices = cfg.right_arm_indices
        elif cfg.active_arm == "left":
            arm_indices = cfg.left_arm_indices
        else:
            arm_indices = cfg.left_arm_indices + cfg.right_arm_indices

        action_indices = list(arm_indices)
        if cfg.include_base_vel:
            action_indices.extend(cfg.base_vel_indices)

        subsampled_rows = row_indices[::stride]
        if len(subsampled_rows) < 2:
            return None

        actions = []
        states = []
        primary_images = []
        wrist_images = []

        for row_idx in subsampled_rows:
            act_full = data_table.column("action")[row_idx].as_py()
            state_full = data_table.column("observation.state")[row_idx].as_py()
            actions.append(np.array([act_full[i] for i in action_indices], dtype=np.float32))
            states.append(np.array([state_full[i] for i in action_indices], dtype=np.float32))

            frame_idx = data_table.column("frame_index")[row_idx].as_py()

            # Read images based on source type
            if image_source == "frames":
                primary_images.append(
                    self._read_frame_png(record_dir, cfg.primary_camera, ep_idx, frame_idx, cfg.primary_size)
                )
                if cfg.wrist_camera:
                    wrist_images.append(
                        self._read_frame_png(record_dir, cfg.wrist_camera, ep_idx, frame_idx, cfg.wrist_size)
                    )
            elif image_source == "video":
                for role, size in [("primary", cfg.primary_size), ("wrist", cfg.wrist_size)]:
                    if role not in video_readers:
                        continue
                    cap = video_readers[role]
                    global_idx = row_indices[0] + frame_idx
                    cap.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, size)
                    else:
                        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    if role == "primary":
                        primary_images.append(frame)
                    else:
                        wrist_images.append(frame)
            else:
                # No images — fill with black
                primary_images.append(np.zeros((cfg.primary_size[1], cfg.primary_size[0], 3), dtype=np.uint8))
                if cfg.wrist_camera:
                    wrist_images.append(np.zeros((cfg.wrist_size[1], cfg.wrist_size[0], 3), dtype=np.uint8))

        raw_actions = np.stack(actions)
        raw_states = np.stack(states)

        # Apply FK conversion if configured
        if cfg.action_mode == "delta_eef":
            from wxai_fk import batch_fk_euler, compute_delta_eef

            # FK on the 6 revolute joints to get EEF poses
            fk_indices = cfg.fk_joint_indices
            joint_angles = raw_actions[:, fk_indices]
            eef_poses = batch_fk_euler(joint_angles, include_mount=cfg.fk_include_mount)

            # Compute delta EEF actions from consecutive poses
            delta_eef = compute_delta_eef(eef_poses)  # (T-1, 6)

            # Append gripper as 7th dim (binarized: >0.01 = closed)
            grip_idx = cfg.gripper_index
            gripper_vals = raw_actions[1:, grip_idx]  # align with delta (T-1)
            gripper_binary = (gripper_vals > 0.01).astype(np.float32)

            final_actions = np.concatenate(
                [delta_eef, gripper_binary[:, None]], axis=-1
            ).astype(np.float32)  # (T-1, 7)

            # EEF state: [x, y, z, roll, pitch, yaw, gripper]
            all_gripper = (raw_actions[:, grip_idx] > 0.01).astype(np.float32)
            final_states = np.concatenate(
                [eef_poses, all_gripper[:, None]], axis=-1
            ).astype(np.float32)  # (T, 7)

            # Trim images to match T-1 actions (drop last frame)
            primary_images = primary_images[:-1]
            if wrist_images:
                wrist_images = wrist_images[:-1]
            final_states = final_states[:-1]  # (T-1, 7)
        else:
            # Raw joint positions
            final_actions = raw_actions
            final_states = raw_states

        return {
            "primary_images": primary_images,
            "wrist_images": wrist_images,
            "actions": final_actions,
            "states": final_states,
            "language": task_text,
        }


# ── RLDS TFRecord Writer (reused from mcap_to_rlds.py) ──────────────────────

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bool_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def _encode_image_jpeg(img):
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def episode_to_tfrecord(episode, episode_id, has_wrist=True):
    """Serialize one episode to tf.train.Example in RLDS nested format."""
    T = len(episode["actions"])
    steps_bytes = []

    for t in range(T):
        sf = {}
        sf["observation/image_0"] = _bytes_feature(
            _encode_image_jpeg(episode["primary_images"][t])
        )
        if has_wrist and episode["wrist_images"]:
            sf["observation/image_1"] = _bytes_feature(
                _encode_image_jpeg(episode["wrist_images"][t])
            )
        sf["observation/state"] = _float_feature(episode["states"][t].tolist())
        sf["action"] = _float_feature(episode["actions"][t].tolist())
        sf["language_instruction"] = _bytes_feature(
            episode["language"].encode("utf-8")
        )
        sf["is_first"] = _bool_feature(t == 0)
        sf["is_last"] = _bool_feature(t == T - 1)
        sf["is_terminal"] = _bool_feature(t == T - 1)
        sf["discount"] = _float_feature([1.0])
        sf["reward"] = _float_feature([1.0 if t == T - 1 else 0.0])

        step_ex = tf.train.Example(
            features=tf.train.Features(feature=sf)
        )
        steps_bytes.append(step_ex.SerializeToString())

    traj_features = {
        "steps": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=steps_bytes)
        ),
        "episode_metadata/episode_id": _int64_feature(episode_id),
        "episode_metadata/file_path": _bytes_feature(
            episode.get("source", f"episode_{episode_id}").encode("utf-8")
        ),
        "episode_metadata/has_image_0": _bool_feature(True),
        "episode_metadata/has_image_1": _bool_feature(has_wrist and bool(episode["wrist_images"])),
        "episode_metadata/has_image_2": _bool_feature(False),
        "episode_metadata/has_image_3": _bool_feature(False),
        "episode_metadata/has_language": _bool_feature(True),
    }
    return tf.train.Example(features=tf.train.Features(feature=traj_features))


def write_tfrecords(examples, path):
    with tf.io.TFRecordWriter(path) as writer:
        for ex in examples:
            writer.write(ex.SerializeToString())
    print(f"  Wrote {len(examples)} episodes to {path}")


def generate_features_json(action_dim, state_dim, primary_size, wrist_size, has_wrist):
    """Generate RLDS features.json for this dataset."""
    obs_features = {
        "image_0": {
            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
            "image": {
                "shape": {"dimensions": [str(primary_size[1]), str(primary_size[0]), "3"]},
                "dtype": "uint8", "encodingFormat": "jpeg",
            },
            "description": "Primary camera RGB.",
        },
        "state": {
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
            "tensor": {
                "shape": {"dimensions": [str(state_dim)]},
                "dtype": "float32", "encoding": "none",
            },
            "description": "Joint positions.",
        },
    }
    if has_wrist:
        obs_features["image_1"] = {
            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
            "image": {
                "shape": {"dimensions": [str(wrist_size[1]), str(wrist_size[0]), "3"]},
                "dtype": "uint8", "encodingFormat": "jpeg",
            },
            "description": "Wrist camera RGB.",
        }

    return {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {"features": {
            "steps": {
                "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                "sequence": {"feature": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {"features": {
                        "action": {
                            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                            "tensor": {"shape": {"dimensions": [str(action_dim)]}, "dtype": "float32", "encoding": "none"},
                            "description": "Joint position actions.",
                        },
                        "observation": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {"features": obs_features},
                        },
                        "language_instruction": {"pythonClassName": "tensorflow_datasets.core.features.text_feature.Text", "text": {}, "description": "Task instruction."},
                        "is_first": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                        "is_last": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                        "is_terminal": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                        "discount": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"}},
                        "reward": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"}},
                    }},
                }, "length": "-1"},
            },
            "episode_metadata": {
                "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                "featuresDict": {"features": {
                    "file_path": {"pythonClassName": "tensorflow_datasets.core.features.text_feature.Text", "text": {}},
                    "has_language": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                    "has_image_0": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                    "has_image_1": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                    "has_image_2": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                    "has_image_3": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}},
                    "episode_id": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"}},
                }},
            },
        }},
    }


def generate_dataset_info(dataset_name, n_train, n_val, train_bytes, val_bytes):
    return {
        "citation": "",
        "description": f"LeRobot to RLDS conversion: {dataset_name}",
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name}.dataset_builder",
        "name": dataset_name,
        "releaseNotes": {"1.0.0": "Converted from LeRobot v3 format."},
        "splits": [
            {"filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
             "name": "train", "numBytes": str(train_bytes), "shardLengths": [str(n_train)]},
            {"filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
             "name": "val", "numBytes": str(val_bytes), "shardLengths": [str(n_val)]},
        ],
        "version": "1.0.0",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_config_from_yaml(yaml_path: str) -> LeRobotConfig:
    """Load config from YAML file."""
    if not HAS_YAML:
        raise ImportError("PyYAML required for config files. Install with: pip install pyyaml")
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return LeRobotConfig(**raw)


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3 datasets to RLDS TFRecord for Octo"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--record_dirs", nargs="+", help="LeRobot record directories")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--language_instruction", type=str)
    parser.add_argument("--active_arm", type=str, choices=["left", "right", "both"])
    parser.add_argument("--primary_camera", type=str)
    parser.add_argument("--wrist_camera", type=str)
    parser.add_argument("--target_hz", type=float)
    parser.add_argument("--val_ratio", type=float)
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = LeRobotConfig()

    # Apply CLI overrides
    for key in ["record_dirs", "output_dir", "dataset_name", "language_instruction",
                "active_arm", "primary_camera", "wrist_camera", "target_hz", "val_ratio"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)

    if not cfg.record_dirs:
        parser.error("Must provide --record_dirs or --config with record_dirs")

    has_wrist = cfg.wrist_camera is not None

    # Read episodes
    print(f"Reading LeRobot records ({len(cfg.record_dirs)} dirs)...")
    print(f"  Active arm: {cfg.active_arm}")
    print(f"  Cameras: primary={cfg.primary_camera}, wrist={cfg.wrist_camera}")
    print(f"  Target Hz: {cfg.target_hz}")

    reader = LeRobotReader(cfg)
    episodes = reader.read_all_records()

    if not episodes:
        print("No episodes extracted, exiting.")
        return

    # Print stats
    action_dim = episodes[0]["actions"].shape[1]
    total_frames = sum(len(ep["actions"]) for ep in episodes)
    print(f"\nExtracted {len(episodes)} episodes, {total_frames} total frames")
    print(f"  Action dim: {action_dim}")
    print(f"  Steps/episode: {[len(ep['actions']) for ep in episodes]}")

    all_actions = np.concatenate([ep["actions"] for ep in episodes])
    all_states = np.concatenate([ep["states"] for ep in episodes])
    print(f"  Action mean: {np.array2string(all_actions.mean(0), precision=4)}")
    print(f"  Action std:  {np.array2string(all_actions.std(0), precision=4)}")
    print(f"  State range: [{np.array2string(all_states.min(0), precision=3)}, {np.array2string(all_states.max(0), precision=3)}]")

    # Split train/val
    n_val = max(1, int(len(episodes) * cfg.val_ratio))
    n_train = len(episodes) - n_val
    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:]
    print(f"\nSplit: {n_train} train, {n_val} val")

    # Serialize to TFRecords
    train_examples = [
        episode_to_tfrecord(ep, i, has_wrist)
        for i, ep in enumerate(tqdm(train_eps, desc="Serializing train"))
    ]
    val_examples = [
        episode_to_tfrecord(ep, i, has_wrist)
        for i, ep in enumerate(tqdm(val_eps, desc="Serializing val"))
    ]

    # Write output
    ds_dir = os.path.join(cfg.output_dir, cfg.dataset_name, "1.0.0")
    os.makedirs(ds_dir, exist_ok=True)

    train_path = os.path.join(ds_dir, f"{cfg.dataset_name}-train.tfrecord-00000-of-00001")
    val_path = os.path.join(ds_dir, f"{cfg.dataset_name}-val.tfrecord-00000-of-00001")
    write_tfrecords(train_examples, train_path)
    write_tfrecords(val_examples, val_path)

    train_bytes = os.path.getsize(train_path)
    val_bytes = os.path.getsize(val_path)

    # Write metadata
    features = generate_features_json(
        action_dim, action_dim, cfg.primary_size, cfg.wrist_size, has_wrist
    )
    with open(os.path.join(ds_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=4)

    ds_info = generate_dataset_info(cfg.dataset_name, n_train, n_val, train_bytes, val_bytes)
    with open(os.path.join(ds_dir, "dataset_info.json"), "w") as f:
        json.dump(ds_info, f, indent=2)

    print(f"\nRLDS dataset written to: {ds_dir}")
    print(f"  Train: {n_train} episodes ({train_bytes / 1e6:.1f} MB)")
    print(f"  Val:   {n_val} episodes ({val_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
