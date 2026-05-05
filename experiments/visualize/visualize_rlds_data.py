# visualize_rlds_data.py: Inspect post-conversion RLDS data quality before training.
# visualize_rlds_data.py: Compares raw 30fps vs downsampled trajectories and action distributions.

"""
Visualize RLDS TFRecord data after conversion to check downsampling quality.

Shows:
1. Raw vs downsampled EEF trajectories (3D overlay per episode)
2. Per-dimension action step sizes — are steps too large after downsampling?
3. Action magnitude distribution — histogram of delta norms
4. Image frame sampling — what the model actually sees at each step

Usage (inside Docker — project code and data mounted at runtime):

    # 1. Visualize marker_pick at 5Hz (default config)
    docker run --rm \
      -v /path/to/octo:/octo \
      -v /path/to/lerobot/data:/data/aloha \
      -v /path/to/output:/data \
      octo-finetune:latest python3 /octo/experiments/visualize/visualize_rlds_data.py \
        --config /octo/experiments/configs/lerobot_marker_pick.yaml \
        --output_dir /data/viz_output/data_quality_marker_pick \
        --max_episodes 5

    # 2. Compare 10Hz vs 5Hz — run on both configs, compare dataset_summary.png
    docker run --rm \
      -v /path/to/octo:/octo \
      -v /path/to/lerobot/data:/data/aloha \
      -v /path/to/output:/data \
      octo-finetune:latest python3 /octo/experiments/visualize/visualize_rlds_data.py \
        --config /octo/experiments/configs/lerobot_marker_pick_10hz.yaml \
        --output_dir /data/viz_output/data_quality_marker_pick_10hz \
        --max_episodes 5

    # 3. End-to-end pipeline: convert -> visualize -> train -> eval
    #    Step 1: Convert LeRobot to RLDS
    docker run --rm -v ...  python3 /octo/experiments/data/lerobot_to_rlds.py --config /octo/experiments/configs/lerobot_marker_pick.yaml
    #    Step 2: Visualize data quality (this script)
    docker run --rm -v ...  python3 /octo/experiments/visualize/visualize_rlds_data.py --config /octo/experiments/configs/lerobot_marker_pick.yaml --output_dir /data/viz_output/data_quality
    #    Step 3: Train (on remote GPU server)
    bash experiments/train/deploy_training.sh aws-L4-server1 /home/ubuntu/torqueagi
    ssh aws-L4-server1 "cd /home/ubuntu/torqueagi/octo && bash experiments/train/train.sh experiments/configs/finetune_marker_pick.yaml"
    #    Step 4: Eval
    docker run --rm --gpus all -v ... python3 /octo/experiments/eval/eval_xarm.py --checkpoint_path=/data/checkpoints/... --data_dir=/data/rlds_output --dataset_name=aloha_marker_pick
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    raise ImportError("pyarrow required")

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml required")

# Reuse FK from the conversion pipeline
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from wxai_fk import batch_fk_euler, compute_delta_eef


def load_raw_episode(data_table, row_indices, right_arm_indices, fk_joint_indices, gripper_index):
    """Load raw 30fps joint data and compute EEF poses + deltas at full rate."""
    actions_raw = []
    for row_idx in row_indices:
        act_full = data_table.column("action")[row_idx].as_py()
        arm_act = np.array([act_full[i] for i in right_arm_indices], dtype=np.float32)
        actions_raw.append(arm_act)
    actions_raw = np.stack(actions_raw)

    # FK at full rate
    joint_angles = actions_raw[:, fk_joint_indices]
    eef_poses = batch_fk_euler(joint_angles, include_mount=False)

    # Delta EEF at full rate
    delta_eef = compute_delta_eef(eef_poses)
    gripper_vals = actions_raw[1:, gripper_index]
    gripper_binary = (gripper_vals > 0.01).astype(np.float32)
    full_actions = np.concatenate([delta_eef, gripper_binary[:, None]], axis=-1)

    return eef_poses, full_actions


def load_downsampled_episode(data_table, row_indices, right_arm_indices, fk_joint_indices,
                              gripper_index, stride):
    """Load downsampled data — exactly what the model trains on."""
    subsampled = row_indices[::stride]
    actions_sub = []
    for row_idx in subsampled:
        act_full = data_table.column("action")[row_idx].as_py()
        arm_act = np.array([act_full[i] for i in right_arm_indices], dtype=np.float32)
        actions_sub.append(arm_act)
    actions_sub = np.stack(actions_sub)

    joint_angles = actions_sub[:, fk_joint_indices]
    eef_poses = batch_fk_euler(joint_angles, include_mount=False)

    delta_eef = compute_delta_eef(eef_poses)
    gripper_vals = actions_sub[1:, gripper_index]
    gripper_binary = (gripper_vals > 0.01).astype(np.float32)
    ds_actions = np.concatenate([delta_eef, gripper_binary[:, None]], axis=-1)

    return eef_poses, ds_actions


def load_downsampled_images(record_dir, primary_camera, row_indices, stride, primary_size):
    """Load downsampled camera frames — what the model sees."""
    lerobot_key = f"observation.images.{primary_camera}"
    video_files = sorted(glob.glob(os.path.join(record_dir, "videos", lerobot_key, "chunk-*", "*.mp4")))
    if not video_files:
        return []

    cap = cv2.VideoCapture(video_files[0])
    subsampled = row_indices[::stride]
    frames = []
    for row_idx in subsampled:
        frame_idx = row_idx - row_indices[0]
        global_idx = row_indices[0] + frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, primary_size)
            frames.append(frame)
    cap.release()
    return frames


def plot_episode_comparison(ep_idx, eef_full, eef_ds, actions_full, actions_ds, output_dir):
    """Plot raw vs downsampled for a single episode."""
    fig = plt.figure(figsize=(20, 15))

    # 1. 3D trajectory comparison
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(eef_full[:, 0]*1000, eef_full[:, 1]*1000, eef_full[:, 2]*1000,
            'b-', alpha=0.4, linewidth=1, label=f'30fps ({len(eef_full)} pts)')
    ax.plot(eef_ds[:, 0]*1000, eef_ds[:, 1]*1000, eef_ds[:, 2]*1000,
            'r.-', linewidth=2, markersize=6, label=f'5Hz ({len(eef_ds)} pts)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('EEF Trajectory: Raw vs Downsampled')
    ax.legend(fontsize=8)

    # 2. Per-step delta magnitudes (position only)
    ax2 = fig.add_subplot(2, 3, 2)
    full_pos_norms = np.linalg.norm(actions_full[:, :3], axis=1) * 1000
    ds_pos_norms = np.linalg.norm(actions_ds[:, :3], axis=1) * 1000
    ax2.plot(full_pos_norms, 'b-', alpha=0.5, label=f'30fps (mean={full_pos_norms.mean():.1f}mm)')
    # Stretch downsampled to same x-axis for visual comparison
    ds_x = np.linspace(0, len(full_pos_norms)-1, len(ds_pos_norms))
    ax2.plot(ds_x, ds_pos_norms, 'r.-', markersize=4, label=f'5Hz (mean={ds_pos_norms.mean():.1f}mm)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Position delta (mm)')
    ax2.set_title('Step Size: Position')
    ax2.legend(fontsize=8)
    ax2.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Bridge ref 10mm')

    # 3. Per-step rotation magnitudes
    ax3 = fig.add_subplot(2, 3, 3)
    full_rot_norms = np.linalg.norm(actions_full[:, 3:6], axis=1)
    ds_rot_norms = np.linalg.norm(actions_ds[:, 3:6], axis=1)
    ax3.plot(np.degrees(full_rot_norms), 'b-', alpha=0.5,
             label=f'30fps (mean={np.degrees(full_rot_norms.mean()):.1f}deg)')
    ds_x = np.linspace(0, len(full_rot_norms)-1, len(ds_rot_norms))
    ax3.plot(ds_x, np.degrees(ds_rot_norms), 'r.-', markersize=4,
             label=f'5Hz (mean={np.degrees(ds_rot_norms.mean()):.1f}deg)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Rotation delta (deg)')
    ax3.set_title('Step Size: Rotation')
    ax3.legend(fontsize=8)

    # 4. Per-dimension action comparison (dx, dy, dz)
    dim_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw']
    for i, name in enumerate(dim_names[:3]):
        ax4 = fig.add_subplot(2, 3, 4 + i)
        ax4.plot(actions_full[:, i]*1000, 'b-', alpha=0.4, linewidth=0.8, label='30fps')
        ds_x = np.linspace(0, len(actions_full)-1, len(actions_ds))
        ax4.plot(ds_x, actions_ds[:, i]*1000, 'r.-', markersize=3, label='5Hz')
        ax4.set_xlabel('Step')
        ax4.set_ylabel(f'{name} (mm)')
        ax4.set_title(f'{name}: Raw vs Downsampled')
        ax4.legend(fontsize=7)

    fig.suptitle(f'Episode {ep_idx} — Data Quality Check (30fps vs 5Hz)', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, f'ep{ep_idx}_raw_vs_downsampled.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_dataset_summary(all_full_actions, all_ds_actions, output_dir):
    """Plot aggregate statistics across all episodes."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    full = np.concatenate(all_full_actions)
    ds = np.concatenate(all_ds_actions)

    # Position delta magnitude histograms
    full_pos = np.linalg.norm(full[:, :3], axis=1) * 1000
    ds_pos = np.linalg.norm(ds[:, :3], axis=1) * 1000

    axes[0, 0].hist(full_pos, bins=50, alpha=0.6, color='blue', label=f'30fps (n={len(full_pos)})')
    axes[0, 0].hist(ds_pos, bins=50, alpha=0.6, color='red', label=f'5Hz (n={len(ds_pos)})')
    axes[0, 0].axvline(x=10, color='g', linestyle='--', label='Bridge ref 10mm')
    axes[0, 0].set_xlabel('Position delta (mm)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Position Step Size Distribution')
    axes[0, 0].legend(fontsize=8)

    # Rotation delta magnitude histograms
    full_rot = np.degrees(np.linalg.norm(full[:, 3:6], axis=1))
    ds_rot = np.degrees(np.linalg.norm(ds[:, 3:6], axis=1))

    axes[0, 1].hist(full_rot, bins=50, alpha=0.6, color='blue', label='30fps')
    axes[0, 1].hist(ds_rot, bins=50, alpha=0.6, color='red', label='5Hz')
    axes[0, 1].set_xlabel('Rotation delta (deg)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Rotation Step Size Distribution')
    axes[0, 1].legend(fontsize=8)

    # Per-dimension box plots for downsampled
    dim_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'grip']
    ds_mm = ds.copy()
    ds_mm[:, :3] *= 1000
    ds_mm[:, 3:6] = np.degrees(ds_mm[:, 3:6])
    bp = axes[0, 2].boxplot([ds_mm[:, i] for i in range(6)], labels=dim_names[:6])
    axes[0, 2].set_title('5Hz Action Distribution (mm / deg)')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Percentile table
    ax_table = axes[1, 0]
    ax_table.axis('off')
    headers = ['Metric', '30fps', '5Hz', 'Ratio']
    rows = [
        ['Pos mean (mm)', f'{full_pos.mean():.1f}', f'{ds_pos.mean():.1f}', f'{ds_pos.mean()/full_pos.mean():.1f}x'],
        ['Pos p50 (mm)', f'{np.median(full_pos):.1f}', f'{np.median(ds_pos):.1f}', f'{np.median(ds_pos)/np.median(full_pos):.1f}x'],
        ['Pos p99 (mm)', f'{np.percentile(full_pos, 99):.1f}', f'{np.percentile(ds_pos, 99):.1f}', f'{np.percentile(ds_pos, 99)/np.percentile(full_pos, 99):.1f}x'],
        ['Rot mean (deg)', f'{full_rot.mean():.2f}', f'{ds_rot.mean():.2f}', f'{ds_rot.mean()/full_rot.mean():.1f}x'],
        ['Steps/ep', f'{len(full)/len(all_full_actions):.0f}', f'{len(ds)/len(all_ds_actions):.0f}', f'{len(ds)/len(full):.2f}'],
        ['Total steps', f'{len(full)}', f'{len(ds)}', ''],
    ]
    table = ax_table.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax_table.set_title('Summary Statistics', fontsize=12, pad=20)

    # Cumulative displacement per episode
    ax_cum = axes[1, 1]
    for i, (fa, da) in enumerate(zip(all_full_actions, all_ds_actions)):
        full_cum = np.cumsum(np.linalg.norm(fa[:, :3], axis=1)) * 1000
        ds_cum = np.cumsum(np.linalg.norm(da[:, :3], axis=1)) * 1000
        ax_cum.plot(np.linspace(0, 1, len(full_cum)), full_cum, 'b-', alpha=0.3, linewidth=0.8)
        ax_cum.plot(np.linspace(0, 1, len(ds_cum)), ds_cum, 'r-', alpha=0.5, linewidth=1.5)
    ax_cum.set_xlabel('Episode progress (normalized)')
    ax_cum.set_ylabel('Cumulative displacement (mm)')
    ax_cum.set_title('Total Path Length: Raw (blue) vs 5Hz (red)')

    # Gripper
    axes[1, 2].hist(full[:, 6], bins=3, alpha=0.6, color='blue', label='30fps')
    axes[1, 2].hist(ds[:, 6], bins=3, alpha=0.6, color='red', label='5Hz')
    axes[1, 2].set_xlabel('Gripper value')
    axes[1, 2].set_title('Gripper Distribution')
    axes[1, 2].legend(fontsize=8)

    fig.suptitle('Dataset Quality Summary — 30fps vs 5Hz Downsampling', fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'dataset_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_image_strip(frames, ep_idx, output_dir):
    """Show what the model sees at each downsampled timestep."""
    if not frames or len(frames) < 2:
        return None
    n = min(10, len(frames))
    indices = np.linspace(0, len(frames)-1, n, dtype=int)
    selected = [frames[i] for i in indices]
    strip = np.concatenate(selected, axis=1)
    path = os.path.join(output_dir, f'ep{ep_idx}_frame_strip.png')
    cv2.imwrite(path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return path


def main():
    parser = argparse.ArgumentParser(description="Visualize RLDS data quality after downsampling")
    parser.add_argument("--config", required=True, help="Path to lerobot conversion YAML config")
    parser.add_argument("--output_dir", default="./data/viz_output/data_quality", help="Output directory")
    parser.add_argument("--max_episodes", type=int, default=5, help="Max episodes to visualize")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    right_arm_indices = cfg.get('right_arm_indices', list(range(7, 14)))
    fk_joint_indices = cfg.get('fk_joint_indices', list(range(0, 6)))
    gripper_index = cfg.get('gripper_index', 6)
    original_fps = cfg.get('original_fps', 30)
    target_hz = cfg.get('target_hz', 5.0)
    stride = max(1, int(round(original_fps / target_hz)))
    primary_camera = cfg.get('primary_camera', 'cam_high')
    primary_size = tuple(cfg.get('primary_size', [256, 256]))

    print(f"Config: {original_fps}fps -> {target_hz}Hz (stride={stride})")
    print(f"Arm indices: {right_arm_indices}, FK joints: {fk_joint_indices}")
    print(f"Records: {len(cfg['record_dirs'])} dirs")

    all_full_actions = []
    all_ds_actions = []
    ep_global = 0

    for record_dir in cfg['record_dirs']:
        record_name = os.path.basename(record_dir)
        print(f"\n  {record_name}...")

        data_files = sorted(glob.glob(os.path.join(record_dir, "data", "chunk-*", "*.parquet")))
        if not data_files:
            print(f"    No parquet files found, skipping")
            continue
        table = pa.concat_tables([pq.read_table(f) for f in data_files])

        episode_groups = {}
        for i in range(len(table)):
            ep_idx = table.column("episode_index")[i].as_py()
            if ep_idx not in episode_groups:
                episode_groups[ep_idx] = []
            episode_groups[ep_idx].append(i)

        for ep_idx in sorted(episode_groups.keys()):
            if ep_global >= args.max_episodes:
                # Still collect stats for summary
                row_indices = episode_groups[ep_idx]
                if len(row_indices) < stride * 2:
                    continue
                _, full_act = load_raw_episode(table, row_indices, right_arm_indices,
                                                fk_joint_indices, gripper_index)
                _, ds_act = load_downsampled_episode(table, row_indices, right_arm_indices,
                                                      fk_joint_indices, gripper_index, stride)
                all_full_actions.append(full_act)
                all_ds_actions.append(ds_act)
                continue

            row_indices = episode_groups[ep_idx]
            if len(row_indices) < stride * 2:
                continue

            eef_full, full_act = load_raw_episode(table, row_indices, right_arm_indices,
                                                   fk_joint_indices, gripper_index)
            eef_ds, ds_act = load_downsampled_episode(table, row_indices, right_arm_indices,
                                                       fk_joint_indices, gripper_index, stride)

            all_full_actions.append(full_act)
            all_ds_actions.append(ds_act)

            path = plot_episode_comparison(ep_global, eef_full, eef_ds, full_act, ds_act, output_dir)
            print(f"    ep{ep_global}: {len(row_indices)} frames -> {len(row_indices)//stride} steps | {path}")

            # Image strip
            frames = load_downsampled_images(record_dir, primary_camera, row_indices, stride, primary_size)
            img_path = plot_image_strip(frames, ep_global, output_dir)
            if img_path:
                print(f"    frames: {img_path}")

            ep_global += 1

    # Dataset summary
    if all_full_actions:
        path = plot_dataset_summary(all_full_actions, all_ds_actions, output_dir)
        print(f"\n  Summary: {path}")

    print(f"\nDone. {len(all_full_actions)} episodes analyzed, outputs in {output_dir}")


if __name__ == "__main__":
    main()
