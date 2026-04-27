# 08_visualize_3d_trajectory.py: 3D point cloud + trajectory visualization for
# 08_visualize_3d_trajectory.py: Octo zero-shot predictions vs GT on xarm mcap data.

"""
Adapts the evaluation/visualization protocol from 3d_flowmatch_actor's
horizon_eval.py for Octo. Reads mcap bags, extracts depth -> point clouds,
runs Octo inference with GT-anchored autoregressive rollout, and produces
3D plots showing predicted vs GT EEF trajectories over the scene point cloud.

Usage (inside octo-inference container):
    python3 examples/08_visualize_3d_trajectory.py \
        --data_dir /data/xarm/default_task \
        --max_episodes 3 \
        --horizon 5
"""

import argparse
import json
import os
from collections import deque

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import jax
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from octo.model.octo_model import OctoModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IM_SIZE_PRIMARY = 256
IM_SIZE_WRIST = 128
WINDOW_SIZE = 2
ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


# ---------------------------------------------------------------------------
# MCAP reading (adapted from horizon_eval.py)
# ---------------------------------------------------------------------------

def read_episode_bag(bag_dir, target_hz=10.0, sync_slop=0.05):
    """Read an mcap episode bag and return synced frames with depth + intrinsics.

    Returns dict with frames (front_rgb, front_depth, wrist_rgb, wrist_depth,
    eef_pose, gripper, wrist_E, timestamp), camera intrinsics, and front camera pose.
    """
    from mcap_ros2.reader import read_ros2_messages
    from pathlib import Path

    bag_dir = Path(bag_dir)
    mcap_files = list(bag_dir.glob("*.mcap"))
    if not mcap_files:
        return None

    mcap_path = str(mcap_files[0])

    # Collect all messages by topic
    topic_data = {}
    for msg in read_ros2_messages(mcap_path):
        topic = msg.channel.topic
        if topic not in topic_data:
            topic_data[topic] = []
        # mcap_ros2 returns log_time as datetime; convert to float seconds
        t = msg.log_time
        if hasattr(t, 'timestamp'):
            t = t.timestamp()  # datetime -> float seconds
        else:
            t = float(t) / 1e9  # nanosecond int -> float seconds
        topic_data[topic].append((t, msg.ros_msg))

    # Sort by time
    for t in topic_data:
        topic_data[t].sort(key=lambda x: x[0])

    # Extract camera intrinsics and front camera pose (static, take first)
    front_info = topic_data.get("/front/camera_info", [])
    wrist_info = topic_data.get("/wrist/camera_info", [])
    front_pose_msgs = topic_data.get("/front/pose", [])

    if not front_info or not wrist_info:
        print("  Missing camera info, skipping")
        return None

    front_K_full = np.array(front_info[0][1].k, dtype=np.float32).reshape(3, 3)
    wrist_K_full = np.array(wrist_info[0][1].k, dtype=np.float32).reshape(3, 3)

    # Front camera extrinsic
    if front_pose_msgs:
        fp = front_pose_msgs[0][1]
        p, q = fp.pose.position, fp.pose.orientation
        front_pose_7d = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
        front_E = _pose_7d_to_4x4(front_pose_7d)
    else:
        front_E = np.eye(4, dtype=np.float32)

    # Get original image sizes for intrinsics scaling
    first_front = topic_data["/front/color/image_raw"][0][1]
    first_wrist = topic_data["/wrist/color/image_raw"][0][1]
    front_orig_w, front_orig_h = first_front.width, first_front.height
    wrist_orig_w, wrist_orig_h = first_wrist.width, first_wrist.height

    # Scale intrinsics to point cloud resolution (256x256)
    pcd_size = 256
    front_K_pcd = _scale_intrinsics(front_K_full, front_orig_w, front_orig_h, pcd_size)
    wrist_K_pcd = _scale_intrinsics(wrist_K_full, wrist_orig_w, wrist_orig_h, pcd_size)

    # Sync and resample at target_hz
    eef_msgs = topic_data.get("/eef_pose", [])
    if len(eef_msgs) < 2:
        return None

    t_start, t_end = eef_msgs[0][0], eef_msgs[-1][0]
    sample_times = np.arange(t_start, t_end, 1.0 / target_hz)

    def _find_closest(target_time, messages, slop):
        best, best_diff = None, float("inf")
        for t, msg in messages:
            diff = abs(t - target_time)
            if diff < best_diff and diff <= slop:
                best_diff, best = diff, msg
        return best

    frames = []
    for t in sample_times:
        eef = _find_closest(t, eef_msgs, sync_slop * 2)
        fr = _find_closest(t, topic_data.get("/front/color/image_raw", []), sync_slop)
        fd = _find_closest(t, topic_data.get("/front/depth/image_raw", []), sync_slop)
        wr = _find_closest(t, topic_data.get("/wrist/color/image_raw", []), sync_slop)
        wd = _find_closest(t, topic_data.get("/wrist/depth/image_raw", []), sync_slop)
        wp = _find_closest(t, topic_data.get("/wrist/pose", []), sync_slop)
        gr = _find_closest(t, topic_data.get("/gripper_position", []), sync_slop * 2)

        if any(x is None for x in [eef, fr, fd, wr, wd]):
            continue

        p, q = eef.pose.position, eef.pose.orientation
        eef_pose = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
        gripper_val = np.float32(gr.data if gr else 0.0)

        wrist_E = np.eye(4, dtype=np.float32)
        if wp:
            wp_p, wp_q = wp.pose.position, wp.pose.orientation
            wrist_E = _pose_7d_to_4x4(
                np.array([wp_p.x, wp_p.y, wp_p.z, wp_q.x, wp_q.y, wp_q.z, wp_q.w], dtype=np.float32)
            )

        frames.append({
            "front_rgb": _decode_image(fr),
            "front_depth": _decode_depth(fd),
            "wrist_rgb": _decode_image(wr),
            "wrist_depth": _decode_depth(wd),
            "wrist_E": wrist_E,
            "eef_pose": eef_pose,
            "gripper": gripper_val,
            "timestamp": t - t_start,
        })

    if len(frames) < 2:
        return None

    return {
        "frames": frames,
        "front_E": front_E,
        "front_K_pcd": front_K_pcd,
        "wrist_K_pcd": wrist_K_pcd,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(msg):
    h, w = msg.height, msg.width
    enc = msg.encoding
    if enc in ("rgb8", "RGB8"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    elif enc in ("bgr8", "BGR8"):
        return cv2.cvtColor(np.frombuffer(msg.data, np.uint8).reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image encoding: {enc}")


def _decode_depth(msg):
    h, w = msg.height, msg.width
    enc = msg.encoding
    if enc == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    elif enc == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    raise ValueError(f"Unsupported depth encoding: {enc}")


def _pose_7d_to_4x4(pose):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def _scale_intrinsics(K, orig_w, orig_h, target):
    K = K.copy()
    K[0, 0] *= target / orig_w
    K[1, 1] *= target / orig_h
    K[0, 2] *= target / orig_w
    K[1, 2] *= target / orig_h
    return K


def depth_to_pointcloud(depth, K, extrinsic, rgb=None):
    """Convert depth image to 3D point cloud in world frame.

    Args:
        depth: (H, W) float32, depth in meters
        K: (3, 3) intrinsic matrix
        extrinsic: (4, 4) camera-to-world transform
        rgb: optional (H, W, 3) uint8 colors

    Returns:
        points: (N, 3) world coordinates
        colors: (N, 3) uint8 colors (if rgb provided)
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()
    valid = (z > 0.01) & (z < 5.0) & np.isfinite(z)

    x = (u.flatten()[valid] - cx) * z[valid] / fx
    y = (v.flatten()[valid] - cy) * z[valid] / fy
    z = z[valid]

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)  # (4, N)
    pts_world = (extrinsic @ pts_cam)[:3].T  # (N, 3)

    colors_out = None
    if rgb is not None:
        colors_out = rgb.reshape(-1, 3)[valid]

    return pts_world, colors_out


def subsample_pointcloud(pts, colors=None, max_points=8000):
    """Subsample point cloud and filter outliers."""
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 5.0).all(axis=1)
    pts = pts[valid]
    if colors is not None:
        colors = colors[valid]
    if len(pts) > max_points:
        idx = np.random.RandomState(42).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]
    return pts, colors


def quat_to_euler(qx, qy, qz, qw):
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_euler("xyz")


def apply_delta_action(current_pose_7d, delta_action):
    """Apply a normalized delta action to get next pose.

    Since Octo outputs normalized actions (not calibrated to xarm),
    we scale the deltas to a reasonable range for visualization.
    """
    # Scale factor: Octo's normalized actions are roughly in [-2, 2]
    # xarm workspace is ~0.3m, so scale position deltas
    pos_scale = 0.005  # 5mm per unit of normalized action
    rot_scale = 0.02   # ~1 degree per unit

    new_pose = current_pose_7d.copy()
    # Position delta
    new_pose[:3] += delta_action[:3] * pos_scale
    # Rotation delta (apply as euler increments)
    curr_euler = quat_to_euler(*current_pose_7d[3:7])
    new_euler = curr_euler + delta_action[3:6] * rot_scale
    new_quat = Rotation.from_euler("xyz", new_euler).as_quat()
    new_pose[3:7] = new_quat
    return new_pose


# ---------------------------------------------------------------------------
# Octo inference connector
# ---------------------------------------------------------------------------

def run_octo_horizon_eval(model, episode_data, task, horizon=5):
    """Run GT-anchored autoregressive Octo inference, mirroring horizon_eval.py.

    Protocol:
      1. Anchor at GT frame N
      2. Feed GT images to Octo, predict action
      3. Integrate predicted action to get predicted EEF pose
      4. For next steps in horizon, keep same images but update predicted pose
      5. Hop to next GT anchor at N+horizon
      6. Repeat

    Returns:
        dict with gt_all, segments (each with anchor, preds, gt, errors),
        and point cloud data for visualization.
    """
    frames = episode_data["frames"]
    N = len(frames)
    front_E = episode_data["front_E"]
    front_K = episode_data["front_K_pcd"]
    wrist_K = episode_data["wrist_K_pcd"]

    gt_all = np.array([f["eef_pose"] for f in frames])  # (N, 7)
    segments = []
    anchor_indices = list(range(0, N - 1, horizon))

    for anchor in tqdm(anchor_indices, desc="  Segments"):
        anchor_pose = gt_all[anchor].copy()

        # Prepare observation images from GT anchor frame
        front_rgb = cv2.resize(frames[anchor]["front_rgb"],
                               (IM_SIZE_PRIMARY, IM_SIZE_PRIMARY))
        wrist_rgb = cv2.resize(frames[anchor]["wrist_rgb"],
                               (IM_SIZE_WRIST, IM_SIZE_WRIST))

        # For window_size=2, use previous frame too (or pad)
        if anchor > 0:
            front_rgb_prev = cv2.resize(frames[anchor - 1]["front_rgb"],
                                        (IM_SIZE_PRIMARY, IM_SIZE_PRIMARY))
            wrist_rgb_prev = cv2.resize(frames[anchor - 1]["wrist_rgb"],
                                        (IM_SIZE_WRIST, IM_SIZE_WRIST))
        else:
            front_rgb_prev = front_rgb
            wrist_rgb_prev = wrist_rgb

        front_window = np.stack([front_rgb_prev, front_rgb])[None]  # (1, 2, H, W, 3)
        wrist_window = np.stack([wrist_rgb_prev, wrist_rgb])[None]

        observation = {
            "image_primary": front_window,
            "image_wrist": wrist_window,
            "timestep_pad_mask": np.array([[anchor > 0, True]]),
        }

        current_pose = anchor_pose.copy()
        seg_preds, seg_gt, seg_errors = [], [], []

        for step in range(horizon):
            target_idx = anchor + step + 1
            if target_idx >= N:
                break

            # Sample action from Octo
            actions = model.sample_actions(
                observation, task,
                rng=jax.random.PRNGKey(anchor * horizon + step),
            )
            delta = np.array(actions[0, 0])  # first action in chunk, (7,)

            # Integrate: apply delta to get predicted next pose
            pred_pose = apply_delta_action(current_pose, delta)
            gt_pose = gt_all[target_idx]

            pos_err = np.linalg.norm(pred_pose[:3] - gt_pose[:3])
            seg_preds.append(pred_pose.copy())
            seg_gt.append(gt_pose.copy())
            seg_errors.append(pos_err)

            # Update current pose for next autoregressive step
            current_pose = pred_pose

        if seg_preds:
            segments.append({
                "anchor_idx": anchor,
                "anchor_pos": anchor_pose[:3].copy(),
                "preds": np.array(seg_preds),
                "gt": np.array(seg_gt),
                "errors": np.array(seg_errors),
            })

    # Generate point clouds from a representative frame (middle of episode)
    mid = N // 2
    pcd_size = 256
    front_depth_resized = cv2.resize(
        frames[mid]["front_depth"], (pcd_size, pcd_size),
        interpolation=cv2.INTER_NEAREST,
    )
    front_rgb_resized = cv2.resize(frames[mid]["front_rgb"], (pcd_size, pcd_size))
    front_pts, front_colors = depth_to_pointcloud(
        front_depth_resized, front_K, front_E, front_rgb_resized,
    )
    front_pts, front_colors = subsample_pointcloud(front_pts, front_colors)

    wrist_depth_resized = cv2.resize(
        frames[mid]["wrist_depth"], (pcd_size, pcd_size),
        interpolation=cv2.INTER_NEAREST,
    )
    wrist_rgb_resized = cv2.resize(frames[mid]["wrist_rgb"], (pcd_size, pcd_size))
    wrist_E_mid = frames[mid]["wrist_E"]
    wrist_pts, wrist_colors = depth_to_pointcloud(
        wrist_depth_resized, wrist_K, wrist_E_mid, wrist_rgb_resized,
    )
    wrist_pts, wrist_colors = subsample_pointcloud(wrist_pts, wrist_colors)

    return {
        "num_frames": N,
        "horizon": horizon,
        "gt_all": gt_all,
        "segments": segments,
        "front_pts": front_pts,
        "front_colors": front_colors,
        "wrist_pts": wrist_pts,
        "wrist_colors": wrist_colors,
    }


# ---------------------------------------------------------------------------
# 3D Visualization (adapted from horizon_eval.py)
# ---------------------------------------------------------------------------

def plot_3d_trajectory(result, episode_name, output_dir):
    """Create 3-panel figure: 3D scene with PCD + trajectories, XY projection, error bars."""
    gt_all = result["gt_all"]
    segments = result["segments"]
    m = result["horizon"]
    front_pts = result["front_pts"]
    front_colors = result["front_colors"]
    wrist_pts = result["wrist_pts"]
    wrist_colors = result["wrist_colors"]

    gt_pos = gt_all[:, :3]
    N = len(gt_pos)

    fig = plt.figure(figsize=(22, 8))

    # ---- Panel 1: 3D scene with point cloud + trajectories ----
    ax1 = fig.add_subplot(131, projection="3d")

    # Point cloud backdrop (low alpha so trajectories are visible)
    if front_pts is not None and len(front_pts) > 0:
        ax1.scatter(
            front_pts[:, 0], front_pts[:, 1], front_pts[:, 2],
            c=front_colors / 255.0 if front_colors is not None else "lightblue",
            s=0.3, alpha=0.15, depthshade=False,
        )
    if wrist_pts is not None and len(wrist_pts) > 0:
        ax1.scatter(
            wrist_pts[:, 0], wrist_pts[:, 1], wrist_pts[:, 2],
            c=wrist_colors / 255.0 if wrist_colors is not None else "lightyellow",
            s=0.3, alpha=0.1, depthshade=False,
        )

    # GT trajectory
    cmap_gt = plt.cm.Greens(np.linspace(0.3, 1.0, N))
    ax1.scatter(
        gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
        c=cmap_gt, s=10, alpha=0.7, depthshade=False, label="GT",
    )
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
             color="#2ecc71", linewidth=1.0, alpha=0.5)

    # Predicted segments
    seg_colors = plt.cm.Reds(np.linspace(0.4, 1.0, max(len(segments), 1)))
    for i, seg in enumerate(segments):
        # Anchor marker
        ax1.scatter(
            *seg["anchor_pos"], color="#00ff88", s=80, marker="^",
            edgecolors="k", linewidths=0.5, zorder=10,
        )
        # Predicted trajectory from anchor
        preds = seg["preds"][:, :3]
        full = np.vstack([seg["anchor_pos"][None], preds])
        ax1.plot(
            full[:, 0], full[:, 1], full[:, 2], "-o",
            color=seg_colors[i], markersize=4, linewidth=1.8, alpha=0.85,
        )
        # Error lines connecting predicted to GT
        for j in range(len(seg["preds"])):
            ax1.plot(
                [seg["preds"][j, 0], seg["gt"][j, 0]],
                [seg["preds"][j, 1], seg["gt"][j, 1]],
                [seg["preds"][j, 2], seg["gt"][j, 2]],
                "--", color="gray", linewidth=0.5, alpha=0.4,
            )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"3D Scene — {episode_name}")
    ax1.view_init(elev=25, azim=-60)

    # ---- Panel 2: XY projection ----
    ax2 = fig.add_subplot(132)

    # Point cloud XY projection as background
    if front_pts is not None and len(front_pts) > 0:
        ax2.scatter(
            front_pts[:, 0], front_pts[:, 1],
            c=front_colors / 255.0 if front_colors is not None else "lightblue",
            s=0.2, alpha=0.1,
        )

    ax2.scatter(gt_pos[:, 0], gt_pos[:, 1], c=cmap_gt, s=8, alpha=0.6, label="GT")
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], color="#2ecc71", linewidth=0.8, alpha=0.4)

    for i, seg in enumerate(segments):
        preds = seg["preds"][:, :3]
        full = np.vstack([seg["anchor_pos"][None], preds])
        ax2.plot(full[:, 0], full[:, 1], "-o", color=seg_colors[i],
                 markersize=4, linewidth=1.5, alpha=0.8)
        ax2.plot(seg["anchor_pos"][0], seg["anchor_pos"][1], "^",
                 color="#00ff88", markersize=8, zorder=10)

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("XY Projection")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Error by rollout step ----
    ax3 = fig.add_subplot(133)
    step_errors = {i: [] for i in range(m)}
    for seg in segments:
        for j, err in enumerate(seg["errors"]):
            step_errors[j].append(err * 100)  # convert to cm

    steps = sorted(step_errors.keys())
    means = [np.mean(step_errors[s]) if step_errors[s] else 0 for s in steps]
    stds = [np.std(step_errors[s]) if step_errors[s] else 0 for s in steps]
    ax3.bar(
        [s + 1 for s in steps], means, yerr=stds, capsize=4,
        color="#3498db", alpha=0.75, edgecolor="#2c3e50",
    )
    ax3.set_xlabel("Step within horizon")
    ax3.set_ylabel("Position Error (cm)")
    ax3.set_title("Error vs Rollout Step")
    ax3.set_xticks([s + 1 for s in steps])
    ax3.grid(True, alpha=0.3, axis="y")

    # Super title with summary stats
    if segments:
        all_errors_cm = np.concatenate([s["errors"] for s in segments]) * 100
        fig.suptitle(
            f"Octo-Small Zero-Shot | {episode_name} | "
            f"{result['num_frames']} frames | horizon={m} | {len(segments)} segments\n"
            f"Mean err: {all_errors_cm.mean():.2f}cm | Max: {all_errors_cm.max():.2f}cm | "
            f"Step-1: {means[0]:.2f}cm | Step-{m}: {means[-1]:.2f}cm",
            fontsize=12, fontweight="bold",
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{episode_name}_3d_horizon_m{m}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {save_path}")
    return save_path


def plot_pointcloud_only(result, episode_name, output_dir):
    """Standalone point cloud + trajectory overview from multiple angles."""
    gt_all = result["gt_all"]
    segments = result["segments"]
    front_pts = result["front_pts"]
    front_colors = result["front_colors"]
    wrist_pts = result["wrist_pts"]
    wrist_colors = result["wrist_colors"]
    gt_pos = gt_all[:, :3]

    fig = plt.figure(figsize=(20, 10))
    views = [
        ("Front view", 0, -90),
        ("Top-down", 90, -90),
        ("Side view", 0, 0),
        ("3/4 view", 30, -45),
    ]

    for idx, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

        if front_pts is not None and len(front_pts) > 0:
            ax.scatter(
                front_pts[:, 0], front_pts[:, 1], front_pts[:, 2],
                c=front_colors / 255.0 if front_colors is not None else "lightblue",
                s=0.5, alpha=0.2, depthshade=False,
            )

        # GT trajectory
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                color="#2ecc71", linewidth=2.0, alpha=0.8, label="GT")

        # Predicted segments
        seg_colors = plt.cm.Reds(np.linspace(0.4, 1.0, max(len(segments), 1)))
        for i, seg in enumerate(segments):
            preds = seg["preds"][:, :3]
            full = np.vstack([seg["anchor_pos"][None], preds])
            ax.plot(full[:, 0], full[:, 1], full[:, 2], "-o",
                    color=seg_colors[i], markersize=3, linewidth=1.5, alpha=0.8)
            ax.scatter(*seg["anchor_pos"], color="#00ff88", s=40, marker="^",
                       edgecolors="k", linewidths=0.3, zorder=10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(f"Point Cloud + Trajectories — {episode_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{episode_name}_pcd_views.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Multi-view saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D visualization of Octo zero-shot on xarm mcap")
    parser.add_argument("--data_dir", type=str, default="/data/xarm/default_task")
    parser.add_argument("--checkpoint", type=str, default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--task_instruction", type=str, default="do the task")
    parser.add_argument("--max_episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/octo/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Octo model
    print(f"Loading model: {args.checkpoint}")
    model = OctoModel.load_pretrained(args.checkpoint)
    task = model.create_tasks(texts=[args.task_instruction])
    print(f"Task: '{args.task_instruction}'")

    # Find episode directories
    import glob
    episode_dirs = sorted(glob.glob(os.path.join(args.data_dir, "episode_*_bag")))
    if args.max_episodes:
        episode_dirs = episode_dirs[:args.max_episodes]

    all_metrics = []

    for ep_dir in episode_dirs:
        ep_name = os.path.basename(ep_dir)
        print(f"\n{'='*60}")
        print(f"Processing {ep_name}")
        print(f"{'='*60}")

        # Read episode
        episode_data = read_episode_bag(ep_dir)
        if episode_data is None:
            print(f"  Failed to read {ep_dir}, skipping")
            continue

        print(f"  {len(episode_data['frames'])} synced frames")

        # Run horizon evaluation
        result = run_octo_horizon_eval(model, episode_data, task, horizon=args.horizon)

        # Generate plots
        plot_3d_trajectory(result, ep_name, args.output_dir)
        plot_pointcloud_only(result, ep_name, args.output_dir)

        # Save data as npz
        npz_path = os.path.join(args.output_dir, f"{ep_name}_horizon_m{args.horizon}.npz")
        save_dict = {
            "gt_all": result["gt_all"],
            "front_pts": result["front_pts"],
            "wrist_pts": result["wrist_pts"],
        }
        if result["front_colors"] is not None:
            save_dict["front_colors"] = result["front_colors"]
        if result["wrist_colors"] is not None:
            save_dict["wrist_colors"] = result["wrist_colors"]
        for i, seg in enumerate(result["segments"]):
            save_dict[f"seg{i}_anchor"] = seg["anchor_pos"]
            save_dict[f"seg{i}_preds"] = seg["preds"]
            save_dict[f"seg{i}_gt"] = seg["gt"]
            save_dict[f"seg{i}_errors"] = seg["errors"]
        np.savez_compressed(npz_path, **save_dict)
        print(f"  Data saved: {npz_path}")

        # Collect metrics
        seg_summaries = []
        for seg in result["segments"]:
            seg_summaries.append({
                "anchor_idx": seg["anchor_idx"],
                "num_steps": len(seg["errors"]),
                "mean_error_cm": float(seg["errors"].mean() * 100),
                "max_error_cm": float(seg["errors"].max() * 100),
                "per_step_error_cm": [float(e * 100) for e in seg["errors"]],
            })
        all_metrics.append({
            "episode": ep_name,
            "horizon": args.horizon,
            "num_frames": result["num_frames"],
            "num_segments": len(result["segments"]),
            "segments": seg_summaries,
        })

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, f"metrics_octo_m{args.horizon}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
