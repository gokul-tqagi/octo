# validate_rlds.py: Visual validation of post-extracted RLDS TFRecord dataset.
# validate_rlds.py: Loads TFRecords, decodes images/actions/states, renders 3D trajectory + image panels.

"""
Validate extracted RLDS TFRecord dataset by visualizing:
  1. 3D EEF trajectory with orientation frames and action arrows
  2. Gripper state color-coded along trajectory (green=open, red=closed)
  3. Decoded RGB image contact sheets (front + wrist)
  4. Action and state statistics sanity checks

Reads the POST-EXTRACTED TFRecords (not raw mcap) to verify the
serialization roundtrip is correct.

Produces:
  - Interactive Open3D window (if display available)
  - Static plotly HTML report (always, for headless/remote use)
  - Image contact sheets as PNG

Usage:
    # Inside Docker (headless, saves to disk):
    python scripts/validate_rlds.py \
        --data_dir /data/rlds \
        --dataset_name xarm_place_toolbox \
        --output_dir /data/validation_output

    # On host with display (interactive Open3D):
    python scripts/validate_rlds.py \
        --data_dir ./data/xarm_place_toolbox_rlds \
        --dataset_name xarm_place_toolbox \
        --interactive
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import open3d as o3d
    HAS_O3D = True
except (ImportError, AttributeError):
    HAS_O3D = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ── RLDS loading ──────────────────────────────────────────────────────────────

def _parse_step(step_bytes):
    """Parse a serialized step Example back into arrays."""
    step_ex = tf.train.Example()
    step_ex.ParseFromString(step_bytes)
    sf = step_ex.features.feature

    result = {}

    # Action
    if "action" in sf:
        result["action"] = np.array(sf["action"].float_list.value, dtype=np.float32)

    # State
    if "observation/state" in sf:
        result["state"] = np.array(
            sf["observation/state"].float_list.value, dtype=np.float32
        )

    # Front image (JPEG bytes -> numpy)
    if "observation/image_0" in sf:
        jpg_bytes = sf["observation/image_0"].bytes_list.value[0]
        img_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        result["front_image"] = cv2.cvtColor(
            cv2.imdecode(img_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )

    # Wrist image
    if "observation/image_1" in sf:
        jpg_bytes = sf["observation/image_1"].bytes_list.value[0]
        img_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        result["wrist_image"] = cv2.cvtColor(
            cv2.imdecode(img_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )

    # Language
    if "language_instruction" in sf:
        result["language"] = sf["language_instruction"].bytes_list.value[0].decode("utf-8")

    return result


def load_rlds_episodes(data_dir, dataset_name, max_episodes=None):
    """Load episodes from RLDS TFRecord by parsing our custom serialization format."""
    if not HAS_TF:
        raise ImportError("tensorflow required. Run inside Docker container.")

    ds_path = os.path.join(data_dir, dataset_name, "1.0.0")
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dataset not found at {ds_path}")

    # Find train tfrecord files
    import glob
    tfrecord_files = sorted(glob.glob(os.path.join(ds_path, "*-train.tfrecord*")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No train tfrecord files in {ds_path}")

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    episodes = []
    for i, raw_record in enumerate(dataset):
        if max_episodes is not None and i >= max_episodes:
            break

        # Parse trajectory-level example
        traj_ex = tf.train.Example()
        traj_ex.ParseFromString(raw_record.numpy())
        features = traj_ex.features.feature

        # "steps" contains serialized step Examples as bytes
        if "steps" not in features:
            print(f"  Episode {i}: no 'steps' key, skipping")
            continue

        steps_bytes = features["steps"].bytes_list.value

        ep_data = {
            "front_images": [],
            "wrist_images": [],
            "states": [],
            "actions": [],
            "language": "",
        }

        for step_bytes in steps_bytes:
            step = _parse_step(step_bytes)
            if "action" in step:
                ep_data["actions"].append(step["action"])
            if "state" in step:
                ep_data["states"].append(step["state"])
            if "front_image" in step:
                ep_data["front_images"].append(step["front_image"])
            if "wrist_image" in step:
                ep_data["wrist_images"].append(step["wrist_image"])
            if "language" in step and step["language"]:
                ep_data["language"] = step["language"]

        ep_data["states"] = np.array(ep_data["states"]) if ep_data["states"] else np.array([])
        ep_data["actions"] = np.array(ep_data["actions"]) if ep_data["actions"] else np.array([])

        episodes.append(ep_data)
        print(f"  Episode {i}: {len(steps_bytes)} steps, "
              f"states {ep_data['states'].shape}, "
              f"actions {ep_data['actions'].shape}, "
              f"lang=\"{ep_data['language'][:50]}\"")

    return episodes


# ── Statistics and sanity checks ──────────────────────────────────────────────

def print_dataset_stats(episodes):
    """Print comprehensive dataset statistics for sanity checking."""
    all_states = np.concatenate([ep["states"] for ep in episodes])
    all_actions = np.concatenate([ep["actions"] for ep in episodes])

    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS ({len(episodes)} episodes)")
    print(f"{'='*60}")

    print(f"\nTotal steps: {len(all_states)}")
    print(f"Steps per episode: {[len(ep['states']) for ep in episodes]}")

    print(f"\n--- States [x, y, z, roll, pitch, yaw, gripper] ---")
    print(f"  Mean:  {np.array2string(all_states.mean(0), precision=4)}")
    print(f"  Std:   {np.array2string(all_states.std(0), precision=4)}")
    print(f"  Min:   {np.array2string(all_states.min(0), precision=4)}")
    print(f"  Max:   {np.array2string(all_states.max(0), precision=4)}")

    pos = all_states[:, :3]
    print(f"\n--- Workspace ---")
    print(f"  X: [{pos[:,0].min():.4f}, {pos[:,0].max():.4f}] = "
          f"{(pos[:,0].max()-pos[:,0].min())*100:.1f} cm")
    print(f"  Y: [{pos[:,1].min():.4f}, {pos[:,1].max():.4f}] = "
          f"{(pos[:,1].max()-pos[:,1].min())*100:.1f} cm")
    print(f"  Z: [{pos[:,2].min():.4f}, {pos[:,2].max():.4f}] = "
          f"{(pos[:,2].max()-pos[:,2].min())*100:.1f} cm")

    print(f"\n--- Actions [dx, dy, dz, droll, dpitch, dyaw, dgripper] ---")
    print(f"  Mean:  {np.array2string(all_actions.mean(0), precision=6)}")
    print(f"  Std:   {np.array2string(all_actions.std(0), precision=6)}")
    print(f"  Min:   {np.array2string(all_actions.min(0), precision=6)}")
    print(f"  Max:   {np.array2string(all_actions.max(0), precision=6)}")

    step_disp = np.linalg.norm(all_actions[:, :3], axis=1)
    print(f"\n--- Per-step displacement ---")
    print(f"  Mean: {step_disp.mean()*1000:.3f} mm")
    print(f"  Std:  {step_disp.std()*1000:.3f} mm")
    print(f"  Max:  {step_disp.max()*1000:.3f} mm")
    print(f"  Median: {np.median(step_disp)*1000:.3f} mm")

    rot_disp = np.linalg.norm(all_actions[:, 3:6], axis=1)
    print(f"\n--- Per-step rotation ---")
    print(f"  Mean: {np.degrees(rot_disp.mean()):.4f} deg")
    print(f"  Max:  {np.degrees(rot_disp.max()):.4f} deg")

    grip = all_states[:, 6]
    print(f"\n--- Gripper ---")
    print(f"  Unique values: {np.unique(np.round(grip, 2))}")

    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    # Check for NaN/Inf
    has_nan = np.any(np.isnan(all_actions)) or np.any(np.isnan(all_states))
    has_inf = np.any(np.isinf(all_actions)) or np.any(np.isinf(all_states))
    print(f"  NaN in data: {'FAIL' if has_nan else 'OK'}")
    print(f"  Inf in data: {'FAIL' if has_inf else 'OK'}")

    # Check for unreasonable rotation deltas (euler wrapping bug)
    max_rot_delta = np.abs(all_actions[:, 3:6]).max()
    print(f"  Max rotation delta: {np.degrees(max_rot_delta):.1f} deg "
          f"{'FAIL (>90 deg, possible euler wrapping)' if np.degrees(max_rot_delta) > 90 else 'OK'}")

    # Check images decode
    front_shape = episodes[0]["front_images"][0].shape if episodes[0]["front_images"] else None
    wrist_shape = episodes[0]["wrist_images"][0].shape if episodes[0]["wrist_images"] else None
    print(f"  Front image shape: {front_shape}")
    print(f"  Wrist image shape: {wrist_shape}")

    # Compare with bridge dataset reference
    print(f"\n--- Bridge Dataset Comparison ---")
    bridge_disp = 10.26  # mm per step
    our_disp = step_disp.mean() * 1000
    ratio = bridge_disp / our_disp if our_disp > 0 else float("inf")
    print(f"  Bridge mean displacement: {bridge_disp:.2f} mm/step")
    print(f"  Our mean displacement:    {our_disp:.2f} mm/step")
    print(f"  Ratio: {ratio:.1f}x {'(acceptable <15x)' if ratio < 15 else '(WARNING: large gap)'}")
    print(f"{'='*60}\n")


# ── Open3D visualization ─────────────────────────────────────────────────────

def create_trajectory_geometries(episode, episode_idx=0, action_scale=50.0):
    """Create Open3D geometries for a single episode trajectory."""
    geometries = []
    states = episode["states"]
    actions = episode["actions"]
    T = len(states)

    # EEF positions
    positions = states[:, :3]

    # Trajectory line
    line_points = o3d.utility.Vector3dVector(positions)
    line_indices = [[i, i + 1] for i in range(T - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = line_points
    line_set.lines = o3d.utility.Vector2iVector(line_indices)

    # Color by gripper state: green=open (0), red=closed (1)
    grip = states[:, 6]
    colors = []
    for i in range(T - 1):
        g = grip[i]
        colors.append([g, 1.0 - g, 0.0])  # red=closed, green=open
    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(line_set)

    # Start/end markers
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    start_sphere.translate(positions[0])
    start_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # blue = start
    geometries.append(start_sphere)

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    end_sphere.translate(positions[-1])
    end_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red = end
    geometries.append(end_sphere)

    # Orientation frames at sampled timesteps
    frame_indices = np.linspace(0, T - 1, min(10, T), dtype=int)
    for idx in frame_indices:
        pos = positions[idx]
        euler = states[idx, 3:6]
        rot = Rotation.from_euler("xyz", euler).as_matrix()

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        frame.transform(transform)
        geometries.append(frame)

    # Action arrows (position deltas, scaled for visibility)
    arrow_indices = np.linspace(0, min(T - 1, len(actions)) - 1, min(15, T), dtype=int)
    for idx in arrow_indices:
        start = positions[idx]
        delta = actions[idx, :3]
        end = start + delta * action_scale

        arrow_points = o3d.utility.Vector3dVector([start, end])
        arrow_lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow = o3d.geometry.LineSet()
        arrow.points = arrow_points
        arrow.lines = arrow_lines
        arrow.colors = o3d.utility.Vector3dVector([[0.0, 0.5, 1.0]])  # cyan
        geometries.append(arrow)

    # World frame origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(origin)

    return geometries


def visualize_open3d(episodes, max_episodes=5, action_scale=50.0):
    """Interactive Open3D visualization of extracted trajectories."""
    if not HAS_O3D:
        print("Open3D not available, skipping interactive visualization")
        return

    all_geometries = []
    for i, ep in enumerate(episodes[:max_episodes]):
        geoms = create_trajectory_geometries(ep, i, action_scale)
        all_geometries.extend(geoms)

    print(f"Launching Open3D viewer with {len(all_geometries)} geometries "
          f"({min(max_episodes, len(episodes))} episodes)...")
    print("  Blue sphere = start, Red sphere = end")
    print("  Line color: green=open gripper, red=closed gripper")
    print("  Cyan arrows = action directions (scaled {action_scale}x)")
    print("  Small coordinate frames = EEF orientation")

    o3d.visualization.draw_geometries(
        all_geometries,
        window_name=f"RLDS Validation: {len(episodes)} episodes",
        width=1280,
        height=720,
    )


# ── Plotly HTML report (always works, even headless) ──────────────────────────

def generate_plotly_report(episodes, output_path):
    """Generate interactive HTML report with plotly (works headless)."""
    if not HAS_PLOTLY:
        print("plotly not available, skipping HTML report")
        return

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "3D EEF Trajectories",
            "Per-step Displacement (mm)",
            "Action Dimensions Over Time",
            "State Dimensions Over Time",
        ],
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for i, ep in enumerate(episodes):
        color = colors[i % len(colors)]
        pos = ep["states"][:, :3]

        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="lines+markers",
                marker=dict(size=2, color=color),
                line=dict(color=color, width=3),
                name=f"Ep {i}",
            ),
            row=1, col=1,
        )

        # Mark start/end
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
                mode="markers",
                marker=dict(size=6, color="blue", symbol="diamond"),
                name=f"Ep {i} start",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]],
                mode="markers",
                marker=dict(size=6, color="red", symbol="diamond"),
                name=f"Ep {i} end",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Per-step displacement
        disp = np.linalg.norm(ep["actions"][:, :3], axis=1) * 1000
        fig.add_trace(
            go.Scatter(
                y=disp, mode="lines",
                line=dict(color=color),
                name=f"Ep {i}",
                showlegend=False,
            ),
            row=1, col=2,
        )

        # Action dims over time (first episode only for clarity)
        if i == 0:
            dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip"]
            for d in range(7):
                fig.add_trace(
                    go.Scatter(
                        y=ep["actions"][:, d],
                        mode="lines",
                        name=dim_names[d],
                    ),
                    row=2, col=1,
                )

            # State dims over time
            state_names = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
            for d in range(7):
                fig.add_trace(
                    go.Scatter(
                        y=ep["states"][:, d],
                        mode="lines",
                        name=state_names[d],
                    ),
                    row=2, col=2,
                )

    fig.update_layout(
        height=900,
        width=1400,
        title_text="RLDS Dataset Validation Report",
    )
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="mm", row=1, col=2)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=2)

    fig.write_html(output_path)
    print(f"Plotly report saved to: {output_path}")


# ── Image contact sheets ─────────────────────────────────────────────────────

def save_contact_sheets(episodes, output_dir, max_episodes=5, samples_per_ep=8):
    """Save image contact sheets for visual spot-checking."""
    os.makedirs(output_dir, exist_ok=True)

    for ep_idx, ep in enumerate(episodes[:max_episodes]):
        front_imgs = ep["front_images"]
        wrist_imgs = ep["wrist_images"]
        T = len(front_imgs)

        if T == 0:
            continue

        # Sample evenly across episode
        indices = np.linspace(0, T - 1, min(samples_per_ep, T), dtype=int)

        # Front camera contact sheet
        if front_imgs:
            h, w = front_imgs[0].shape[:2]
            cols = min(4, len(indices))
            rows = (len(indices) + cols - 1) // cols
            sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

            for pos, idx in enumerate(indices):
                r, c = divmod(pos, cols)
                img = front_imgs[idx]
                # Add step number overlay
                img_copy = img.copy()
                cv2.putText(img_copy, f"t={idx}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                sheet[r*h:(r+1)*h, c*w:(c+1)*w] = img_copy

            path = os.path.join(output_dir, f"ep{ep_idx}_front.png")
            cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

        # Wrist camera contact sheet
        if wrist_imgs:
            h, w = wrist_imgs[0].shape[:2]
            cols = min(4, len(indices))
            rows = (len(indices) + cols - 1) // cols
            sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

            for pos, idx in enumerate(indices):
                r, c = divmod(pos, cols)
                img = wrist_imgs[idx]
                img_copy = img.copy()
                cv2.putText(img_copy, f"t={idx}", (3, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                sheet[r*h:(r+1)*h, c*w:(c+1)*w] = img_copy

            path = os.path.join(output_dir, f"ep{ep_idx}_wrist.png")
            cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    print(f"Contact sheets saved to: {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate extracted RLDS TFRecord dataset visually"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing the RLDS dataset (parent of <dataset_name>/1.0.0/)",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="xarm_place_toolbox",
        help="Name of the RLDS dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./validation_output",
        help="Directory for saving validation outputs",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=10,
        help="Maximum episodes to load for validation",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Launch interactive Open3D viewer (requires display)",
    )
    parser.add_argument(
        "--action_scale", type=float, default=50.0,
        help="Scale factor for action arrows in 3D view",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load episodes
    print(f"Loading RLDS dataset: {args.dataset_name}")
    episodes = load_rlds_episodes(args.data_dir, args.dataset_name, args.max_episodes)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes loaded, exiting.")
        return

    # Print statistics and sanity checks
    print_dataset_stats(episodes)

    # Save image contact sheets (always works)
    sheets_dir = os.path.join(args.output_dir, "contact_sheets")
    save_contact_sheets(episodes, sheets_dir)

    # Generate plotly HTML report (works headless)
    report_path = os.path.join(args.output_dir, "validation_report.html")
    generate_plotly_report(episodes, report_path)

    # Interactive Open3D (only if requested and display available)
    if args.interactive:
        visualize_open3d(episodes, action_scale=args.action_scale)
    else:
        print("\nSkipping interactive Open3D (use --interactive to enable)")

    print(f"\nValidation outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
