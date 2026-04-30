# analyze_lerobot_dataset.py: Comparative distribution analysis for LeRobot v3 datasets.
# analyze_lerobot_dataset.py: Validates data quality and compares records for training compatibility.

"""
Analyze and compare LeRobot v3 dataset records for distribution consistency.

Reports per-record and per-episode statistics:
  - Frame counts, durations, FPS consistency
  - EEF trajectory length and step displacement via FK
  - Gripper behavior (transitions, open/close ratio)
  - Joint limit violations, NaN/Inf, timestamp gaps
  - Cross-record distribution comparison

Designed to run on any new data collection before training to catch:
  - Incompatible records (different robots, action dims, tasks)
  - Bad episodes (NaN, jumps, stuck joints, missing images)
  - Distribution shifts between collection sessions

Usage:
    python scripts/analyze_lerobot_dataset.py \
        --data_dir /path/to/aloha \
        --robot_type bi_widowxai_follower_robot \
        --arm right \
        --fk_module wxai_fk

    # Exclude specific records:
    python scripts/analyze_lerobot_dataset.py \
        --data_dir /path/to/aloha \
        --exclude mobileai-test_record

    # Compare only specific records:
    python scripts/analyze_lerobot_dataset.py \
        --data_dir /path/to/aloha \
        --records pickplace pickplace2 pickplace3
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    raise ImportError("pyarrow required. Install with: pip install pyarrow")


# ── FK module loader ──────────────────────────────────────────────────────────

def load_fk_module(fk_module_name):
    """Dynamically load an FK module that provides batch_fk_euler and compute_delta_eef."""
    if fk_module_name is None:
        return None
    try:
        mod = __import__(fk_module_name)
        return mod
    except ImportError:
        print("Warning: FK module '{}' not found. Skipping EEF analysis.".format(fk_module_name))
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def find_records(data_dir, include=None, exclude=None):
    """Find valid LeRobot records in data_dir."""
    records = []
    for name in sorted(os.listdir(data_dir)):
        rec_path = os.path.join(data_dir, name)
        if not os.path.isdir(rec_path):
            continue
        if not os.path.exists(os.path.join(rec_path, "meta", "info.json")):
            continue
        if include and not any(inc in name for inc in include):
            continue
        if exclude and any(exc in name for exc in exclude):
            continue
        records.append(name)
    return records


def load_record_data(data_dir, record_name):
    """Load parquet data and metadata for a single record."""
    rec_path = os.path.join(data_dir, record_name)

    with open(os.path.join(rec_path, "meta", "info.json")) as f:
        info = json.load(f)

    # Task text
    tasks_table = pq.read_table(os.path.join(rec_path, "meta", "tasks.parquet"))
    tasks_dict = tasks_table.to_pydict()
    task_text = tasks_dict.get("__index_level_0__", ["unknown"])[0]

    # Data
    data_files = sorted(
        [os.path.join(dp, f)
         for dp, dn, fn in os.walk(os.path.join(rec_path, "data"))
         for f in fn if f.endswith(".parquet") and not f.endswith(".truncated")]
    )
    data = pa.concat_tables([pq.read_table(f) for f in data_files])

    # Image source
    has_videos = os.path.isdir(os.path.join(rec_path, "videos"))
    has_images = os.path.isdir(os.path.join(rec_path, "images"))
    img_source = "video" if has_videos else ("frames" if has_images else "none")

    # Cameras
    if has_videos:
        cameras = sorted(os.listdir(os.path.join(rec_path, "videos")))
    elif has_images:
        cameras = sorted(os.listdir(os.path.join(rec_path, "images")))
    else:
        cameras = []

    # Group by episode
    ep_groups = {}
    for i in range(len(data)):
        ep = data.column("episode_index")[i].as_py()
        if ep not in ep_groups:
            ep_groups[ep] = []
        ep_groups[ep].append(i)

    return {
        "info": info,
        "task": task_text,
        "data": data,
        "ep_groups": ep_groups,
        "img_source": img_source,
        "cameras": cameras,
    }


# ── Per-episode analysis ─────────────────────────────────────────────────────

def analyze_episode(data, rows, arm_indices, gripper_index, fk_mod, joint_limits, fps):
    """Analyze a single episode. Returns stats dict and list of issues."""
    n_frames = len(rows)
    issues = []

    # Extract actions and timestamps
    actions = np.array([data.column("action")[r].as_py() for r in rows])
    timestamps = np.array([data.column("timestamp")[r].as_py() for r in rows])

    # Joint data for selected arm
    joints_full = actions[:, arm_indices]
    gripper = actions[:, gripper_index] if gripper_index is not None else None

    # NaN/Inf check
    if np.any(np.isnan(joints_full)) or np.any(np.isinf(joints_full)):
        issues.append("NaN/Inf in joint data")
        return {"n_frames": n_frames, "issues": issues}, issues

    # Joint limit check
    if joint_limits:
        for j, (lo, hi) in enumerate(joint_limits):
            if j >= joints_full.shape[1]:
                break
            jmin, jmax = joints_full[:, j].min(), joints_full[:, j].max()
            if jmin < lo - 0.1 or jmax > hi + 0.1:
                issues.append("joint_{} out of limits [{:.3f},{:.3f}] vs [{},{}]".format(
                    j, jmin, jmax, lo, hi))

    # Duration and FPS
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    actual_fps = (n_frames - 1) / duration if duration > 0 else 0

    # Timestamp gaps
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        expected_dt = 1.0 / fps
        gap_count = np.sum(np.abs(dt - expected_dt) > expected_dt * 2)
        if gap_count > 0:
            issues.append("{} timestamp gaps".format(gap_count))

    # FK-based EEF analysis
    eef_stats = {}
    fk_joints = joints_full[:, :6] if joints_full.shape[1] >= 6 else None
    if fk_mod and fk_joints is not None:
        eef = fk_mod.batch_fk_euler(fk_joints)
        diffs = np.diff(eef[:, :3], axis=0)
        step_disp = np.linalg.norm(diffs, axis=1)
        traj_length = step_disp.sum()
        displacement = np.linalg.norm(eef[-1, :3] - eef[0, :3])

        # Big jump detection (>5cm in one step at native Hz)
        big_jumps = np.where(step_disp > 0.05)[0]
        if len(big_jumps) > 0:
            issues.append("{} big jumps (>5cm/step)".format(len(big_jumps)))

        eef_stats = {
            "traj_length_cm": traj_length * 100,
            "displacement_cm": displacement * 100,
            "step_disp_mean_mm": step_disp.mean() * 1000,
            "step_disp_max_mm": step_disp.max() * 1000,
            "eef_start": eef[0, :3].tolist(),
            "eef_end": eef[-1, :3].tolist(),
        }

    # Gripper analysis
    grip_stats = {}
    if gripper is not None:
        grip_binary = (gripper > 0.01).astype(int)
        grip_changes = int(np.sum(np.abs(np.diff(grip_binary))))
        grip_closed_pct = float(grip_binary.mean() * 100)
        grip_stats = {
            "grip_changes": grip_changes,
            "grip_closed_pct": grip_closed_pct,
        }

    # Joint variance (detect stuck joints)
    joint_stds = joints_full.std(axis=0)
    stuck_joints = np.where(joint_stds < 1e-6)[0]
    if len(stuck_joints) > 0 and len(stuck_joints) < joints_full.shape[1]:
        # Only flag if SOME joints are stuck (all stuck = idle arm, expected)
        issues.append("stuck joints: {}".format(stuck_joints.tolist()))

    stats = {
        "n_frames": n_frames,
        "duration_s": duration,
        "actual_fps": actual_fps,
        **eef_stats,
        **grip_stats,
        "issues": issues,
    }
    return stats, issues


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_dataset(args):
    """Run full dataset analysis."""
    fk_mod = load_fk_module(args.fk_module)

    # Arm index mapping
    arm_map = {
        "right": (list(range(7, 13)), 13),
        "left": (list(range(0, 6)), 6),
    }
    arm_indices, gripper_index = arm_map.get(args.arm, (list(range(7, 13)), 13))

    # wxai joint limits
    joint_limits = [
        (-3.1, 3.1), (0, 3.15), (0, 2.36),
        (-1.58, 1.58), (-1.58, 1.58), (-3.15, 3.15),
    ]

    records = find_records(args.data_dir, args.records, args.exclude)
    if not records:
        print("No records found in {}".format(args.data_dir))
        return

    print("Analyzing {} records in {}\n".format(len(records), args.data_dir))

    # ── Per-record summary table ──────────────────────────────────────────
    all_record_stats = []

    for rec_name in records:
        rec = load_record_data(args.data_dir, rec_name)
        info = rec["info"]
        robot = info["robot_type"]
        adim = info["features"]["action"]["shape"][0]
        fps = info["fps"]

        ep_stats_list = []
        all_issues = []

        for ep_idx in sorted(rec["ep_groups"].keys()):
            rows = rec["ep_groups"][ep_idx]
            stats, issues = analyze_episode(
                rec["data"], rows, arm_indices, gripper_index,
                fk_mod, joint_limits, fps,
            )
            stats["ep_idx"] = ep_idx
            ep_stats_list.append(stats)
            for iss in issues:
                all_issues.append("ep{}: {}".format(ep_idx, iss))

        all_record_stats.append({
            "name": rec_name,
            "robot": robot,
            "adim": adim,
            "task": rec["task"],
            "img_source": rec["img_source"],
            "cameras": rec["cameras"],
            "n_episodes": len(rec["ep_groups"]),
            "total_frames": sum(s["n_frames"] for s in ep_stats_list),
            "fps": fps,
            "ep_stats": ep_stats_list,
            "issues": all_issues,
        })

    # ── Print comparison table ────────────────────────────────────────────
    print("=" * 130)
    header = "{:40s} {:22s} {:>4s} {:>6s} {:>9s} {:>9s}".format(
        "Record", "Robot", "Eps", "Frames", "Frm/Ep", "Dur/Ep(s)")
    if fk_mod:
        header += " {:>11s} {:>11s}".format("Traj/Ep(cm)", "Step(mm)")
    header += " {:>8s}".format("Status")
    print(header)
    print("-" * 130)

    for rs in all_record_stats:
        eps = rs["ep_stats"]
        n_eps = rs["n_episodes"]
        robot_str = "{}({}d)".format(rs["robot"][:18], rs["adim"])

        mean_frames = np.mean([e["n_frames"] for e in eps])
        mean_dur = np.mean([e["duration_s"] for e in eps])

        line = "{:40s} {:22s} {:4d} {:6d} {:9.0f} {:9.1f}".format(
            rs["name"], robot_str, n_eps, rs["total_frames"], mean_frames, mean_dur)

        if fk_mod and "traj_length_cm" in eps[0]:
            mean_traj = np.mean([e["traj_length_cm"] for e in eps])
            mean_step = np.mean([e["step_disp_mean_mm"] for e in eps])
            line += " {:11.1f} {:11.2f}".format(mean_traj, mean_step)

        status = "CLEAN" if not rs["issues"] else "{} ISSUES".format(len(rs["issues"]))
        line += " {:>8s}".format(status)
        print(line)

    # ── Compatibility check ───────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("COMPATIBILITY CHECK")
    print("=" * 130)

    tasks = set(rs["task"] for rs in all_record_stats)
    robots = set(rs["robot"] for rs in all_record_stats)
    adims = set(rs["adim"] for rs in all_record_stats)
    img_sources = set(rs["img_source"] for rs in all_record_stats)

    print("Tasks:        {} {}".format(
        "MATCH" if len(tasks) == 1 else "MISMATCH", tasks))
    print("Robots:       {} {}".format(
        "MATCH" if len(robots) == 1 else "MIXED (check arm compatibility)", robots))
    print("Action dims:  {} {}".format(
        "MATCH" if len(adims) == 1 else "MIXED (need index remapping)", adims))
    print("Image source: {} {}".format(
        "MATCH" if len(img_sources) == 1 else "MIXED (both supported)", img_sources))

    # ── Distribution comparison ───────────────────────────────────────────
    if fk_mod and any("traj_length_cm" in rs["ep_stats"][0] for rs in all_record_stats):
        print("\n" + "=" * 130)
        print("DISTRIBUTION COMPARISON")
        print("=" * 130)

        for rs in all_record_stats:
            eps = rs["ep_stats"]
            trajs = [e["traj_length_cm"] for e in eps if "traj_length_cm" in e]
            steps = [e["step_disp_mean_mm"] for e in eps if "step_disp_mean_mm" in e]
            grips = [e.get("grip_changes", 0) for e in eps]
            durations = [e["duration_s"] for e in eps]

            print("\n  {} ({} episodes):".format(rs["name"], len(eps)))
            print("    Traj length (cm): mean={:.1f}, std={:.1f}, range=[{:.1f}, {:.1f}]".format(
                np.mean(trajs), np.std(trajs), min(trajs), max(trajs)))
            print("    Step disp (mm):   mean={:.2f}, std={:.2f}, range=[{:.2f}, {:.2f}]".format(
                np.mean(steps), np.std(steps), min(steps), max(steps)))
            print("    Duration (s):     mean={:.1f}, std={:.1f}, range=[{:.1f}, {:.1f}]".format(
                np.mean(durations), np.std(durations), min(durations), max(durations)))
            print("    Grip changes/ep:  mean={:.1f}, range=[{}, {}]".format(
                np.mean(grips), min(grips), max(grips)))

    # ── Issues summary ────────────────────────────────────────────────────
    total_issues = sum(len(rs["issues"]) for rs in all_record_stats)
    if total_issues > 0:
        print("\n" + "=" * 130)
        print("ISSUES ({} total)".format(total_issues))
        print("=" * 130)
        for rs in all_record_stats:
            if rs["issues"]:
                print("\n  {}:".format(rs["name"]))
                for iss in rs["issues"]:
                    print("    - {}".format(iss))
    else:
        print("\nNo issues found across all records.")

    # ── Final summary ─────────────────────────────────────────────────────
    total_eps = sum(rs["n_episodes"] for rs in all_record_stats)
    total_frames = sum(rs["total_frames"] for rs in all_record_stats)
    print("\n" + "=" * 130)
    print("TOTAL: {} records, {} episodes, {} frames, {} issues".format(
        len(all_record_stats), total_eps, total_frames, total_issues))
    if total_issues == 0:
        print("Dataset is READY for training.")
    else:
        print("Review issues above before training.")
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LeRobot v3 dataset distribution for training compatibility"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing LeRobot record subdirectories")
    parser.add_argument("--records", nargs="+", default=None,
                        help="Only analyze records containing these substrings")
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="Exclude records containing these substrings")
    parser.add_argument("--arm", type=str, default="right", choices=["left", "right"],
                        help="Which arm to analyze")
    parser.add_argument("--fk_module", type=str, default="wxai_fk",
                        help="FK module name (must provide batch_fk_euler). None to skip EEF analysis.")
    args = parser.parse_args()

    if args.fk_module == "None":
        args.fk_module = None

    analyze_dataset(args)


if __name__ == "__main__":
    main()
