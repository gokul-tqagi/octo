# eval_xarm.py: Evaluate finetuned Octo on xarm val episodes.
# eval_xarm.py: Rolls out action predictions and compares to ground truth trajectories.

"""
Evaluate a finetuned Octo checkpoint on xarm validation episodes.

Loads val episodes from our custom TFRecords, runs the model's action
prediction at each timestep, and compares predicted trajectory to
ground truth. Produces a plotly HTML report and prints error metrics.

Usage:
    python scripts/eval_xarm.py \
        --checkpoint_path=./checkpoints/xarm_place_toolbox \
        --data_dir=./data/xarm_place_toolbox_rlds \
        --output_dir=./eval_output
"""

import glob
import json
import os

from absl import app, flags, logging
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to finetuned Octo checkpoint.")
flags.DEFINE_string("data_dir", None, "Path to RLDS dataset directory.")
flags.DEFINE_string("dataset_name", "xarm_place_toolbox", "Name of the dataset.")
flags.DEFINE_string("output_dir", "./eval_output", "Directory for eval outputs.")
flags.DEFINE_integer("max_episodes", 5, "Max val episodes to evaluate.")


def load_val_episodes(data_dir, dataset_name):
    """Load val episodes from custom TFRecords."""
    ds_path = os.path.join(data_dir, dataset_name, "1.0.0")
    pattern = os.path.join(ds_path, f"{dataset_name}-val.tfrecord*")
    tfrecord_files = sorted(glob.glob(pattern))

    if not tfrecord_files:
        raise FileNotFoundError(f"No val tfrecords: {pattern}")

    episodes = []
    dataset = tf.data.TFRecordDataset(tfrecord_files)

    for raw_record in dataset:
        traj_ex = tf.train.Example()
        traj_ex.ParseFromString(raw_record.numpy())
        steps_bytes = traj_ex.features.feature["steps"].bytes_list.value

        ep = {"front": [], "wrist": [], "state": [], "action": [], "language": ""}
        for sb in steps_bytes:
            step_ex = tf.train.Example()
            step_ex.ParseFromString(sb)
            sf = step_ex.features.feature

            front_jpg = sf["observation/image_0"].bytes_list.value[0]
            front = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(front_jpg, np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            wrist_jpg = sf["observation/image_1"].bytes_list.value[0]
            wrist = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(wrist_jpg, np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            state = np.array(sf["observation/state"].float_list.value, dtype=np.float32)
            action = np.array(sf["action"].float_list.value, dtype=np.float32)
            lang = sf["language_instruction"].bytes_list.value[0].decode("utf-8")

            ep["front"].append(front)
            ep["wrist"].append(wrist)
            ep["state"].append(state)
            ep["action"].append(action)
            ep["language"] = lang

        ep["front"] = np.stack(ep["front"])
        ep["wrist"] = np.stack(ep["wrist"])
        ep["state"] = np.stack(ep["state"])
        ep["action"] = np.stack(ep["action"])
        episodes.append(ep)

    return episodes


def rollout_episode(model, episode, stats_path):
    """Run model predictions through an episode and collect predicted actions."""
    # Load normalization stats
    with open(stats_path) as f:
        stats = json.load(f)

    action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
    action_std = np.array(stats["action"]["std"], dtype=np.float32)
    action_mask = np.array([True, True, True, True, True, True, False])
    proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
    proprio_std = np.array(stats["proprio"]["std"], dtype=np.float32)

    T = len(episode["front"])
    task = model.create_tasks(texts=[episode["language"]])

    predicted_actions = []
    rng = jax.random.PRNGKey(0)

    for t in range(T - 1):
        # Normalize proprio
        proprio = (episode["state"][t] - proprio_mean) / (proprio_std + 1e-8)

        # Build observation dict for single timestep
        observation = {
            "image_primary": episode["front"][t][None, None],  # (1, 1, H, W, 3)
            "image_wrist": episode["wrist"][t][None, None],    # (1, 1, H, W, 3)
            "proprio": proprio[None, None],                     # (1, 1, 7)
            "timestep_pad_mask": np.array([[True]]),             # (1, 1)
            "pad_mask_dict": {
                "image_primary": np.array([True]),
                "image_wrist": np.array([True]),
                "proprio": np.array([True]),
            },
        }

        # Run model
        rng, sample_rng = jax.random.split(rng)
        actions = model.sample_actions(
            jax.tree_map(lambda x: jnp.array(x), observation),
            task,
            rng=sample_rng,
        )
        # actions shape: (1, 1, action_horizon, 7)
        # Take first action from the chunk
        pred_action = np.array(actions[0, 0, 0])

        # Denormalize action
        denorm_action = np.where(
            action_mask,
            pred_action * (action_std + 1e-8) + action_mean,
            pred_action,
        )
        predicted_actions.append(denorm_action)

    return np.stack(predicted_actions)


def compute_metrics(gt_actions, pred_actions):
    """Compute evaluation metrics between GT and predicted action sequences."""
    # Per-step position error
    pos_error = np.linalg.norm(
        gt_actions[:, :3] - pred_actions[:, :3], axis=1
    )
    # Per-step rotation error
    rot_error = np.linalg.norm(
        gt_actions[:, 3:6] - pred_actions[:, 3:6], axis=1
    )

    # Cumulative trajectory from actions
    gt_traj = np.cumsum(gt_actions[:, :3], axis=0)
    pred_traj = np.cumsum(pred_actions[:, :3], axis=0)
    traj_error = np.linalg.norm(gt_traj - pred_traj, axis=1)

    return {
        "pos_error_mean_mm": float(pos_error.mean() * 1000),
        "pos_error_max_mm": float(pos_error.max() * 1000),
        "rot_error_mean_deg": float(np.degrees(rot_error.mean())),
        "rot_error_max_deg": float(np.degrees(rot_error.max())),
        "traj_error_mean_mm": float(traj_error.mean() * 1000),
        "traj_error_final_mm": float(traj_error[-1] * 1000),
    }


def generate_eval_report(all_results, output_path):
    """Generate plotly HTML eval report."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logging.warning("plotly not available, skipping HTML report")
        return

    n_eps = len(all_results)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "3D Trajectories (GT vs Predicted)",
            "Per-step Position Error (mm)",
            "Cumulative Trajectory Divergence (mm)",
            "Action Predictions vs GT (Episode 0)",
        ],
    )

    colors_gt = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    colors_pred = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22", "#7f7f7f"]

    for i, res in enumerate(all_results):
        gt = res["gt_actions"]
        pred = res["pred_actions"]

        # Cumulative trajectories
        gt_traj = np.cumsum(gt[:, :3], axis=0)
        pred_traj = np.cumsum(pred[:, :3], axis=0)

        # 3D trajectories
        fig.add_trace(
            go.Scatter3d(
                x=gt_traj[:, 0]*1000, y=gt_traj[:, 1]*1000, z=gt_traj[:, 2]*1000,
                mode="lines", line=dict(color=colors_gt[i % 5], width=4),
                name=f"Ep {i} GT",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=pred_traj[:, 0]*1000, y=pred_traj[:, 1]*1000, z=pred_traj[:, 2]*1000,
                mode="lines", line=dict(color=colors_pred[i % 5], width=4, dash="dash"),
                name=f"Ep {i} Pred",
            ),
            row=1, col=1,
        )

        # Per-step position error
        pos_err = np.linalg.norm(gt[:, :3] - pred[:, :3], axis=1) * 1000
        fig.add_trace(
            go.Scatter(y=pos_err, mode="lines", name=f"Ep {i}",
                       line=dict(color=colors_gt[i % 5])),
            row=1, col=2,
        )

        # Cumulative trajectory error
        traj_err = np.linalg.norm(gt_traj - pred_traj, axis=1) * 1000
        fig.add_trace(
            go.Scatter(y=traj_err, mode="lines", name=f"Ep {i}",
                       line=dict(color=colors_gt[i % 5]), showlegend=False),
            row=2, col=1,
        )

    # Action dim comparison for episode 0
    if all_results:
        gt0 = all_results[0]["gt_actions"]
        pred0 = all_results[0]["pred_actions"]
        dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip"]
        for d in range(min(6, gt0.shape[1])):
            fig.add_trace(
                go.Scatter(y=gt0[:, d], mode="lines", name=f"{dim_names[d]} GT",
                           line=dict(dash="solid")),
                row=2, col=2,
            )
            fig.add_trace(
                go.Scatter(y=pred0[:, d], mode="lines", name=f"{dim_names[d]} pred",
                           line=dict(dash="dash")),
                row=2, col=2,
            )

    fig.update_layout(height=900, width=1400, title_text="Octo xarm Evaluation")
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(title_text="mm", row=1, col=2)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)

    fig.write_html(output_path)
    logging.info(f"Eval report saved to: {output_path}")


def main(_):
    assert FLAGS.checkpoint_path, "Must provide --checkpoint_path"
    assert FLAGS.data_dir, "Must provide --data_dir"

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Load model
    logging.info(f"Loading model from {FLAGS.checkpoint_path}...")
    model = OctoModel.load_pretrained(FLAGS.checkpoint_path)

    # Load val episodes
    logging.info("Loading validation episodes...")
    val_episodes = load_val_episodes(FLAGS.data_dir, FLAGS.dataset_name)
    logging.info(f"Loaded {len(val_episodes)} val episodes")

    stats_path = os.path.join(FLAGS.checkpoint_path, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        logging.warning(f"No dataset_statistics.json at {stats_path}, "
                        "predictions will not be denormalized correctly")

    # Rollout evaluation
    all_results = []
    for i, ep in enumerate(val_episodes[:FLAGS.max_episodes]):
        logging.info(f"Evaluating episode {i} ({len(ep['front'])} steps)...")
        pred_actions = rollout_episode(model, ep, stats_path)
        gt_actions = ep["action"][:-1]  # GT has T actions, rollout produces T-1

        # Trim to same length
        min_len = min(len(gt_actions), len(pred_actions))
        gt_actions = gt_actions[:min_len]
        pred_actions = pred_actions[:min_len]

        metrics = compute_metrics(gt_actions, pred_actions)
        all_results.append({
            "gt_actions": gt_actions,
            "pred_actions": pred_actions,
            "metrics": metrics,
        })

        logging.info(f"  Episode {i} metrics:")
        for k, v in metrics.items():
            logging.info(f"    {k}: {v:.2f}")

    # Aggregate metrics
    if all_results:
        logging.info("\n=== Aggregate Metrics ===")
        metric_keys = all_results[0]["metrics"].keys()
        for k in metric_keys:
            vals = [r["metrics"][k] for r in all_results]
            logging.info(f"  {k}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")

    # Generate report
    report_path = os.path.join(FLAGS.output_dir, "eval_report.html")
    generate_eval_report(all_results, report_path)

    # Save metrics JSON
    metrics_out = []
    for i, r in enumerate(all_results):
        metrics_out.append({"episode": i, **r["metrics"]})
    with open(os.path.join(FLAGS.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    logging.info(f"\nEval complete. Outputs in {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)
