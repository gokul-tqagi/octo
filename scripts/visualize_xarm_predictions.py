# visualize_xarm_predictions.py: Visualize finetuned Octo predictions using the
# visualize_xarm_predictions.py: author's visualization_lib.py approach (per-dim action plots + 3D trajectory).

"""
Visualize finetuned Octo predictions on xarm val episodes using the same
visualization approach as the author's VisualizationCallback:

1. Per-dimension action comparison (GT vs predicted) with action chunk overlay
2. 3D plotly trajectory with predicted action arrows
3. Manipulation metrics (MSE, XYZ angle accuracy, gripper accuracy)

Adapted from octo/utils/visualization_lib.py to work with our custom
TFRecords (bypasses TFDS) and saves to disk (bypasses wandb).

Usage:
    python scripts/visualize_xarm_predictions.py \
        --checkpoint_path=./checkpoints/xarm_place_toolbox \
        --data_dir=./data/xarm_place_toolbox_rlds \
        --output_dir=./viz_output
"""

import glob
import json
import os
from functools import partial

from absl import app, flags, logging
import cv2
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.visualization_lib import (
    add_manipulation_metrics,
    add_unnormalized_info,
    plot_trajectory_actions,
    plot_trajectory_overview_mpl,
    unnormalize,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to finetuned Octo checkpoint.")
flags.DEFINE_string("data_dir", None, "Path to RLDS dataset directory.")
flags.DEFINE_string("dataset_name", "xarm_place_toolbox", "Name of the dataset.")
flags.DEFINE_string("output_dir", "./viz_output", "Directory for visualization outputs.")
flags.DEFINE_integer("max_episodes", 3, "Max val episodes to visualize.")
flags.DEFINE_integer("samples_per_state", 8, "Number of action samples per state.")


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


def build_traj_dict(episode, stats, window_size=1):
    """Build a trajectory dict in the format visualization_lib expects.

    The author's run_policy_on_trajectory expects:
        traj["observation"] — dict with image_primary, image_wrist, proprio, timestep_pad_mask, pad_mask_dict
        traj["action"] — (T, window_size, action_horizon=1, action_dim)
        traj["task"]["language_instruction"] — list of byte strings
    """
    T = len(episode["front"])
    action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
    action_std = np.array(stats["action"]["std"], dtype=np.float32)
    proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
    proprio_std = np.array(stats["proprio"]["std"], dtype=np.float32)

    # Normalize actions and proprio to match what the model sees
    norm_actions = (episode["action"] - action_mean) / (action_std + 1e-8)
    # Gripper dim (6) is not normalized
    norm_actions[:, 6] = episode["action"][:, 6]

    norm_proprio = (episode["state"] - proprio_mean) / (proprio_std + 1e-8)

    traj = {
        "observation": {
            "image_primary": episode["front"][:, None],      # (T, 1, H, W, 3)
            "image_wrist": episode["wrist"][:, None],         # (T, 1, H, W, 3)
            "proprio": norm_proprio[:, None],                  # (T, 1, 7)
            "timestep_pad_mask": np.ones((T, 1), dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.ones((T, 1), dtype=bool),
                "image_wrist": np.ones((T, 1), dtype=bool),
                "proprio": np.ones((T, 1), dtype=bool),
            },
        },
        "action": norm_actions[:, None, None, :],  # (T, 1, 1, 7)
        "task": {
            "language_instruction": [episode["language"].encode("utf-8")] * T,
        },
    }
    return traj


def run_policy_on_traj(model, traj, text_processor, samples_per_state=8):
    """Run the model on each timestep, collecting multi-sample predictions.

    Mimics visualization_lib.run_policy_on_trajectory but adapted for
    our batch format.
    """
    T = len(traj["action"])
    task = model.create_tasks(texts=[traj["task"]["language_instruction"][0].decode("utf-8")])

    all_pred_actions = []
    rng = jax.random.PRNGKey(0)

    for t in range(T):
        obs_t = jax.tree_map(lambda x: x[t:t+1], traj["observation"])

        sample_actions = []
        for s in range(samples_per_state):
            rng, sample_rng = jax.random.split(rng)
            actions = model.sample_actions(
                jax.tree_map(lambda x: jnp.array(x), obs_t),
                task,
                rng=sample_rng,
            )
            # actions: (batch=1, action_horizon, action_dim)
            sample_actions.append(np.array(actions[0]))  # (action_horizon, 7)

        all_pred_actions.append(np.stack(sample_actions))  # (n_samples, action_horizon, 7)

    pred_actions_chunk = np.stack(all_pred_actions)  # (T, n_samples, action_horizon, 7)
    horizon = traj["observation"]["image_primary"].shape[1]

    logging.info(f"  pred_actions_chunk shape: {pred_actions_chunk.shape}")
    logging.info(f"  traj action shape: {traj['action'].shape}")

    info = {
        "n": np.array(T),
        "pred_actions_chunk": pred_actions_chunk,
        "pred_actions": pred_actions_chunk[:, :, 0],     # (T, n_samples, 7) — first step only
        "actions": traj["action"][:, horizon - 1, 0],     # (T, 7) — GT actions
        "proprio": traj["observation"]["proprio"][:, horizon - 1],  # (T, 7)
    }
    logging.info(f"  info pred_actions: {info['pred_actions'].shape}, actions: {info['actions'].shape}")
    return info


def save_mpl_figure(fig, path):
    """Save matplotlib figure to file."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_overview(traj, info, unnorm_pred_actions_chunk, unnorm_actions, episode_idx):
    """Replicate the author's plot_trajectory_overview_mpl but save to disk."""
    act = unnorm_pred_actions_chunk[:, :, :1]  # single-step predictions
    n_act_dims = traj["action"].shape[-1]
    grid_size = int(np.ceil(np.sqrt(n_act_dims + 1)))

    fig = plt.figure(figsize=(grid_size * 5, grid_size * 5))
    gs = gridspec.GridSpec(grid_size, grid_size)

    # MSE plot
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(info["mse"].mean(axis=1))
    ax.set_ylabel("MSE")
    ax.set_title("Normalized MSE")

    dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip"]
    for i in range(n_act_dims):
        ax = fig.add_subplot(gs[(i + 1) // grid_size, (i + 1) % grid_size])
        ax.plot(unnorm_actions[:, i], label="GT action", color="tab:blue")

        # Plot predicted action samples
        chunk_length = act.shape[2]
        for t in range(act.shape[0]):
            step_idx, chunk_idx = divmod(t, chunk_length)
            pred_i = act[int(step_idx * chunk_length), :, chunk_idx, i]
            x = np.full(pred_i.shape[0], t)
            ax.scatter(x, pred_i, color="tab:red", s=4, alpha=0.5)
            if chunk_idx == 0 and (act.shape[0] // chunk_length) <= 20:
                ax.axvline(t, color="red", linestyle="--", alpha=0.2)

        ax.set_ylabel(dim_names[i] if i < len(dim_names) else f"dim {i}")
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"Episode {episode_idx}: {traj['task']['language_instruction'][0].decode('utf-8')}")
    fig.tight_layout()
    return fig


def main(_):
    assert FLAGS.checkpoint_path, "Must provide --checkpoint_path"
    assert FLAGS.data_dir, "Must provide --data_dir"

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Load model and stats
    logging.info(f"Loading model from {FLAGS.checkpoint_path}...")
    model = OctoModel.load_pretrained(FLAGS.checkpoint_path)
    text_processor = model.text_processor

    stats_path = os.path.join(FLAGS.checkpoint_path, "dataset_statistics.json")
    with open(stats_path) as f:
        stats = json.load(f)

    # Build normalization stats dict in visualization_lib's expected format
    norm_stats = {
        "action": {
            "mean": np.array(stats["action"]["mean"]),
            "std": np.array(stats["action"]["std"]),
            "mask": np.array([True, True, True, True, True, True, False]),
        },
        "proprio": {
            "mean": np.array(stats["proprio"]["mean"]),
            "std": np.array(stats["proprio"]["std"]),
        },
    }

    # Load val episodes
    logging.info("Loading validation episodes...")
    val_episodes = load_val_episodes(FLAGS.data_dir, FLAGS.dataset_name)
    logging.info(f"Loaded {len(val_episodes)} val episodes")

    all_metrics = []

    for ep_idx, episode in enumerate(val_episodes[:FLAGS.max_episodes]):
        logging.info(f"Processing episode {ep_idx} ({len(episode['front'])} steps, "
                     f"{FLAGS.samples_per_state} samples/state)...")

        # Build trajectory dict
        traj = build_traj_dict(episode, stats)

        # Run policy
        info = run_policy_on_traj(model, traj, text_processor, FLAGS.samples_per_state)

        # Add unnormalized info and manipulation metrics (author's functions)
        info = add_unnormalized_info(info, norm_stats)
        info = add_manipulation_metrics(info)

        # Print metrics
        metrics = {
            "mse": float(info["mse"].mean()),
            "xyz_angle_accuracy": float(info["xyz_angle_accuracy"].mean()),
            "gripper_correct": float(info["gripper_correct"].mean()),
        }
        logging.info(f"  MSE: {metrics['mse']:.4f}")
        logging.info(f"  XYZ angle accuracy: {metrics['xyz_angle_accuracy']:.4f}")
        logging.info(f"  Gripper accuracy: {metrics['gripper_correct']:.4f}")
        all_metrics.append({"episode": ep_idx, **metrics})

        # 1. Per-dimension action overview (author's approach)
        fig = plot_trajectory_overview(
            traj, info,
            info["unnorm_pred_actions_chunk"],
            info["unnorm_actions"],
            ep_idx,
        )
        overview_path = os.path.join(FLAGS.output_dir, f"ep{ep_idx}_action_overview.png")
        save_mpl_figure(fig, overview_path)
        logging.info(f"  Saved: {overview_path}")

        # 2. 3D trajectory plot (author's approach) — only if proprio available
        if "unnorm_proprio" in info:
            plotly_fig = plot_trajectory_actions(
                unnorm_pred_actions=info["unnorm_pred_actions"],
                unnorm_actions=info["unnorm_actions"],
                unnorm_proprio=info["unnorm_proprio"],
            )
            plotly_path = os.path.join(FLAGS.output_dir, f"ep{ep_idx}_3d_trajectory.html")
            plotly_fig.write_html(plotly_path)
            logging.info(f"  Saved: {plotly_path}")

        # 3. With action chunks
        fig_chunk = plot_trajectory_overview(
            traj, info,
            info["unnorm_pred_actions_chunk"][:, :, :],  # all chunks
            info["unnorm_actions"],
            ep_idx,
        )
        chunk_path = os.path.join(FLAGS.output_dir, f"ep{ep_idx}_action_chunks.png")
        save_mpl_figure(fig_chunk, chunk_path)
        logging.info(f"  Saved: {chunk_path}")

        # 4. Image strip
        images = episode["front"]
        indices = np.linspace(0, len(images) - 1, 5, dtype=int)
        strip = np.concatenate([images[i] for i in indices], axis=1)
        strip_path = os.path.join(FLAGS.output_dir, f"ep{ep_idx}_image_strip.png")
        cv2.imwrite(strip_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        logging.info(f"  Saved: {strip_path}")

    # Save aggregate metrics
    with open(os.path.join(FLAGS.output_dir, "viz_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    logging.info(f"\nVisualization complete. Outputs in {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)
