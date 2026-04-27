# 07_zero_shot_mcap.py: Zero-shot action prediction on xarm mcap trajectories
# using pretrained Octo-Small model. Loads mcap data, runs inference, and plots predicted vs GT actions.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm

from octo.model.octo_model import OctoModel
from mcap_dataloader import load_dataset


ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
WINDOW_SIZE = 2  # Octo was trained with history window of 2


def run_zero_shot_inference(
    model,
    episode,
    task,
    use_wrist=True,
):
    """Run zero-shot inference on a single episode.

    Args:
        model: Loaded OctoModel.
        episode: Dict from mcap_dataloader.load_episode().
        task: Task dict from model.create_tasks().
        use_wrist: Whether to include wrist camera observations.

    Returns:
        pred_actions: (T, action_horizon, 7) predicted action chunks.
        true_actions: (T, 7) ground truth delta actions.
    """
    front_images = episode["front_images"]
    wrist_images = episode["wrist_images"]
    n_steps = len(front_images)

    pred_actions = []
    true_actions = []

    for step in tqdm.trange(n_steps - WINDOW_SIZE, desc="Inference"):
        # Stack window of observations
        front_window = front_images[step : step + WINDOW_SIZE][None]  # (1, W, H, W, 3)

        observation = {
            "image_primary": front_window,
            "timestep_pad_mask": np.full((1, WINDOW_SIZE), True, dtype=bool),
        }

        if use_wrist:
            wrist_window = wrist_images[step : step + WINDOW_SIZE][None]
            observation["image_wrist"] = wrist_window

        # Sample actions (normalized — we skip unnormalization since our GT actions
        # are in a different scale/space than the bridge dataset stats)
        actions = model.sample_actions(
            observation,
            task,
            rng=jax.random.PRNGKey(step),
        )
        actions = actions[0]  # remove batch dim -> (action_horizon, 7)
        pred_actions.append(actions)

        # Ground truth action at the last timestep in the window
        gt_idx = step + WINDOW_SIZE - 1
        if gt_idx < len(episode["actions"]):
            true_actions.append(episode["actions"][gt_idx])

    pred_actions = np.array(pred_actions)
    true_actions = np.array(true_actions)

    # Trim to same length
    min_len = min(len(pred_actions), len(true_actions))
    return pred_actions[:min_len], true_actions[:min_len]


def plot_actions(pred_actions, true_actions, episode_name, save_path):
    """Plot predicted vs ground truth actions for one episode."""
    fig, axes = plt.subplots(len(ACTION_DIM_LABELS), 1, figsize=(14, 3 * len(ACTION_DIM_LABELS)))
    fig.suptitle(f"Zero-Shot Octo-Small Predictions vs GT — {episode_name}", fontsize=14)

    for dim, (ax, label) in enumerate(zip(axes, ACTION_DIM_LABELS)):
        # Plot first action from each predicted chunk
        ax.plot(pred_actions[:, 0, dim], label="predicted (normalized)", alpha=0.8)
        ax.plot(true_actions[:, dim], label="ground truth (delta)", alpha=0.8)
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {save_path}")


def compute_metrics(pred_actions, true_actions):
    """Compute basic comparison metrics between predicted and GT actions.

    Since the model outputs normalized actions and GT is in delta-EEF space,
    we compute correlation (direction agreement) rather than MSE.
    """
    metrics = {}
    for dim, label in enumerate(ACTION_DIM_LABELS):
        pred = pred_actions[:, 0, dim]
        gt = true_actions[:, dim]

        # Pearson correlation — measures if the model captures the right trends
        if np.std(pred) > 1e-8 and np.std(gt) > 1e-8:
            corr = np.corrcoef(pred, gt)[0, 1]
        else:
            corr = 0.0

        # Direction agreement — fraction of timesteps where pred and GT move in same direction
        pred_diff = np.diff(pred)
        gt_diff = np.diff(gt)
        if len(pred_diff) > 0:
            direction_agree = np.mean(np.sign(pred_diff) == np.sign(gt_diff))
        else:
            direction_agree = 0.0

        metrics[label] = {"correlation": corr, "direction_agreement": direction_agree}

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Octo inference on xarm mcap data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/xarm/default_task",
        help="Path to xarm mcap episode directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="hf://rail-berkeley/octo-small-1.5",
        help="Octo model checkpoint path",
    )
    parser.add_argument(
        "--task_instruction",
        type=str,
        default="do the task",
        help="Language instruction for the task",
    )
    parser.add_argument("--max_episodes", type=int, default=3, help="Max episodes to evaluate")
    parser.add_argument("--output_dir", type=str, default="/octo/results", help="Output directory for plots")
    parser.add_argument("--no_wrist", action="store_true", help="Disable wrist camera input")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = OctoModel.load_pretrained(args.checkpoint)
    print(model.get_pretty_spec())

    # Create task
    task = model.create_tasks(texts=[args.task_instruction])
    print(f"\nTask instruction: '{args.task_instruction}'")

    # Load data
    print(f"\nLoading episodes from {args.data_dir}...")
    episodes = load_dataset(args.data_dir, max_episodes=args.max_episodes)

    # Run inference on each episode
    all_metrics = {}
    for ep in episodes:
        ep_name = ep["name"]
        print(f"\n{'='*60}")
        print(f"Running inference on {ep_name} ({len(ep['front_images'])} frames)")
        print(f"{'='*60}")

        pred_actions, true_actions = run_zero_shot_inference(
            model, ep, task, use_wrist=not args.no_wrist
        )

        print(f"Predicted actions shape: {pred_actions.shape}")
        print(f"Ground truth actions shape: {true_actions.shape}")

        # Plot
        save_path = os.path.join(args.output_dir, f"{ep_name}_actions.png")
        plot_actions(pred_actions, true_actions, ep_name, save_path)

        # Metrics
        metrics = compute_metrics(pred_actions, true_actions)
        all_metrics[ep_name] = metrics
        print("\nMetrics (correlation / direction agreement):")
        for label, m in metrics.items():
            print(f"  {label:>8s}: corr={m['correlation']:+.3f}  dir_agree={m['direction_agreement']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY across all episodes")
    print(f"{'='*60}")
    for label in ACTION_DIM_LABELS:
        corrs = [all_metrics[ep][label]["correlation"] for ep in all_metrics]
        dirs = [all_metrics[ep][label]["direction_agreement"] for ep in all_metrics]
        print(
            f"  {label:>8s}: avg_corr={np.mean(corrs):+.3f}  avg_dir_agree={np.mean(dirs):.3f}"
        )

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
