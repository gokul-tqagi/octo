# finetune_xarm.py: Finetune Octo-Small on xarm place-object-in-toolbox dataset.
# finetune_xarm.py: Loads custom TFRecords directly, bypassing TFDS.

"""
Finetune Octo-Small on the xarm place-object-in-toolbox dataset.

Loads our custom TFRecords directly via tf.data (bypasses TFDS which
can't read our nested serialization format). Keeps Octo's pretrained
observation space (front + wrist cameras) and action space (7-dim delta EEF).

Usage:
    python experiments/train/finetune_xarm.py \
        --pretrained_path=hf://rail-berkeley/octo-small-1.5 \
        --data_dir=./data/xarm_place_toolbox_rlds \
        --save_dir=./checkpoints/xarm_place_toolbox
"""

import glob
import json
import os
import uuid
from datetime import datetime

from absl import app, flags, logging
import cv2
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm

from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to YAML config file (CLI flags override).")
flags.DEFINE_string("pretrained_path", None, "Path to pre-trained Octo checkpoint.")
flags.DEFINE_string("data_dir", None, "Path to RLDS dataset directory.")
flags.DEFINE_string("save_dir", None, "Directory for saving checkpoints.")
flags.DEFINE_string("dataset_name", "xarm_place_toolbox", "Name of the dataset.")
flags.DEFINE_integer("batch_size", 32, "Batch size for finetuning.")
flags.DEFINE_integer("num_steps", 5000, "Number of training steps.")
flags.DEFINE_float("learning_rate", 3e-4, "Peak learning rate.")
flags.DEFINE_integer("action_horizon", 4, "Action chunking horizon.")
flags.DEFINE_integer("window_size", 1, "Observation window size.")
flags.DEFINE_bool("freeze_transformer", False, "Freeze pretrained transformer.")
flags.DEFINE_integer("log_interval", 50, "Steps between log prints.")
flags.DEFINE_integer("save_interval", 1000, "Steps between checkpoint saves.")
flags.DEFINE_bool("wandb", False, "Enable wandb logging.")


# ── Custom TFRecord data loader ──────────────────────────────────────────────

def parse_step(step_bytes):
    """Parse a single serialized step Example."""
    step_ex = tf.train.Example()
    step_ex.ParseFromString(step_bytes.numpy())
    sf = step_ex.features.feature

    action = np.array(sf["action"].float_list.value, dtype=np.float32)
    state = np.array(sf["observation/state"].float_list.value, dtype=np.float32)

    # Decode JPEG images
    front_jpg = sf["observation/image_0"].bytes_list.value[0]
    front = cv2.imdecode(
        np.frombuffer(front_jpg, np.uint8), cv2.IMREAD_COLOR
    )
    front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)

    wrist_jpg = sf["observation/image_1"].bytes_list.value[0]
    wrist = cv2.imdecode(
        np.frombuffer(wrist_jpg, np.uint8), cv2.IMREAD_COLOR
    )
    wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)

    language = sf["language_instruction"].bytes_list.value[0].decode("utf-8")

    return front, wrist, state, action, language


def load_all_episodes(data_dir, dataset_name, split="train"):
    """Load all episodes from our custom TFRecords into memory."""
    ds_path = os.path.join(data_dir, dataset_name, "1.0.0")
    pattern = os.path.join(ds_path, f"{dataset_name}-{split}.tfrecord*")
    tfrecord_files = sorted(glob.glob(pattern))

    if not tfrecord_files:
        raise FileNotFoundError(f"No {split} tfrecords found: {pattern}")

    episodes = []
    dataset = tf.data.TFRecordDataset(tfrecord_files)

    for raw_record in dataset:
        traj_ex = tf.train.Example()
        traj_ex.ParseFromString(raw_record.numpy())
        steps_bytes = traj_ex.features.feature["steps"].bytes_list.value

        ep = {"front": [], "wrist": [], "state": [], "action": [], "language": ""}
        for sb in steps_bytes:
            front, wrist, state, action, lang = parse_step(
                tf.constant(sb)
            )
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


def compute_dataset_statistics(episodes):
    """Compute action and proprio normalization statistics."""
    all_actions = np.concatenate([ep["action"] for ep in episodes])
    all_states = np.concatenate([ep["state"] for ep in episodes])

    stats = {
        "action": {
            "mean": all_actions.mean(0).tolist(),
            "std": all_actions.std(0).tolist(),
            "min": all_actions.min(0).tolist(),
            "max": all_actions.max(0).tolist(),
            "p01": np.percentile(all_actions, 1, axis=0).tolist(),
            "p99": np.percentile(all_actions, 99, axis=0).tolist(),
        },
        "proprio": {
            "mean": all_states.mean(0).tolist(),
            "std": all_states.std(0).tolist(),
            "min": all_states.min(0).tolist(),
            "max": all_states.max(0).tolist(),
            "p01": np.percentile(all_states, 1, axis=0).tolist(),
            "p99": np.percentile(all_states, 99, axis=0).tolist(),
        },
        "num_transitions": sum(len(ep["action"]) for ep in episodes),
        "num_trajectories": len(episodes),
    }
    return stats


def normalize(val, mean, std, mask=None):
    """Normalize with optional mask (don't normalize masked dims)."""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    result = (val - mean) / (std + 1e-8)
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        result = np.where(mask, result, val)
    return result


def make_batch_generator(episodes, stats, batch_size, action_horizon, window_size):
    """Generator that yields batches in Octo's expected format.

    For each sample, picks a random episode and random timestep, then constructs
    the observation window and action chunk.
    """
    action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
    action_std = np.array(stats["action"]["std"], dtype=np.float32)
    # Don't normalize gripper (last dim)
    action_mask = np.array([True, True, True, True, True, True, False])

    proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
    proprio_std = np.array(stats["proprio"]["std"], dtype=np.float32)

    n_episodes = len(episodes)
    ep_lengths = [len(ep["action"]) for ep in episodes]

    while True:
        batch_obs_primary = []
        batch_obs_wrist = []
        batch_obs_proprio = []
        batch_obs_pad_mask = []
        batch_actions = []
        batch_action_pad_mask = []
        batch_language = []

        for _ in range(batch_size):
            # Sample random episode and timestep
            ep_idx = np.random.randint(n_episodes)
            ep = episodes[ep_idx]
            T = ep_lengths[ep_idx]
            t = np.random.randint(0, T)

            # Observation window (window_size frames ending at t)
            obs_primary = []
            obs_wrist = []
            obs_proprio = []
            obs_pad = []

            for w in range(window_size):
                idx = t - (window_size - 1 - w)
                if idx < 0:
                    # Pad with first frame
                    obs_primary.append(ep["front"][0])
                    obs_wrist.append(ep["wrist"][0])
                    obs_proprio.append(
                        normalize(ep["state"][0], proprio_mean, proprio_std)
                    )
                    obs_pad.append(False)
                else:
                    obs_primary.append(ep["front"][idx])
                    obs_wrist.append(ep["wrist"][idx])
                    obs_proprio.append(
                        normalize(ep["state"][idx], proprio_mean, proprio_std)
                    )
                    obs_pad.append(True)

            batch_obs_primary.append(np.stack(obs_primary))
            batch_obs_wrist.append(np.stack(obs_wrist))
            batch_obs_proprio.append(np.stack(obs_proprio))
            batch_obs_pad_mask.append(np.array(obs_pad))

            # Action chunk: action_horizon steps starting at t
            act_chunk = []
            act_pad = []
            for h in range(action_horizon):
                idx = t + h
                if idx < T:
                    act = normalize(ep["action"][idx], action_mean, action_std, action_mask)
                    act_chunk.append(act)
                    act_pad.append(True)
                else:
                    act_chunk.append(np.zeros(7, dtype=np.float32))
                    act_pad.append(False)

            batch_actions.append(np.stack(act_chunk))
            # action_pad_mask is (H, A) — same shape as action chunk, per-dim mask
            act_pad_2d = np.stack([
                np.ones(7, dtype=bool) if p else np.zeros(7, dtype=bool)
                for p in act_pad
            ])
            batch_action_pad_mask.append(act_pad_2d)
            batch_language.append(ep["language"].encode("utf-8"))

        # pad_mask_dict values must be (B, window_size) to match observation shapes
        W = window_size
        yield {
            "observation": {
                "image_primary": np.stack(batch_obs_primary),    # (B, W, 256, 256, 3)
                "image_wrist": np.stack(batch_obs_wrist),        # (B, W, 128, 128, 3)
                "proprio": np.stack(batch_obs_proprio),          # (B, W, 7)
                "timestep_pad_mask": np.stack(batch_obs_pad_mask),  # (B, W)
                "pad_mask_dict": {
                    "image_primary": np.ones((batch_size, W), dtype=bool),
                    "image_wrist": np.ones((batch_size, W), dtype=bool),
                    "proprio": np.ones((batch_size, W), dtype=bool),
                },
            },
            "task": {
                "language_instruction": batch_language,
                "pad_mask_dict": {
                    "language_instruction": np.ones(batch_size, dtype=bool),
                },
            },
            "action": np.stack(batch_actions)[:, None, :, :],    # (B, W, H, 7)
            "action_pad_mask": np.stack(batch_action_pad_mask)[:, None, :, :],  # (B, W, H, A)
            "dataset_name": ["xarm_place_toolbox"] * batch_size,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(_):
    # Load YAML config if provided, then let CLI flags override
    if FLAGS.config:
        import yaml
        with open(FLAGS.config) as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            if hasattr(FLAGS, key) and FLAGS[key].value == FLAGS[key].default:
                FLAGS[key].value = val

    assert FLAGS.pretrained_path, "Must provide --pretrained_path or --config with pretrained_path"
    assert FLAGS.data_dir, "Must provide --data_dir or --config with data_dir"

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    if FLAGS.wandb:
        import wandb as wb
        wb.init(name="finetune_xarm", project="octo_finetune_xarm")

    # Load data
    logging.info("Loading dataset...")
    train_episodes = load_all_episodes(FLAGS.data_dir, FLAGS.dataset_name, "train")
    val_episodes = load_all_episodes(FLAGS.data_dir, FLAGS.dataset_name, "val")
    stats = compute_dataset_statistics(train_episodes)

    total_frames = stats["num_transitions"]
    logging.info(
        f"Loaded {len(train_episodes)} train, {len(val_episodes)} val episodes "
        f"({total_frames} total frames)"
    )

    # Generate unique training run ID
    run_id = f"{FLAGS.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Save stats for inference later
    if FLAGS.save_dir:
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        with open(os.path.join(FLAGS.save_dir, "dataset_statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)

    # Set up training log file alongside checkpoints
    log_file = None
    if FLAGS.save_dir:
        log_path = os.path.join(FLAGS.save_dir, f"train_{run_id}.log")
        log_file = open(log_path, "w")
        log_file.write(f"run_id: {run_id}\n")
        log_file.write(f"pretrained_path: {FLAGS.pretrained_path}\n")
        log_file.write(f"dataset: {FLAGS.dataset_name}\n")
        log_file.write(f"batch_size: {FLAGS.batch_size}\n")
        log_file.write(f"num_steps: {FLAGS.num_steps}\n")
        log_file.write(f"learning_rate: {FLAGS.learning_rate}\n")
        log_file.write(f"freeze_transformer: {FLAGS.freeze_transformer}\n")
        log_file.write(f"train_episodes: {len(train_episodes)}\n")
        log_file.write(f"val_episodes: {len(val_episodes)}\n")
        log_file.write(f"total_frames: {total_frames}\n")
        log_file.write(f"---\n")
        log_file.write(f"step,loss\n")
        log_file.flush()
        logging.info(f"Training log: {log_path}")

    # Create data iterators
    train_iter = make_batch_generator(
        train_episodes, stats, FLAGS.batch_size,
        FLAGS.action_horizon, FLAGS.window_size,
    )

    # Load pretrained model
    logging.info("Loading pretrained Octo model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # Get config — keep the pretrained architecture, just use our data
    config = pretrained_model.config
    text_processor = pretrained_model.text_processor

    # Process first batch to get example shapes
    example_batch = next(train_iter)
    example_batch = process_text(example_batch, text_processor)
    del example_batch["dataset_name"]

    # Initialize model from config with our data
    logging.info("Initializing model for finetuning...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=stats,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    # Optimizer with cosine schedule
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(0.0, FLAGS.learning_rate, 500),
            optax.cosine_decay_schedule(FLAGS.learning_rate, FLAGS.num_steps - 500),
        ],
        [500],
    )
    tx = optax.adamw(lr_schedule, weight_decay=0.01)
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)

    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)

    train_state = TrainState.create(
        rng=jax.random.PRNGKey(42),
        model=model,
        tx=tx,
    )

    # Loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True,
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # Training loop
    logging.info(f"Starting finetuning for {FLAGS.num_steps} steps...")
    logging.info(f"  Batch size: {FLAGS.batch_size}")
    logging.info(f"  Action horizon: {FLAGS.action_horizon}")
    logging.info(f"  Window size: {FLAGS.window_size}")
    logging.info(f"  Freeze transformer: {FLAGS.freeze_transformer}")

    for step in tqdm.tqdm(range(FLAGS.num_steps), dynamic_ncols=True):
        batch = next(train_iter)
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]

        train_state, update_info = train_step(train_state, batch)

        if (step + 1) % FLAGS.log_interval == 0:
            info = jax.device_get(update_info)
            loss = float(info["loss"])
            tqdm.tqdm.write(f"  Step {step+1}/{FLAGS.num_steps} | loss: {loss:.4f}")

            if log_file:
                log_file.write(f"{step+1},{loss:.6f}\n")
                log_file.flush()

            if FLAGS.wandb:
                wb.log(
                    flax.traverse_util.flatten_dict({"training": info}, sep="/"),
                    step=step,
                )

        if FLAGS.save_dir and (step + 1) % FLAGS.save_interval == 0:
            save_path = FLAGS.save_dir
            logging.info(f"Saving checkpoint at step {step+1} to {save_path}")
            train_state.model.save_pretrained(
                step=step, checkpoint_path=save_path,
            )

    # Final save
    if FLAGS.save_dir:
        train_state.model.save_pretrained(
            step=FLAGS.num_steps - 1, checkpoint_path=FLAGS.save_dir,
        )
        logging.info(f"Training complete. Final checkpoint saved to {FLAGS.save_dir}")

    if log_file:
        log_file.close()


if __name__ == "__main__":
    app.run(main)
