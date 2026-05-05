# eval_compare_aloha.py: Compare two finetuned Octo checkpoints on a common ALOHA val episode.
# eval_compare_aloha.py: Loads val episode 0 from 55ep val set and evaluates both models.

import glob
import json
import os

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache


VAL_TFRECORD = "/data/rlds_55ep/aloha_pack_red_bag_55ep/1.0.0/aloha_pack_red_bag_55ep-val.tfrecord-00000-of-00001"

MODELS = {
    "10ep_headonly": {
        "checkpoint": "/checkpoints/aloha_pack_red_bag",
        "step":       9999,
        "stats":      "/checkpoints/aloha_pack_red_bag/dataset_statistics.json",
    },
    "55ep_headonly_20k": {
        "checkpoint": "/checkpoints/aloha_55ep_headonly",
        "step":       19999,
        "stats":      "/checkpoints/aloha_55ep_headonly/dataset_statistics.json",
    },
}


def load_episode(tfrecord_path, episode_idx=0):
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    for i, raw in enumerate(dataset):
        if i != episode_idx:
            continue
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        steps_bytes = ex.features.feature["steps"].bytes_list.value

        ep = {"state": [], "action": [], "language": ""}
        for sb in steps_bytes:
            step = tf.train.Example()
            step.ParseFromString(sb)
            sf = step.features.feature
            ep["state"].append(np.array(sf["observation/state"].float_list.value, dtype=np.float32))
            ep["action"].append(np.array(sf["action"].float_list.value, dtype=np.float32))
            if not ep["language"]:
                ep["language"] = sf["language_instruction"].bytes_list.value[0].decode("utf-8")

        ep["state"] = np.stack(ep["state"])
        ep["action"] = np.stack(ep["action"])
        print(f"Loaded episode {episode_idx}: {len(ep['state'])} steps, action_dim={ep['action'].shape[1]}")
        print(f"  Language: {ep['language']}")
        return ep

    raise ValueError(f"Episode {episode_idx} not found in {tfrecord_path}")


def rollout(model, episode, stats):
    action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
    action_std  = np.array(stats["action"]["std"],  dtype=np.float32)
    proprio_mean = np.array(stats["proprio"]["mean"], dtype=np.float32)
    proprio_std  = np.array(stats["proprio"]["std"],  dtype=np.float32)

    T = len(episode["state"])
    task = model.create_tasks(texts=[episode["language"]])
    rng = jax.random.PRNGKey(42)
    preds = []

    for t in range(T - 1):
        proprio = (episode["state"][t] - proprio_mean) / (proprio_std + 1e-8)
        obs = {
            "image_primary": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
            "image_wrist":   np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
            "proprio":       proprio[None, None],
            "timestep_pad_mask": np.array([[True]]),
            "pad_mask_dict": {
                "image_primary": np.array([[True]]),
                "image_wrist":   np.array([[True]]),
                "proprio":       np.array([[True]]),
            },
        }
        rng, key = jax.random.split(rng)
        actions = model.sample_actions(jax.tree_map(lambda x: jnp.array(x), obs), task, rng=key)
        pred = np.array(actions[0, 0, 0])
        # denormalize
        pred_denorm = pred * (action_std + 1e-8) + action_mean
        preds.append(pred_denorm)

    return np.stack(preds)


def metrics(gt, pred):
    pos_err  = np.linalg.norm(gt[:, :3] - pred[:, :3], axis=1) * 1000
    rot_err  = np.degrees(np.linalg.norm(gt[:, 3:6] - pred[:, 3:6], axis=1))
    gt_traj  = np.cumsum(gt[:, :3], axis=0)
    pr_traj  = np.cumsum(pred[:, :3], axis=0)
    traj_err = np.linalg.norm(gt_traj - pr_traj, axis=1) * 1000
    mse      = float(np.mean((gt - pred) ** 2))
    return {
        "MSE":              round(mse, 4),
        "pos_err_mean_mm":  round(float(pos_err.mean()), 2),
        "pos_err_max_mm":   round(float(pos_err.max()),  2),
        "rot_err_mean_deg": round(float(rot_err.mean()), 2),
        "traj_final_mm":    round(float(traj_err[-1]),   2),
    }


def main():
    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    episode = load_episode(VAL_TFRECORD, episode_idx=0)
    gt = episode["action"][:-1]

    print("\n" + "="*55)
    print(f"Val episode 0 | {len(gt)} steps | GT action dim={gt.shape[1]}")
    print("="*55)

    results = {}
    for name, cfg in MODELS.items():
        print(f"\nLoading {name} from {cfg['checkpoint']} step={cfg['step']} ...")
        model = OctoModel.load_pretrained(cfg["checkpoint"], step=cfg["step"])
        with open(cfg["stats"]) as f:
            stats = json.load(f)
        pred = rollout(model, episode, stats)
        min_len = min(len(gt), len(pred))
        m = metrics(gt[:min_len], pred[:min_len])
        results[name] = m
        print(f"  {name}: {m}")

    print("\n" + "="*55)
    print("COMPARISON SUMMARY")
    print("="*55)
    keys = list(next(iter(results.values())).keys())
    header = f"{'Metric':<22}" + "".join(f"{n:<22}" for n in results)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<22}" + "".join(f"{results[n][k]:<22}" for n in results)
        print(row)

    with open("/eval_output/compare_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /eval_output/compare_metrics.json")


if __name__ == "__main__":
    main()
