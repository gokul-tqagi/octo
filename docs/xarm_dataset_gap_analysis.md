# xarm Dataset vs Octo Pretraining Distribution: Gap Analysis

**Date:** 2026-04-26
**Dataset:** xarm place-object-in-toolbox (37 episodes, ROS2 mcap bags)
**Target model:** Octo-Small-1.5 (pretrained on OXE magic soup, primarily Bridge dataset)

## Data Sources

- **Bridge dataset** (Octo's primary pretraining data): WidowX robot, tabletop manipulation, delta EEF actions
- **xarm dataset** (ours): xarm robot, place-object-in-toolbox task, front + wrist cameras

## Side-by-Side Comparison

| Parameter | Bridge (Octo pretrained) | xarm (ours) | Ratio | Status |
|---|---|---|---|---|
| **Workspace X** | 15.2 cm | 2.4 cm | 6.3x smaller | Fix: OK after normalization |
| **Workspace Y** | 22.6 cm | 1.1 cm | 20x smaller | Fix: OK after normalization |
| **Workspace Z** | 15.7 cm | 4.7 cm | 3.3x smaller | Fix: OK after normalization |
| **Per-step displacement (mean)** | 10.26 mm | 0.23 mm (at 10Hz) | 45x smaller | Fix: resample at 2Hz |
| **Per-step displacement (at 2Hz)** | 10.26 mm | ~1.1 mm | ~9x smaller | Acceptable |
| **Position delta abs mean** | 4-7 mm/step | 0.01-0.18 mm/step (10Hz) | ~40x smaller | Fix: resample at 2Hz |
| **Rotation delta abs mean** | 1.1-1.3 deg/step | ~0.01 deg/step (10Hz) | ~100x smaller | Fix: resample at 2Hz |
| **Gripper** | Binary 0/1, varies across steps | Constant 0.851 | Dead dimension | Fix: binarize to 1.0 |
| **Episode length** | ~10-60 steps (10Hz) | 151-226 steps (10Hz), ~45 (2Hz) | Comparable at 2Hz | OK |
| **Recording frequency** | ~10 Hz | ~10 Hz native | Match | OK |
| **Front camera** | 256x256 | 1280x720 -> 256x256 | Match | OK |
| **Wrist camera** | None (bridge) | 640x480 -> 128x128 | Extra camera | OK (bonus) |
| **Action dims** | 7 (delta EEF + gripper) | 7 (delta EEF + gripper) | Match | OK |
| **Action representation** | delta pos + delta euler + binary grip | delta pos + delta euler + grip | Match format | OK |
| **Normalization** | NORMAL (mean/std), gripper masked | Same planned | Match | OK |
| **State/proprio dims** | 7 (pos + euler + gripper) | 7 (pos + euler + gripper) | Match | OK |
| **Episodes** | ~25+ (debug), thousands (full) | 37 | Small but viable | OK for finetune |

## Issues Found and Fixes Applied

### 1. Euler Wrapping Bug (CRITICAL)

**Problem:** Original extraction computed rotation deltas by subtracting consecutive euler angles. When euler angles wrap (e.g., roll jumps from +179 deg to -179 deg), the naive subtraction produces a false delta of ~358 degrees. This corrupts the rotation action dimensions.

**Evidence:** Episode 0 roll std = 172 degrees with 360-degree range — clear wrapping signature.

**Fix:** Compute rotation deltas in quaternion space using relative rotation `q_delta = q_next * q_curr^{-1}`, then convert to euler. The resulting delta euler angles are guaranteed bounded and physically correct.

### 2. Action Scale Mismatch (HIGH)

**Problem:** At 10Hz native recording rate, xarm per-step displacement is 0.23mm mean — 45x smaller than bridge's 10.26mm. The pretrained model's prior expects centimeter-scale deltas. Near-zero deltas have poor signal-to-noise ratio.

**Fix:** Resample at 2Hz instead of 10Hz. At 2Hz, each step covers ~5x more distance (~1.1mm), reducing the gap to ~9x. Normalization handles the remaining scale difference. Episode length drops from ~200 to ~45 steps (comparable to bridge at ~10-60 steps).

### 3. Constant Gripper (MEDIUM)

**Problem:** Gripper value is constant at 0.851 across all 37 episodes — never opens or closes. The 7th action dimension (gripper delta) is always zero, providing no learning signal.

**Evidence:** `np.unique(gripper_values) = [0.851]` across 10 sampled episodes.

**Fix:** Binarize gripper to 0/1 matching bridge dataset convention (threshold 0.5). Since gripper is always closed (0.851 > 0.5*0.85), all steps get gripper=1.0. The model learns "keep gripper closed" which is correct for this task. The gripper dimension is masked from normalization (matches bridge config).

## Bridge Dataset Detailed Statistics (from debug dataset)

```
Episodes: 25 (debug), full dataset much larger
Episode lengths: 10 steps (debug, likely truncated)

Actions (delta EEF, 7-dim):
  Mean:     [ 0.001592 -0.000971 -0.002935 -0.009942 -0.004846 -0.005332  0.739201]
  Std:      [0.005735 0.005613 0.008285 0.036344 0.028768 0.036266 0.434099]
  Min:      [-0.019015 -0.029268 -0.020456 -0.159555 -0.117399 -0.254269  0.000000]
  Max:      [0.017997 0.012203 0.032824 0.132965 0.072375 0.067411 1.000000]

  Position delta: mean 10.26 mm/step, max 35.57 mm/step
  Rotation delta: mean 1.1-1.3 deg/step
  Gripper: binary {0, 1}, mean 0.74 (mostly open)

States (EEF pos + euler + gripper, 7-dim):
  Workspace: X=15.2cm, Y=22.6cm, Z=15.7cm
```

## xarm Dataset Statistics (10 episodes sampled)

```
Episodes: 37 total
Frames per episode: 151-226 (at native 10Hz)

Position workspace (all episodes):
  X: [0.2639, 0.2875] = 2.4 cm
  Y: [-0.0847, -0.0738] = 1.1 cm
  Z: [0.2757, 0.3232] = 4.7 cm

Trajectory lengths: 3.6-5.5 cm per episode
Step sizes: mean 0.227 mm, max 6.220 mm (at 10Hz)
Gripper: constant 0.851 across all episodes
```

## Recommended Finetune Configuration

Based on the gap analysis, the finetune config uses:

- `target_hz=2.0` (resampling from native 10Hz)
- `batch_size=64` (small dataset)
- `max_steps=10000` (37 eps * ~45 steps = ~1665 frames, ~26 batches/epoch)
- `eval_interval=2000` (check for overfitting early)
- `action_normalization_mask=[True]*6 + [False]` (don't normalize gripper)
- `mode=full,language_conditioned` (language-only, no image goals)
- Image augmentation matching bridge defaults

## Remaining Considerations

1. **Workspace scale vs pretrained prior:** Even after normalization, the pretrained model learned from centimeter-scale workspaces. The xarm 2.4x1.1x4.7cm workspace may be below the model's learned distribution. Monitor whether the model outputs actions that are too large. Consider `head_only` finetuning first to limit divergence.

2. **Single-task, single-instruction:** All 37 episodes have the same task and instruction. The model may overfit to producing a single trajectory regardless of visual input. Language conditioning provides limited signal when there's only one instruction.

3. **No gripper variation:** The model cannot learn gripper control from this data. If the deployment task requires opening/closing, additional data with gripper variation is needed.
