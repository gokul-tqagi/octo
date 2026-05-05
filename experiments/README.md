# Octo Finetuning Experiments

Custom training pipeline for finetuning Octo-Small on ALOHA bimanual manipulation tasks.
Built on top of the [original octo repo](https://github.com/octo-models/octo) (kept untouched in `scripts/`).

## Directory Structure

```
experiments/
├── configs/                    # YAML configs for each stage
│   ├── lerobot_*.yaml          # Data conversion configs (source format, Hz, cameras)
│   └── finetune_*.yaml         # Training configs (steps, lr, freeze mode)
│
├── data/                       # Stage 1: Data conversion & inspection
│   ├── lerobot_to_rlds.py      # LeRobot v3 (parquet+mp4) -> RLDS TFRecord
│   ├── mcap_to_rlds.py         # ROS2 mcap bags -> RLDS TFRecord
│   ├── validate_rlds.py        # Visual validation of converted datasets
│   ├── analyze_lerobot_dataset.py  # Distribution analysis of raw LeRobot data
│   ├── inspect_lerobot_data.py # Quick inspection of LeRobot parquet files
│   ├── wxai_fk.py              # Forward kinematics for ALOHA xarm
│   └── xarm_standardization_transforms.py  # Octo data transform for xarm actions
│
├── train/                      # Stage 2: Training
│   ├── finetune_xarm.py        # Core finetuning script (loads TFRecords, trains Octo)
│   ├── train.sh                # End-to-end: sync to server + launch training
│   └── deploy_training.sh      # Sync data/code/Docker image to remote GPU server
│
├── eval/                       # Stage 3: Evaluation
│   ├── eval_xarm.py            # Run model on val episodes, compute trajectory metrics
│   ├── eval.sh                 # End-to-end: eval on server + download + update Excel
│   ├── compile_results.py      # Build/update Excel spreadsheet across all runs
│   └── eval_compare_aloha.py   # Compare multiple checkpoints side-by-side
│
├── visualize/                  # Stage 4: Visualization
│   ├── visualize_rlds_data.py  # Pre-training: raw vs downsampled data quality
│   └── visualize_xarm_predictions.py  # Post-training: predicted vs GT trajectories
│
└── run_xarm_pipeline.sh        # Legacy: full mcap->train pipeline (single script)
```

## Typical Workflow

### 1. Convert data (local machine)

```bash
docker run --rm -v $(pwd):/octo -v /path/to/lerobot/data:/data/aloha -v /path/to/output:/data/rlds_output \
  octo-finetune:latest python3 /octo/experiments/data/lerobot_to_rlds.py \
    --config /octo/experiments/configs/lerobot_marker_pick_10hz.yaml
```

### 2. Visualize data quality (local machine)

```bash
docker run --rm -v $(pwd):/octo -v /path/to/lerobot/data:/data/aloha -v /path/to/output:/data \
  octo-finetune:latest python3 /octo/experiments/visualize/visualize_rlds_data.py \
    --config /octo/experiments/configs/lerobot_marker_pick_10hz.yaml \
    --output_dir /data/viz_output/data_quality
```

### 3. Train (remote GPU server)

```bash
# Syncs data + code to server, launches training in a screen session
bash experiments/train/train.sh \
  experiments/configs/finetune_marker_pick_10hz.yaml \
  aws-L4-server1 \
  /home/ubuntu/torqueagi
```

### 4. Eval + download results

```bash
# Runs eval on server, downloads results, updates Excel spreadsheet
bash experiments/eval/eval.sh \
  experiments/configs/finetune_marker_pick_10hz.yaml \
  aws-L4-server1 \
  /home/ubuntu/torqueagi
```

## Config Naming Convention

- `lerobot_<task>.yaml` -- data conversion config (source dirs, Hz, camera selection)
- `lerobot_<task>_<hz>hz.yaml` -- data conversion at a specific sampling rate
- `finetune_<task>.yaml` -- training config (baseline head_only from pretrained)
- `finetune_<task>_<variant>.yaml` -- training variant (10hz, 40k, full, etc.)

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sampling rate | 10Hz over 5Hz | 2x data, halved step size, ~45% better pos accuracy |
| Finetune mode | head_only | Sufficient for 50-75 episode datasets, avoids overfit |
| Action space | delta EEF (6D + gripper) | More generalizable than raw joint positions |
| Training steps | 20k | Diminishing returns beyond this for head_only |

## Results

Run `compile_results.py` to generate/update the Excel spreadsheet:

```bash
docker run --rm -v $(pwd):/octo -v /path/to/data:/data \
  octo-finetune:latest bash -c "pip install openpyxl -q && \
    python3 /octo/experiments/eval/compile_results.py \
      --data_dir /data --config_dir /octo/experiments/configs \
      --output /data/octo_finetune_results.xlsx"
```
