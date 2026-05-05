#!/bin/bash
# eval.sh: Run eval on a remote server's checkpoints and download results locally.
# eval.sh: Runs eval + viz on server, syncs results back, updates Excel spreadsheet.

set -euo pipefail

OCTO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_DATA_DIR="/home/gokul/data"
DOCKER_IMAGE="octo-finetune:latest"

usage() {
    echo "Usage: $0 <finetune_config> <server> <dest_dir>"
    echo "       $0 <finetune_config> local"
    echo ""
    echo "  finetune_config  Finetune YAML config used for training"
    echo "  server           SSH config host name (e.g. aws-L4-server1)"
    echo "  dest_dir         Remote base directory (e.g. /home/ubuntu/torqueagi)"
    echo ""
    echo "Examples:"
    echo "  $0 experiments/configs/finetune_marker_pick_10hz.yaml aws-L4-server1 /home/ubuntu/torqueagi"
    echo "  $0 experiments/configs/finetune_lipbalm_10hz.yaml local"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

CONFIG="$1"
MODE="$2"
CONFIG_PATH="$OCTO_DIR/$CONFIG"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

# Parse config values
DATASET_NAME=$(grep 'dataset_name:' "$CONFIG_PATH" | head -1 | awk '{print $2}' | tr -d '"')
SAVE_DIR=$(grep 'save_dir:' "$CONFIG_PATH" | head -1 | awk '{print $2}' | tr -d '"')
CHECKPOINT_NAME=$(basename "$SAVE_DIR")

echo "=== Octo Eval ==="
echo "  Config:     $CONFIG"
echo "  Dataset:    $DATASET_NAME"
echo "  Checkpoint: $CHECKPOINT_NAME"
echo ""

if [ "$MODE" = "local" ]; then
    DATA_DIR="$(cd "$OCTO_DIR/../data" 2>/dev/null && pwd || echo "$LOCAL_DATA_DIR")"

    echo "[1/3] Running eval locally..."
    docker run --rm --gpus all --shm-size 8g -v "$OCTO_DIR":/octo -v "$DATA_DIR":/data -v ~/.cache:/root/.cache -e HF_HOME=/root/.cache/huggingface "$DOCKER_IMAGE" bash -c "pip install -e /octo -q && python3 /octo/experiments/eval/eval_xarm.py --checkpoint_path=/data/checkpoints/$CHECKPOINT_NAME --data_dir=/data/rlds_output --dataset_name=$DATASET_NAME --output_dir=/data/eval_output/$CHECKPOINT_NAME --max_episodes=7"

    echo "[2/3] Running viz locally..."
    docker run --rm --gpus all --shm-size 8g -v "$OCTO_DIR":/octo -v "$DATA_DIR":/data -v ~/.cache:/root/.cache -e HF_HOME=/root/.cache/huggingface "$DOCKER_IMAGE" bash -c "pip install -e /octo -q && python3 /octo/experiments/visualize/visualize_xarm_predictions.py --checkpoint_path=/data/checkpoints/$CHECKPOINT_NAME --data_dir=/data/rlds_output --dataset_name=$DATASET_NAME --output_dir=/data/viz_output/$CHECKPOINT_NAME --max_episodes=3 --samples_per_state=8"

    # Update results spreadsheet
    echo "[3/3] Updating results spreadsheet..."
    docker run --rm -v "$OCTO_DIR":/octo -v "$DATA_DIR":/data "$DOCKER_IMAGE" bash -c "pip install openpyxl -q && python3 /octo/experiments/eval/compile_results.py --data_dir /data --config_dir /octo/experiments/configs --output /data/octo_finetune_results.xlsx --run $CHECKPOINT_NAME"

    echo ""
    echo "Done. Results at:"
    echo "  Eval:  $DATA_DIR/eval_output/$CHECKPOINT_NAME/"
    echo "  Viz:   $DATA_DIR/viz_output/$CHECKPOINT_NAME/"
    echo "  Excel: $DATA_DIR/octo_finetune_results.xlsx"
else
    if [ $# -ne 3 ]; then
        usage
    fi
    SERVER="$2"
    DEST_DIR="$3"

    # Check training status
    echo "Checking training status on $SERVER..."
    RUNNING=$(ssh "$SERVER" "docker ps --format '{{.Image}}' | grep -c octo-finetune || true")
    if [ "$RUNNING" -gt 0 ]; then
        echo "WARNING: Training container still running on $SERVER. Wait for it to finish."
        ssh "$SERVER" "docker ps --format '{{.Names}} {{.Status}}'"
        exit 1
    fi

    # Check checkpoints exist
    CKPT_COUNT=$(ssh "$SERVER" "ls -d $DEST_DIR/data/checkpoints/$CHECKPOINT_NAME/*/default 2>/dev/null | wc -l")
    if [ "$CKPT_COUNT" -eq 0 ]; then
        echo "Error: No checkpoints found at $SERVER:$DEST_DIR/data/checkpoints/$CHECKPOINT_NAME/"
        exit 1
    fi
    echo "Found $CKPT_COUNT checkpoints on $SERVER"

    # Print training log summary if available
    LOG_FILE=$(ssh "$SERVER" "ls $DEST_DIR/data/checkpoints/$CHECKPOINT_NAME/train_*.log 2>/dev/null | head -1")
    if [ -n "$LOG_FILE" ]; then
        echo ""
        echo "--- Training Log Summary ---"
        ssh "$SERVER" "head -10 $LOG_FILE"
        echo "..."
        ssh "$SERVER" "tail -5 $LOG_FILE"
        echo "---"
        echo ""
    fi

    # 1. Run eval on server
    echo "[1/5] Running eval on $SERVER..."
    ssh "$SERVER" "docker run --rm --gpus all --shm-size 8g -v $DEST_DIR/octo:/octo -v $DEST_DIR/data:/data -v ~/.cache:/root/.cache -e HF_HOME=/root/.cache/huggingface $DOCKER_IMAGE bash -c 'pip install -e /octo -q && python3 /octo/experiments/eval/eval_xarm.py --checkpoint_path=/data/checkpoints/$CHECKPOINT_NAME --data_dir=/data/rlds_output --dataset_name=$DATASET_NAME --output_dir=/data/eval_output/$CHECKPOINT_NAME --max_episodes=7'"

    # 2. Run viz on server
    echo "[2/5] Running viz on $SERVER..."
    ssh "$SERVER" "docker run --rm --gpus all --shm-size 8g -v $DEST_DIR/octo:/octo -v $DEST_DIR/data:/data -v ~/.cache:/root/.cache -e HF_HOME=/root/.cache/huggingface $DOCKER_IMAGE bash -c 'pip install -e /octo -q && python3 /octo/experiments/visualize/visualize_xarm_predictions.py --checkpoint_path=/data/checkpoints/$CHECKPOINT_NAME --data_dir=/data/rlds_output --dataset_name=$DATASET_NAME --output_dir=/data/viz_output/$CHECKPOINT_NAME --max_episodes=3 --samples_per_state=8'"

    # 3. Download results
    echo "[3/5] Downloading results..."
    mkdir -p "$LOCAL_DATA_DIR/eval_output/$CHECKPOINT_NAME" "$LOCAL_DATA_DIR/viz_output/$CHECKPOINT_NAME" 2>/dev/null || docker run --rm -v "$LOCAL_DATA_DIR":/data "$DOCKER_IMAGE" bash -c "mkdir -p /data/eval_output/$CHECKPOINT_NAME /data/viz_output/$CHECKPOINT_NAME && chown -R 1000:1000 /data/eval_output/$CHECKPOINT_NAME /data/viz_output/$CHECKPOINT_NAME"
    rsync -avz "$SERVER:$DEST_DIR/data/eval_output/$CHECKPOINT_NAME/" "$LOCAL_DATA_DIR/eval_output/$CHECKPOINT_NAME/"
    rsync -avz "$SERVER:$DEST_DIR/data/viz_output/$CHECKPOINT_NAME/" "$LOCAL_DATA_DIR/viz_output/$CHECKPOINT_NAME/"

    # Download training log
    if [ -n "$LOG_FILE" ]; then
        rsync -avz "$SERVER:$LOG_FILE" "$LOCAL_DATA_DIR/eval_output/$CHECKPOINT_NAME/"
    fi

    # 4. Print results
    echo ""
    echo "[4/5] Results downloaded."
    echo ""
    METRICS="$LOCAL_DATA_DIR/eval_output/$CHECKPOINT_NAME/eval_metrics.json"
    if [ -f "$METRICS" ]; then
        echo "--- Eval Metrics ---"
        python3 -c "
import json, numpy as np
with open('$METRICS') as f:
    data = json.load(f)
keys = ['pos_error_mean_mm', 'pos_error_max_mm', 'rot_error_mean_deg', 'rot_error_max_deg', 'traj_error_mean_mm', 'traj_error_final_mm']
print(f'  Episodes: {len(data)}')
for k in keys:
    vals = [d[k] for d in data]
    print(f'  {k}: {np.mean(vals):.2f} (std={np.std(vals):.2f})')
"
        echo "---"
    fi

    # 5. Update results spreadsheet
    echo "[5/5] Updating results spreadsheet..."
    docker run --rm -v "$OCTO_DIR":/octo -v "$LOCAL_DATA_DIR":/data "$DOCKER_IMAGE" bash -c "pip install openpyxl -q && python3 /octo/experiments/eval/compile_results.py --data_dir /data --config_dir /octo/experiments/configs --output /data/octo_finetune_results.xlsx --run $CHECKPOINT_NAME"

    echo ""
    echo "Results at:"
    echo "  Eval:  $LOCAL_DATA_DIR/eval_output/$CHECKPOINT_NAME/"
    echo "  Viz:   $LOCAL_DATA_DIR/viz_output/$CHECKPOINT_NAME/"
    echo "  Excel: $LOCAL_DATA_DIR/octo_finetune_results.xlsx"
fi
