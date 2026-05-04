#!/bin/bash
# train.sh: End-to-end remote training — sync data/code to server, launch training in screen.
# train.sh: Also supports local mode when no server is specified.

set -euo pipefail

OCTO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RLDS_DIR="/home/gokul/data/rlds_output"
DOCKER_IMAGE="octo-finetune:latest"

usage() {
    echo "Usage: $0 <finetune_config> <server> <dest_dir>"
    echo "       $0 <finetune_config> local"
    echo ""
    echo "  finetune_config  Finetune YAML config (relative to octo repo root)"
    echo "  server           SSH config host name (e.g. aws-L4-server1)"
    echo "  dest_dir         Remote base directory (e.g. /home/ubuntu/torqueagi)"
    echo ""
    echo "Examples:"
    echo "  $0 scripts/configs/finetune_marker_pick_10hz.yaml aws-L4-server1 /home/ubuntu/torqueagi"
    echo "  $0 scripts/configs/finetune_lipbalm_10hz.yaml local"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

CONFIG="$1"
MODE="$2"

if [ ! -f "$OCTO_DIR/$CONFIG" ]; then
    echo "Error: Config not found: $OCTO_DIR/$CONFIG"
    exit 1
fi

# Extract dataset_name from config for screen session naming
DATASET_NAME=$(grep 'dataset_name:' "$OCTO_DIR/$CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
SESSION_NAME="train_${DATASET_NAME}"

if [ "$MODE" = "local" ]; then
    # --- Local mode ---
    DATA_DIR="$LOCAL_RLDS_DIR/.."
    if [ -d "$OCTO_DIR/../data" ]; then
        DATA_DIR="$(cd "$OCTO_DIR/../data" && pwd)"
    fi

    echo "=== Local Training ==="
    echo "  Config:  $CONFIG"
    echo "  Dataset: $DATASET_NAME"
    echo "  Data:    $DATA_DIR"
    echo ""

    docker run --rm --gpus all --shm-size 8g -v "$OCTO_DIR":/octo -v "$DATA_DIR":/data -v ~/.cache:/root/.cache -e HF_HOME=/root/.cache/huggingface "$DOCKER_IMAGE" bash -c "pip install -e /octo && python3 /octo/scripts/finetune_xarm.py --config /octo/$CONFIG"
else
    # --- Remote mode ---
    if [ $# -ne 3 ]; then
        usage
    fi
    SERVER="$2"
    DEST_DIR="$3"

    echo "=== Remote Training ==="
    echo "  Config:  $CONFIG"
    echo "  Dataset: $DATASET_NAME"
    echo "  Server:  $SERVER:$DEST_DIR"
    echo ""

    # 1. Sync RLDS data
    echo "[1/3] Syncing RLDS data..."
    ssh "$SERVER" "mkdir -p $DEST_DIR/data/rlds_output"
    rsync -avz "$LOCAL_RLDS_DIR/" "$SERVER:$DEST_DIR/data/rlds_output/"

    # 2. Sync code
    echo "[2/3] Syncing octo codebase..."
    ssh "$SERVER" "mkdir -p $DEST_DIR/octo"
    rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='checkpoints/' --exclude='/data/' --exclude='.eggs/' --exclude='*.egg-info/' "$OCTO_DIR/" "$SERVER:$DEST_DIR/octo/"

    # 3. Launch training in screen
    echo "[3/3] Launching training in screen session '$SESSION_NAME'..."
    ssh "$SERVER" "sudo mkdir -p /run/screen && sudo chmod 777 /run/screen 2>/dev/null; cd $DEST_DIR/octo && screen -dmS $SESSION_NAME bash -c 'docker run --rm --gpus all --shm-size 8g -v $DEST_DIR/octo:/octo -v $DEST_DIR/data:/data -v ~/.cache:/root/.cache -e HF_HOME=/root/.cache/huggingface $DOCKER_IMAGE bash -c \"pip install -e /octo && python3 /octo/scripts/finetune_xarm.py --config /octo/$CONFIG\"'"

    sleep 3
    ssh "$SERVER" "screen -ls; docker ps --format '{{.Names}} {{.Image}} {{.Status}}'"

    echo ""
    echo "Training launched. Monitor with:"
    echo "  ssh $SERVER 'screen -r $SESSION_NAME'"
    echo ""
    echo "When done, run eval with:"
    echo "  bash scripts/eval.sh $CONFIG $SERVER $DEST_DIR"
fi
