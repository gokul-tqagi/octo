#!/bin/bash
# train.sh: Launch Octo finetuning inside Docker container.
# train.sh: Handles pip install of mounted code and passes config to finetune script.

set -euo pipefail

usage() {
    echo "Usage: $0 <config_yaml>"
    echo ""
    echo "  config_yaml  Path to finetune config (relative to octo repo root)"
    echo ""
    echo "Example:"
    echo "  $0 scripts/configs/finetune_marker_pick.yaml"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

CONFIG="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OCTO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-detect data dir: check sibling data/ or parent data/
if [ -d "$OCTO_DIR/../data" ]; then
    DATA_DIR="$(cd "$OCTO_DIR/../data" && pwd)"
elif [ -d "$OCTO_DIR/data" ]; then
    DATA_DIR="$OCTO_DIR/data"
else
    echo "Error: Cannot find data directory. Expected at $OCTO_DIR/../data or $OCTO_DIR/data"
    exit 1
fi

echo "=== Octo Training ==="
echo "  Config: $CONFIG"
echo "  Octo:   $OCTO_DIR"
echo "  Data:   $DATA_DIR"
echo ""

docker run --rm --gpus all --shm-size 8g -v "$OCTO_DIR":/octo -v "$DATA_DIR":/data -v ~/.cache:/root/.cache -e HF_HOME=/root/.cache/huggingface octo-finetune:latest bash -c "pip install -e /octo && python3 /octo/scripts/finetune_xarm.py --config /octo/$CONFIG"
