#!/bin/bash
# deploy_training.sh: Sync training data, Docker image, and code to a remote server.
# deploy_training.sh: Handles rsync of RLDS tfrecords, Docker image export, and octo codebase.

set -euo pipefail

usage() {
    echo "Usage: $0 <server> <dest_dir>"
    echo ""
    echo "  server    SSH config host name (from your ~/.ssh/config)"
    echo "  dest_dir  Remote base directory (e.g. /home/ubuntu/torqueagi)"
    echo ""
    echo "Example:"
    echo "  $0 aws-L4-server1 /home/ubuntu/torqueagi"
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

SERVER="$1"
DEST_DIR="$2"

OCTO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RLDS_DIR="/home/gokul/data/rlds_output"
DOCKER_IMAGE="octo-finetune:latest"
DOCKER_TAR="/tmp/octo-finetune.tar"

echo "=== Deploy training to ${SERVER}:${DEST_DIR} ==="
echo "  Octo code:   ${OCTO_DIR}"
echo "  RLDS data:   ${LOCAL_RLDS_DIR}"
echo "  Docker image: ${DOCKER_IMAGE}"
echo ""

# --- 1. Sync RLDS TFRecords ---
echo "[1/4] Syncing RLDS tfrecords..."
ssh "${SERVER}" "mkdir -p ${DEST_DIR}/data/rlds_output"
rsync -avz --progress \
    "${LOCAL_RLDS_DIR}/" \
    "${SERVER}:${DEST_DIR}/data/rlds_output/"
echo "  Done."

# --- 2. Sync Octo codebase ---
echo "[2/4] Syncing octo codebase..."
ssh "${SERVER}" "mkdir -p ${DEST_DIR}/octo"
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints/' \
    --exclude='/data/' \
    --exclude='.eggs/' \
    --exclude='*.egg-info/' \
    "${OCTO_DIR}/" \
    "${SERVER}:${DEST_DIR}/octo/"
echo "  Done."

# --- 3. Export and transfer Docker image ---
echo "[3/4] Exporting Docker image (this may take a few minutes)..."
docker save "${DOCKER_IMAGE}" -o "${DOCKER_TAR}"
echo "  Transferring to ${SERVER}..."
rsync -avz --progress "${DOCKER_TAR}" "${SERVER}:${DEST_DIR}/"
echo "  Loading image on remote..."
ssh "${SERVER}" "docker load -i ${DEST_DIR}/octo-finetune.tar && rm ${DEST_DIR}/octo-finetune.tar"
echo "  Done."

# --- 4. Print remote training command ---
echo ""
echo "[4/4] Deploy complete. Run training on ${SERVER} with:"
echo ""
echo "  ssh ${SERVER}"
echo ""
echo "  docker run --rm --gpus all --shm-size 8g \\"
echo "    -v ${DEST_DIR}/octo:/octo \\"
echo "    -v ${DEST_DIR}/data:/data \\"
echo "    -v ~/.cache:/root/.cache \\"
echo "    -e HF_HOME=/root/.cache/huggingface \\"
echo "    ${DOCKER_IMAGE} \\"
echo "    python3 /octo/scripts/finetune_xarm.py \\"
echo "      --config /octo/scripts/configs/finetune_marker_pick.yaml"
echo ""
