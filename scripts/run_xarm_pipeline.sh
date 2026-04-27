#!/bin/bash
# run_xarm_pipeline.sh: Build Octo container, extract mcap→RLDS, and optionally finetune.
# run_xarm_pipeline.sh: End-to-end pipeline for xarm place-object-in-toolbox data.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
OCTO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="octo-finetune"
CONTAINER_NAME="octo-xarm-pipeline"

# Data paths (host)
BAG_DIR="/home/gokul/tqagi/3d_flowmatch_actor/data/xarm/place_object_in_toolbox"
RLDS_OUTPUT_DIR="${OCTO_DIR}/data/xarm_place_toolbox_rlds"
CHECKPOINT_DIR="${OCTO_DIR}/checkpoints/xarm_place_toolbox"

# Extraction params
DATASET_NAME="xarm_place_toolbox"
LANGUAGE_INSTRUCTION="place object in toolbox"
TARGET_HZ=2.0
SYNC_SLOP=0.05
VAL_RATIO=0.1

# Finetune params
PRETRAINED_MODEL="hf://rail-berkeley/octo-small-1.5"
FINETUNE_MODE="full,language_conditioned"
# ──────────────────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 [build|extract|finetune|all]"
    echo ""
    echo "  build    - Build the Docker image"
    echo "  extract  - Extract mcap bags to RLDS TFRecord"
    echo "  finetune - Run Octo finetuning on extracted data"
    echo "  all      - Run build + extract + finetune"
    exit 1
}

build() {
    echo "=== Building Docker image: ${IMAGE_NAME} ==="
    docker build -t "${IMAGE_NAME}" -f "${OCTO_DIR}/Dockerfile" "${OCTO_DIR}"
    echo "=== Build complete ==="
}

extract() {
    echo "=== Extracting mcap bags → RLDS TFRecords ==="
    echo "  Bag dir:    ${BAG_DIR}"
    echo "  Output dir: ${RLDS_OUTPUT_DIR}"
    echo "  Target Hz:  ${TARGET_HZ}"
    echo "  Sync slop:  ${SYNC_SLOP}s"

    mkdir -p "${RLDS_OUTPUT_DIR}"

    docker run --rm \
        --name "${CONTAINER_NAME}-extract" \
        -v "${BAG_DIR}:/data/bags:ro" \
        -v "${RLDS_OUTPUT_DIR}:/data/rlds_output" \
        -v "${OCTO_DIR}:/octo" \
        -e PYTHONUNBUFFERED=1 \
        "${IMAGE_NAME}" \
        python3 /octo/scripts/mcap_to_rlds.py \
            --bag_dir /data/bags \
            --output_dir /data/rlds_output \
            --dataset_name "${DATASET_NAME}" \
            --language_instruction "${LANGUAGE_INSTRUCTION}" \
            --target_hz "${TARGET_HZ}" \
            --sync_slop "${SYNC_SLOP}" \
            --val_ratio "${VAL_RATIO}"

    echo "=== Extraction complete ==="
    echo "  Dataset: ${RLDS_OUTPUT_DIR}/${DATASET_NAME}/1.0.0/"
}

finetune() {
    echo "=== Starting Octo finetuning ==="
    echo "  RLDS data:   ${RLDS_OUTPUT_DIR}"
    echo "  Pretrained:  ${PRETRAINED_MODEL}"
    echo "  Checkpoints: ${CHECKPOINT_DIR}"
    echo "  Mode:        ${FINETUNE_MODE}"

    mkdir -p "${CHECKPOINT_DIR}"

    docker run --rm --gpus all --shm-size 8g \
        --name "${CONTAINER_NAME}-finetune" \
        -v "${RLDS_OUTPUT_DIR}:/data/rlds:ro" \
        -v "${CHECKPOINT_DIR}:/checkpoints" \
        -v "${OCTO_DIR}:/octo" \
        -e PYTHONUNBUFFERED=1 \
        "${IMAGE_NAME}" \
        python3 /octo/scripts/finetune.py \
            --config=/octo/scripts/configs/finetune_xarm_config.py:"${FINETUNE_MODE}" \
            --config.pretrained_path="${PRETRAINED_MODEL}" \
            --config.dataset_kwargs.data_dir=/data/rlds \
            --config.save_dir=/checkpoints

    echo "=== Finetuning complete ==="
    echo "  Checkpoints at: ${CHECKPOINT_DIR}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
COMMAND="${1:-}"

case "${COMMAND}" in
    build)    build ;;
    extract)  extract ;;
    finetune) finetune ;;
    all)      build && extract && finetune ;;
    *)        usage ;;
esac
