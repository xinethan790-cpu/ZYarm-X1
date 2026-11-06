#!/usr/bin/env bash
# =============================================================================
# ZYarm-X1 ACT 正式訓練腳本（10 epochs，雙 L20 DDP）
# =============================================================================
# 使用方式：
#   bash scripts/train_act.sh
#
# 功能：
#   - ACT 從零訓練（resnet18 ImageNet 預訓練 backbone）
#   - 數據增強：亮度、對比度、飽和度、色相、銳利度、仿射（每幀隨機 3 種）
#   - 自動續訓：checkpoints/last 完整則續訓，否則清舊輸出全新訓練
#   - TensorBoard 事件檔：outputs/tensorboard/zyarm_act/
#
# 注意：
#   - ACTConfig 無 dtype 字段（fp32 訓練，模型小，L20 綽綽有餘）
#   - 正式訓練須待用戶明確指令才啟動
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- 路徑設定 ----
PYTHON=/data/home/cloudsway/.conda/envs/embody/bin/python
LEROBOT_SRC=/data/home/cloudsway/lerobot-main/src
DATASET_ROOT="$PROJECT_DIR/data/zyarm_pick_place_v2"
DATASET_REPO=zyarm_pick_place
OUTPUT_DIR="$PROJECT_DIR/outputs/train/zyarm_act"
RUN_NAME=zyarm_act
TB_LOG_DIR="$PROJECT_DIR/outputs/tensorboard/zyarm_act"
VIDEO_BACKEND=pyav

# ---- 訓練超參數（可用環境變數覆寫）----
BATCH_SIZE="${ACT_BATCH_SIZE:-32}"        # 每卡
NUM_GPUS="${ACT_NUM_GPUS:-2}"
EPOCHS="${ACT_EPOCHS:-10}"
SAVE_FREQ="${ACT_SAVE_FREQ:-5000}"
KEEP_LAST="${ACT_KEEP_LAST:-1000000}"     # 超大 = 全留
LOG_FREQ=200
NUM_WORKERS=4

# ---- 依資料集幀數換算 STEPS ----
[ -f "${DATASET_ROOT}/meta/info.json" ] || { echo "ERROR: 找不到 ${DATASET_ROOT}/meta/info.json"; exit 1; }
TOTAL_FRAMES=$(${PYTHON} -c "import json;print(json.load(open('${DATASET_ROOT}/meta/info.json'))['total_frames'])")
EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS))
STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))
STEPS="${ACT_STEPS:-$(( STEPS_PER_EPOCH * EPOCHS ))}"

# ---- 續訓偵測 ----
RESUME_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
RESUME_MODE=false
START_STEP=0
if [ -e "${RESUME_CONFIG}" ]; then
    RESUME_MODE=true
    START_STEP=$(${PYTHON} -c "import json;print(json.load(open('${OUTPUT_DIR}/checkpoints/last/training_state/training_step.json'))['step'])")
fi

# ---- 重複進程檢查 ----
EXISTING=$(ps aux | grep "[l]erobot_train" | wc -l)
if [ "${EXISTING}" -gt 0 ]; then
    echo "ERROR: 偵測到 ${EXISTING} 個 lerobot_train 進程，先確認是否重複（ps -o pid,ppid）："
    ps aux | grep "[l]erobot_train" | head
    exit 1
fi

# ---- 訓練日誌 ----
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "${LOG_DIR}" "$PROJECT_DIR/tmp"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH}"
export TMPDIR="$PROJECT_DIR/tmp"

# ---- 顯示設定 ----
echo "=========================================="
echo "  ZYarm-X1 ACT Training"
echo "=========================================="
echo "Python:   ${PYTHON}"
echo "Dataset:  ${DATASET_ROOT} (${TOTAL_FRAMES} frames)"
echo "Output:   ${OUTPUT_DIR}"
echo "Training: steps=${STEPS} (${EPOCHS} ep x ${STEPS_PER_EPOCH}) batch=${BATCH_SIZE}/GPU x ${NUM_GPUS} GPUs (eff=${EFFECTIVE_BATCH})"
echo "          save_freq=${SAVE_FREQ} keep_last=${KEEP_LAST} log_freq=${LOG_FREQ} workers=${NUM_WORKERS}"
echo "Augmentation: enable=true max=3 random_order=true"
if [ "${RESUME_MODE}" = "true" ]; then
    echo "Mode:    Resume（從 step ${START_STEP} 續訓）"
else
    echo "Mode:    Fresh（全新訓練）"
fi
echo "TensorBoard events: ${TB_LOG_DIR}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

set +e

COMMON_ARGS=(
    --dataset.repo_id="${DATASET_REPO}"
    --dataset.root="${DATASET_ROOT}"
    --output_dir="${OUTPUT_DIR}"
    --job_name="${RUN_NAME}"
    --policy.type=act
    --policy.device=cuda
    --policy.push_to_hub=false
    --dataset.video_backend="${VIDEO_BACKEND}"
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=3
    --dataset.image_transforms.random_order=true
    --num_workers="${NUM_WORKERS}"
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --save_freq="${SAVE_FREQ}"
    --keep_last_checkpoints="${KEEP_LAST}"
    --log_freq="${LOG_FREQ}"
)

if [ "${RESUME_MODE}" = "true" ]; then
    TRAIN_ARGS=(
        --resume=true
        --config_path="${RESUME_CONFIG}"
    )
else
    rm -rf "${OUTPUT_DIR}"
    TRAIN_ARGS=()
fi

${PYTHON} -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes "${NUM_GPUS}" \
    -m lerobot.scripts.lerobot_train \
    "${COMMON_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "training PID: ${TRAIN_PID}"

# 背景啟動 TensorBoard 監控（完全 detach，複用 lerobot-main 的 tb_monitor）
mkdir -p "${TB_LOG_DIR}"
nohup ${PYTHON} ${LEROBOT_SRC}/tb_monitor.py --log_file="${LOG_FILE}" --tb_log_dir="${TB_LOG_DIR}" \
    > "$PROJECT_DIR/tmp/tb_monitor_${RUN_NAME}.log" 2>&1 < /dev/null &
echo "tb_monitor PID: $! (events: ${TB_LOG_DIR})"

tail -f "${LOG_FILE}" &
TAIL_PID=$!
wait "${TRAIN_PID}" 2>/dev/null; TRAIN_EXIT=$?
kill "${TAIL_PID}" 2>/dev/null || true

echo ""
echo "✅ 訓練結束！exit code: ${TRAIN_EXIT}"
echo "   檢查點: ${OUTPUT_DIR}/checkpoints/"
exit ${TRAIN_EXIT}
