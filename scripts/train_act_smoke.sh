#!/usr/bin/env bash
# Step 3：ACT 訓練冒煙（100 步）。
# 目的：驗證 數據集可讀、ACT 特徵自動推斷、resnet18 加載、loss 下降、checkpoint 完整。
# 用法：bash scripts/train_act_smoke.sh
set +e
cd "$(dirname "$0")/.." || exit 1

PY=/data/home/cloudsway/.conda/envs/embody/bin/python
export PYTHONPATH=/data/home/cloudsway/lerobot-main/src
export TMPDIR="$PWD/tmp"
mkdir -p tmp logs outputs/train

OUTPUT_DIR="$PWD/outputs/train/act_smoke"
LOG="logs/act_smoke_$(date +%Y%m%d_%H%M%S).log"

# 冒煙每次都全新跑
rm -rf "$OUTPUT_DIR"

$PY -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=zyarm_pick_place \
    --dataset.root="$PWD/data/zyarm_pick_place_v2" \
    --dataset.video_backend=pyav \
    --batch_size=8 \
    --steps=100 \
    --log_freq=20 \
    --output_dir="$OUTPUT_DIR" \
    --job_name=act_smoke \
    > "$LOG" 2>&1 &
TRAIN_PID=$!

echo "smoke training started, pid=$TRAIN_PID, log=$LOG"
tail -n +1 -f "$LOG" &
TAIL_PID=$!

wait $TRAIN_PID
RC=$?
kill $TAIL_PID 2>/dev/null

echo "training exited with code $RC"
if [ $RC -ne 0 ]; then
    echo "--- last 40 lines of log ---"
    tail -n 40 "$LOG"
    exit $RC
fi

# checkpoint 完整性（0.6.1 布局：last -> 000100/{pretrained_model,training_state}）
LAST="$OUTPUT_DIR/checkpoints/last"
MISSING=0
for f in \
    "$LAST/pretrained_model/model.safetensors" \
    "$LAST/pretrained_model/config.json" \
    "$LAST/pretrained_model/policy_preprocessor.json" \
    "$LAST/pretrained_model/policy_postprocessor.json" \
    "$LAST/pretrained_model/train_config.json" \
    "$LAST/training_state/training_step.json"
do
    if [ ! -f "$f" ]; then
        echo "[FAIL] missing $f"
        MISSING=1
    fi
done
if [ $MISSING -eq 0 ]; then
    echo "[PASS] checkpoint complete: $LAST"
    exit 0
else
    exit 1
fi
