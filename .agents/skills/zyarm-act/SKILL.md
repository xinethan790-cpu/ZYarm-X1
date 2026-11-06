# zyarm-act skill — ZYarm-X1 ACT 訓練專案手冊

專案：用 ZYarm-X1 真機搖操數據（70 集 pick-and-place）訓練 ACT 策略。範圍：數據修復 → 訓練 → 離線評估。

## 快速入口

```bash
cd /data/home/cloudsway/ZYarm-X1
PY=/data/home/cloudsway/.conda/envs/embody/bin/python
export PYTHONPATH=/data/home/cloudsway/lerobot-main/src

# 環境冒煙
$PY tests/test_env.py

# 數據診斷
$PY scripts/diagnose_dataset.py

# 數據修復（先生成 data/zyarm_pick_place_v2/）
$PY scripts/fix_dataset.py [--episodes 0]   # 單集 PoC 後再去掉 --episodes 全量

# 修復數據集驗收
$PY tests/test_fixed_dataset.py [--full]

# 訓練冒煙 / 正式訓練
bash scripts/train_act_smoke.sh
bash scripts/train_act.sh          # 須用戶明確指令

# 離線評估
$PY scripts/eval_act_dataset.py --checkpoint outputs/train/zyarm_act/checkpoints/last
```

## 詳細信息

- 架構與數據流：`reference/architecture.md`
- 已知問題與排查：`reference/troubleshooting.md`
- 專案總覽與當前狀態：項目根 `AGENTS.md`
