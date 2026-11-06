# Troubleshooting

（Step 6 按實際踩坑補全；先記已確認的两條）

## `FrameTimestampError: ... violate the tolerance`

- **根因**：原始採集視頻的幀 pts 相對於 k/30 理想網格抖動 4~21ms（實測 ep0：第 100 幀 +9ms、第 1000 幀 +9.6ms、第 2000 幀 +21ms），LeRobot pyav 解碼容差僅 0.0001s；且多數集視頻尾部比 manifest 長度短 0.8~2s。
- **解法**：一律使用修復後數據集 `data/zyarm_pick_place_v2/`（CFR30 重編碼 + 尾幀補齊）；不要直接指向原始 `Pick-place-object/...` 目錄。
- **驗證**：`tests/test_fixed_dataset.py`。

## `ModuleNotFoundError: No module named 'lerobot'`

- 用 embody 環境並 `export PYTHONPATH=/data/home/cloudsway/lerobot-main/src`；系統 python（3.14）無 torch/lerobot。
