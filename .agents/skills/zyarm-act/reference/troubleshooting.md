# Troubleshooting

（Step 6 按實際踩坑補全；先記已確認的两條）

## `FrameTimestampError: ... violate the tolerance`

- **根因**：原始採集視頻的幀 pts 相對於 k/30 理想網格抖動 4~21ms（實測 ep0：第 100 幀 +9ms、第 1000 幀 +9.6ms、第 2000 幀 +21ms），LeRobot pyav 解碼容差僅 0.0001s；且多數集視頻尾部比 manifest 長度短 0.8~2s。
- **解法**：一律使用修復後數據集 `data/zyarm_pick_place_v2/`（CFR30 重編碼 + 尾幀補齊）；不要直接指向原始 `Pick-place-object/...` 目錄。
- **驗證**：`tests/test_fixed_dataset.py`。

## `ModuleNotFoundError: No module named 'lerobot'`

- 用 embody 環境並 `export PYTHONPATH=/data/home/cloudsway/lerobot-main/src`；系統 python（3.14）無 torch/lerobot。

## 數據集隨機訪問很慢（~240ms/幀，GPU 飢餓）

- **根因**：LeRobot 0.6.1 的 pyav 解碼每次查詢重新 `av.open` + seek 到最近關鍵幀 + 向前解碼到目標。關鍵幀間隔（GOP）決定平均多解碼多少幀。x264 默認 GOP 250 → 平均 +125 幀 → ~240ms/次。
- **解法**：視頻重編碼時用 `-g 30`（GOP 30，~60ms/次）。已固化在 `scripts/fix_dataset.py`。若日後自建 dataset，編碼參數务必帶 `-g 30 -bf 0`。
- 訓練時可用 `--num_workers`（頂層字段）並行讀取進一步提速。

## `draccus DecodingError: The fields dtype are not valid for ACTConfig`

- `--policy.dtype` 只存在於 PI05Config/PI0Config（0.6.1），**ACTConfig 沒有**。ACT 一律 fp32 訓練，不傳該旗標。

## `ImportError: cannot import name 'ACTPolicy' from 'lerobot.policies'`

- 0.6.1 頂層只導出 `ACTConfig`，modeling class 懶加載。要載入權重：
  `from lerobot.policies.act.modeling_act import ACTPolicy`
  或走 factory：`from lerobot.policies import get_policy_class, make_policy`。

## `draccus DecodingError: The fields type are not valid for ACTConfig`（`ACTConfig.from_pretrained`）

- checkpoint 的 `config.json` 頂層有註冊表級 `"type": "act"` 字段，draccus 解到具體 class 直接報錯；**官方 `from_pretrained` 第一步也會炸**，只是官方推理走 CLI 配 `--policy.pretrained_path` 從不觸發。
- **解法**（照官方 `from_pretrained` 第二步行動）：`json.load` → `config.pop("type")` → 寫臨時 json → `draccus.parse(ACTConfig, tmp, args=[])`（包在 `draccus.config_type("json")` 內）。`scripts/eval_act_dataset.py` 已固化此路徑。

## 其它 0.6.1 API 坑

- `ds.episodes` 為 None → 用 `ds.meta.episodes`。
- `--keep_last` 不存在 → 正確字段名 `--keep_last_checkpoints`。
- `--num_workers` 是頂層字段，不是 `--dataset.num_workers`。
- x264 默認 B 幀讓 ffprobe 包序 ≠ 顯示序，pts 網格檢查會誤判 → 編碼加 `-bf 0`。
- Bash 工具的 600s 超时会殺長任務 → 長測試/訓練一律 `nohup … > log 2>&1 < /dev/null &` 脫離 + 日誌帶 `flush=True` 進度。
