# 2026-11-06 ACT 訓練方案（已批准執行）

> 備註：本檔為 plan mode 批准方案的存檔（know why）。執行記錄見 git log 與 AGENTS.md。

## 背景與目標

用 ZYarm-X1 真機搖操數據（70 集 / 224,658 幀 / 7 DoF / 單 overhead 相機 640×480 / 30fps，
`Pick-place-object/.../zyarm_lerobot_dataset`）訓練 **ACT** 策略。範圍 = 數據修復 + 訓練 + 離線評估；
真機部署待機器人接入後另立方案（用戶指示：先訓練，訓練完成後接機器人驗證）。

## 關鍵決策與理由

| 決策 | 理由 |
|---|---|
| 算法 = ACT（LeRobot 0.6.1 內建 `policies/act/`，ACT 原作者實現） | 用戶目標即復現 ACT；70 集小數據量下 ACT 最合適（訓練快、無需預訓練權重、部署簡單） |
| 復用 `lerobot-main/src`（只讀）+ `embody` 環境 + 雙 L20 | 已有完整 0.6.1 工作環境（含本地 dataset 加載修補），不重複建設 |
| 數據修復 = CFR 30fps 重編碼 + 尾幀克隆補齊 + **GOP 30** | 原始視頻 pts 抖動 4~69ms（pyav 容差 0.1ms）+ 尾部欠幀 → LeRobot 無法讀取；GOP 30 解決 pyav 隨機訪問慢的問題（見下） |
| 項目落位 `/data/home/cloudsway/ZYarm-X1/`，全自包含 | 用戶開發規範：不向項目外寫任何內容 |
| 正式訓練 fp32（不用 bf16） | 實測 `--policy.dtype` 不是 ACTConfig 字段（pi05 專屬），傳入直接報 DecodingError；ACT 模型小，fp32 無壓力且與原論文一致 |
| 輸入/輸出特徵不手配 | `dataset_to_policy_features` 從 dataset 自動推斷（observation.* → STATE/VISUAL，action → ACTION），實測可行 |

## 踩坑記錄（執行中發現，重要）

1. **FrameTimestampError**：原始視頻 pts 抖動 + 尾部欠幀 → 幾乎所有幀讀取失敗。解法：fix_dataset.py。
2. **pyav 隨機訪問成本 = f(關鍵幀間隔)**：LeRobot 0.6.1 的 pyav 解碼每次查詢都重新 `av.open` + seek 到最近關鍵幀 + 向前解碼。
   x264 默認 GOP 250 → 平均多解 125 幀 → **240ms/次**（10 epoch 估算 ~10h+ 且 GPU 飢餓）；
   **GOP 30 → ~60ms/次**。已將 `-g 30` 固化進 fix_dataset.py。
3. **B 幀導致包序 ≠ 顯示序**：ffprobe 包序索引檢查會誤判；`-bf 0` 讓解碼序 == 顯示序。
4. **`ds.episodes` 為 None**：0.6.1 用 `ds.meta.episodes`。
5. **`--policy.dtype` 對 ACT 無效**：只存在於 PI05Config/PI0Config，ACTConfig 沒有（draccus 直接報錯）。
6. **`--keep_last` 實際字段名是 `--keep_last_checkpoints`**；`--num_workers` 是頂層字段（非 `--dataset.num_workers`）。
7. **長測試/訓練必須 nohup 脫離**：Bash 工具的 600s 超时会殺掉前台轉背景的長任務；且 stdout 緩衝導致日誌延遲，測試腳本要帶 `flush=True` 進度輸出。

## 步驟（TDD）

0. 項目骨架 + 環境冒煙 → **完成**
1. 數據診斷量化（`scripts/diagnose_dataset.py`）→ **完成**（報告 logs/diagnose_20261106_160908.json）
2. 數據修復（`scripts/fix_dataset.py`，CFR30 + 補幀 + GOP30）+ 驗收（`tests/test_fixed_dataset.py`）
3. ACT 冒煙 100 步（`scripts/train_act_smoke.sh`）
4. 正式訓練 10 epochs（`scripts/train_act.sh`，雙卡 DDP batch 32/卡，自動續訓；**等用戶指令啟動**）
5. 離線評估（`scripts/eval_act_dataset.py`：dataset 回放 + GT 對比）
6. 收尾（AGENTS.md / skill / commit）

## 超參預設

- ACT 默認：chunk_size=100、n_action_steps=100、resnet18（ImageNet 預訓練，~45MB 首次下載）、
  dim_model=512、4 enc/1 dec 層、VAE on（latent 32）、lr=1e-5、dropout 0.1、kl_weight 10
- 訓練：batch 32/卡 × 2 卡（總 64）、10 epochs ≈ 35,100 步、save_freq 5000、num_workers 4（可調大）
- 數據增強：亮度/對比度/飽和度/色相/銳利度/仿射，每幀隨機 3 種（沿用 lerobot-main 慣例）
