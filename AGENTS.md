# AGENTS.md — ZYarm-X1 ACT 訓練專案

## 專案概覽

用 ZYarm-X1 單臂真機搖操採集的 pick-and-place 數據（70 集 / 224,658 幀 / 7 自由度 / 單 overhead 相機 640×480 / 30fps）訓練 **ACT**（Action Chunking with Transformers）策略。本專案範圍 = 數據修復 → ACT 訓練 → 離線評估；真機部署待機器人接入後另立方案。

## 環境資訊

| 項目 | 值 |
|---|---|
| **Python** | `/data/home/cloudsway/.conda/envs/embody/bin/python`（3.10.20，主 conda `embody`，**不可用系統 python 3.14**） |
| **LeRobot** | 0.6.1 原始碼，**只讀引用** `/data/home/cloudsway/lerobot-main/src`（`export PYTHONPATH=/data/home/cloudsway/lerobot-main/src`），含該專案五處本地修補（本地 dataset 加載等），不可還原 |
| **PyTorch** | 2.7.1+cu128 |
| **GPU** | 2 × NVIDIA L20，各 46 GB |
| **ffmpeg** | `/usr/bin/ffmpeg`（視頻修復用） |
| **網路** | 可連外網；ACT resnet18 ImageNet 權重首次自動下載 ~45MB |

## 數據集

| 數據集 | 位置 | 狀態 |
|---|---|---|
| **原始採集**（只讀，絕不寫入） | `Pick-place-object/Pick-place-object/zyarm_lerobot_dataset/` | 70 集；視頻時間戳抖動 4~21ms + 尾部欠幀，**LeRobot 無法直接讀取** |
| **修復版**（訓練用） | `data/zyarm_pick_place_v2/` | 視頻重編碼為嚴格 CFR 30fps + 尾幀克隆補齊至 manifest 聲明幀數 + **GOP 30**（pyav 隨機訪問 ~60ms/次）+ 無 B 幀；meta/data 拷貝自原始 |

原始數據格式：LeRobot v3.0 佈局 + 自定義 `manifest.json`（`zyarm_lerobot_v1` schema，採集工具產物，LeRobot 不讀取）。特徵：`observation.state(7,)`、`action(7,)`（joint0~5 為弧度關節角、joint6 為夾爪 0~1）、`observation.images.overhead(3,480,640)` h264。

## 目錄結構

```
ZYarm-X1/
├── .agents/skills/zyarm-act/     # 專案 skill：SKILL.md + reference/（架構、troubleshooting）
├── Pick-place-object/            # 原始採集數據（只讀，gitignored）
├── data/zyarm_pick_place_v2/     # 修復後數據集（gitignored）
├── scripts/                      # 診斷/修復/訓練/評估腳本
├── tests/                        # 每步驗收測試（TDD）
├── docs/                         # 文檔與 plan 備份
├── logs/                         # 運行日誌（gitignored）
├── outputs/                      # 訓練/評估輸出（gitignored）
├── tmp/                          # 暫存（pip cache、TMPDIR，gitignored）
└── AGENTS.md                     # 本檔案
```

## 執行規則

1. **所有命令用 embody 環境**：`/data/home/cloudsway/.conda/envs/embody/bin/python`，並 `export PYTHONPATH=/data/home/cloudsway/lerobot-main/src`。
2. **項目自包含**：不向項目目錄外寫任何內容（日誌、暫存、pip cache 一律指到 `tmp/`/`logs/`）；`lerobot-main` 與原始數據目錄只讀。
3. **訓練腳本執行模式**（沿用 lerobot-main 紀律）：`training_cmd > logfile 2>&1 &` 背景執行 + `tail -f` 觀察；**禁用 `tee`/`set -o pipefail`**；監控程序（tb_monitor）用 `nohup … < /dev/null &` 完全 detach。
4. **執行前查重**：`ps aux | grep lerobot_train` 確認無重複進程（PPID 相同的為 DataLoader workers，正常）。
5. **TDD**：每個步驟先有 `tests/` 測試並通過，才進入下一步；每步完成即 git commit（繁體中文 message）。
6. **正式訓練須待用戶明確指令才啟動**（smoke 通過後停住等指令）。

## 當前狀態

- 2026-11-06：項目建立，環境冒煙通過中（見下方步驟狀態）。

| 步驟 | 內容 | 狀態 |
|---|---|---|
| 0 | 項目骨架 + 環境冒煙 | 進行中 |
| 1 | 數據診斷量化 | 待做 |
| 2 | 數據修復（CFR30 + 補幀） | 待做 |
| 3 | ACT 訓練冒煙（100 步） | 待做 |
| 4 | 正式訓練（10 epochs） | 待做 |
| 5 | 離線評估（dataset 回放） | 待做 |
| 6 | 收尾文檔 | 待做 |
