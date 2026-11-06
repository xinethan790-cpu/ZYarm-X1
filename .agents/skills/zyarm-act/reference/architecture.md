# 架構與數據流

（Step 6 補全；先記骨架）

## 數據流

```
原始採集 (Pick-place-object/.../zyarm_lerobot_dataset, 只讀)
  │  問題：視頻 pts 抖動 4~21ms（容差 0.1ms）+ 尾部欠幀 0.8~2s
  ▼
scripts/fix_dataset.py  （ffmpeg CFR30 重編碼 + tpad 尾幀補齊）
  ▼
data/zyarm_pick_place_v2/  （LeRobot v3.0，可被 LeRobotDataset 讀取）
  ▼
lerobot_train --policy.type=act  （LeRobot 0.6.1 @ lerobot-main/src，只讀）
  │  特徵自動推斷：observation.state(7) + overhead(3,480,640) → action(7)
  ▼
outputs/train/zyarm_act/checkpoints/last
  ▼
scripts/eval_act_dataset.py  （dataset 回放 + GT 動作對比）
  ▼
outputs/eval/  報告（為真機驗證建立基線）
```

## 關鍵約定

- ACT 默認超參：chunk_size=100、n_action_steps=100、resnet18（ImageNet 預訓練）、dim_model=512、VAE on、lr=1e-5
- 部署時 30Hz 下 chunk=100 = 3.33s 重新規劃；若真機節奏太長可部署期只執行 chunk 前 50 步
