# ZYarm-X1 ACT 训练项目

用 ZYarm-X1 单臂真机遥操采集的 pick-and-place 数据训练 **ACT**（Action Chunking with Transformers）策略。
项目范围：**数据修复 → ACT 训练 → 离线评估**；真机部署待机器人接入后另立方案。

## 项目进度（截至 2026-11-16）

| 步骤 | 内容 | 状态 |
|---|---|---|
| 0 | 项目骨架 + 环境冒烟 | ✅ 完成 |
| 1 | 数据诊断量化 | ✅ 完成（70 集：帧数差 -63~+347、时间戳抖动 p95 59ms / max 69ms、41 集尾部欠帧中位 0.65s；报告 `logs/diagnose_20261106_160908.json`） |
| 2 | 数据修复（CFR30 重编码 + GOP30 + 尾帧克隆补帧） | ✅ 完成（全量深扫 224,658 帧逐帧解码通过，修复版在 `data/zyarm_pick_place_v2/`） |
| 3 | ACT 训练冒烟（100 步，单卡） | ✅ 完成（loss 26.4 → 4.3，checkpoint 6 文件齐全） |
| 4 | 正式训练（20 epochs ≈ 70,220 步，双卡 DDP，支持断点续训） | ⏸ **搁置待启动**：`scripts/train_act.sh` 就绪；2026-11-06 定时启动因双卡被 vLLM 推理服务占用而搁置，截至 2026-11-16 GPU 仍被占用，需待显存释放后执行 `ACT_EPOCHS=20 bash scripts/train_act.sh` |
| 5 | 离线评估（dataset 回放，`scripts/eval_act_dataset.py`） | ✅ 脚本就绪 + 单集试跑通过；训练完成后跑全量 70 集 |
| 6 | 收尾文档 | 进行中（skill troubleshooting 已补全 LeRobot 0.6.1 API 坑） |

## 数据

| 数据集 | 位置 | 说明 |
|---|---|---|
| 原始采集（只读，gitignored） | `Pick-place-object/Pick-place-object/zyarm_lerobot_dataset/` | 70 集 / 224,658 帧 / 30fps；视频时间戳抖动 4~21ms + 尾部欠帧，LeRobot 无法直接读取 |
| 修复版（训练用，gitignored） | `data/zyarm_pick_place_v2/` | 严格 CFR 30fps + 尾帧补齐 + GOP 30（pyav 随机访问 ~60ms/次）+ 无 B 帧 |

数据格式：LeRobot v3.0 布局。特征 `observation.state(7,)`、`action(7,)`（joint0~5 弧度关节角、joint6 夹爪 0~1）、`observation.images.overhead(3,480,640)` h264。

## 环境要求

| 项目 | 值 |
|---|---|
| Python | 3.10（conda env `embody`，**不要用系统 python 3.14**） |
| LeRobot | 0.6.1 源码（只读引用，含本项目五处本地修补，`export PYTHONPATH=.../lerobot-main/src`） |
| PyTorch | 2.7.1+cu128 |
| GPU | 2 × NVIDIA L20（46GB） |
| ffmpeg | `/usr/bin/ffmpeg`（视频修复用） |

## 目录结构

```
ZYarm-X1/
├── .agents/skills/zyarm-act/     # 项目 skill：SKILL.md + reference/（架构、troubleshooting）
├── Pick-place-object/            # 原始采集数据（只读，gitignored）
├── data/zyarm_pick_place_v2/     # 修复后数据集（gitignored）
├── scripts/                      # 诊断 / 修复 / 训练 / 评估脚本
├── tests/                        # 每步验收测试（TDD）
├── docs/                         # 文档与 plan 备份
├── logs/                         # 运行日志（gitignored）
├── outputs/                      # 训练/评估输出（gitignored）
├── tmp/                          # 暂存（pip cache、TMPDIR，gitignored）
├── AGENTS.md                     # 项目执行规则与状态（供 AI agent 使用）
└── README.md                     # 本文件
```

## 脚本一览

| 脚本 | 用途 |
|---|---|
| `scripts/diagnose_dataset.py` | 数据诊断：逐集核对声明帧数 vs 实际视频帧数、时间戳抖动统计 |
| `scripts/fix_dataset.py` | 数据修复：视频重编码 CFR30 + 尾帧克隆补帧（`-g 30` 指定 GOP） |
| `scripts/train_act_smoke.sh` | ACT 冒烟训练（100 步，验证环境与脚本） |
| `scripts/train_act.sh` | 正式训练：双卡 DDP，支持断点续训，`ACT_EPOCHS=20` 控制 epoch 数 |
| `scripts/eval_act_dataset.py` | 离线评估：dataset 回放，计算动作误差指标 |

## 备注

- 训练/数据/日志/输出目录均 gitignored，本仓库只含代码、脚本、测试与文档。
- 启动训练前必须 `nvidia-smi` 确认 GPU 空闲；本机存在 root 级 vLLM 推理服务，显存冲突时先报告负责人，不得擅自 kill。
- 详细执行规则见 `AGENTS.md`，踩坑记录见 `.agents/skills/zyarm-act/reference/troubleshooting.md`。
