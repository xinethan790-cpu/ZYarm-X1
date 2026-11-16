# AGENTS.md — ZYarm-X1 ACT 训练项目

## 项目概览

用 ZYarm-X1 单臂真机遥操采集的 pick-and-place 数据（70 集 / 224,658 帧 / 7 自由度 / 单 overhead 相机 640×480 / 30fps）训练 **ACT**（Action Chunking with Transformers）策略。本项目范围 = 数据修复 → ACT 训练 → 离线评估；真机部署待机器人接入后另立方案。

## 环境信息

| 项目 | 值 |
|---|---|
| **Python** | `/data/home/cloudsway/.conda/envs/embody/bin/python`（3.10.20，主 conda `embody`，**不可用系统 python 3.14**） |
| **LeRobot** | 0.6.1 源码，**只读引用** `/data/home/cloudsway/lerobot-main/src`（`export PYTHONPATH=/data/home/cloudsway/lerobot-main/src`），含该项目五处本地修补（本地 dataset 加载等），不可还原 |
| **PyTorch** | 2.7.1+cu128 |
| **GPU** | 2 × NVIDIA L20，各 46 GB |
| **ffmpeg** | `/usr/bin/ffmpeg`（视频修复用） |
| **网络** | 可连外网；ACT resnet18 ImageNet 权重首次自动下载 ~45MB |
| **GitHub** | `xinethan790-cpu/ZYarm-X1`（公开，origin=SSH `git@github.com`）；2026-11-16 首次推送，**含原始采集数据（~2.4GB）**，仓库总大小 ≈2.5GB（5GB 软限的一半），勿再往仓库加大文件。注意：本机 github.com HTTPS 间歇性断连，**推送一律走 SSH**（稳定）；api.github.com 稳定可用 |

## 数据集

| 数据集 | 位置 | 状态 |
|---|---|---|
| **原始采集**（只读，绝不写入，已入库） | `Pick-place-object/Pick-place-object/zyarm_lerobot_dataset/` | 70 集 / ~2.4GB；视频时间戳抖动 4~21ms + 尾部欠帧，**LeRobot 无法直接读取** |
| **修复版**（训练用） | `data/zyarm_pick_place_v2/` | 视频重编码为严格 CFR 30fps + 尾帧克隆补齐至 manifest 声明帧数 + **GOP 30**（pyav 随机访问 ~60ms/次）+ 无 B 帧；meta/data 拷贝自原始；2026-11-06 全量深扫通过（224,658 帧逐帧解码 OK） |

原始数据格式：LeRobot v3.0 布局 + 自定义 `manifest.json`（`zyarm_lerobot_v1` schema，采集工具产物，LeRobot 不读取）。特征：`observation.state(7,)`、`action(7,)`（joint0~5 为弧度关节角、joint6 为夹爪 0~1）、`observation.images.overhead(3,480,640)` h264。

## 目录结构

```
ZYarm-X1/
├── .agents/skills/zyarm-act/     # 项目 skill：SKILL.md + reference/（架构、troubleshooting）
├── Pick-place-object/            # 原始采集数据（只读，已入库）
├── data/zyarm_pick_place_v2/     # 修复后数据集（gitignored）
├── scripts/                      # 诊断/修复/训练/评估脚本
├── tests/                        # 每步验收测试（TDD）
├── docs/                         # 文档与 plan 备份
├── logs/                         # 运行日志（gitignored）
├── outputs/                      # 训练/评估输出（gitignored）
├── tmp/                          # 暂存（pip cache、TMPDIR，gitignored）
└── AGENTS.md                     # 本文件
```

## 执行规则

1. **所有命令用 embody 环境**：`/data/home/cloudsway/.conda/envs/embody/bin/python`，并 `export PYTHONPATH=/data/home/cloudsway/lerobot-main/src`。
2. **项目自包含**：不向项目目录外写任何内容（日志、暂存、pip cache 一律指到 `tmp/`/`logs/`）；`lerobot-main` 与原始数据目录只读。
3. **训练脚本执行模式**（沿用 lerobot-main 纪律）：`training_cmd > logfile 2>&1 &` 后台执行 + `tail -f` 观察；**禁用 `tee`/`set -o pipefail`**；监控程序（tb_monitor）用 `nohup … < /dev/null &` 完全 detach。
4. **执行前查重**：`ps aux | grep lerobot_train` 确认无重复进程（PPID 相同的为 DataLoader workers，正常）。
5. **TDD**：每个步骤先有 `tests/` 测试并通过，才进入下一步；每步完成即 git commit（繁体中文 message）。
6. **正式训练须待用户明确指令才启动**（smoke 通过后停住等指令）。

## 当前状态（2026-11-09 更新）

| 步骤 | 内容 | 状态 |
|---|---|---|
| 0 | 项目骨架 + 环境冒烟 | ✅ 完成 |
| 1 | 数据诊断量化 | ✅ 完成（报告 `logs/diagnose_20261106_160908.json`） |
| 2 | 数据修复（CFR30 + GOP30 + 补帧） | ✅ 完成（全量深扫 224,658 帧通过） |
| 3 | ACT 训练冒烟（100 步） | ✅ 完成（loss 26.4→4.3，checkpoint 6 文件齐全） |
| 4 | 正式训练（20 epochs ≈ 70,220 步，双卡 DDP） | ⏸ 脚本就绪，**待启动**：2026-11-06 19:30 定时启动因双卡被 vLLM Qwen38-27B 推理服务占用而搁置，一次性 cron 已失效；用户指令后手动启动 `ACT_EPOCHS=20 bash scripts/train_act.sh` |
| 5 | 离线评估（dataset 回放） | ✅ 脚本就绪 + 单集试跑通过（`scripts/eval_act_dataset.py`）；训练完成后跑全量 70 集 |
| 6 | 收尾文档 | 进行中（skill troubleshooting 已补全 0.6.1 API 坑） |

**GPU 冲突教训（2026-11-06）**：启动前必须 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 确认无其他进程占显存。本机存在 root 级 vLLM 推理服务（`vllm serve`，TP=2，每卡 92% 显存），若占用则训练必 OOM；发现冲突**先报告用户，不得擅自 kill**。
