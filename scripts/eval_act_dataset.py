#!/usr/bin/env python
"""Step 5：離線評估 —— dataset 回放 + GT 動作對比。

對每集從頭回放：逐幀把 (image, state) 送入 policy（經 preprocessor 前處理，
與部署路徑一致），用 select_action 取單步動作（chunk 隊列由 policy 內部管理，
與真機執行方式相同），與 GT action 對比。

指標：
- 每關節（joint0~5）與夾爪（joint6）分開統計 MAE
- 整體 MAE / 平均歐氏距離
- 夾爪 Pearson 相關係數與二值化（>0.5）一致率

報告寫入 outputs/eval/<run>/metrics.json（逐集 + 汇总）。

用法（必須用 embody 環境 + lerobot PYTHONPATH）：
    PYTHONPATH=/data/home/cloudsway/lerobot-main/src \
        /data/home/cloudsway/.conda/envs/embody/bin/python scripts/eval_act_dataset.py \
        --checkpoint outputs/train/zyarm_act/checkpoints/last
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "outputs/train/zyarm_act/checkpoints/last")
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data/zyarm_pick_place_v2")
    parser.add_argument("--episodes", type=str, default=None,
                        help="只評估指定集（逗號分隔），如 '0,1,2'")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import draccus
    import tempfile

    from lerobot.policies import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets import LeRobotDataset

    model_dir = args.checkpoint / "pretrained_model"
    if not model_dir.exists():
        print(f"ERROR: 找不到 {model_dir}（checkpoint 路徑應指向 checkpoints/last 一級）")
        sys.exit(1)

    print(f"loading policy from {model_dir} ...")
    policy = ACTPolicy.from_pretrained(model_dir)
    # 0.6.1 的 ACTConfig.from_pretrained 会被 config.json 頂層 "type" 字段卡住
    # (draccus DecodingError)；官方 from_pretrained 也是先 pop 再 parse，照搬。
    with open(model_dir / "config.json") as f:
        _raw = json.load(f)
    _raw.pop("type")
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
        json.dump(_raw, f)
        _tmp_cfg = f.name
    with draccus.config_type("json"):
        policy_cfg = draccus.parse(ACTConfig, _tmp_cfg, args=[])
    policy_cfg.device = "cuda"
    policy.to("cuda").eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_dir,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    input_keys = [k for k in policy_cfg.input_features]
    print(f"policy loaded. input keys: {input_keys}, chunk_size={policy_cfg.chunk_size}")

    ds = LeRobotDataset(repo_id="zyarm_pick_place", root=args.dataset, video_backend="pyav")
    rows = list(ds.meta.episodes)
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        rows = [r for r in rows if r["episode_index"] in wanted]

    out_dir = args.out or PROJECT_ROOT / "outputs/eval" / f"zyarm_act_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    action_dim = policy_cfg.output_features["action"].shape[0]
    assert action_dim == 7, f"expected 7-dim action, got {action_dim}"
    per_episode = []
    t0 = time.time()

    with torch.no_grad():
        for r in rows:
            ep_idx = r["episode_index"]
            lo, hi = r["dataset_from_index"], r["dataset_to_index"]
            policy.reset()
            pred_joints = np.zeros((0, action_dim))
            gt_joints = np.zeros((0, action_dim))
            dists = []
            for t in range(lo, hi):
                sample = ds[t]
                obs = {k: sample[k] for k in input_keys}
                batch = preprocessor(obs)
                pred = policy.select_action(batch).cpu().numpy().reshape(1, -1)
                gt = sample["action"].cpu().numpy().reshape(1, -1)
                pred_joints = np.concatenate([pred_joints, pred])
                gt_joints = np.concatenate([gt_joints, gt])
                dists.append(float(np.linalg.norm(pred - gt)))
            n = hi - lo
            mae = np.abs(pred_joints - gt_joints)  # (n, 7)
            row = {
                "episode_index": ep_idx,
                "frames": n,
                "mae_per_joint": [round(float(mae[:, j].mean()), 5) for j in range(action_dim)],
                "mae_gripper": round(float(mae[:, 6].mean()), 5),
                "mae_arm": round(float(mae[:, :6].mean()), 5),
                "mean_euclid_dist": round(float(np.mean(dists)), 5),
                "gripper_pearson": round(pearson(gt_joints[:, 6], pred_joints[:, 6]), 4),
                "gripper_binary_agree": round(float(np.mean((gt_joints[:, 6] > 0.5) == (pred_joints[:, 6] > 0.5))), 4),
            }
            per_episode.append(row)
            print(f"  ep{ep_idx}: {n} frames, mae_arm={row['mae_arm']}, "
                  f"mae_gripper={row['mae_gripper']}, agree={row['gripper_binary_agree']} "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    mae_arm = float(np.mean([r["mae_arm"] for r in per_episode]))
    mae_grip = float(np.mean([r["mae_gripper"] for r in per_episode]))
    summary = {
        "checkpoint": str(args.checkpoint),
        "num_episodes": len(per_episode),
        "total_frames": sum(r["frames"] for r in per_episode),
        "mean_mae_arm": round(mae_arm, 5),
        "mean_mae_gripper": round(mae_grip, 5),
        "mean_gripper_pearson": round(float(np.nanmean([r["gripper_pearson"] for r in per_episode])), 4),
        "mean_gripper_binary_agree": round(float(np.mean([r["gripper_binary_agree"] for r in per_episode])), 4),
        "mean_euclid_dist": round(float(np.mean([r["mean_euclid_dist"] for r in per_episode])), 5),
        "elapsed_s": round(time.time() - t0, 1),
    }
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "summary": summary, "episodes": per_episode}
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nreport: {out_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
