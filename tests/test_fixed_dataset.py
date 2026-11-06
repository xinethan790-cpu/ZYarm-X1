#!/usr/bin/env python
"""Step 2 驗收：修復後數據集測試。

三項檢查：
1. 70 個視頻幀數 == manifest 聲明長度 N（ffprobe 包計數）
2. 每集抽 5 個位置（0/100/中點/N-10/N-1）的 pts 與 k/30 網格誤差 < 1ms
3. LeRobotDataset 加載成功，且「逐集最後一幀 + 每 10 幀抽樣」全部解碼無 FrameTimestampError
   （--full 跑全量 224,658 幀，耗時較長）

用法（必須用 embody 環境 + lerobot PYTHONPATH）：
    PYTHONPATH=/data/home/cloudsway/lerobot-main/src \
        /data/home/cloudsway/.conda/envs/embody/bin/python tests/test_fixed_dataset.py [--full]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data/zyarm_pick_place_v2"
FPS = 30.0
PTS_TOL_S = 0.001


def ffprobe_pts(video: Path) -> list[float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return sorted(float(x) for x in out.strip().splitlines() if x and x != "N/A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--full", action="store_true", help="全量解碼所有幀（耗時）")
    parser.add_argument("--sample-step", type=int, default=10,
                        help="抽樣步長（每 N 幀取一幀），默認 10")
    args = parser.parse_args()

    manifest = json.loads((args.root / "manifest.json").read_text())
    episodes = manifest["episodes"]
    failures: list[str] = []

    # --- 檢查 1+2：幀數與 pts 網格 ---
    for ep in episodes:
        i, n = ep["video_file_index"], ep["length"]
        video = args.root / "videos/observation.images.overhead/chunk-000" / f"file-{i:03d}.mp4"
        if not video.exists():
            failures.append(f"ep{i}: missing {video.name}")
            continue
        pts = ffprobe_pts(video)
        if len(pts) != n:
            failures.append(f"ep{i}: frame count {len(pts)} != {n}")
            continue
        for k in {0, min(100, n - 1), n // 2, n - 10, n - 1}:
            err = abs(pts[k] - k / FPS)
            if err > PTS_TOL_S:
                failures.append(f"ep{i}: frame {k} pts err {err*1000:.2f}ms > 1ms")

    # --- 檢查 3：LeRobot 解碼 ---
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset(repo_id="zyarm_pick_place", root=args.root, video_backend="pyav")
    eps = ds.meta.episodes
    total = ds.num_frames
    if args.full:
        indices = list(range(total))
        mode = "FULL"
    else:
        indices = set()
        for row in eps:
            indices.add(row["dataset_to_index"] - 1)  # 逐集最後一幀
        indices.update(range(0, total, args.sample_step))
        indices = sorted(indices)
        mode = "sampled (last-of-episode + every 10)"

    print(f"dataset: {total} frames, checking {len(indices)} frames ({mode}) ...")
    checked = 0
    for idx in indices:
        try:
            s = ds[idx]
            img = s["observation.images.overhead"]
            assert img.shape == (3, 480, 640), f"bad image shape {img.shape}"
            checked += 1
            if checked % 2000 == 0:
                print(f"  ... {checked}/{len(indices)} frames OK", flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append(f"frame {idx}: {type(e).__name__}: {str(e)[:100]}")
            if len(failures) > 30:
                print("too many failures, stopping", file=sys.stderr)
                break

    if failures:
        print(f"[FAIL] {len(failures)} failure(s):")
        for f in failures[:30]:
            print("  ", f)
        sys.exit(1)
    print(f"[PASS] {len(eps)} episodes frame-count+pts grid OK; {checked} frames decoded OK")


if __name__ == "__main__":
    main()
