#!/usr/bin/env python
"""Step 1：原始數據集診斷（可複跑）。

對 manifest 中每一集統計：
- 視頻實際幀數（ffprobe 包計數）與 manifest 聲明長度 N
- 首幀 / 末幀 pts
- 各幀 pts 相對於 k/30 理想網格的偏差 |pts_k - k/30|：max / p95 / p50
- 尾部欠幀時長（N 大於實際幀數時）

報告寫入 logs/diagnose_<timestamp>.json，摘要打印到 stdout。

用法：
    /data/home/cloudsway/.conda/envs/embody/bin/python scripts/diagnose_dataset.py \
        [--root 原始數據集目錄]
"""
import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "Pick-place-object/Pick-place-object/zyarm_lerobot_dataset"
FPS = 30.0


def ffprobe_pts(video_path: Path) -> list[float]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time", "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    pts = []
    for line in out.strip().splitlines():
        line = line.strip()
        if line and line != "N/A":
            pts.append(float(line))
    return pts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def diagnose_episode(ep: dict, root: Path) -> dict:
    vidx = ep["video_file_index"]
    declared = ep["length"]
    video = root / "videos/observation.images.overhead/chunk-000" / f"file-{vidx:03d}.mp4"
    if not video.exists():
        return {"episode_index": ep["episode_index"], "video_file_index": vidx,
                "declared_length": declared, "error": "video missing"}
    pts = ffprobe_pts(video)
    n = len(pts)
    diffs = []
    for k in range(min(n, declared)):
        diffs.append(abs(pts[k] - k / FPS))
    return {
        "episode_index": ep["episode_index"],
        "video_file_index": vidx,
        "declared_length": declared,
        "video_frames": n,
        "frame_delta": n - declared,
        "tail_gap_s": round(max(0.0, (declared - n) / FPS), 3) if n < declared else 0.0,
        "first_pts": pts[0] if pts else None,
        "last_pts": pts[-1] if pts else None,
        "grid_diff_max_s": round(max(diffs), 4) if diffs else None,
        "grid_diff_p95_s": round(percentile(diffs, 0.95), 4) if diffs else None,
        "grid_diff_p50_s": round(percentile(diffs, 0.50), 4) if diffs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    manifest = json.loads((args.root / "manifest.json").read_text())
    results = [diagnose_episode(ep, args.root) for ep in manifest["episodes"]]

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    summary = {
        "dataset_root": str(args.root),
        "num_episodes": len(results),
        "episodes_with_errors": len(errors),
        "total_declared_frames": sum(r["declared_length"] for r in ok),
        "total_video_frames": sum(r["video_frames"] for r in ok),
        "frame_delta_min": min(r["frame_delta"] for r in ok),
        "frame_delta_max": max(r["frame_delta"] for r in ok),
        "grid_diff_max_overall_s": max(r["grid_diff_max_s"] for r in ok if r["grid_diff_max_s"]),
        "grid_diff_p95_overall_s": max(r["grid_diff_p95_s"] for r in ok if r["grid_diff_p95_s"]),
        "episodes_with_tail_gap": sum(1 for r in ok if r["tail_gap_s"] > 0),
        "max_tail_gap_s": max(r["tail_gap_s"] for r in ok),
        "median_tail_gap_s": statistics.median(r["tail_gap_s"] for r in ok),
    }
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "summary": summary, "episodes": results}

    out_dir = PROJECT_ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"diagnose_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"report: {out_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        print("episodes with errors:", errors, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
