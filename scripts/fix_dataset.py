#!/usr/bin/env python
"""Step 2：數據修復 —— 視頻重編碼為嚴格 CFR 30fps + 尾幀克隆補齊。

背景：原始採集視頻的幀 pts 相對於 k/30 網格抖動（max 69ms），且部分集
尾部欠幀（最多 2.1s）或超幀（最多 +347 幀）。LeRobot pyav 解碼容差 0.1ms，
直接讀取會拋 FrameTimestampError。本腳本輸出嚴格 CFR 30fps 視頻：

- 輸出幀數 == manifest 聲明長度 N（不足用最後一幀克隆補齊，多餘截斷）
- 第 k 幀 pts 精確等於 k/30

原始數據只讀，產物寫入 --out（默認 data/zyarm_pick_place_v2/）。
meta/、data/、manifest.json 原樣拷貝（幀數/長度與補齊後視頻一致，無需改動）。

用法：
    # 單集 PoC（ep0）
    /data/home/cloudsway/.conda/envs/embody/bin/python scripts/fix_dataset.py --episodes 0
    # 全量（支持中斷重跑：已達標的視頻自動跳過）
    /data/home/cloudsway/.conda/envs/embody/bin/python scripts/fix_dataset.py
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = PROJECT_ROOT / "Pick-place-object/Pick-place-object/zyarm_lerobot_dataset"
DEFAULT_OUT = PROJECT_ROOT / "data/zyarm_pick_place_v2"
FPS = 30


def frame_count(video: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    return int(json.loads(r.stdout)["streams"][0]["nb_read_packets"])


def copy_static_files(src: Path, out: Path) -> None:
    """拷貝 meta/、data/、manifest.json（冪等）。"""
    for item in ("meta", "data"):
        s, d = src / item, out / item
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
    manifest = src / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, out / "manifest.json")


def fix_episode(src_video: Path, out_video: Path, n_declared: int,
                crf: int, preset: str, log: Path) -> str:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    if out_video.exists():
        existing = frame_count(out_video)
        if existing == n_declared:
            return "skip(already ok)"

    n_src = frame_count(src_video)
    pad_s = max(0.0, (n_declared - n_src) / FPS)

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src_video)]
    if pad_s > 0:
        cmd += ["-vf", f"tpad=stop_mode=clone:stop_duration={pad_s:.6f}"]
    cmd += [
        "-r", str(FPS), "-vsync", "cfr",
        "-frames:v", str(n_declared),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        # 不用 B 幀：解碼序 == 顯示序，ffprobe 包序即幀序，檢查工具不會誤判
        "-bf", "0",
        "-pix_fmt", "yuv420p", "-an",
        str(out_video),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.write_text(f"FAILED ffmpeg cmd:\n{' '.join(cmd)}\n\nstderr:\n{r.stderr}\n")
        raise RuntimeError(f"ffmpeg failed for {src_video.name} (log: {log})")
    n_out = frame_count(out_video)
    elapsed = time.time() - t0
    note = f"pad={pad_s:.2f}s" if pad_s > 0 else ("trunc" if n_src > n_declared else "exact")
    print(f"  file-{int(src_video.stem.split('-')[1]):03d}.mp4: {n_src} -> {n_out} frames "
          f"[{note}] {elapsed:.1f}s", flush=True)
    if n_out != n_declared:
        raise RuntimeError(f"{out_video.name}: got {n_out} frames, expected {n_declared}")
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--episodes", type=str, default=None,
                        help="只處理指定集（逗號分隔的 episode_index），如 '0' 或 '0,3,5'")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="medium")
    args = parser.parse_args()

    manifest = json.loads((args.src / "manifest.json").read_text())
    episodes = manifest["episodes"]
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        episodes = [ep for ep in episodes if ep["episode_index"] in wanted]

    copy_static_files(args.src, args.out)

    log = PROJECT_ROOT / "logs" / "fix_dataset_last_error.txt"
    t0 = time.time()
    for ep in episodes:
        i = ep["video_file_index"]
        n = ep["length"]
        src_video = args.src / "videos/observation.images.overhead/chunk-000" / f"file-{i:03d}.mp4"
        out_video = args.out / "videos/observation.images.overhead/chunk-000" / f"file-{i:03d}.mp4"
        status = fix_episode(src_video, out_video, n, args.crf, args.preset, log)
        if status == "skip(already ok)":
            print(f"  file-{i:03d}.mp4: skip (already {n} frames)", flush=True)
    print(f"done: {len(episodes)} episode(s) in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
