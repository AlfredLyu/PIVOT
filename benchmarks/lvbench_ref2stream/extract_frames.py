"""Extract frames from LVBench videos in the layout expected by the runners.

Output layout:
    <out_dir>/<video_id>/frame_000001.png, frame_000002.png, ...

Frame `frame_%06d.png` with index i corresponds to second (i - 1) of the video
at the given fps (1 fps for PIVOT / generic-MLLM baselines; the runners then
sub-sample with --frame_stride 2 to obtain the 0.5 fps stream used in the paper).

LVBench videos are expected at <video_dir>/<video_id>.mp4, where video_id is the
8-digit index used in data/lvbench_ref2stream.json (see
data/video_id_to_youtube_key.json for the corresponding YouTube ids).

Usage:
    python benchmarks/lvbench_ref2stream/extract_frames.py \
        --video_dir data/lvbench/videos --out_dir data/lvbench/frames_1fps --fps 1
"""
import argparse
import os
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video_dir", required=True, help="Directory with <video_id>.mp4 files.")
    p.add_argument("--out_dir", required=True, help="Output root for per-video frame folders.")
    p.add_argument("--fps", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    videos = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    for i, name in enumerate(videos):
        video_id = os.path.splitext(name)[0]
        out = os.path.join(args.out_dir, video_id)
        if os.path.isdir(out) and os.listdir(out):
            print(f"[{i + 1}/{len(videos)}] skip {video_id} (exists)")
            continue
        os.makedirs(out, exist_ok=True)
        print(f"[{i + 1}/{len(videos)}] extracting {video_id} ...", flush=True)
        subprocess.run(
            ["ffmpeg", "-i", os.path.join(args.video_dir, name), "-vf", f"fps={args.fps}",
             "-start_number", "1", os.path.join(out, "frame_%06d.png"), "-y", "-loglevel", "warning"],
            check=True,
        )


if __name__ == "__main__":
    main()
