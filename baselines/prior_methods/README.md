# Prior streaming methods on LVBench-Ref2Stream

These adapters run prior streaming video LLMs under our streaming protocol (no future frames, no
task-type labels, questions revealed at their query times). They depend on the original code bases,
so copy each file into the corresponding official repository and run it with that repo's environment.
Both write one JSON per question directly into `--result_dir`; point `--result_dir` at
`<run_dir>/question_results` so that `python evaluation/compute_metrics.py <run_dir>` can score it.

| File | Method | Where to put it | Notes |
|---|---|---|---|
| `dispider_lvbench.py` | [Dispider](https://github.com/Mark12Ding/Dispider) | repo root | Reads raw `.mp4` videos. The model only sees the video up to the clip containing the query time; the answer time is the end of that clip. Paper setting: `--num_clips_cap 30`. |
| `timechat_online_lvbench.py` | [TimeChat-Online](https://github.com/yaolinli/TimeChat-Online) (`wyccccc/TimeChatOnline-7B`) | `eval/scripts/` | Reads 2 fps frames (`extract_frames.py --fps 2`). Multi-tier FIFO visual memory; per chunk the model answers or defers each active question. |

Set `PIVOT_ROOT=/path/to/PIVOT` so the scripts find the benchmark JSON, or pass the paths explicitly:

```bash
# Dispider (inside the Dispider repo). If the checkpoint's config.json points mm_compressor to an
# absolute path, update it to <ckpt>/stream_compressor first.
PIVOT_ROOT=/path/to/PIVOT python dispider_lvbench.py \
    --model_path /path/to/Dispider_ckpt \
    --video_dir /path/to/PIVOT/data/lvbench/videos \
    --result_dir /path/to/PIVOT/results/dispider/question_results \
    --num_clips_cap 30

# TimeChat-Online (inside the TimeChat-Online repo)
PIVOT_ROOT=/path/to/PIVOT python eval/scripts/timechat_online_lvbench.py \
    --video_dir /path/to/PIVOT/data/lvbench/frames_2fps \
    --result_dir /path/to/PIVOT/results/timechat_online/question_results
```
