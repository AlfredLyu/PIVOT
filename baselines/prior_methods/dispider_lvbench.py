"""LVBench-Ref2Stream evaluation for Dispider under the streaming protocol.

The video is truncated at the clip containing ask_time, so the model never sees future frames.
The released checkpoint has no usable response head, so each question is answered at the end of
that clip (no deferral).

Usage (copy this file into the root of the official Dispider repo,
https://github.com/Mark12Ding/Dispider, and run inside its environment):

  PIVOT_ROOT=/path/to/PIVOT python dispider_lvbench.py \
      --model_path /path/to/Dispider_ckpt \
      --video_dir /path/to/LVBench/videos \
      --result_dir /path/to/PIVOT/results/dispider/question_results \
      --num_clips_cap 30
"""

import argparse
import json
import os
import os.path as osp
import re
import sys
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from PIL import Image

THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from decord import VideoReader
from transformers import StoppingCriteria, StoppingCriteriaList

from dispider.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_ANS_TOKEN,
    DEFAULT_TODO_TOKEN,
)
from dispider.conversation import conv_templates
from dispider.model.builder import load_pretrained_model
from dispider.mm_utils import tokenizer_image_token, get_model_name_from_path

_PIVOT_ROOT = os.environ.get("PIVOT_ROOT", ".")  # path to the PIVOT repo checkout
DEFAULT_QUESTIONS = os.path.join(_PIVOT_ROOT, "benchmarks", "lvbench_ref2stream", "data", "lvbench_ref2stream.json")
DEFAULT_VIDEO_DIR = os.path.join(_PIVOT_ROOT, "data", "lvbench", "videos")
DEFAULT_SUBSET = ""  # optional {"video_ids": [...]}; empty = all videos
DEFAULT_RESULT_DIR = os.path.join(_PIVOT_ROOT, "results", "dispider")
DEFAULT_MODEL_PATH = ""  # local Dispider checkpoint dir (required)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None):
        super().__init__()
        self.stops = stops or []

    def __call__(self, input_ids, scores):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def parse_timestamp(ts: str) -> int:
    parts = ts.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"bad timestamp: {ts}")
    return int(parts[0]) * 60 + int(parts[1])


def normalize_mcq(text: Any) -> str:
    s = str(text).strip()
    if not s:
        return ""
    for pat in (
        re.compile(r"^\s*\(?\s*([ABCD])\s*\)?\s*[\.,;:!\?]*\s*$", re.IGNORECASE),
        re.compile(r"^\s*\(?\s*([ABCD])\s*\)?\s*[\.\):,\-]", re.IGNORECASE),
        re.compile(r"\(([ABCD])\)", re.IGNORECASE),
        re.compile(r"\boption\s*([ABCD])\b", re.IGNORECASE),
        re.compile(r"\b(?:answer|choice)\s*[:\-]?\s*\(?([ABCD])\)?\b", re.IGNORECASE),
    ):
        m = pat.search(s)
        if m:
            return f"({m.group(1).upper()})"
    return ""


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    return [(int(np.round(seg_size * i)) + int(np.round(seg_size * (i + 1)))) // 2 for i in range(desired_num_frames)]


def get_seq_time(vr, frame_idx, num_clip):
    frm_per_clip = len(frame_idx) // num_clip
    key_frame = [
        [frame_idx[i * frm_per_clip], frame_idx[i * frm_per_clip + frm_per_clip - 1]]
        for i in range(num_clip)
    ]
    times = vr.get_frame_timestamp(key_frame)
    return np.hstack([times[:, 0, 0], times[:, 1, 1]])


def safe_get_batch(vr, frame_idx):
    idx = [int(i) for i in frame_idx if 0 <= int(i) < len(vr)]
    if not idx:
        raise RuntimeError("safe_get_batch: no valid frame indices.")
    try:
        return vr.get_batch(idx).asnumpy(), idx
    except Exception as exc:
        frames, good = [], []
        for i in idx:
            try:
                frames.append(vr[i].asnumpy())
                good.append(i)
            except Exception:
                continue
        if not frames:
            raise RuntimeError("safe_get_batch: all frames failed.") from exc
        return np.stack(frames, axis=0), good


def load_video(vis_path, num_frm=16, max_clip=30):
    vr = VideoReader(vis_path, num_threads=1)
    total_frame_num = len(vr)
    fps = float(vr.get_avg_fps()) or 30.0
    total_time = total_frame_num / fps

    num_clip = max(1, min(max_clip, int(np.round(total_time / num_frm))))
    total_num_frm = num_frm * num_clip
    raw_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array, frame_idx = safe_get_batch(vr, raw_idx)

    actual = img_array.shape[0]
    if actual != total_num_frm:
        num_clip = max(1, actual // num_frm)
        total_num_frm = num_clip * num_frm
        img_array = img_array[:total_num_frm]
        frame_idx = frame_idx[:total_num_frm]
    time_idx = get_seq_time(vr, frame_idx, num_clip)

    a, H, W, _ = img_array.shape
    if H != W:
        t = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        t = torch.nn.functional.interpolate(t, size=(min(H, W), min(H, W)))
        img_array = t.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape(
        (1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1])
    )
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]
    return clip_imgs, time_idx, num_clip


def preprocess_time(time_idx, num_clip, tokenizer):
    time_idx = time_idx.reshape(2, num_clip)
    seqs = []
    for i in range(num_clip):
        start = int(np.round(time_idx[0, i]))
        end = int(np.round(time_idx[1, i]))
        s = f"This contains a clip sampled in {start} to {end} seconds" + DEFAULT_IMAGE_TOKEN
        seqs.append(tokenizer_image_token(s, tokenizer, return_tensors="pt"))
    return seqs


def preprocess_question(questions, tokenizer):
    return [
        tokenizer_image_token(q + DEFAULT_TODO_TOKEN, tokenizer, return_tensors="pt")
        for q in questions
    ]


class VideoStream:
    def __init__(self, model_path):
        model_name = get_model_name_from_path(osp.expanduser(model_path))
        self.tokenizer, self.model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map="auto", device="cuda",
        )
        self.image_processor, self.time_tokenizer = image_processor
        self.image_processor_large = self.image_processor
        if self.time_tokenizer.pad_token is None:
            self.time_tokenizer.pad_token = "<pad>"
        stops = [torch.tensor(self.tokenizer("<|im_end|>").input_ids).cuda()]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stops)])

    def prepare_video(self, video_path, num_clips_cap=30, num_frames=16):
        frames, time_idx, num_clips = load_video(video_path, num_frames, max_clip=num_clips_cap)
        video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video = video.view(num_clips, num_frames, *video.shape[1:])
        video_large = self.image_processor_large.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_large = video_large.view(num_clips, num_frames, *video_large.shape[1:])[:, :1].contiguous()
        seqs_pre = preprocess_time(time_idx, num_clips, self.time_tokenizer)
        seqs = torch.nn.utils.rnn.pad_sequence(
            seqs_pre, batch_first=True, padding_value=self.time_tokenizer.pad_token_id
        )
        compress_mask = seqs.ne(self.time_tokenizer.pad_token_id)
        return {
            "video": video, "video_large": video_large,
            "seqs": seqs, "compress_mask": compress_mask,
            "time_idx": time_idx, "num_clips": num_clips,
        }

    def _qa_inputs(self, question_text):
        if self.model.config.mm_use_im_start_end:
            qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                  + "\n" + question_text)
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question_text
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        question_ids = preprocess_question([question_text], self.time_tokenizer)
        question_ids = torch.nn.utils.rnn.pad_sequence(
            question_ids, batch_first=True, padding_value=self.time_tokenizer.pad_token_id
        )
        qs_mask = question_ids.ne(self.time_tokenizer.pad_token_id)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, question_ids, qs_mask

    def run_streaming(self, video_cache, prompt, ask_time_sec, max_new_tokens=64):
        """Truncate the video at the clip containing ask_time and run the model
        on that prefix only. Returns (out_text, insert_position, num_clips_used,
        clip_times_used)."""
        input_ids, question_ids, qs_mask = self._qa_inputs(prompt)
        input_ids = input_ids.unsqueeze(0).to(device="cuda", non_blocking=True)

        time_idx = video_cache["time_idx"]
        num_clips_total = video_cache["num_clips"]
        clip_times = time_idx.reshape(2, num_clips_total)

        # Find the clip that contains ask_time (or the next clip if it's between).
        insert_position = 0
        for i in range(num_clips_total):
            if clip_times[0, i] <= ask_time_sec <= clip_times[1, i]:
                insert_position = i
                break
            if ask_time_sec > clip_times[1, i]:
                insert_position = i + 1
        insert_position = max(0, min(insert_position, num_clips_total - 1))

        # Keep only clips up to the query time.
        n_keep = insert_position + 1
        video = video_cache["video"][:n_keep]
        video_large = video_cache["video_large"][:n_keep]
        seqs = video_cache["seqs"][:n_keep]
        compress_mask = video_cache["compress_mask"][:n_keep]
        clip_times_used = clip_times[:, :n_keep]

        rel_insert = n_keep - 1

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video.to(dtype=torch.float16, device="cuda", non_blocking=True),
                images_large=video_large.to(dtype=torch.float16, device="cuda", non_blocking=True),
                seqs=seqs.to(device="cuda", non_blocking=True),
                compress_mask=compress_mask.to(device="cuda", non_blocking=True),
                qs=question_ids.to(device="cuda", non_blocking=True),
                qs_mask=qs_mask.to(device="cuda", non_blocking=True),
                ans_token=self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt")
                .input_ids.to(device="cuda", non_blocking=True),
                todo_token=self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt")
                .input_ids.to(device="cuda", non_blocking=True),
                q_id=None,
                insert_position=rel_insert,
                ans_position=[],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria,
                use_cache=True,
            )
        out = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return out, insert_position, n_keep, clip_times_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions_path", default=DEFAULT_QUESTIONS)
    p.add_argument("--video_dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--subset_path", default=DEFAULT_SUBSET)
    p.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_clips_cap", type=int, default=30)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--video_id", default=None)
    p.add_argument("--limit_questions", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    if args.subset_path:
        with open(args.subset_path) as f:
            subset_ids = set(json.load(f)["video_ids"])
    else:
        subset_ids = None
    if args.video_id:
        subset_ids = {args.video_id}

    with open(args.questions_path) as f:
        tasks = json.load(f)
    if subset_ids is not None:
        tasks = [t for t in tasks if t["video_id"] in subset_ids]
    subset_ids = {t["video_id"] for t in tasks}
    if args.limit_questions > 0:
        tasks = tasks[: args.limit_questions]
    print(f"loaded {len(tasks)} subset questions across {len(subset_ids)} videos")

    print("loading model...")
    streamer = VideoStream(args.model_path)
    print("model ready")

    by_video: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for t in tasks:
        by_video.setdefault(t["video_id"], []).append(t)

    written = skipped = failed = 0
    start = time.time()
    pbar = tqdm.tqdm(total=len(tasks), desc="dispider-stream")

    for video_id, video_tasks in by_video.items():
        if not args.overwrite:
            pending = [
                t for t in video_tasks
                if not osp.exists(osp.join(args.result_dir, f"{t['uid']}.json"))
            ]
            if not pending:
                skipped += len(video_tasks)
                pbar.update(len(video_tasks))
                continue
            video_tasks = pending  # Only run pending questions
        video_path = osp.join(args.video_dir, f"{video_id}.mp4")
        if not osp.exists(video_path):
            for t in video_tasks:
                err = {
                    "qid": t["uid"], "uid": t["uid"], "video_id": video_id,
                    "status": "error", "error": f"video missing: {video_path}",
                    "question": t["question"], "gt_answer": t["answer"],
                    "task_type": t["type"],
                }
                with open(osp.join(args.result_dir, f"{t['uid']}.json"), "w") as f:
                    json.dump(err, f, indent=2, ensure_ascii=False)
                failed += 1
            pbar.update(len(video_tasks))
            continue

        try:
            video_cache = streamer.prepare_video(
                video_path, num_clips_cap=args.num_clips_cap, num_frames=args.num_frames
            )
        except Exception as e:
            print(f"[ERR prepare_video] video={video_id}: {e}")
            traceback.print_exc()
            for t in video_tasks:
                err = {
                    "qid": t["uid"], "uid": t["uid"], "video_id": video_id,
                    "status": "error", "error": f"prepare_video: {e}",
                    "question": t["question"], "gt_answer": t["answer"],
                    "task_type": t["type"],
                }
                with open(osp.join(args.result_dir, f"{t['uid']}.json"), "w") as f:
                    json.dump(err, f, indent=2, ensure_ascii=False)
                failed += 1
            pbar.update(len(video_tasks))
            continue

        for task in video_tasks:
            uid = task["uid"]
            out_path = osp.join(args.result_dir, f"{uid}.json")
            try:
                ask_time_sec = parse_timestamp(task["ask_time"])
                beg_ref, end_ref = task["time_reference"].split("-")
                beg_ref_sec = parse_timestamp(beg_ref)
                end_ref_sec = parse_timestamp(end_ref)
                task_type = task["type"]
                prompt_text = task["question"].rstrip() + " Best option: ("

                output, insert_position, n_clips_used, clip_times_used = streamer.run_streaming(
                    video_cache, prompt_text, ask_time_sec,
                    max_new_tokens=args.max_new_tokens,
                )
                pred_strict = normalize_mcq(output)

                # answer_time = end of the clip in which ask_time falls.
                answer_time = int(round(float(clip_times_used[1, n_clips_used - 1])))

                if task_type in ("Past", "Present"):
                    gt_answer_time = ask_time_sec
                else:
                    gt_answer_time = end_ref_sec

                video_last_processed_sec = answer_time
                record = {
                    "qid": uid, "uid": uid, "video_id": video_id,
                    "question": task["question"], "gt_answer": task["answer"],
                    "task_type": task_type, "status": "answered_online",
                    "pred_answer": pred_strict if pred_strict else output,
                    "pred_answer_raw": output,
                    "pred_answer_strict": pred_strict,
                    "answer_time": answer_time,
                    "answer_source": "streaming_truncated",
                    "ask_time": ask_time_sec,
                    "ask_time_raw": ask_time_sec,
                    "beg_ref": beg_ref_sec,
                    "end_ref": end_ref_sec,
                    "gt_answer_time_raw": gt_answer_time,
                    "gt_answer_time": gt_answer_time,
                    "video_last_processed_sec": video_last_processed_sec,
                    "sampled_num_frames": n_clips_used * args.num_frames,
                    "insert_position": int(insert_position),
                    "n_clips_used": int(n_clips_used),
                    "n_clips_total": int(video_cache["num_clips"]),
                    "model": "Dispider-streaming-truncated",
                    "max_new_tokens": args.max_new_tokens,
                }
                with open(out_path, "w") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                written += 1
                pbar.update(1)
            except Exception as e:
                failed += 1
                print(f"[ERR] uid={uid}: {e}")
                traceback.print_exc()
                err = {
                    "qid": uid, "uid": uid, "video_id": video_id,
                    "question": task["question"], "gt_answer": task["answer"],
                    "task_type": task["type"], "status": "error",
                    "error": str(e),
                }
                with open(out_path, "w") as f:
                    json.dump(err, f, indent=2, ensure_ascii=False)
                pbar.update(1)
        del video_cache
        torch.cuda.empty_cache()
    pbar.close()
    elapsed = time.time() - start
    print(f"DONE  written={written} skipped={skipped} failed={failed} "
          f"elapsed={int(elapsed)//60}m{int(elapsed)%60}s")


if __name__ == "__main__":
    main()
