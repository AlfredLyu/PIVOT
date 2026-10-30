"""LVBench-Ref2Stream evaluation for TimeChat-Online (wyccccc/TimeChatOnline-7B)
under the streaming protocol, without text memory.

Each chunk (16 frames at 2 fps), the model sees the observed prefix through a multi-tier FIFO
visual memory and, for every active question, either answers (answer_time = chunk end) or
defers. Questions still pending at the end of the video are unanswered; task types are hidden.

Usage (copy this file to eval/scripts/ inside the official TimeChat-Online
repo, https://github.com/yaolinli/TimeChat-Online, and run with its
environment: torch 2.6, transformers 4.50, flash-attn 2.8.3):

  PIVOT_ROOT=/path/to/PIVOT python eval/scripts/timechat_online_lvbench.py \
      --video_dir /path/to/LVBench/frames_2fps \
      --result_dir /path/to/PIVOT/results/timechat_online/question_results
"""

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

_PIVOT_ROOT = os.environ.get("PIVOT_ROOT", ".")  # path to the PIVOT repo checkout
DEFAULT_TASK_JSON = os.path.join(_PIVOT_ROOT, "benchmarks", "lvbench_ref2stream", "data", "lvbench_ref2stream.json")
DEFAULT_VIDEO_DIR = os.path.join(_PIVOT_ROOT, "data", "lvbench", "frames_2fps")
DEFAULT_SUBSET = ""  # optional {"video_ids": [...]}; empty = all videos
DEFAULT_RESULT_DIR = os.path.join(_PIVOT_ROOT, "results", "timechat_online")
DEFAULT_CKPT = "wyccccc/TimeChatOnline-7B"

MIN_PIXELS = 224 * 224
MAX_PIXELS = 224 * 224

# 2 fps stream, 16 frames (8 s) per chunk.
SOURCE_FPS = 2
FRAME_STRIDE = 1
CHUNK_SIZE = 16
TIER1_SECONDS = 50
TIER2_SECONDS = 100
TIER3_MAX_FRAMES = 100


def parse_timestamp(ts: str) -> int:
    parts = ts.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"bad timestamp: {ts}")
    return int(parts[0]) * 60 + int(parts[1])


_OPTION_PATTERNS = (
    re.compile(r"^\s*\(?\s*([ABCD])\s*\)?\s*[\.,;:!\?]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*([ABCD])\s*\)?\s*[\.\):,\-]", re.IGNORECASE),
    re.compile(r"\(([ABCD])\)", re.IGNORECASE),
    re.compile(r"\boption\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\b(?:answer|choice)\s*[:\-]?\s*\(?([ABCD])\)?\b", re.IGNORECASE),
)


def normalize_choice(text: Any) -> str:
    s = str(text).strip()
    if not s:
        return ""
    for pat in _OPTION_PATTERNS:
        m = pat.search(s)
        if m:
            return f"({m.group(1).upper()})"
    return ""


def list_video_frames(video_dir: str) -> List[str]:
    frames = [
        osp.join(video_dir, n) for n in os.listdir(video_dir)
        if n.endswith(".png") or n.endswith(".jpg")
    ]
    frames.sort()
    return frames


def frame_path_to_second(frame_path: str, source_fps: int) -> float:
    base = osp.basename(frame_path)
    if base.startswith("frame_"):
        idx = int(base.replace("frame_", "").rsplit(".", 1)[0]) - 1
        return float(idx)
    idx = int(base.rsplit(".", 1)[0])
    return idx / source_fps


def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def multitier_prefix_select(
    prefix_2fps: List[Any],
    *,
    tier1_seconds: int = 50,
    tier2_seconds: int = 100,
    tier3_max_frames: int = 100,
    source_fps: int = 2,
) -> List[Any]:
    """Multi-tier FIFO memory over the observed prefix: 2 fps recent, 1 fps medium, and a capped 0.5 fps (coarsening) long-term tier."""
    if not prefix_2fps:
        return []
    if source_fps != 2:
        raise NotImplementedError("multitier_prefix_select assumes source_fps == 2")

    t1_n = tier1_seconds * source_fps
    # Tier 1: most recent frames at 2 fps.
    if len(prefix_2fps) <= t1_n:
        return list(prefix_2fps)
    tier1 = list(prefix_2fps[-t1_n:])
    rem1 = prefix_2fps[:-t1_n]

    # Tier 2: the preceding tier2_seconds at 1 fps.
    t2_max = tier2_seconds
    tier2_pool = rem1[::2]
    if len(tier2_pool) > t2_max:
        tier2 = list(tier2_pool[-t2_max:])
        rem2 = rem1[: -2 * t2_max]
    else:
        tier2 = list(tier2_pool)
        rem2 = []

    # Tier 3: everything older at 0.5 fps, halved until within tier3_max_frames.
    if rem2:
        stride = 4
        tier3_pool = rem2[::stride]
        while len(tier3_pool) > tier3_max_frames:
            tier3_pool = tier3_pool[::2]
        tier3 = list(tier3_pool)
    else:
        tier3 = []

    return tier3 + tier2 + tier1


def extract_json(text: str) -> Any:
    """Return whatever JSON value appears in text (dict / list / None).
    Tolerates leading/trailing markdown fences and surrounding prose."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find an object first, then a list.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        l = text.find(open_ch)
        r = text.rfind(close_ch)
        if l != -1 and r != -1 and l < r:
            try:
                return json.loads(text[l:r + 1])
            except Exception:
                continue
    return None


SYSTEM_PROMPT = """You are a streaming video QA assistant. You watch a long video chunk by chunk in temporal order. At each step you see the video frames from the very beginning up to the current point.

You receive a list of active multiple-choice questions. For each, decide whether the visible video so far already gives enough evidence to answer (commit) or whether you should wait for more future frames (defer).

Output JSON only, with this schema:
{
  "question_updates": [
    {"qid": "<id>", "decision": "answer" | "need_more_frames", "answer": "(A)" | "(B)" | "(C)" | "(D)" | ""}
  ]
}

Rules:
- If decision is "answer", `answer` MUST be exactly one of (A), (B), (C), (D).
- If decision is "need_more_frames", `answer` MUST be the empty string.
- Output JSON only, no commentary, no markdown fences.
"""


def call_model(
    model,
    processor,
    frames: List[str],
    user_text: str,
    *,
    drop_method: str,
    drop_threshold: float,
    drop_absolute: bool,
    dr_save_path: str,
    max_new_tokens: int,
    min_px: int,
    max_px: int,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "min_pixels": min_px,
                    "max_pixels": max_px,
                    "max_frames": len(frames),
                    "min_frames": 1,
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            drop_method=drop_method,
            drop_threshold=drop_threshold,
            drop_absolute=drop_absolute,
            dr_save_path=dr_save_path,
        )
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def make_question_record(
    item: Dict[str, Any],
    *,
    status: str,
    pred_answer: str,
    pred_answer_raw: str,
    answer_time: Optional[int],
    answer_source: str,
    evidence: str,
    ask_time: int,
    beg_ref: int,
    end_ref: int,
    gt_answer_time: int,
    video_last_processed_sec: int,
    sampled_frame_seconds: List[int],
    decision_history: List[Dict[str, Any]],
    chunk_size: int,
    frame_stride: int,
    source_fps: int,
    tier1_seconds: int,
    tier2_seconds: int,
    tier3_max_frames: int,
) -> Dict[str, Any]:
    pred_strict = normalize_choice(pred_answer)
    return {
        "qid": item["uid"],
        "uid": item["uid"],
        "video_id": item["video_id"],
        "question": item["question"],
        "gt_answer": item["answer"],
        "task_type": item["type"],
        "status": status,
        "pred_answer": pred_strict if pred_strict else pred_answer,
        "pred_answer_raw": pred_answer_raw,
        "pred_answer_strict": pred_strict,
        "answer_time": answer_time,
        "answer_source": answer_source,
        "evidence": evidence,
        "ask_time_raw": ask_time,
        "ask_time": ask_time,
        "beg_ref": beg_ref,
        "end_ref": end_ref,
        "gt_answer_time_raw": gt_answer_time,
        "gt_answer_time": gt_answer_time,
        "video_last_processed_sec": video_last_processed_sec,
        "sampled_num_frames": len(sampled_frame_seconds),
        "sampled_frame_seconds": sampled_frame_seconds,
        "decision_history": decision_history,
        "model": "TimeChatOnline-7B",
        "chunk_size": chunk_size,
        "frame_stride": frame_stride,
        "source_fps": source_fps,
        "tier1_seconds": tier1_seconds,
        "tier2_seconds": tier2_seconds,
        "tier3_max_frames": tier3_max_frames,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_json", default=DEFAULT_TASK_JSON)
    p.add_argument("--video_dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--subset_path", default=DEFAULT_SUBSET)
    p.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--ckpt_path", default=DEFAULT_CKPT)

    p.add_argument("--drop_method", default="feature", choices=["feature", "pixel", "none"])
    p.add_argument("--drop_threshold", type=float, default=0.5)
    p.add_argument("--drop_relative", action="store_true")
    p.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    p.add_argument("--max_pixels", type=int, default=MAX_PIXELS)

    p.add_argument("--source_fps", type=int, default=SOURCE_FPS)
    p.add_argument("--frame_stride", type=int, default=FRAME_STRIDE,
                   help="Stride over the source frames; effective stream FPS = source_fps / stride.")
    p.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                   help="Sampled frames per streaming chunk.")
    p.add_argument("--tier1_seconds", type=int, default=TIER1_SECONDS,
                   help="Multi-tier memory: tier-1 holds the most recent N seconds at 2 fps.")
    p.add_argument("--tier2_seconds", type=int, default=TIER2_SECONDS,
                   help="Multi-tier memory: tier-2 holds the next N seconds at 1 fps.")
    p.add_argument("--tier3_max_frames", type=int, default=TIER3_MAX_FRAMES,
                   help="Multi-tier memory: tier-3 holds the long-term remainder at "
                        "0.5 fps (halved when exceeding this cap).")
    p.add_argument("--max_defers_per_question", type=int, default=30,
                   help="If a question is deferred this many times, give up and "
                        "mark it unanswered (avoids endless deferring on long videos).")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--video_id", default=None)
    p.add_argument("--limit_videos", type=int, default=-1)
    p.add_argument("--limit_questions_per_video", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    drop_absolute = not args.drop_relative

    os.makedirs(args.result_dir, exist_ok=True)
    log_dir = osp.join(args.result_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    curr = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = osp.join(log_dir, f"run_{curr}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)5s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"ckpt: {args.ckpt_path}")
    logger.info(f"streaming: source_fps={args.source_fps} stride={args.frame_stride} "
                f"chunk_size={args.chunk_size} "
                f"memory=tier1={args.tier1_seconds}s@2fps,tier2={args.tier2_seconds}s@1fps,"
                f"tier3<={args.tier3_max_frames}f@0.5fps")
    logger.info(f"DTD: method={args.drop_method} threshold={args.drop_threshold} "
                f"absolute={drop_absolute}")

    if args.subset_path:
        with open(args.subset_path) as f:
            subset_ids = set(json.load(f)["video_ids"])
    else:
        subset_ids = None
    if args.video_id:
        subset_ids = {args.video_id}

    with open(args.task_json) as f:
        tasks = json.load(f)
    if subset_ids is not None:
        tasks = [t for t in tasks if t["video_id"] in subset_ids]
    subset_ids = {t["video_id"] for t in tasks}
    logger.info(f"subset videos: {len(subset_ids)}")

    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        by_video[t["video_id"]].append(t)

    video_ids = sorted(by_video.keys())
    if args.limit_videos > 0:
        video_ids = video_ids[: args.limit_videos]
    if args.limit_questions_per_video > 0:
        for vid in video_ids:
            by_video[vid] = by_video[vid][: args.limit_questions_per_video]
    total_q = sum(len(by_video[v]) for v in video_ids)
    logger.info(f"to evaluate: {len(video_ids)} videos, {total_q} questions")

    torch.manual_seed(1234)
    logger.info("loading model + processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    drop_dir = osp.join(args.result_dir, "drop")
    os.makedirs(drop_dir, exist_ok=True)
    dr_save_path = osp.join(drop_dir, f"drop_{curr}.jsonl")

    written = skipped_videos = failed = 0
    overall_start = time.time()
    pbar = tqdm(total=total_q, desc="streaming-tco")

    for video_id in video_ids:
        items = by_video[video_id]

        existing_terminal = []
        for it in items:
            out_path = osp.join(args.result_dir, f"{it['uid']}.json")
            if osp.exists(out_path) and not args.overwrite:
                try:
                    with open(out_path) as f:
                        prev = json.load(f)
                    if prev.get("status") in ("answered_online", "unanswered"):
                        existing_terminal.append(it["uid"])
                except Exception:
                    pass
        if len(existing_terminal) == len(items) and not args.overwrite:
            logger.info(f"[skip] video={video_id} all {len(items)} questions terminal")
            skipped_videos += 1
            pbar.update(len(items))
            continue

        video_dir = osp.join(args.video_dir, video_id)
        if not osp.isdir(video_dir):
            logger.error(f"[err] video dir missing: {video_dir}")
            failed += len(items)
            for it in items:
                with open(osp.join(args.result_dir, f"{it['uid']}.json"), "w") as f:
                    json.dump({
                        "qid": it["uid"], "uid": it["uid"], "video_id": it["video_id"],
                        "status": "error",
                        "error": f"video dir missing: {video_dir}",
                        "question": it["question"], "gt_answer": it["answer"],
                        "task_type": it["type"],
                    }, f, indent=2, ensure_ascii=False)
            pbar.update(len(items))
            continue

        all_frames = list_video_frames(video_dir)
        if not all_frames:
            logger.error(f"[err] no frames in {video_dir}")
            failed += len(items)
            pbar.update(len(items))
            continue

        sampled = all_frames[:: args.frame_stride] if args.frame_stride > 1 else list(all_frames)
        if not sampled:
            sampled = all_frames[:1]
        sampled_secs = [frame_path_to_second(p, args.source_fps) for p in sampled]
        max_processed_sec = int(sampled_secs[-1])
        chunks = chunk_list(sampled, args.chunk_size)

        # End-of-chunk index (into sampled[]) for each chunk.
        chunk_end_idxs = []
        running = 0
        for ck in chunks:
            running += len(ck)
            chunk_end_idxs.append(running - 1)

        # Pre-build per-question records.
        q_records: Dict[str, Dict[str, Any]] = {}
        ordered_qids: List[str] = []
        for it in items:
            beg_ref, end_ref = it["time_reference"].split("-")
            beg_ref_sec = parse_timestamp(beg_ref)
            end_ref_sec = parse_timestamp(end_ref)
            ask_time_raw = parse_timestamp(it["ask_time"])
            ask_time = min(ask_time_raw, max_processed_sec)
            if it["type"] in ("Past", "Present"):
                gt_answer_time_raw = ask_time_raw
            else:
                gt_answer_time_raw = end_ref_sec
            gt_answer_time = min(gt_answer_time_raw, max_processed_sec)
            q_records[it["uid"]] = {
                "item": it,
                "beg_ref": beg_ref_sec,
                "end_ref": end_ref_sec,
                "ask_time": ask_time,
                "ask_time_raw": ask_time_raw,
                "gt_answer_time": gt_answer_time,
                "gt_answer_time_raw": gt_answer_time_raw,
                "status": "unseen",
                "pred_answer": "",
                "pred_answer_raw": "",
                "answer_time": None,
                "answer_source": "",
                "evidence": "",
                "decision_history": [],
            }
            ordered_qids.append(it["uid"])
        ordered_qids.sort(key=lambda qid: q_records[qid]["ask_time"])

        next_q_idx = 0
        pending_qids: List[str] = []

        video_t0 = time.time()
        n_calls = 0
        for chunk_idx, chunk in enumerate(chunks):
            chunk_secs = [frame_path_to_second(p, args.source_fps) for p in chunk]
            chunk_start_sec = int(min(chunk_secs))
            chunk_end_sec = int(max(chunk_secs))

            while next_q_idx < len(ordered_qids) and \
                    q_records[ordered_qids[next_q_idx]]["ask_time"] <= chunk_end_sec:
                qid_act = ordered_qids[next_q_idx]
                if q_records[qid_act]["status"] == "unseen":
                    q_records[qid_act]["status"] = "pending"
                    pending_qids.append(qid_act)
                next_q_idx += 1

            active_qids = [qid for qid in pending_qids if q_records[qid]["status"] == "pending"]
            if not active_qids:
                continue  # No model call this chunk.

            # Cumulative prefix: all sampled frames up to chunk_end (inclusive).
            prefix_end_idx = chunk_end_idxs[chunk_idx]
            prefix_frames_full = sampled[: prefix_end_idx + 1]
            prefix_frames = multitier_prefix_select(
                prefix_frames_full,
                tier1_seconds=args.tier1_seconds,
                tier2_seconds=args.tier2_seconds,
                tier3_max_frames=args.tier3_max_frames,
                source_fps=args.source_fps,
            )

            active_payload = [
                {"qid": qid, "question": q_records[qid]["item"]["question"]}
                for qid in active_qids
            ]
            user_text = (
                f"You have seen frames from second 0 up to second {chunk_end_sec}.\n\n"
                f"Active unanswered questions:\n"
                f"{json.dumps(active_payload, ensure_ascii=False, indent=2)}\n\n"
                f"For each question above, decide whether to answer now or wait. "
                f"Output JSON only."
            )

            try:
                raw = call_model(
                    model, processor,
                    prefix_frames, user_text,
                    drop_method=args.drop_method,
                    drop_threshold=args.drop_threshold,
                    drop_absolute=drop_absolute,
                    dr_save_path=dr_save_path,
                    max_new_tokens=args.max_new_tokens,
                    min_px=args.min_pixels,
                    max_px=args.max_pixels,
                )
            except Exception as e:
                logger.warning(f"[chunk-err] video={video_id} chunk={chunk_idx}: {e}")
                raw = ""
            n_calls += 1

            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                qu_list = parsed.get("question_updates")
            elif isinstance(parsed, list):
                qu_list = parsed
            else:
                qu_list = None
            qu_by_qid: Dict[str, Dict[str, Any]] = {}
            if isinstance(qu_list, list):
                for upd in qu_list:
                    if not isinstance(upd, dict):
                        continue
                    qid_u = str(upd.get("qid", "")).strip()
                    if qid_u in active_qids and qid_u not in qu_by_qid:
                        qu_by_qid[qid_u] = upd

            for qid in active_qids:
                rec = q_records[qid]
                upd = qu_by_qid.get(qid)
                if upd:
                    decision = str(upd.get("decision", "")).strip().lower()
                    raw_ans = str(upd.get("answer", "")).strip()
                else:
                    decision = ""
                    raw_ans = ""
                strict = normalize_choice(raw_ans)
                committed = decision == "answer" and strict
                rec["decision_history"].append({
                    "chunk_idx": chunk_idx,
                    "chunk_end_sec": chunk_end_sec,
                    "n_prefix_frames": len(prefix_frames),
                    "raw_update": upd,
                    "decision": "answer" if committed else
                                "need_more_frames" if decision == "need_more_frames" else
                                "missing_or_invalid",
                    "answer": strict,
                })
                if committed:
                    rec["status"] = "answered_online"
                    rec["pred_answer_raw"] = raw_ans
                    rec["pred_answer"] = strict
                    rec["answer_time"] = chunk_end_sec
                    rec["answer_source"] = "streaming"
                    rec["evidence"] = ""
                else:
                    # Count defers (and missing/invalid as defer for the cap).
                    defer_count = sum(
                        1 for h in rec["decision_history"]
                        if h.get("decision") in ("need_more_frames", "missing_or_invalid")
                    )
                    if defer_count >= args.max_defers_per_question:
                        rec["status"] = "unanswered"
                        rec["answer_source"] = "max_defers_reached"

            pending_qids = [qid for qid in pending_qids if q_records[qid]["status"] == "pending"]

        for qid in ordered_qids:
            rec = q_records[qid]
            if rec["status"] in ("unseen", "pending"):
                rec["status"] = "unanswered"
                rec["answer_source"] = "video_end_unanswered"

        n_answered = sum(1 for q in q_records.values() if q["status"] == "answered_online")
        n_unanswered = sum(1 for q in q_records.values() if q["status"] == "unanswered")
        for qid in ordered_qids:
            rec = q_records[qid]
            out_path = osp.join(args.result_dir, f"{qid}.json")
            payload = make_question_record(
                rec["item"],
                status=rec["status"],
                pred_answer=rec["pred_answer"],
                pred_answer_raw=rec["pred_answer_raw"],
                answer_time=rec["answer_time"],
                answer_source=rec["answer_source"],
                evidence=rec["evidence"],
                ask_time=rec["ask_time"],
                beg_ref=rec["beg_ref"],
                end_ref=rec["end_ref"],
                gt_answer_time=rec["gt_answer_time"],
                video_last_processed_sec=max_processed_sec,
                sampled_frame_seconds=[int(s) for s in sampled_secs],
                decision_history=rec["decision_history"],
                chunk_size=args.chunk_size,
                frame_stride=args.frame_stride,
                source_fps=args.source_fps,
                tier1_seconds=args.tier1_seconds,
                tier2_seconds=args.tier2_seconds,
                tier3_max_frames=args.tier3_max_frames,
            )
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            written += 1
            pbar.update(1)

        elapsed_video = time.time() - video_t0
        logger.info(
            f"[done video] {video_id} chunks={len(chunks)} model_calls={n_calls} "
            f"answered={n_answered} unanswered={n_unanswered} "
            f"elapsed={elapsed_video:.1f}s"
        )

    pbar.close()
    elapsed = time.time() - overall_start
    logger.info(
        f"DONE  written={written} skipped_videos_all_terminal={skipped_videos} "
        f"failed={failed} elapsed={int(elapsed)//60}m{int(elapsed)%60}s"
    )


if __name__ == "__main__":
    main()
