"""
FIFO visual-context wrapper baseline for streaming long-video QA.

Each chunk:
  1. Sample new frames (chunk_size frames at frame_stride from 1fps source).
  2. Append new frames to a FIFO frame buffer; evict oldest until the buffer fits
     the visual-token budget: max_frames = floor(max_visual_tokens / cframe).
  3. For each active question, call the MLLM with (FIFO frames, question) and
     receive either an answer letter or a defer signal. Defer keeps the question
     active for the next chunk; the model is queried again with the new buffer.

No text memory, no captioning, no planner, no reflection — only the most recent
frames as visual context. Matches the "Generic MLLMs with FIFO visual-context
wrapper" baseline in the paper.
"""

import os
import json
import base64
import re
import random
import traceback
import time
import threading
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sys.path.insert(0, os.path.join(_REPO_ROOT, "pivot"))
DEFAULT_TASK_PATH = os.path.join(_REPO_ROOT, "benchmarks", "lvbench_ref2stream", "data", "lvbench_ref2stream.json")
DEFAULT_FRAMES_DIR = os.environ.get("LVBENCH_FRAMES_DIR", os.path.join(_REPO_ROOT, "data", "lvbench", "frames_1fps"))
DEFAULT_OUTPUT_ROOT = os.path.join(_REPO_ROOT, "results")

from openai import OpenAI

try:
    from llm_api import call_llm as _llm_call_llm, unified_text as _llm_unified_text, unified_cost as _llm_unified_cost, provider_for as _llm_provider_for
    _LLM_API_AVAILABLE = True
except Exception as _llm_e:
    _LLM_API_AVAILABLE = False
    print(f"[llm_api] import failed: {_llm_e}")


MODEL_PRICING = {
    # API models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    # Open-source models (local inference)
    "qwen2.5-vl-7b": {"input": 0.0, "output": 0.0},
    "llava-onevision-7b": {"input": 0.0, "output": 0.0},
    "internvl3-8b": {"input": 0.0, "output": 0.0},
}

# Visual tokens per frame (cframe); buffer size = floor(max_visual_tokens / cframe).
DEFAULT_CFRAME = {
    "gpt-4o": 1175,
    "gemini-3.1-flash-lite-preview": 1100,
    "qwen2.5-vl-7b": 252,
    "llava-onevision-7b": 196,
    "internvl3-8b": 256,
}


# ===== Args =====
parser = argparse.ArgumentParser(
    description="FIFO visual-context wrapper baseline for streaming video QA."
)
parser.add_argument("--chunk_size", type=int, default=16,
                    help="Number of frames per chunk (matches main protocol).")
parser.add_argument("--frame_stride", type=int, default=2,
                    help="Use every N-th frame from 1fps source (stride=2 → 0.5 fps effective).")
parser.add_argument("--model_name", type=str, default="gemini-3.1-flash-lite-preview",
                    choices=sorted(MODEL_PRICING.keys()))
parser.add_argument("--reasoning_effort", type=str, default="medium",
                    choices=["low", "medium", "high"])
parser.add_argument("--max_visual_tokens", type=int, default=50000,
                    help="Visual-context budget for the FIFO buffer (in tokens).")
parser.add_argument("--cframe", type=int, default=0,
                    help="Estimated visual tokens per sampled frame. 0 = use DEFAULT_CFRAME[model_name].")
parser.add_argument("--num_workers", type=int, default=6,
                    help="Worker threads for video-level parallelism. Set to 1 for HF backends "
                         "(local GPU inference is GPU-bound and does not benefit from threading).")
parser.add_argument("--hf_model_path", type=str, default="",
                    help="Override HF model path/repo for OS backends (default: well-known repo for the model_name).")
parser.add_argument("--hf_max_pixels", type=int, default=200704,
                    help="Qwen2.5-VL max_pixels (default 448²=200704). Limits per-frame visual tokens to ~252.")
parser.add_argument("--hf_device", type=str, default="auto",
                    help="device_map for HF model (default: auto).")
parser.add_argument("--hf_dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="torch dtype for HF backend.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--subset_path", type=str, default="",
                    help='Optional JSON {"video_ids": [...]} restricting the run to a video subset (default: all videos).')
parser.add_argument("--result_dir_suffix", type=str, default="",
                    help="appended to result_dir name")
parser.add_argument("--max_output_tokens", type=int, default=8192,
                    help="Output-token budget per LLM call. Late chunks may have many active questions, "
                         "each contributing one JSON entry (~80-100 tokens). 8192 fits ~80 questions; "
                         "increase if you see truncated JSON.")
parser.add_argument("--gemini_thinking_budget", type=int, default=0,
                    help="Gemini thinking budget (0 disables thinking).")
parser.add_argument("--task_path", type=str, default=DEFAULT_TASK_PATH,
                    help="LVBench-Ref2Stream question file (default: one-variant-per-source-QA benchmark).")
parser.add_argument("--frames_dir", type=str, default=DEFAULT_FRAMES_DIR,
                    help="Root dir of pre-extracted 1fps frames: <frames_dir>/<video_id>/frame_XXXXXX.png")
parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                    help="Root directory for result folders.")
args = parser.parse_args()


MODEL_NAME = args.model_name
CHUNK_SIZE = args.chunk_size
FRAME_STRIDE = args.frame_stride
REASONING_EFFORT = args.reasoning_effort
SEED = args.seed
MAX_VISUAL_TOKENS = args.max_visual_tokens
CFRAME = args.cframe if args.cframe > 0 else DEFAULT_CFRAME.get(MODEL_NAME, 256)
# Per-request image cap for APIs that reject very large image counts.
_API_IMAGE_CAP = {
    "gpt-4o": 125,
}
_api_cap = _API_IMAGE_CAP.get(MODEL_NAME, 10**9)
MAX_FRAMES_IN_BUFFER = max(1, min(MAX_VISUAL_TOKENS // CFRAME, _api_cap))

TEXT_INPUT_PRICE_PER_1M = MODEL_PRICING[MODEL_NAME]["input"]
TEXT_OUTPUT_PRICE_PER_1M = MODEL_PRICING[MODEL_NAME]["output"]

GEMINI_THINKING_BUDGET = args.gemini_thinking_budget

print(f"[FIFO config] model={MODEL_NAME} cframe={CFRAME} max_visual_tokens={MAX_VISUAL_TOKENS}")
formula_cap = MAX_VISUAL_TOKENS // CFRAME
print(f"  → max_frames_in_buffer = {MAX_FRAMES_IN_BUFFER} "
      f"(formula={formula_cap}, api_cap={_api_cap if _api_cap < 10**9 else 'unbounded'})")
print(f"  chunk_size={CHUNK_SIZE} frame_stride={FRAME_STRIDE}")

task_path = args.task_path
videos_dir = args.frames_dir

safe_model_name = MODEL_NAME.replace(".", "p").replace("-", "_")
result_dir = os.path.join(
    args.output_root, "fifo",
    f"{safe_model_name}_vt{MAX_VISUAL_TOKENS}_cf{CFRAME}_stride{FRAME_STRIDE}_chunk{CHUNK_SIZE}{args.result_dir_suffix}",
)
question_result_dir = os.path.join(result_dir, "question_results")
video_trace_dir = os.path.join(result_dir, "video_traces")
video_summary_dir = os.path.join(result_dir, "video_summaries")
os.makedirs(question_result_dir, exist_ok=True)
os.makedirs(video_trace_dir, exist_ok=True)
os.makedirs(video_summary_dir, exist_ok=True)

_openai_client = None
def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "MISSING_OPENAI_API_KEY"),
        )
    return _openai_client

TERMINAL_QUESTION_STATUSES = {"answered_online", "unanswered"}
VALID_STRICT_CHOICES = {"(A)", "(B)", "(C)", "(D)"}


# ===== Frame / file utils =====
def parse_timestamp(ts: str) -> int:
    parts = ts.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Unrecognized timestamp format: {ts}")
    m, s = parts
    return int(m) * 60 + int(s)


def list_video_frames(video_dir: str) -> List[str]:
    frames = []
    for name in os.listdir(video_dir):
        if name.endswith(".png") and name.startswith("frame_"):
            frames.append(os.path.join(video_dir, name))
    frames.sort()
    return frames


def downsample_frames(frame_paths: List[str], stride: int) -> List[str]:
    if stride <= 1:
        return list(frame_paths)
    return frame_paths[::stride]


def frame_path_to_second(frame_path: str) -> int:
    base = os.path.basename(frame_path)
    idx = int(base.replace("frame_", "").replace(".png", ""))
    return idx - 1


def encode_image_base64(image_path: str) -> str:
    """Base64-encode a frame; for OpenAI, re-encode as JPEG (same resolution) to keep requests small."""
    if MODEL_NAME.startswith("gpt-4o"):
        from PIL import Image
        import io
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def response_to_text(resp) -> str:
    if resp is None:
        return ""
    if _LLM_API_AVAILABLE:
        t = _llm_unified_text(resp)
        if t:
            return t
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    try:
        outputs = getattr(resp, "output", None)
        if outputs:
            texts = []
            for item in outputs:
                content = getattr(item, "content", [])
                for c in content or []:
                    if getattr(c, "type", None) == "output_text":
                        texts.append(getattr(c, "text", ""))
            return "\n".join(texts).strip()
    except Exception:
        pass
    return ""


def estimate_cost_from_response(resp) -> float:
    if _LLM_API_AVAILABLE:
        return _llm_unified_cost(resp, MODEL_NAME)
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0.0
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cost = input_tokens / 1_000_000 * TEXT_INPUT_PRICE_PER_1M
    cost += output_tokens / 1_000_000 * TEXT_OUTPUT_PRICE_PER_1M
    return cost


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and l < r:
        try:
            return json.loads(text[l:r + 1])
        except Exception:
            pass
    raise ValueError(f"Failed to parse JSON from model output:\n{text[:500]}")


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def question_result_path(qid: str) -> str:
    return os.path.join(question_result_dir, f"{qid}.json")


def video_summary_path(video_id: str) -> str:
    return os.path.join(video_summary_dir, f"{video_id}.json")


def video_full_trace_path(video_id: str) -> str:
    return os.path.join(video_trace_dir, f"{video_id}_trace.json")


def question_status_is_terminal(status: str) -> bool:
    return status in TERMINAL_QUESTION_STATUSES


def load_existing_terminal_question_result(qid: str) -> Optional[Dict[str, Any]]:
    data = load_json_if_exists(question_result_path(qid))
    if not data:
        return None
    if question_status_is_terminal(str(data.get("status", "")).strip()):
        return data
    return None


def video_all_questions_processed(expected_qids: List[str]) -> bool:
    if not expected_qids:
        return False
    for qid in expected_qids:
        if load_existing_terminal_question_result(qid) is None:
            return False
    return True


def normalize_choice_answer(text: Any) -> str:
    s = str(text).strip()
    if not s:
        return ""
    patterns = [
        r"^\s*\(?\s*([ABCD])\s*\)?\s*[\.,;:!\?]*\s*$",
        r"^\s*(?:the\s+)?(?:answer|option)\s*(?:is|=|:)??\s*\(?\s*([ABCD])\s*\)?\s*[\.,;:!\?]*\s*$",
        r"\(([ABCD])\)",
        r"\boption\s*([ABCD])\b",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return f"({m.group(1).upper()})"
    return ""


# ===== Question record prep =====
def compute_gt_answer_time(item: Dict[str, Any], end_ref: int) -> int:
    if item["type"] in ["Past", "Present"]:
        return parse_timestamp(item["ask_time"])
    return end_ref


def prepare_question_records(video_items, max_processed_sec):
    questions = []
    for item in video_items:
        qid = item["uid"]
        existing_terminal = load_existing_terminal_question_result(qid)
        beg_ref_str, end_ref_str = item["time_reference"].split("-")
        beg_ref = parse_timestamp(beg_ref_str)
        end_ref = parse_timestamp(end_ref_str)
        ask_time_raw = parse_timestamp(item["ask_time"])
        ask_time = min(ask_time_raw, max_processed_sec)
        gt_answer_time_raw = compute_gt_answer_time(item, end_ref)
        gt_answer_time = min(gt_answer_time_raw, max_processed_sec)

        base_record = {
            "qid": qid, "video_id": item["video_id"], "question": item["question"],
            "gt_answer": item["answer"], "task_type": item["type"],
            "ask_time_raw": ask_time_raw, "ask_time": ask_time,
            "beg_ref": beg_ref, "end_ref": end_ref,
            "gt_answer_time_raw": gt_answer_time_raw, "gt_answer_time": gt_answer_time,
            "video_last_processed_sec": max_processed_sec,
            "status": "unseen",
            "pred_answer": "", "pred_answer_raw": "", "pred_answer_strict": "",
            "answer_time": None, "answer_source": "", "evidence": "",
            "decision_history": [], "loaded_from_existing_result": False,
        }
        if existing_terminal is not None:
            ep = str(existing_terminal.get("pred_answer", ""))
            base_record.update({
                "status": str(existing_terminal.get("status", "unseen")),
                "pred_answer": ep,
                "pred_answer_raw": str(existing_terminal.get("pred_answer_raw", ep)),
                "pred_answer_strict": str(existing_terminal.get("pred_answer_strict", normalize_choice_answer(ep))),
                "answer_time": existing_terminal.get("answer_time"),
                "answer_source": str(existing_terminal.get("answer_source", "")),
                "evidence": str(existing_terminal.get("evidence", "")),
                "decision_history": existing_terminal.get("decision_history", []),
                "loaded_from_existing_result": True,
            })
        questions.append(base_record)
    questions.sort(key=lambda x: (x["ask_time"], x["qid"]))
    return questions


# ===== Question id obfuscation =====
import hashlib

def obfuscate_qid(qid: str) -> str:
    """Map a qid to an opaque hash so the model never sees the Past/Present/Future suffix."""
    h = hashlib.md5(qid.encode("utf-8")).hexdigest()[:6]
    return f"q{h}"


# ===== Backend abstraction =====
# API backend (llm_api) or local HuggingFace backend, created lazily.
MODEL_BACKEND = None


def _is_oss_model(name: str) -> bool:
    return name.startswith(("qwen2.5-vl", "llava-onevision", "internvl3"))


def get_backend():
    """Return the global model backend, initialising it on first use."""
    global MODEL_BACKEND
    if MODEL_BACKEND is not None:
        return MODEL_BACKEND
    if _is_oss_model(MODEL_NAME):
        MODEL_BACKEND = HFBackend(MODEL_NAME)
    else:
        MODEL_BACKEND = APIBackend()
    return MODEL_BACKEND


class APIBackend:
    """llm_api wrapper for API models. call() returns (text, in_tokens, out_tokens, cost)."""

    def call(self, system_prompt: str, content: List[Dict[str, Any]],
             max_output_tokens: int = 4096) -> Tuple[str, int, int, float]:
        max_retries = 5
        base_delay = 2.0
        if _LLM_API_AVAILABLE:
            resp = _llm_call_llm(
                model=MODEL_NAME,
                system_prompt=system_prompt,
                user_content=content,
                max_output_tokens=max_output_tokens,
                reasoning_effort=REASONING_EFFORT,
                temperature=0.0,
                seed=42,
                gemini_thinking_budget=GEMINI_THINKING_BUDGET,
                max_retries=max_retries,
                base_delay=base_delay,
            )
        else:
            client = _get_openai_client()
            for attempt in range(max_retries):
                try:
                    resp = client.responses.create(
                        model=MODEL_NAME,
                        input=[
                            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                            {"role": "user", "content": content},
                        ],
                        reasoning={"effort": REASONING_EFFORT},
                        max_output_tokens=max_output_tokens,
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"[APIBackend] retry {attempt+1}: {e} (sleep {delay:.1f}s)")
                    time.sleep(delay)
        text = response_to_text(resp)
        cost = estimate_cost_from_response(resp)
        # Best-effort token usage extraction (only API path has this).
        in_tok = 0
        out_tok = 0
        try:
            from llm_api import unified_usage as _unified_usage
            in_tok, out_tok = _unified_usage(resp)
        except Exception:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                in_tok = getattr(usage, "input_tokens", 0) or 0
                out_tok = getattr(usage, "output_tokens", 0) or 0
        return text, int(in_tok), int(out_tok), float(cost)


# Default HuggingFace checkpoints (override with --hf_model_path).
_OSS_DEFAULT_PATHS = {
    "qwen2.5-vl-7b": [
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ],
    "llava-onevision-7b": [
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    ],
    "internvl3-8b": [
        "OpenGVLab/InternVL3-8B",
    ],
}


def _resolve_hf_path(model_name: str) -> str:
    if args.hf_model_path:
        return args.hf_model_path
    for cand in _OSS_DEFAULT_PATHS.get(model_name, []):
        if os.path.isdir(cand) or "/" in cand:
            return cand
    raise RuntimeError(f"No HF path configured for {model_name}; pass --hf_model_path.")


class HFBackend:
    """Local HuggingFace backend for Qwen2.5-VL, LLaVA-OneVision and InternVL3."""

    def __init__(self, model_name: str):
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"torch import failed: {e}") from e

        path = _resolve_hf_path(model_name)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.hf_dtype]
        device_map = args.hf_device

        self.model_name = model_name
        self.path = path
        self.torch = torch
        self.dtype = dtype
        self.device_map = device_map
        self.family = (
            "qwen" if model_name.startswith("qwen2.5-vl") else
            "llava-ov" if model_name.startswith("llava-onevision") else
            "internvl"
        )

        print(f"[HFBackend] loading {model_name} from {path} (dtype={args.hf_dtype}, device={device_map})...")
        if self.family == "qwen":
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(
                path, trust_remote_code=True, max_pixels=args.hf_max_pixels,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
            ).eval()
        elif self.family == "llava-ov":
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
            ).eval()
        elif self.family == "internvl":
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            self.model = AutoModel.from_pretrained(
                path, torch_dtype=dtype, device_map=device_map,
                trust_remote_code=True, low_cpu_mem_usage=True,
            ).eval()
        print(f"[HFBackend] {model_name} loaded.")

    # ---------- helpers ----------
    @staticmethod
    def _parse_content(content: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Split a content list into (text_blocks, image_paths)."""
        text_blocks: List[str] = []
        image_paths: List[str] = []
        for it in content:
            t = it.get("type", "")
            if t == "input_text":
                text_blocks.append(it.get("text", ""))
            elif t == "input_image":
                ip = it.get("image_path")
                if ip:
                    image_paths.append(ip)
        return text_blocks, image_paths

    # ---------- per-family generate ----------
    def call(self, system_prompt: str, content: List[Dict[str, Any]],
             max_output_tokens: int = 4096) -> Tuple[str, int, int, float]:
        text_blocks, image_paths = self._parse_content(content)
        user_text = "\n".join(text_blocks)
        if self.family == "qwen":
            return self._call_qwen(system_prompt, user_text, image_paths, max_output_tokens)
        if self.family == "llava-ov":
            return self._call_llava_ov(system_prompt, user_text, image_paths, max_output_tokens)
        if self.family == "internvl":
            return self._call_internvl(system_prompt, user_text, image_paths, max_output_tokens)
        raise RuntimeError(f"unknown HF family: {self.family}")

    def _call_qwen(self, system_prompt: str, user_text: str, image_paths: List[str],
                   max_output_tokens: int) -> Tuple[str, int, int, float]:
        from PIL import Image
        torch = self.torch
        # Build Qwen messages: one {"type":"image"} per frame, then the user text.
        content = []
        images = []
        for fp in image_paths:
            content.append({"type": "image", "image": fp})
            images.append(Image.open(fp).convert("RGB"))
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images if images else None,
                                padding=True, return_tensors="pt")
        device = self.model.device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_output_tokens,
                                      do_sample=False, temperature=0.0)
        n_in = inputs["input_ids"].shape[1]
        gen_ids = out[0][n_in:]
        text_out = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text_out, int(n_in), int(len(gen_ids)), 0.0

    def _call_llava_ov(self, system_prompt: str, user_text: str, image_paths: List[str],
                       max_output_tokens: int) -> Tuple[str, int, int, float]:
        from PIL import Image
        torch = self.torch
        # LLaVA-OneVision video mode: pass frames as a "video" entry (list of PIL images).
        frames = [Image.open(fp).convert("RGB") for fp in image_paths]
        conv = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": user_text},
            ]},
        ]
        prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self.processor(
            videos=[frames] if frames else None,
            text=prompt,
            return_tensors="pt",
        )
        device = self.model.device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_output_tokens,
                                      do_sample=False, temperature=0.0)
        n_in = inputs["input_ids"].shape[1]
        gen = out[0][n_in:]
        text_out = self.processor.batch_decode(gen.unsqueeze(0), skip_special_tokens=True)[0].strip()
        return text_out, int(n_in), int(len(gen)), 0.0

    def _call_internvl(self, system_prompt: str, user_text: str, image_paths: List[str],
                       max_output_tokens: int) -> Tuple[str, int, int, float]:
        # InternVL3: stack per-frame pixel_values; one patch per frame.
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        torch = self.torch

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if image_paths:
            tensors = [transform(Image.open(fp)) for fp in image_paths]
            pixel_values = torch.stack(tensors).to(self.dtype).to(self.model.device)
            num_patches_list = [1] * len(image_paths)
        else:
            pixel_values = None
            num_patches_list = None

        full_question = (
            f"{system_prompt}\n\n{user_text}"
            if system_prompt else user_text
        )
        # Insert one <image> token per frame so the model knows to attend each.
        if image_paths:
            image_tags = "".join([f"Frame-{i+1}: <image>\n" for i in range(len(image_paths))])
            full_question = f"{system_prompt}\n\n{image_tags}\n{user_text}"

        gen_config = dict(max_new_tokens=max_output_tokens, do_sample=False, temperature=0.0)
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer, pixel_values, full_question, gen_config,
                num_patches_list=num_patches_list,
                history=None, return_history=False,
            )
        # Token usage not directly exposed by InternVL chat; we approximate via tokenizer.
        in_tok = len(self.tokenizer(full_question).input_ids)
        out_tok = len(self.tokenizer(response).input_ids)
        return str(response).strip(), int(in_tok), int(out_tok), 0.0


# ===== FIFO buffer =====
def update_fifo_buffer(buffer: List[str], new_frames: List[str], max_frames: int) -> List[str]:
    """Append new_frames to buffer, then evict oldest until len(buffer) <= max_frames."""
    buffer = list(buffer) + list(new_frames)
    if len(buffer) > max_frames:
        buffer = buffer[-max_frames:]
    return buffer


# ===== Per-question prompt + LLM call =====
SYSTEM_PROMPT = """You are a streaming video question-answering assistant.

You receive:
- A buffer of recent frames from a video stream, in chronological order. Frame timestamps (in seconds from the start of the video) are given before each frame.
- A list of one or more multiple-choice questions about the video. Each question has a qid (opaque hash) and four options (A), (B), (C), (D).

For each question, decide whether the buffered frames provide enough evidence to answer.
- If yes, output the correct option letter.
- If no (the question refers to an event not yet observed, or the relevant moment was already evicted from the buffer and cannot be inferred), defer. The question will be retried with a newer buffer in a later chunk.
Different questions may have different decisions in the same call.

Output a single JSON object with this schema:
{
  "question_updates": [
    {
      "qid": "<qid string from the input list>",
      "decision": "answer" or "defer",
      "answer": "(A)/(B)/(C)/(D)" when decision=answer, empty string when decision=defer,
      "reasoning": "brief one-sentence justification"
    },
    ...
  ]
}

Rules:
- Include exactly one entry per active question, matched by qid.
- If decision="answer", the answer field MUST be exactly one of (A), (B), (C), (D).
- If decision="defer", the answer field MUST be an empty string.
- Use only the provided frames as evidence. Do not invent details. Do not use general world knowledge.
- Do not refuse based on inability to see future frames; if the buffered evidence already determines the answer, commit.
"""


def build_image_inputs(frame_paths: List[str]) -> List[Dict[str, Any]]:
    """Build image inputs (image_path for the HF backend, base64 for APIs)."""
    items = []
    for fp in frame_paths:
        sec = frame_path_to_second(fp)
        items.append({"type": "input_text", "text": f"Frame timestamp: {sec}s"})
        if _is_oss_model(MODEL_NAME):
            items.append({"type": "input_image", "image_path": fp})
        else:
            b64 = encode_image_base64(fp)
            items.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
                "detail": "auto",
            })
    return items


def call_fifo_chunk_batch(active_questions: List[Dict[str, Any]],
                          fifo_frames: List[str]) -> Tuple[Dict[str, Dict[str, Any]], float, str, int, int]:
    """One LLM call per chunk for all active questions. Returns (results_by_qid, cost, raw_text, in_tok, out_tok)."""
    obf_to_qid = {obfuscate_qid(q["qid"]): q["qid"] for q in active_questions}
    serialized = [{"qid": obfuscate_qid(q["qid"]), "question": q["question"]} for q in active_questions]

    pre_blocks = [
        f"Buffered frames in chronological order ({len(fifo_frames)} frames; the oldest may have been evicted from the buffer):",
    ]
    post_blocks = [
        f"\nActive questions ({len(serialized)}):",
        json.dumps(serialized, ensure_ascii=False, indent=2),
        "\nReturn the JSON object now. Include exactly one update per qid above.",
    ]

    content: List[Dict[str, Any]] = []
    for t in pre_blocks:
        content.append({"type": "input_text", "text": t})
    if fifo_frames:
        content.extend(build_image_inputs(fifo_frames))
    for t in post_blocks:
        content.append({"type": "input_text", "text": t})

    backend = get_backend()
    raw, in_tok, out_tok, cost = backend.call(
        system_prompt=SYSTEM_PROMPT,
        content=content,
        max_output_tokens=args.max_output_tokens,
    )

    try:
        parsed = extract_json(raw)
    except ValueError:
        parsed = {"question_updates": []}

    results: Dict[str, Dict[str, Any]] = {}
    updates = parsed.get("question_updates", []) if isinstance(parsed, dict) else []
    if isinstance(updates, list):
        for upd in updates:
            if not isinstance(upd, dict):
                continue
            obf_qid = str(upd.get("qid", "")).strip()
            qid = obf_to_qid.get(obf_qid, obf_qid)
            if qid not in obf_to_qid.values() or qid in results:
                continue
            decision = str(upd.get("decision", "defer")).strip().lower()
            if decision == "need_more_frames":
                decision = "defer"
            if decision not in {"answer", "defer"}:
                decision = "defer"
            raw_answer = str(upd.get("answer", "")).strip()
            strict_answer = normalize_choice_answer(raw_answer)
            if decision != "answer":
                raw_answer = ""
                strict_answer = ""
            results[qid] = {
                "decision": decision,
                "answer": strict_answer if strict_answer else raw_answer,
                "raw_answer": raw_answer,
                "strict_answer": strict_answer,
                "reasoning": str(upd.get("reasoning", "")).strip(),
            }

    # Default to defer for any active question the model omitted.
    for q in active_questions:
        if q["qid"] not in results:
            results[q["qid"]] = {
                "decision": "defer", "answer": "", "raw_answer": "",
                "strict_answer": "", "reasoning": "(no update)",
            }
    return results, cost, raw, in_tok, out_tok


# ===== Main video processing =====
def evaluate_one_video(video_id: str, video_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    video_dir = os.path.join(videos_dir, video_id)
    all_frame_paths_1fps = list_video_frames(video_dir)
    if not all_frame_paths_1fps:
        raise RuntimeError(f"No frames found in {video_dir}")

    frame_paths = downsample_frames(all_frame_paths_1fps, FRAME_STRIDE)
    if not frame_paths:
        raise RuntimeError(f"No frames after downsampling in {video_dir}")

    frame_seconds = [frame_path_to_second(p) for p in frame_paths]
    max_processed_sec = frame_seconds[-1]
    chunks = chunk_list(frame_paths, CHUNK_SIZE)
    questions = prepare_question_records(video_items, max_processed_sec=max_processed_sec)
    qid_to_question = {q["qid"]: q for q in questions}

    fifo_buffer: List[str] = []
    video_cost = 0.0
    trace = []
    pending_qids: List[str] = []
    next_question_idx = 0
    preloaded_terminal_qids = [q["qid"] for q in questions if question_status_is_terminal(q["status"])]

    for chunk_idx, chunk in enumerate(chunks):
        chunk_secs = [frame_path_to_second(p) for p in chunk]
        chunk_end_sec = max(chunk_secs)

        # Update FIFO buffer with the new chunk's frames.
        fifo_buffer = update_fifo_buffer(fifo_buffer, chunk, MAX_FRAMES_IN_BUFFER)

        # Activate newly arrived questions.
        newly_arrived_qids = []
        while next_question_idx < len(questions) and questions[next_question_idx]["ask_time"] <= chunk_end_sec:
            q = questions[next_question_idx]
            if q["status"] == "unseen":
                q["status"] = "pending"
                pending_qids.append(q["qid"])
                newly_arrived_qids.append(q["qid"])
            next_question_idx += 1

        pending_qid_set = set(pending_qids)
        active_questions = [q for q in questions if q["qid"] in pending_qid_set and q["status"] == "pending"]


        answered_now: List[str] = []
        per_q_calls: List[Dict[str, Any]] = []

        if active_questions:
            try:
                results_by_qid, call_cost, raw_resp, in_tok, out_tok = call_fifo_chunk_batch(
                    active_questions, fifo_buffer,
                )
                video_cost += call_cost
            except Exception as e:
                print(f"[fifo_chunk_batch chunk={chunk_idx}] error: {e}")
                traceback.print_exc()
                results_by_qid = {q["qid"]: {"decision": "defer", "answer": "", "raw_answer": "",
                                              "strict_answer": "", "reasoning": f"error: {e}"}
                                  for q in active_questions}
                call_cost = 0.0; raw_resp = ""; in_tok = 0; out_tok = 0

            # Apply outcomes.
            for q in active_questions:
                qid = q["qid"]
                out = results_by_qid[qid]
                per_q_calls.append({
                    "qid": qid,
                    "decision": out["decision"],
                    "answer": out["answer"],
                    "reasoning": out["reasoning"],
                })
                q["decision_history"].append({
                    "chunk_idx": chunk_idx,
                    "chunk_end_sec": chunk_end_sec,
                    "decision": out["decision"],
                    "evidence": out["reasoning"],
                })
                if out["decision"] == "answer" and out["strict_answer"]:
                    q["status"] = "answered_online"
                    q["pred_answer_raw"] = out["raw_answer"]
                    q["pred_answer_strict"] = out["strict_answer"]
                    q["pred_answer"] = out["strict_answer"]
                    q["answer_time"] = chunk_end_sec
                    q["answer_source"] = "fifo_baseline"
                    q["evidence"] = out["reasoning"]
                    answered_now.append(qid)

            pending_qids = [qid for qid in pending_qids if qid_to_question[qid]["status"] == "pending"]

        trace.append({
            "chunk_idx": chunk_idx,
            "chunk_secs": chunk_secs,
            "newly_arrived_qids": newly_arrived_qids,
            "active_qids": [q["qid"] for q in active_questions],
            "answered_now_qids": answered_now,
            "fifo_buffer_size": len(fifo_buffer),
            "fifo_buffer_oldest_sec": frame_path_to_second(fifo_buffer[0]) if fifo_buffer else None,
            "fifo_buffer_newest_sec": frame_path_to_second(fifo_buffer[-1]) if fifo_buffer else None,
            "per_q_calls": per_q_calls,
            "accumulated_video_cost": video_cost,
        })

    # End of video: mark remaining as unanswered.
    for q in questions:
        if q["status"] in {"unseen", "pending"}:
            q["status"] = "unanswered"
            q["pred_answer"] = ""
            q["pred_answer_raw"] = ""
            q["pred_answer_strict"] = ""
            q["answer_time"] = None
            q["answer_source"] = "video_end_unanswered"
            q["evidence"] = ""

    return {
        "video_id": video_id,
        "status": "finished",
        "num_questions": len(questions),
        "num_answered_online": len([q for q in questions if q["status"] == "answered_online"]),
        "num_unanswered": len([q for q in questions if q["status"] == "unanswered"]),
        "sampled_num_frames": len(frame_paths),
        "sampled_frame_seconds": frame_seconds,
        "chunk_size": CHUNK_SIZE,
        "frame_stride": FRAME_STRIDE,
        "reasoning_effort": REASONING_EFFORT,
        "max_visual_tokens": MAX_VISUAL_TOKENS,
        "cframe": CFRAME,
        "max_frames_in_buffer": MAX_FRAMES_IN_BUFFER,
        "video_cost": video_cost,
        "preloaded_terminal_qids": preloaded_terminal_qids,
        "questions": questions,
        "trace": trace,
    }


def make_question_output_record(question, video_result):
    return {
        "qid": question["qid"], "video_id": question["video_id"],
        "question": question["question"], "gt_answer": question["gt_answer"],
        "task_type": question["task_type"], "status": question["status"],
        "pred_answer": question["pred_answer"],
        "pred_answer_raw": question.get("pred_answer_raw", question["pred_answer"]),
        "pred_answer_strict": question.get("pred_answer_strict", normalize_choice_answer(question["pred_answer"])),
        "answer_time": question["answer_time"], "answer_source": question["answer_source"],
        "evidence": question["evidence"],
        "ask_time_raw": question["ask_time_raw"], "ask_time": question["ask_time"],
        "beg_ref": question["beg_ref"], "end_ref": question["end_ref"],
        "gt_answer_time_raw": question["gt_answer_time_raw"], "gt_answer_time": question["gt_answer_time"],
        "video_last_processed_sec": question["video_last_processed_sec"],
        "loaded_from_existing_result": question["loaded_from_existing_result"],
        "decision_history": question["decision_history"],
        "chunk_size": video_result.get("chunk_size"),
        "frame_stride": video_result.get("frame_stride"),
        "reasoning_effort": video_result.get("reasoning_effort"),
        "max_visual_tokens": video_result.get("max_visual_tokens"),
        "cframe": video_result.get("cframe"),
        "max_frames_in_buffer": video_result.get("max_frames_in_buffer"),
        "sampled_num_frames": video_result.get("sampled_num_frames"),
        "sampled_frame_seconds": video_result.get("sampled_frame_seconds"),
        "video_run_status": video_result.get("status"),
        "video_cost": video_result.get("video_cost"),
    }


def write_video_artifacts(video_result):
    video_id = video_result["video_id"]
    with open(video_full_trace_path(video_id), "w") as f:
        json.dump(video_result.get("trace", []), f, indent=2, ensure_ascii=False)

    video_summary = {
        "video_id": video_id, "status": video_result.get("status"),
        "num_questions": video_result.get("num_questions"),
        "num_answered_online": video_result.get("num_answered_online"),
        "num_unanswered": video_result.get("num_unanswered"),
        "video_cost": video_result.get("video_cost"),
        "max_visual_tokens": video_result.get("max_visual_tokens"),
        "cframe": video_result.get("cframe"),
        "max_frames_in_buffer": video_result.get("max_frames_in_buffer"),
        "sampled_num_frames": video_result.get("sampled_num_frames"),
        "chunk_size": video_result.get("chunk_size"),
        "frame_stride": video_result.get("frame_stride"),
    }
    with open(video_summary_path(video_id), "w") as f:
        json.dump(video_summary, f, indent=2, ensure_ascii=False)

    for q in video_result.get("questions", []):
        with open(question_result_path(q["qid"]), "w") as f:
            json.dump(make_question_output_record(q, video_result), f, indent=2, ensure_ascii=False)


# ===== Main =====
def main():
    with open(task_path, "r") as f:
        data = json.load(f)

    video_to_items = defaultdict(list)
    for item in data:
        video_to_items[item["video_id"]].append(item)

    video_ids = list(video_to_items.keys())
    random.seed(SEED)
    random.shuffle(video_ids)

    SUBSET_PATH = args.subset_path
    if SUBSET_PATH:
        with open(SUBSET_PATH, "r") as f:
            subset_video_ids = set(json.load(f)["video_ids"])
        video_ids = [v for v in video_ids if v in subset_video_ids]
    print(f"Subset: {len(video_ids)}/{len(video_to_items)} videos (from {SUBSET_PATH})")

    cost_lock = threading.Lock()
    global_total_cost = 0.0
    summary_results = []

    def process_video(idx, video_id):
        nonlocal global_total_cost

        expected_qids = [item["uid"] for item in video_to_items[video_id]]
        if video_all_questions_processed(expected_qids):
            print(f"[SKIP] video_id={video_id} all questions already processed.")
            with cost_lock:
                summary_results.append({"video_id": video_id, "status": "skipped_already_processed",
                                        "num_questions": len(expected_qids)})
            return

        with cost_lock:
            current_cost = global_total_cost

        print(f"\n===== Video {idx + 1}/{len(video_ids)}: {video_id} =====")
        print(f"Cost: ${current_cost:.4f} | Questions: {len(video_to_items[video_id])}")

        try:
            res = evaluate_one_video(video_id, video_to_items[video_id])
            write_video_artifacts(res)
            with cost_lock:
                summary_results.append({
                    "video_id": res["video_id"], "status": res["status"],
                    "num_questions": res.get("num_questions"),
                    "num_answered_online": res.get("num_answered_online"),
                    "num_unanswered": res.get("num_unanswered"),
                    "video_cost": res.get("video_cost"),
                })
                global_total_cost += res["video_cost"]
                current_cost = global_total_cost
            print(f"[DONE] {video_id} | Cost: ${res['video_cost']:.4f} | Total: ${current_cost:.4f}")
            print(f"  Answered: {res.get('num_answered_online',0)} | Unanswered: {res.get('num_unanswered',0)}")
        except Exception as e:
            with cost_lock:
                summary_results.append({"video_id": video_id, "status": "error", "error": str(e)})
            print(f"[ERROR] {video_id}: {e}")
            traceback.print_exc()

    num_workers = args.num_workers
    if num_workers <= 1:
        for idx, video_id in enumerate(video_ids):
            process_video(idx, video_id)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for idx, video_id in enumerate(video_ids):
                futures[executor.submit(process_video, idx, video_id)] = video_id
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[UNEXPECTED ERROR] {futures[fut]}: {e}")

    with cost_lock:
        final_cost = global_total_cost
        final_results = list(summary_results)

    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "num_videos_processed_or_skipped": len(final_results),
            "total_cost": final_cost,
            "model_name": MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "frame_stride": FRAME_STRIDE,
            "reasoning_effort": REASONING_EFFORT,
            "num_workers": num_workers,
            "hf_max_pixels": args.hf_max_pixels,
            "hf_dtype": args.hf_dtype,
            "hf_device": args.hf_device,
            "max_visual_tokens": MAX_VISUAL_TOKENS,
            "cframe": CFRAME,
            "max_frames_in_buffer": MAX_FRAMES_IN_BUFFER,
            "subset_path": SUBSET_PATH,
            "video_order": video_ids,
            "result_dir": result_dir,
            "results": final_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n===== ALL DONE =====")
    print(f"Processed/skipped: {len(final_results)} videos")
    print(f"Total cost: ${final_cost:.4f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
