import os
import json
import base64
import re
import random
import traceback
import time
import threading
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sys.path.insert(0, os.path.join(_REPO_ROOT, "pivot"))
DEFAULT_TASK_PATH = os.path.join(_REPO_ROOT, "benchmarks", "lvbench_ref2stream", "data", "lvbench_ref2stream.json")
DEFAULT_FRAMES_DIR = os.environ.get("LVBENCH_FRAMES_DIR", os.path.join(_REPO_ROOT, "data", "lvbench", "frames_1fps"))
DEFAULT_OUTPUT_ROOT = os.path.join(_REPO_ROOT, "results")

from openai import OpenAI

# Route model calls through llm_api (OpenAI + Gemini).
try:
    from llm_api import call_llm as _llm_call_llm, unified_text as _llm_unified_text, unified_cost as _llm_unified_cost, unified_usage as _llm_unified_usage, provider_for as _llm_provider_for
    _LLM_API_AVAILABLE = True
except Exception as _llm_e:
    _LLM_API_AVAILABLE = False
    print(f"[llm_api] import failed: {_llm_e}")


MODEL_PRICING = {
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
}

parser = argparse.ArgumentParser(
    description="PIVOT: planner-controlled streaming agent on LVBench-Ref2Stream."
)
parser.add_argument("--chunk_size", type=int, default=16, help="Number of frames per chunk.")
parser.add_argument("--frame_stride", type=int, default=2, help="Use every N-th frame from 1fps.")
parser.add_argument(
    "--model_name", type=str, default="gemini-3.1-flash-lite-preview",
    choices=sorted(MODEL_PRICING.keys()),
)
parser.add_argument(
    "--reasoning_effort", type=str, default="medium",
    choices=["low", "medium", "high"],
)
parser.add_argument("--max_memory_tokens", type=int, default=50000,
                    help="Total token budget for memory (scene_log + character_registry + question_evidence).")
parser.add_argument("--max_tool_rounds", type=int, default=5,
                    help="Maximum tool call rounds per question before auto-deferring.")
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--subset_path", type=str, default="",
                    help='Optional JSON {"video_ids": [...]} restricting the run to a video subset (default: all videos).')
parser.add_argument("--result_dir_suffix", type=str, default="", help="appended to result_dir name")
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
MAX_MEMORY_TOKENS = args.max_memory_tokens
MAX_TOOL_ROUNDS = args.max_tool_rounds
MAX_RETRIEVER_CALLS = 2  # per question per chunk

RETRIEVER_MODEL_NAME = MODEL_NAME
RETRIEVER_REASONING_EFFORT = "low"

TEXT_INPUT_PRICE_PER_1M = MODEL_PRICING[MODEL_NAME]["input"]
TEXT_OUTPUT_PRICE_PER_1M = MODEL_PRICING[MODEL_NAME]["output"]

# Per-video API call / token statistics (thread-safe).
_video_stats_lock = threading.Lock()
_video_stats: Dict[str, Dict[str, int]] = {}
_current_video_id = threading.local()


def _record_api_call(resp) -> None:
    vid = getattr(_current_video_id, "value", None)
    if vid is None:
        return
    try:
        in_tok, out_tok = _llm_unified_usage(resp) if _LLM_API_AVAILABLE else (0, 0)
    except Exception:
        in_tok, out_tok = 0, 0
    with _video_stats_lock:
        s = _video_stats.setdefault(vid, {"api_calls": 0, "input_tokens": 0, "output_tokens": 0})
        s["api_calls"] += 1
        s["input_tokens"] += int(in_tok or 0)
        s["output_tokens"] += int(out_tok or 0)

task_path = args.task_path
videos_dir = args.frames_dir

safe_model_name = MODEL_NAME.replace(".", "p").replace("-", "_")
result_dir = os.path.join(
    args.output_root, "pivot",
    f"{safe_model_name}{args.result_dir_suffix}",
)
question_result_dir = os.path.join(result_dir, "question_results")
video_trace_dir = os.path.join(result_dir, "video_traces")
video_summary_dir = os.path.join(result_dir, "video_summaries")
os.makedirs(question_result_dir, exist_ok=True)
os.makedirs(video_trace_dir, exist_ok=True)
os.makedirs(video_summary_dir, exist_ok=True)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "MISSING_OPENAI_API_KEY"),
)

TERMINAL_QUESTION_STATUSES = {"answered_online", "unanswered"}
VALID_STRICT_CHOICES = {"(A)", "(B)", "(C)", "(D)"}


# ===== Utils =====
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
                content = getattr(item, "content", None)
                if content:
                    for c in content:
                        if getattr(c, "type", None) == "output_text":
                            texts.append(getattr(c, "text", ""))
                elif getattr(item, "type", None) == "message":
                    content = getattr(item, "content", [])
                    for c in content:
                        if getattr(c, "type", None) == "output_text":
                            texts.append(getattr(c, "text", ""))
            return "\n".join(texts).strip()
    except Exception:
        pass
    return ""


def estimate_cost_from_response(resp) -> float:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0.0
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cost = input_tokens / 1_000_000 * TEXT_INPUT_PRICE_PER_1M
    cost += output_tokens / 1_000_000 * TEXT_OUTPUT_PRICE_PER_1M
    return cost


def _fix_json_string(raw: str) -> str:
    fixed = raw.replace('\\\\"', '\\"')
    fixed = re.sub(r'\\\\+"', r'\\"', fixed)
    result = []
    in_string = False
    escape = False
    i = 0
    while i < len(fixed):
        ch = fixed[i]
        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if in_string and ch in ('\n', '\r'):
            rest = fixed[i:].lstrip('\n\r\t ')
            if rest and rest[0] in ('}', ']') or re.match(r'"[^"]*"\s*:', rest) or re.match(r',\s*["\[{}\]]', rest):
                result.append('"')
                in_string = False
            else:
                result.append(' ')
            i += 1
            continue
        if in_string and ch == '\t':
            result.append(' ')
            i += 1
            continue
        result.append(ch)
        i += 1
    fixed = ''.join(result)
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    return fixed


def _repair_truncated_json(raw: str) -> str:
    raw = _fix_json_string(raw)
    in_string = False
    escape = False
    stack = []
    for ch in raw:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    closing = ""
    if in_string:
        closing += '"'
    for bracket in reversed(stack):
        if bracket == '{':
            closing += '}'
        elif bracket == '[':
            closing += ']'
    repaired = raw + closing
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    for _ in range(20):
        try:
            json.loads(repaired)
            return repaired
        except Exception:
            pass
        m = re.search(r',\s*\{[^{]*$', repaired)
        if not m:
            m = re.search(r',\s*"[^"]*"?\s*[}\]]*\s*$', repaired)
        if not m:
            m = re.search(r',\s*[^,]*$', repaired)
        if m:
            repaired = repaired[:m.start()]
            repaired = repaired.rstrip()
            opens = repaired.count('[') - repaired.count(']')
            opens_b = repaired.count('{') - repaired.count('}')
            if opens > 0:
                repaired += ']' * opens
            if opens_b > 0:
                repaired += '}' * opens_b
            continue
        break
    return repaired


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and l < r:
        chunk = text[l:r + 1]
        try:
            return json.loads(chunk)
        except Exception:
            pass
        try:
            return json.loads(_fix_json_string(chunk))
        except Exception:
            pass
    if l != -1:
        truncated = text[l:]
        try:
            return json.loads(_repair_truncated_json(truncated))
        except Exception:
            pass
    try:
        return json.loads(_fix_json_string(text))
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


def video_input_trace_path(video_id: str) -> str:
    return os.path.join(video_trace_dir, f"{video_id}_input_trace.json")


def question_status_is_terminal(status: str) -> bool:
    return status in TERMINAL_QUESTION_STATUSES


def load_existing_terminal_question_result(qid: str) -> Optional[Dict[str, Any]]:
    data = load_json_if_exists(question_result_path(qid))
    if not data:
        return None
    status = str(data.get("status", "")).strip()
    if question_status_is_terminal(status):
        return data
    return None


def video_all_questions_processed(expected_qids: List[str]) -> bool:
    if not expected_qids:
        return False
    for qid in expected_qids:
        if load_existing_terminal_question_result(qid) is None:
            return False
    return True


def sanitize_qid(qid: str) -> str:
    """Deterministic opaque identifier for a question. Hides task type entirely.
    E.g., '55_Past' / '55_Present' / '55_Future' all map to distinct opaque hex hashes.
    """
    import hashlib
    h = hashlib.md5(str(qid).encode("utf-8")).hexdigest()[:6]
    return f"q{h}"


def strip_question_options(question_text: str) -> str:
    """Strip answer options from question text, returning only the question stem."""
    m = re.search(r'\n\s*\(A\)', question_text)
    if m:
        return question_text[:m.start()].strip()
    return question_text.strip()


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


# ===== Memory: Token-budgeted reservoir sampling with character registry =====
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def memory_token_count(mem: Dict[str, Any]) -> int:
    total = 80
    for entry in mem.get("scene_log", []):
        total += estimate_tokens(f"[{entry['time_range']}]: {entry['caption']}\n")
    for char_entry in mem.get("character_registry", []):
        total += estimate_tokens(f"  {char_entry['name']}: {char_entry['description']}\n")
    for qid, ev in mem.get("question_evidence", {}).items():
        total += estimate_tokens(f"  [{qid}]: {ev}\n")
    return total


def init_memory_state() -> Dict[str, Any]:
    return {
        "scene_log": [],
        "character_registry": [],
        "question_evidence": {},
        "_n_scenes_seen": 0,
        "_n_chars_seen": 0,
    }


def _reservoir_accept(reservoir_size: int, n_seen: int, rng: random.Random) -> Tuple[bool, Optional[int]]:
    if reservoir_size == 0:
        return True, None
    if rng.random() < reservoir_size / n_seen:
        return True, rng.randint(0, reservoir_size - 1)
    return False, None


def _trim_memory_to_budget(mem: Dict[str, Any], rng: random.Random) -> None:
    while memory_token_count(mem) > MAX_MEMORY_TOKENS:
        scene_log = mem["scene_log"]
        if len(scene_log) > 0:
            idx = rng.randint(0, len(scene_log) - 1)
            scene_log.pop(idx)
        else:
            char_reg = mem["character_registry"]
            qe = mem["question_evidence"]
            if len(char_reg) > 0:
                idx = rng.randint(0, len(char_reg) - 1)
                char_reg.pop(idx)
            elif qe:
                longest_qid = max(qe, key=lambda k: len(qe[k]))
                ev = qe[longest_qid]
                half = ev[:len(ev) // 2].strip()
                if half:
                    qe[longest_qid] = half
                else:
                    del qe[longest_qid]
            else:
                break


def normalize_memory_state(mem: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    if not isinstance(mem, dict):
        mem = {}
    scene_log = mem.get("scene_log", [])
    character_registry = mem.get("character_registry", [])
    question_evidence = mem.get("question_evidence", {})
    if not isinstance(scene_log, list):
        scene_log = []
    if not isinstance(character_registry, list):
        character_registry = []
    if not isinstance(question_evidence, dict):
        question_evidence = {}

    valid_log = []
    for entry in scene_log:
        if isinstance(entry, dict) and "time_range" in entry and "caption" in entry:
            valid_log.append({
                "time_range": str(entry["time_range"]),
                "caption": str(entry["caption"]).strip(),
            })
        elif isinstance(entry, str):
            valid_log.append({"time_range": "?", "caption": entry.strip()})

    valid_chars = []
    seen_names = set()
    for entry in character_registry:
        if isinstance(entry, dict) and "name" in entry and "description" in entry:
            name = str(entry["name"]).strip()
            if name and name not in seen_names:
                seen_names.add(name)
                valid_chars.append({
                    "name": name,
                    "description": str(entry["description"]).strip(),
                })

    norm_qe = {}
    for qid, evidence in question_evidence.items():
        ev = str(evidence).strip()
        if ev:
            norm_qe[str(qid)] = ev

    result = {
        "scene_log": valid_log,
        "character_registry": valid_chars,
        "question_evidence": norm_qe,
        "_n_scenes_seen": mem.get("_n_scenes_seen", len(valid_log)),
        "_n_chars_seen": mem.get("_n_chars_seen", len(valid_chars)),
    }
    _trim_memory_to_budget(result, rng)
    return result


_TIME_RANGE_RE = re.compile(r"(\d+)\s*s?\s*[-–]\s*(\d+)\s*s?")


def _parse_time_range(tr: str) -> Optional[Tuple[int, int]]:
    if not tr:
        return None
    m = _TIME_RANGE_RE.search(str(tr))
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2))
    if end < start:
        start, end = end, start
    return start, end


def add_scene_entry(mem: Dict[str, Any], time_range: str, caption: str, rng: random.Random) -> None:
    """Add one scene-log entry per chunk, using reservoir sampling when over budget."""
    caption = str(caption).strip()
    time_range = str(time_range)
    if not caption:
        return

    mem["_n_scenes_seen"] = mem.get("_n_scenes_seen", len(mem["scene_log"])) + 1
    n = mem["_n_scenes_seen"]
    new_entry = {"time_range": time_range, "caption": caption}

    mem["scene_log"].append(new_entry)

    if memory_token_count(mem) <= MAX_MEMORY_TOKENS:
        return

    # Over budget: reservoir sampling.
    mem["scene_log"].pop()
    k = len(mem["scene_log"])
    accept, replace_idx = _reservoir_accept(k, n, rng)
    if accept and replace_idx is not None:
        mem["scene_log"][replace_idx] = new_entry


def update_character_registry(mem: Dict[str, Any], new_chars: List[Dict[str, str]], rng: random.Random) -> None:
    if not isinstance(new_chars, list):
        return
    existing_names = {c["name"] for c in mem["character_registry"]}
    for char_entry in new_chars:
        if not isinstance(char_entry, dict):
            continue
        name = str(char_entry.get("name", "")).strip()
        desc = str(char_entry.get("description", "")).strip()
        if not name or not desc:
            continue
        if name in existing_names:
            for c in mem["character_registry"]:
                if c["name"] == name:
                    c["description"] = desc
                    break
            continue
        mem["_n_chars_seen"] = mem.get("_n_chars_seen", len(mem["character_registry"])) + 1
        n = mem["_n_chars_seen"]
        mem["character_registry"].append({"name": name, "description": desc})
        existing_names.add(name)
        if memory_token_count(mem) <= MAX_MEMORY_TOKENS:
            continue
        mem["character_registry"].pop()
        existing_names.discard(name)
        k = len(mem["character_registry"])
        accept, replace_idx = _reservoir_accept(k, n, rng)
        if accept and replace_idx is not None:
            evicted = mem["character_registry"][replace_idx]
            existing_names.discard(evicted["name"])
            mem["character_registry"][replace_idx] = {"name": name, "description": desc}
            existing_names.add(name)


def _scene_sort_key(entry: Dict[str, str]) -> int:
    tr = entry.get("time_range", "")
    m = re.match(r'(\d+)', tr)
    return int(m.group(1)) if m else 0


def memory_state_to_text(mem: Dict[str, Any]) -> str:
    if mem["scene_log"]:
        sorted_log = sorted(mem["scene_log"], key=_scene_sort_key)
        log_lines = [f"[{e['time_range']}]: {e['caption']}" for e in sorted_log]
        scene_text = "\n".join(log_lines)
    else:
        scene_text = "(empty - no scenes processed yet)"

    if mem["character_registry"]:
        char_lines = [f"  {c['name']}: {c['description']}" for c in mem["character_registry"]]
        char_text = "\n".join(char_lines)
    else:
        char_text = "(none)"

    if mem["question_evidence"]:
        qe_lines = [f"  [{qid}]: {ev}" for qid, ev in mem["question_evidence"].items()]
        qe_text = "\n".join(qe_lines)
    else:
        qe_text = "(none)"

    return (
        f"(Total memory budget: ~{MAX_MEMORY_TOKENS} tokens shared across the three sections below — "
        "SCENE LOG, CHARACTER REGISTRY, QUESTION EVIDENCE — front-truncated when exceeded.)\n\n"
        "=== SCENE LOG (per-scene captions) ===\n"
        f"{scene_text}\n\n"
        "=== CHARACTER REGISTRY (consistent character/object tracking) ===\n"
        f"{char_text}\n\n"
        "=== QUESTION EVIDENCE (running notes per active question) ===\n"
        f"{qe_text}"
    )


# ===== Question helpers =====
def compute_gt_answer_time(item: Dict[str, Any], end_ref: int) -> int:
    if item["type"] in ["Past", "Present"]:
        return parse_timestamp(item["ask_time"])
    return end_ref


def prepare_question_records(
    video_items: List[Dict[str, Any]],
    max_processed_sec: int,
) -> List[Dict[str, Any]]:
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
            "qid": qid,
            "video_id": item["video_id"],
            "question": item["question"],
            "gt_answer": item["answer"],
            "task_type": item["type"],
            "ask_time_raw": ask_time_raw,
            "ask_time": ask_time,
            "beg_ref": beg_ref,
            "end_ref": end_ref,
            "gt_answer_time_raw": gt_answer_time_raw,
            "gt_answer_time": gt_answer_time,
            "video_last_processed_sec": max_processed_sec,
            "status": "unseen",
            "pred_answer": "",
            "pred_answer_raw": "",
            "pred_answer_strict": "",
            "answer_time": None,
            "answer_source": "",
            "evidence": "",
            "decision_history": [],
            "loaded_from_existing_result": False,
        }

        if existing_terminal is not None:
            existing_pred_answer = str(existing_terminal.get("pred_answer", ""))
            existing_pred_answer_raw = str(existing_terminal.get("pred_answer_raw", existing_pred_answer))
            existing_pred_answer_strict = str(
                existing_terminal.get("pred_answer_strict", normalize_choice_answer(existing_pred_answer))
            )
            base_record.update({
                "status": str(existing_terminal.get("status", base_record["status"])),
                "pred_answer": existing_pred_answer,
                "pred_answer_raw": existing_pred_answer_raw,
                "pred_answer_strict": existing_pred_answer_strict,
                "answer_time": existing_terminal.get("answer_time"),
                "answer_source": str(existing_terminal.get("answer_source", "")),
                "evidence": str(existing_terminal.get("evidence", "")),
                "decision_history": existing_terminal.get("decision_history", []),
                "loaded_from_existing_result": True,
            })
        questions.append(base_record)

    questions.sort(key=lambda x: (x["ask_time"], x["qid"]))
    return questions


# ===== OpenAI API call helpers =====
def build_image_inputs(frame_paths: List[str]) -> List[Dict[str, Any]]:
    items = []
    for fp in frame_paths:
        sec = frame_path_to_second(fp)
        b64 = encode_image_base64(fp)
        items.append({"type": "input_text", "text": f"Frame timestamp: {sec}s"})
        items.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}",
            "detail": "low",
        })
    return items


def _create_response_with_retry(
    system_prompt: str,
    content: List[Dict[str, Any]],
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_output_tokens: int = 32000,
    model: str = None,
    reasoning_effort: str = None,
):
    _model = model or MODEL_NAME
    _effort = reasoning_effort or REASONING_EFFORT
    if _LLM_API_AVAILABLE:
        return _llm_call_llm(
            model=_model,
            system_prompt=system_prompt,
            user_content=content,
            max_output_tokens=max_output_tokens,
            reasoning_effort=_effort,
            temperature=0.0,
            seed=42,
            gemini_thinking_budget=0,
        )
    for attempt in range(max_retries):
        try:
            return client.responses.create(
                model=_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": content},
                ],
                reasoning={"effort": _effort},
                max_output_tokens=max_output_tokens,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
                                                                    
            print(f"[API] error (attempt {attempt + 1}): {e}. Retry in {delay:.1f}s...")
            time.sleep(delay)


def call_model_json(
    system_prompt: str,
    user_text_blocks: List[str],
    frame_paths: List[str] = None,
    image_inputs: List[Dict[str, Any]] = None,
    max_output_tokens: int = 32000,
    model: str = None,
    reasoning_effort: str = None,
) -> Tuple[Dict[str, Any], float, str]:
    """Call the model and parse JSON. image_inputs (pre-encoded) takes precedence over frame_paths."""
    _model = model or MODEL_NAME
    content = []
    for t in user_text_blocks:
        content.append({"type": "input_text", "text": t})
    if image_inputs is not None:
        content.extend(image_inputs)
    elif frame_paths:
        content.extend(build_image_inputs(frame_paths))

    resp = _create_response_with_retry(
        system_prompt, content, max_output_tokens=max_output_tokens,
        model=_model, reasoning_effort=reasoning_effort,
    )
    _record_api_call(resp)
    raw_text = response_to_text(resp)
    cost = _estimate_cost_for_model(_model, resp)
    try:
        parsed = extract_json(raw_text)
    except ValueError:
        print(f"[call_model_json] JSON parse failed, retrying...")
        try:
            resp2 = _create_response_with_retry(
                system_prompt, content, max_output_tokens=max_output_tokens,
                model=_model, reasoning_effort=reasoning_effort,
            )
            _record_api_call(resp2)
            raw_text = response_to_text(resp2)
            cost += _estimate_cost_for_model(_model, resp2)
            parsed = extract_json(raw_text)
        except Exception:
            print(f"[call_model_json] JSON parse failed twice; returning empty dict. raw_text[:200]={raw_text[:200]!r}")
            parsed = {}
    return parsed, cost, raw_text


def _estimate_cost_for_model(model_name: str, resp) -> float:
    if _LLM_API_AVAILABLE:
        return _llm_unified_cost(resp, model_name)
    """Estimate cost using the correct pricing for the given model."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0.0
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING[MODEL_NAME])
    cost = input_tokens / 1_000_000 * pricing["input"]
    cost += output_tokens / 1_000_000 * pricing["output"]
    return cost


# ===== Tool Prompts =====

MEMORY_MANAGER_SYSTEM = f"""You are a Memory Manager for a streaming video understanding system.
You process video sequentially in chunks. You CANNOT go back to re-watch earlier frames.
Your job is to observe the current chunk of video frames and update the structured memory.
The memory (scene_log + character_registry + question_evidence) is all the system has from earlier chunks.

=== MEMORY STRUCTURE (token budget: ~{MAX_MEMORY_TOKENS} tokens total) ===
1. scene_log: per-chunk captions. Each entry has a time_range and a self-contained caption.
   Old entries are randomly evicted when the budget is exceeded, so each caption must stand on its own.
2. character_registry: persistent tracking of recurring characters and key objects using <name> format.
   This is an IDENTITY CATALOG — its purpose is to let you re-identify the same person/object across scenes.
   Entries MUST describe stable visual identity ONLY: appearance, clothing, hairstyle, hair color, skin tone,
   approximate age, gender, height/build, distinguishing marks, accessories, role/occupation if visible, and
   a short semantic label (e.g., "the cook", "the bride", "the goalkeeper"). They must NOT describe what the
   character is currently doing, their posture, momentary emotions, or what just happened to them — those
   belong in scene_log, not here.
3. question_evidence: per-question running notes of evidence observed so far.

=== FRAME INPUT STRUCTURE (OVERLAPPING CHUNKS) ===
To avoid information loss at chunk boundaries, frames are provided using an OVERLAPPING scheme.
Each step, you receive frames spanning TWO consecutive chunks (the previous chunk + the current chunk).
All frames are presented equally — there is no distinction between "context" and "current" frames.
The user message will tell you the CURRENT CHUNK time range (e.g., "Current chunk [64s-94s]").
Your caption must cover the FULL time range of ALL frames you see (both chunks combined).
For the very first chunk, there is no previous chunk, so you only receive one chunk of frames.

This overlap means:
- The tail end of the previous chunk and the beginning of the current chunk are both visible to you,
  so you can accurately describe events that span the boundary between chunks.
- Each frame will be shown to you in TWO consecutive steps (once as part of the "current" chunk, and once as part of the "previous" chunk in the next step). This is intentional — it ensures no boundary artifacts.
- Your caption should describe what happens across ALL the frames you see, as one coherent scene entry.

=== YOUR TASK (do all of these in ONE pass per chunk) ===
A. CAPTIONING of the provided frames:
   - Produce exactly ONE caption covering ALL frames shown to you (both the previous chunk and the current chunk).
   - The time_range in your output should span from the earliest frame to the latest frame you see.
   - QUESTION-AGNOSTIC: Caption based on what is VISUALLY PRESENT. Do NOT skew or narrow the caption to favor the active questions shown to you. Record ALL salient visual facts — including details, characters, and events that no current question asks about — because future questions may ask about them, and the same facts must be recorded consistently regardless of which questions happen to be active right now.
   - Caption should be DETAIL-PRESERVING. Even when the overall scene continues across chunks, record the FINE-GRAINED actions and observations that could distinguish later questions. Brief summaries like "the standoff continues" lose information; prefer concrete logs of WHAT each subject does and what changes about objects/positions/numbers in the visible frames.
   - Captions must record (when present and non-trivial):
     * EXACT COUNTS of countable items (people, objects, instances): write the precise number, not "several" or "a few". If you can see 4 people, write "4 people"; if a sequence has 8 contestants, count them and state "8".
     * specific visual details: colors, text/numbers on screen, spatial relations, object states, items appearing/disappearing
     * WHO does WHAT to WHOM, and the RESULT/OUTCOME of actions — including SMALL/MICRO actions (a hand brushing an object, an item falling, a glance, a posture shift). These small actions are often the answer to later questions; do NOT skip them just because the broader scene is "continuing".
     * dialogue content, emotional states, and reactions when they convey meaningful information
     * locations, settings, and environmental changes
     * temporal relationships: what happens before/after/during what
   - CRITICAL — CHARACTER REFERENCE RULE: Every time a human (or a character, or key recurring object) appears in a caption, you MUST refer to them by a <name> tag that matches an existing character_registry entry. Do NOT use generic noun phrases like "the man in the coat", "a woman", "the judge", or "someone". If the person is new, FIRST add them to character_updates with a stable <name> and identity description, THEN use that <name> in the caption. This guarantees that later questions about "the cook" or "the protagonist" can be matched against scene_log by <name>.

B. CHARACTER REGISTRY UPDATES:
   - Include NEW characters (not yet in the registry) and EXISTING characters whose STABLE IDENTITY information changed (e.g., they took off a hat, their full outfit became visible for the first time).
   - Description fields contain ONLY stable-identity attributes:
     * physical appearance: hair (length, color, style), skin tone, approximate age, gender, height/build
     * clothing and accessories (what they are consistently wearing in this video)
     * distinguishing marks (tattoos, scars, glasses, hat, etc.)
     * a short semantic label of their role/function in the video (e.g., "the cook", "the host", "the #7 player")
     * CHOOSING THE TAG — depends on whether the video has an OBVIOUS sequence of participants taking turns:
       Case 1 — COMPETITION/SEQUENCE context (talent show, awards ceremony, debate, panel discussion, contestants queueing, judges/voters in a fixed panel): use sequential number tags for the participants: <performer_1>, <performer_2>, <contestant_1>, <judge_1>, <speaker_1>, etc.
         - Counting rule: the FIRST distinct participant in that role is #1, the next NEW one is #2, etc. Do NOT renumber, skip, or restart.
         - Stability rule: once registered with order# K, that K is PERMANENT — same person keeps the same tag for the rest of the video.
         - Do NOT use their real names as the tag (e.g., NOT <performer_sandro_machado>). Real names from on-screen text go in the description field.
       Case 2 — NON-COMPETITION context (movies, vlogs, documentaries, interviews, sports games, news, plot-driven content): use SEMANTIC role tags or real names — NOT <performer_N>.
         - Role-based: <protagonist>, <antagonist>, <narrator>, <vlogger>, <reporter>, <chef>, <coach>, <interviewer>, <child_son>, <wife>, etc.
         - When a stable real name is visible on screen (lower-third, name plate, jersey, scoreboard), prefer including it in the tag: <player_bruno>, <chef_anna>, <coach_smith>, <speaker_obama>. This makes question matching natural.
         - Sequential numbering does NOT apply here — DO NOT default to <performer_N> just because someone appears. A vlogger is `<vlogger>`, not `<performer_1>`.
       Tie-breaker for ambiguous cases: if the video DOES include a numbered sequence within otherwise plot content (e.g., "the 8th sword-fight in the movie"), still use sequential tags for THAT subset only; protagonists and recurring named characters keep their semantic/named tags.
     * ON-SCREEN NAMES: if a name appears on screen (lower-third text, scoreboard, name plate), record it in the character_registry (e.g., "<contestant_3> aka 'John Smith'"). On-screen names are important identifiers.
   - Do NOT write about: current posture, current action, what they just did, emotional state, or plot events.
   - Bad example: "previously seated and lifting a bowl; by ≈286s shifts position and begins to rise". [This is current action, not identity — belongs in scene_log.]
   - Good example: "elderly man, ~60s, wearing a wide-brim straw hat, round glasses, grey kimono. Role: protagonist / elder villager."
   - COMPLETENESS: Do NOT miss important recurring characters. Every contestant, performer, judge, speaker, or major participant MUST be registered. Missing a character means questions about them become unanswerable.
   - Do NOT repeat unchanged characters.
   - FULL-REPLACEMENT RULE (IMPORTANT): When you submit an update for an EXISTING character, the system REPLACES the stored description entirely with whatever you write — it does NOT merge. So you MUST write a complete, self-contained description that includes EVERY still-valid attribute from the previous description PLUS your new observations. Do NOT write only the delta.
     * WRONG (will destroy prior info): {{"name": "<protagonist>", "description": "now without the straw hat"}}
     * RIGHT (full identity restated): {{"name": "<protagonist>", "description": "elderly man ~60s, round glasses, grey kimono, short grey hair, stocky build — previously wearing a wide-brim straw hat, now bareheaded. Role: protagonist / elder villager."}}
     * If you have no new stable-identity information to add for an existing character, DO NOT include them in character_updates at all (no-op is the correct default).

C. QUESTION EVIDENCE UPDATES:
   - For each active question, maintain a short running note of directly observed evidence.
   - Keep notes factual and specific (cite timestamps when possible).

=== OUTPUT JSON (strict schema) ===
{{
  "time_range": "Xs-Ys",
  "caption": "dense, self-contained caption covering ALL frames you see (both chunks). Every human/recurring-object reference MUST be a <name> from the character_registry. Length should be content-adaptive.",
  "character_updates": [
    {{"name": "<person_a>", "description": "stable identity attributes ONLY — appearance, clothing, accessories, role/label. No current actions, no postures, no events."}}
  ],
  "question_evidence": {{"qid": "updated evidence notes", ...}}
}}

=== RULES ===
- Always output valid JSON matching the schema.
- character_updates: ONLY NEW or CHANGED characters, ONLY stable identity attributes. Include sequence numbers and on-screen names. Do NOT miss key characters (contestants, performers, judges).
- caption: covers ALL frames you see (both chunks combined). time_range should span from the earliest to latest frame timestamp.
- Captions: every human/key-object reference is a <name> matching character_registry. No generic descriptions.
- Captions: content-adaptive length — detailed for rich/novel scenes, brief for simple/repetitive scenes. Do NOT pad with redundant description."""


ANSWER_PROPOSER_SYSTEM = """You are a Question Analyzer for a streaming video understanding system.
You are given:
1. The current chunk's video frames (visual input).
2. Text-based context information (from memory retrieval — includes relevant scene captions, character descriptions, and question evidence from earlier in the video).
3. A specific question to analyze.

=== YOUR REASONING PROCESS ===
Follow these steps IN ORDER:

1. TARGET IDENTIFICATION — read the question carefully and identify what is being asked. Map any people/objects to a <name> in the character descriptions when possible (e.g., "the cook" → <cook>). Note whether the question requires DIRECT OBSERVATION or REASONING/INFERENCE.

2. PRECONDITION CHECK (do this BEFORE looking at options):
   Questions often reference a specific event, entity, or moment. Before answering, briefly check whether that reference is supported by what you have seen. Preconditions can be met by current frames OR by earlier observations in the retrieved context — memory counts just as much as live frames, and partial/imperfect memory entries are still evidence.
   Compact handles for common reference types:
     - ORDINALS ("the Nth X"): count sequential entries in memory (character_registry order or temporal order of appearances in scene_log); if N entries exist OR you can infer ordering, precondition is met.
     - TEMPORAL anchors ("after X"): search memory for evidence that X has happened; if recorded, precondition is met.
     - SUPERLATIVES ("last to leave", "who turns first"): require the relevant sequence to have observably ended in memory or current frames.
     - Specific on-screen details (numbers, text, scores): rely on what was literally captured in captions; if the caption recorded the detail, use it directly.
   When memory contains any plausible match for the question's subject or event, proceed to option analysis on that basis — do not demand a perfect textual match, and do not wait hoping for a cleaner frame later (the video rarely re-labels past moments).

3. EVIDENCE LOOKUP — list every piece of evidence from the retrieved context that bears on the question. Quote the key phrase from each.

4. OPTION ANALYSIS — for each of (A)(B)(C)(D): SUPPORTED (cite specific context entry or current frame), CONTRADICTED (cite evidence against), or INSUFFICIENT.

5. REASONING STEP — after gathering evidence, actively connect the dots. Many questions require combining 2+ pieces of evidence or inferring from observed events. Do NOT refuse to reason. Inference from concrete observations is legitimate — it is not "guessing".
   IMPORTANT: base ALL reasoning on VIDEO EVIDENCE ONLY. Do NOT use general/world knowledge.

6. DECIDE — prefer committing whenever the evidence so far lets you pick the best-supported option. Defer only when the question genuinely depends on frames you have not yet seen.
   Commit ("answer") when any of these hold:
     - At least one piece of context/frame evidence bears on the question and supports one option more than the others.
     - You can chain a few pieces of evidence into an inference that favors one option.
     - You can rule out some options by observation and the remaining option(s) are more plausible than the rest.
   Defer ("need_more_frames") only when the question refers to an event that still clearly lies in the future of this video (it has not yet occurred on screen and is not in memory), and seeing more frames is realistically likely to change your answer.
   Keep in mind:
     - You don't need full certainty to commit — when one option is meaningfully better supported than the others, that's usually enough; if no option clearly stands out, deferring is reasonable.
     - Evidence about already-past events tends to be in memory by now; for events that are clearly still upcoming, additional frames may genuinely help.
     - Both committing and deferring have costs — answering wrong is just as bad as leaving unanswered, so weigh whether more frames would actually change your best-supported choice.

7. AVOID HALLUCINATION — do not invent facts. Do not use general/world knowledge. Do not mark an option CONTRADICTED without a video observation supporting it. Committing with imperfect but best-available evidence is not hallucinating — it is choosing the best-supported option, which is what this MCQ task requires.

=== OUTPUT JSON (strict schema) ===
{
  "chain_of_thought": "TARGET: [what is being asked]. PRECONDITION CHECK: [...]. EVIDENCE LOOKUP: [...]. REASONING: [...]. OPTION ANALYSIS: (A) ..., (B) ..., (C) ..., (D) .... DECISION: [commit/wait and why].",
  "option_analysis": {
    "(A)": {"assessment": "SUPPORTED/CONTRADICTED/INSUFFICIENT", "evidence": "..."},
    "(B)": {"assessment": "SUPPORTED/CONTRADICTED/INSUFFICIENT", "evidence": "..."},
    "(C)": {"assessment": "SUPPORTED/CONTRADICTED/INSUFFICIENT", "evidence": "..."},
    "(D)": {"assessment": "SUPPORTED/CONTRADICTED/INSUFFICIENT", "evidence": "..."}
  },
  "decision": "answer" or "need_more_frames",
  "answer": "(A)/(B)/(C)/(D) if decision=answer, else empty string",
  "confidence": 0.0 to 1.0,
  "reasoning_summary": "Brief summary of why this answer was chosen or why more frames are needed."
}

=== RULES ===
- If decision="answer", the answer field MUST be exactly (A), (B), (C), or (D).
- If decision="need_more_frames", the answer field MUST be an empty string.
- For reasoning questions (why/what might/how), actively chain evidence — do not refuse just because it needs inference.
- Do NOT hallucinate — only use information visible in the frames or provided in the text context. Picking the best-supported option among (A)/(B)/(C)/(D) on imperfect-but-relevant evidence is NOT hallucination — it is the correct behavior for this MCQ task.
- NEVER use general/world knowledge to answer. Only use what you observed in the video.
- Prefer committing whenever the evidence so far lets you rank one option above the others. Each chunk you defer reduces your credit if the answer is in fact already determinable. Defer only when the question's required event has clearly not yet occurred AND memory genuinely has nothing related to it.
- IMPORTANT: For questions whose subject already appears in retrieved context or character_registry — even imperfectly — commit on the first chunk where you can rank options. Memory of past chunks does NOT improve later, so waiting wastes credit.

"""


REFLECTION_AGENT_SYSTEM = """You are a Reflection Agent for a streaming video understanding system.
Your job is to critically evaluate a proposed answer to a video understanding question.

You are given:
1. The question being answered.
2. The full history of tool calls and their outputs for this question.
3. The proposed answer and the reasoning behind it.
4. The current chunk's visual frames.

=== PIPELINE CONTEXT (so you can interpret the tool history) ===
The agent gathering evidence for this question can call several tools, but the tool history shown to you will only contain calls made BEFORE this reflection step. The relevant ones you may see:
- **memory_retriever**: searches the agent's memory (scene captions, character registry, per-question evidence notes accumulated over earlier chunks). The output lists relevant past scenes/characters quoted verbatim from memory.
- **answer_proposer**: a vision+text agent that takes a question and a text context, plus the current chunk's frames, and returns one of (A)/(B)/(C)/(D) along with chain-of-thought reasoning, per-option assessment, confidence, and a decision (`answer` or `need_more_frames`). The proposed_answer you are reviewing typically comes from this tool's output.
- **reflection_agent**: prior reflection calls on this same question (you are this tool — multi-round reflection is allowed but rare).

Each entry in the tool history includes the tool name, an input_summary, and an output_summary. Use these to understand what evidence the agent actually has, separate from how the agent's reasoning is framing it.

Your task:
1. Review the reasoning chain — is it logically sound? Are there gaps or unsupported leaps?
2. Cross-check the reasoning's VISUAL CLAIMS against the actual frames. The earlier answer-proposer may have hallucinated details (e.g., described a judge's clothing or action that is NOT actually visible). When reasoning quotes a visual fact, verify it against the frames.
3. Check whether the proposed answer is consistent with the textual evidence from memory AND the visual evidence from the current frames.
4. Consider whether any alternative answer might be better supported.
5. Assign a confidence score to the proposed answer.

=== OUTPUT JSON (strict schema) ===
{
  "reasoning_review": "Assessment of the reasoning chain: is it logically valid? Any gaps?",
  "evidence_consistency": "Does the answer align with visual and textual evidence?",
  "alternative_consideration": "Are there reasons to suspect another option might be better? Describe the doubt without naming a letter.",
  "confidence": 0.0 to 1.0,
  "comment": "Overall assessment and any specific concerns.",
  "recommended_action": "accept" or "reject"
}

=== RULES ===
- Be critical but fair. The goal is to catch errors, not to be contrarian.
- confidence should be between 0.0 and 1.0.
- If recommended_action is "reject", explain clearly in comment what is wrong and what additional information or analysis might help. Do NOT propose a specific (A)/(B)/(C)/(D) letter as a replacement — that responsibility belongs to a follow-up answer_proposer call.
- Do NOT hallucinate — only use information from the tool history, the proposed reasoning, and the visual frames."""


CORE_PLANNER_SYSTEM = f"""You are the Core Planner for a single question in a streaming video understanding agent.
You orchestrate a set of tools to gather evidence and answer ONE question about a video.

=== AVAILABLE TOOLS ===
You call ONE tool at a time by outputting a JSON action. You receive its output, then decide the next action.

1. **memory_retriever**
   - Input: {{"tool": "memory_retriever", "query": "<text describing what to look up in memory>"}}
   - Function: Searches the memory (scene_log, character_registry, question_evidence) for information relevant to the query. Uses a single-stage LLM reranker over the full memory to select the most relevant scenes, characters, and evidence.
   - Output: LLM-selected relevant scene captions, character descriptions, and question-evidence entries.
   - Input format requirement: `query` should usually be the ORIGINAL question stem (with options stripped) so retrieval is stable across chunks. You may use a different query if you need to follow up on a specific aspect that the first retrieval missed.

2. **answer_proposer**
   - Input: {{"tool": "answer_proposer", "question": "<MCQ-formatted question text>", "context": "<text-based context to give the answer-proposer>"}}
   - Function: Analyzes the `question` given the text context and the current chunk's visual frames, then returns one of (A)/(B)/(C)/(D) along with a decision.
   - Output: Chain of thought, answer letter (one of (A)/(B)/(C)/(D)), confidence, decision (`answer` or `need_more_frames`).
   - Input format requirement: `question` MUST be in MCQ format — include the stem AND all four (A)/(B)/(C)/(D) option lines. You may rephrase the stem or rewrite the options if it helps the answer-proposer focus, BUT the four options you propose must remain semantically equivalent to the original MCQ's (A)/(B)/(C)/(D) and stay in the same order. The answer-proposer's letter answer will be used directly as the letter for the original MCQ — if you reorder or change the options' meaning, the letter will be wrong. When in doubt, copy the original MCQ verbatim.

3. **reflection_agent**
   - Input: {{"tool": "reflection_agent", "proposed_answer": "(A)/(B)/(C)/(D)", "reasoning": "<reasoning behind the proposed answer>"}}
   - Function: Critically reviews a proposed answer based on the tool-call history, your reasoning, AND the current chunk's frames. The reflection agent cross-checks the reasoning's visual claims against the actual frames (catching visual hallucinations) and also evaluates the logical structure of the reasoning. It returns only an accept/reject verdict + a confidence score + a comment — it does NOT propose an alternative letter.
   - Output: `recommended_action` (accept/reject), `confidence`, `comment`.
   - When to use: REQUIRED before every finalize_answer (see RULES). If reflection rejects, do NOT just guess a different letter yourself — go back through the pipeline (re-call answer_proposer with augmented context, or defer for more frames).

4. **finalize_answer**
   - Input: {{"tool": "finalize_answer", "answer": "(A)/(B)/(C)/(D)", "evidence": "<summary of evidence>"}}
   - Function: Marks the question as answered. This is FINAL — you cannot change the answer later.
   - Input format requirement: `answer` MUST be exactly one of "(A)", "(B)", "(C)", "(D)". When the answer-proposer has just returned a confident letter, use that letter.

5. **defer_question**
   - Input: {{"tool": "defer_question", "reason": "<why more frames are needed>"}}
   - Function: Marks the question as need_more_frames. The question will be revisited when the next chunk arrives.

=== IMPORTANT: YOU DO NOT HAVE DIRECT ACCESS TO MEMORY OR FRAMES ===
You do NOT see the memory contents or the visual frames of the current chunk yourself. To look up information from past chunks, call memory_retriever. To get a question answered against the current chunk's frames + a text context, call answer_proposer. The user message only tells you the question, the chunk header, and the tool-call history so far.

=== RULES ===
- Output ONLY the JSON action object — no other text.
- You MUST call exactly ONE tool per response.
- You have a maximum of {MAX_TOOL_ROUNDS} tool calls per question per chunk. If you exhaust them without finalizing or deferring, the question is auto-deferred.
- memory_retriever is capped at {MAX_RETRIEVER_CALLS} calls per question per chunk. After that, additional retriever calls will return an error.
- finalize_answer.answer MUST be exactly "(A)", "(B)", "(C)", or "(D)"."""


# ===== Tool Implementations =====

def tool_memory_manager(
    frame_paths: List[str],
    memory_state: Dict[str, Any],
    active_questions: List[Dict[str, Any]],
    chunk_time_range: str,
    mem_rng: random.Random,
    image_inputs: List[Dict[str, Any]] = None,
    prev_chunk_image_inputs: List[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float, str]:
    """Update memory from the current chunk (prev_chunk_image_inputs are prepended as overlap)."""
    memory_text = memory_state_to_text(memory_state)

    chunk_secs = [frame_path_to_second(p) for p in frame_paths]
    if prev_chunk_image_inputs:
        total_num_frames = "previous chunk + current chunk"
        overlap_note = (
            f"Current chunk [{chunk_time_range}] — {len(frame_paths)} current frames. "
            f"Previous chunk frames are also included for overlap context. "
            f"Produce ONE caption covering ALL frames you see."
        )
    else:
        overlap_note = (
            f"This is the first chunk — {len(frame_paths)} frames, range {chunk_time_range}. "
            f"No previous chunk overlap. Produce ONE caption covering all frames."
        )

    q_list_text = ""
    if active_questions:
        q_entries = [{"qid": sanitize_qid(q["qid"]), "question": q["question"]} for q in active_questions]
        q_list_text = f"\nActive questions (update evidence if you observe relevant information):\n{json.dumps(q_entries, ensure_ascii=False, indent=2)}"

    user_blocks = [
        f"Current memory state:\n{memory_text}\n",
        overlap_note,
        q_list_text if q_list_text else "No active questions yet.",
        "\nReminders: "
        "(a) every human/character/key-object in scene captions MUST be a <name> from character_registry — no generic 'the man in the coat'. "
        "(b) character_registry descriptions hold STABLE IDENTITY only (appearance, clothing, role); do not put current actions or recent events there. "
        "(c) Tag choice: COMPETITION/SEQUENCE context → sequential tags <performer_N>/<contestant_N>/<judge_N> (order PERMANENT, never renumber). NON-COMPETITION → SEMANTIC tags <protagonist>/<vlogger>/<chef> or real-name tags <player_bruno>. Don't default to <performer_N> for movies/vlogs/docs. "
        "(d) Caption is QUESTION-AGNOSTIC: describe what is visually present, not what the questions ask about. "
        "(e) produce ONE caption covering ALL frames (both chunks combined).",
    ]

    combined_image_inputs = []
    if prev_chunk_image_inputs:
        combined_image_inputs.extend(prev_chunk_image_inputs)
    if image_inputs is not None:
        combined_image_inputs.extend(image_inputs)

    parsed, cost, raw_text = call_model_json(
        system_prompt=MEMORY_MANAGER_SYSTEM,
        user_text_blocks=user_blocks,
        image_inputs=combined_image_inputs if combined_image_inputs else None,
        frame_paths=frame_paths if not combined_image_inputs else None,
    )

    caption = str(parsed.get("caption", "")).strip()
    scene_time_range = str(parsed.get("time_range", "")).strip()
    if not caption:
        scenes = parsed.get("scenes", [])
        if isinstance(scenes, list) and scenes:
            caption = str(scenes[0].get("caption", "")).strip()
            if not scene_time_range:
                scene_time_range = str(scenes[0].get("time_range", "")).strip()
    if not scene_time_range:
        scene_time_range = chunk_time_range
    if caption:
        add_scene_entry(memory_state, scene_time_range, caption, mem_rng)

    new_chars = parsed.get("character_updates", parsed.get("character_registry", []))
    update_character_registry(memory_state, new_chars, mem_rng)

    new_qe = parsed.get("question_evidence", {})
    if isinstance(new_qe, dict):
        memory_state["question_evidence"].update(new_qe)

    memory_state_normalized = normalize_memory_state(memory_state, mem_rng)
    memory_state.update(memory_state_normalized)

    return parsed, cost, raw_text


# ===== Memory Retrieval (LLM rerank, lexical fallback) =====

def _simple_tokens(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens (including <name> tags)."""
    return re.findall(r"[a-zA-Z0-9_<>]+", text.lower())


def _overlap_score(query_tokens: "Counter", text: str) -> int:
    """Count overlapping tokens between pre-computed query tokens and a text string."""
    ttok = Counter(_simple_tokens(text))
    return sum(min(query_tokens[w], ttok[w]) for w in query_tokens)


# --- Ordinal normalization ---
_ORDINAL_MAP = {
    "1st": "1", "first": "1",
    "2nd": "2", "second": "2",
    "3rd": "3", "third": "3",
    "4th": "4", "fourth": "4",
    "5th": "5", "fifth": "5",
    "6th": "6", "sixth": "6",
    "7th": "7", "seventh": "7",
    "8th": "8", "eighth": "8",
    "9th": "9", "ninth": "9",
    "10th": "10", "tenth": "10",
    "11th": "11", "eleventh": "11",
    "12th": "12", "twelfth": "12",
    "13th": "13", "14th": "14", "15th": "15",
    "16th": "16", "17th": "17", "18th": "18",
    "19th": "19", "20th": "20",
    "last": "last", "final": "last",
}
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


def _extract_ordinals(text: str) -> List[str]:
    result = []
    lower = text.lower()
    for word, num in _ORDINAL_MAP.items():
        if word in lower:
            result.append(num)
    for m in _ORDINAL_RE.finditer(text):
        result.append(m.group(1))
    return list(set(result))


def _ordinal_variants(num: str, base_words: List[str]) -> List[str]:
    variants = [f"#{num}", num]
    for w in base_words:
        w = w.lower().rstrip("s")
        variants.extend([
            f"<{w}_{num}>",
            f"{w}_{num}",
            f"{w} #{num}",
            f"{num}th {w}" if not num.endswith(("1", "2", "3")) else f"{num} {w}",
        ])
    return variants


# --- Temporal keyword expansion ---
_TEMPORAL_EXPANSIONS = {
    "after": ["concludes", "concluded", "ends", "ended", "finishes", "finished",
              "over", "done", "complete", "completed", "following", "later", "next", "afterwards"],
    "before": ["begins", "began", "starts", "started", "prior", "earlier",
               "preceding", "about to", "preparing"],
    "during": ["while", "meanwhile", "ongoing", "in progress", "throughout",
               "amid", "amidst"],
    "ceremony": ["award", "presentation", "prize", "medal", "trophy",
                  "recognition", "honor", "honours"],
    "performance": ["performs", "performed", "performing", "sings", "sang",
                     "dances", "danced", "plays", "played", "act", "routine"],
    "competition": ["contest", "contestant", "compete", "competing", "round",
                     "match", "game", "tournament"],
    "leaves": ["left", "leaving", "departs", "departed", "exits", "exited",
               "walks away", "goes away"],
    "arrives": ["arrived", "arriving", "enters", "entered", "comes", "came",
                "appears", "appeared", "shows up"],
    "wins": ["won", "winner", "victory", "champion", "first place"],
    "loses": ["lost", "loser", "defeated", "eliminated"],
    "last": ["final", "latest", "ultimate", "concluding", "closing", "end"],
    "first": ["initial", "opening", "beginning", "earliest", "starts"],
}


def _expand_query_tokens(query: str, characters: List[Dict[str, str]]) -> Counter:
    base_tokens = _simple_tokens(query)
    expanded = list(base_tokens)
    lower_query = query.lower()

    ordinals = _extract_ordinals(query)
    if ordinals:
        base_nouns = []
        for tok in base_tokens:
            if tok.isalpha() and len(tok) > 2 and tok not in _ORDINAL_MAP and tok not in {
                "the", "and", "who", "what", "which", "that", "this", "does", "did",
                "was", "were", "has", "had", "how", "why", "when", "where",
            }:
                base_nouns.append(tok)
        for num in ordinals:
            variants = _ordinal_variants(num, base_nouns)
            expanded.extend(_simple_tokens(" ".join(variants)))

    for keyword, expansions in _TEMPORAL_EXPANSIONS.items():
        if keyword in lower_query:
            expanded.extend(expansions)

    for char in characters:
        char_name = char.get("name", "").lower()
        char_desc = char.get("description", "").lower()
        char_name_bare = char_name.strip("<>")
        char_desc_tokens = set(_simple_tokens(char_desc))
        query_token_set = set(base_tokens)
        overlap = query_token_set & char_desc_tokens - {
            "the", "a", "an", "and", "or", "in", "on", "of", "to", "is", "was",
            "with", "for", "at", "by", "from", "man", "woman", "person",
        }
        if overlap or char_name_bare in lower_query or char_name in lower_query:
            expanded.extend(_simple_tokens(char_name))
            expanded.append(char_name_bare)
            for dt in char_desc_tokens:
                if len(dt) > 3 and dt not in {"with", "wearing", "about", "from", "this", "that", "role"}:
                    expanded.append(dt)

    return Counter(expanded)


_RERANK_SYSTEM = """You are a retrieval reranker for a video understanding system.
Given a query and candidate scenes, characters, and evidence from memory, select the most relevant ones.

CRITICAL: If NONE of the candidates contain information relevant to the query, you MUST return
empty lists. Do NOT fabricate connections or hallucinate relevance.
The query may ask about an event that has NOT happened yet in the video — in that case, the
correct response is to return empty lists.

Output JSON:
{
  "selected_scene_indices": [0, 2, 5],
  "selected_character_indices": [0, 1, 3]
}

Rules:
- selected_scene_indices: indices (0-based) into the candidate scenes list. Select only scenes that contain DIRECT evidence about the query topic. Return empty list [] if no scene is relevant.
- selected_character_indices: indices (0-based) into the candidate characters list. Select at most 3-5 characters DIRECTLY relevant to the query. Return empty list [] if no character is relevant.
- Do NOT invent or assume information not present in the candidates.
- Be HIGHLY selective — most queries only need 1-3 characters and a few scenes."""


def _llm_rerank_retrieval(
    query: str,
    candidate_scenes: List[Dict[str, str]],
    candidate_characters: List[Dict[str, str]],
    candidate_evidence: Dict[str, str],
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Use RETRIEVER_MODEL_NAME to rerank retrieval candidates. Returns (result_dict, cost) or (None, 0) on failure."""
    scene_lines = []
    for i, s in enumerate(candidate_scenes):
        scene_lines.append(f"  [{i}] [{s.get('time_range', '?')}]: {s.get('caption', '')}")
    char_lines = []
    for i, c in enumerate(candidate_characters):
        char_lines.append(f"  [{i}] {c.get('name', '?')}: {c.get('description', '')}")
    ev_lines = []
    for qid, ev in candidate_evidence.items():
        ev_lines.append(f"  [{sanitize_qid(qid)}]: {ev}")

    user_text = (
        f"Query: {query}\n\n"
        f"Candidate scenes ({len(candidate_scenes)}):\n" + ("\n".join(scene_lines) if scene_lines else "  (none)") + "\n\n"
        f"Candidate characters ({len(candidate_characters)}):\n" + ("\n".join(char_lines) if char_lines else "  (none)") + "\n\n"
        f"Candidate evidence:\n" + ("\n".join(ev_lines) if ev_lines else "  (none)")
    )

    try:
        parsed, cost, _ = call_model_json(
            system_prompt=_RERANK_SYSTEM,
            user_text_blocks=[user_text],
            max_output_tokens=32000,
            model=RETRIEVER_MODEL_NAME,
            reasoning_effort=RETRIEVER_REASONING_EFFORT,
        )
        return parsed, cost
    except Exception as e:
        print(f"[LLM Rerank] Error: {e}")
        return None, 0.0


def tool_memory_retriever(
    query: str,
    memory_state: Dict[str, Any],
    qid: str = "",
) -> Tuple[Dict[str, Any], float, str]:
    """LLM-based memory retriever — single-stage LLM rerank over ALL scenes."""
    all_chars = list(memory_state.get("character_registry", []))
    all_scenes = list(memory_state.get("scene_log", []))

    qe = memory_state.get("question_evidence", {})
    candidate_ev = {}
    if qid:
        # QE keys are sanitized ("Q83") but qid passed in is raw ("83_Past")
        sqid = sanitize_qid(qid)
        if sqid in qe:
            candidate_ev = {sqid: qe[sqid]}
        elif qid in qe:
            candidate_ev = {qid: qe[qid]}

    rerank_result, rerank_cost = _llm_rerank_retrieval(
        query, all_scenes, all_chars, candidate_ev,
    )

    if isinstance(rerank_result, dict):
        scene_indices = rerank_result.get("selected_scene_indices", [])
        char_indices = rerank_result.get("selected_character_indices", [])

        selected_scenes = []
        if isinstance(scene_indices, list):
            for idx in scene_indices:
                if isinstance(idx, int) and 0 <= idx < len(all_scenes):
                    selected_scenes.append(all_scenes[idx])
        if not selected_scenes:
            query_tokens = _expand_query_tokens(query, all_chars)
            scored = [(_overlap_score(query_tokens, s.get("caption","")), s) for s in all_scenes]
            scored.sort(key=lambda x: x[0], reverse=True)
            selected_scenes = [s for _, s in scored[:8] if _ > 0]

        MAX_SELECTED_CHARS = 10
        selected_chars = []
        if isinstance(char_indices, list):
            for idx in char_indices[:MAX_SELECTED_CHARS]:
                if isinstance(idx, int) and 0 <= idx < len(all_chars):
                    selected_chars.append(all_chars[idx])
        if not selected_chars:
            query_tokens = _expand_query_tokens(query, all_chars)
            scored = [((_overlap_score(query_tokens, c.get("name","") + " " + c.get("description","")), i), c) for i, c in enumerate(all_chars)]
            scored.sort(key=lambda x: x[0], reverse=True)
            selected_chars = [c for _, c in scored[:MAX_SELECTED_CHARS]]

        summary_parts = []
        if selected_scenes:
            scene_strs = [f"[{s.get('time_range', '?')}]: {s.get('caption', '')}" for s in selected_scenes]
            summary_parts.append("Retrieved scenes:\n" + "\n".join(scene_strs))
        else:
            summary_parts.append("No relevant scenes found in memory for this query.")
        if selected_chars:
            char_strs = [f"{c.get('name', '?')}: {c.get('description', '')}" for c in selected_chars]
            summary_parts.append("Retrieved characters:\n" + "\n".join(char_strs))
        if candidate_ev:
            ev_strs = [f"[{sanitize_qid(qid)}]: {ev}" for qid, ev in candidate_ev.items()]
            summary_parts.append("Question evidence:\n" + "\n".join(ev_strs))

        summary = "\n\n".join(summary_parts)

        result = {
            "relevant_scenes": selected_scenes,
            "relevant_evidence": candidate_ev,
            "relevant_characters": selected_chars,
            "summary": summary,
        }
        return result, rerank_cost, "llm_reranked_retrieval"

    # Fallback: lexical retrieval.
    query_tokens = _expand_query_tokens(query, all_chars)
    scored_fallback = [(_overlap_score(query_tokens, s.get("caption","")), s) for s in all_scenes]
    scored_fallback.sort(key=lambda x: x[0], reverse=True)
    ranked_scenes = [s for _, s in scored_fallback[:8] if _ > 0]
    summary_parts = []
    if ranked_scenes:
        scene_strs = [f"[{s.get('time_range', '?')}]: {s.get('caption', '')}" for s in ranked_scenes]
        summary_parts.append("Relevant scenes:\n" + "\n".join(scene_strs))
    else:
        summary_parts.append("No scenes matched the query.")
    if all_chars:
        char_strs = [f"{c.get('name', '?')}: {c.get('description', '')}" for c in all_chars]
        summary_parts.append("Character registry (all):\n" + "\n".join(char_strs))
    if candidate_ev:
        ev_strs = [f"[{sanitize_qid(qid)}]: {ev}" for qid, ev in candidate_ev.items()]
        summary_parts.append("Relevant evidence:\n" + "\n".join(ev_strs))

    summary = "\n\n".join(summary_parts)

    result = {
        "relevant_scenes": ranked_scenes,
        "relevant_evidence": candidate_ev,
        "relevant_characters": all_chars,
        "summary": summary,
    }
    return result, 0.0, "lexical_fallback"


def tool_answer_proposer(
    qid: str,
    question_text: str,
    context: str,
    image_inputs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], float, str]:
    """Call the Answer Proposer with visual frames and text context."""
    user_blocks = [
        f"Question (qid={sanitize_qid(qid)}):\n{question_text}",
        f"\nContext information:\n{context}",
    ]

    parsed, cost, raw_text = call_model_json(
        system_prompt=ANSWER_PROPOSER_SYSTEM,
        user_text_blocks=user_blocks,
        image_inputs=image_inputs,
        max_output_tokens=32000,
    )
    return parsed, cost, raw_text


def tool_reflection_agent(
    qid: str,
    question_text: str,
    proposed_answer: str,
    reasoning: str,
    tool_history: List[Dict[str, Any]],
    image_inputs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], float, str]:
    """Verify a proposed answer against the tool history and the current chunk's frames."""
    history_summary = []
    for entry in tool_history:
        history_summary.append({
            "tool": entry.get("tool"),
            "input_summary": entry.get("input_summary", ""),
            "output_summary": entry.get("output_summary", ""),
        })

    user_blocks = [
        f"Question (qid={sanitize_qid(qid)}):\n{question_text}",
        f"\nProposed answer: {proposed_answer}",
        f"\nReasoning behind the proposed answer:\n{reasoning}",
        f"\nTool call history for this question:\n{json.dumps(history_summary, ensure_ascii=False, indent=2)}",
    ]

    parsed, cost, raw_text = call_model_json(
        system_prompt=REFLECTION_AGENT_SYSTEM,
        user_text_blocks=user_blocks,
        image_inputs=image_inputs,
        max_output_tokens=4000,
    )
    return parsed, cost, raw_text


# ===== Per-Question Planner Loop (parallelized across questions) =====

MAX_ROUNDS_PER_QUESTION = MAX_TOOL_ROUNDS


def _process_single_question(
    q: Dict[str, Any],
    memory_state: Dict[str, Any],
    image_inputs: List[Dict[str, Any]],
    chunk_time_range: str,
    chunk_idx: int,
    video_id: str = None,
) -> Dict[str, Any]:
    """Run the Core Planner loop for one question (up to MAX_ROUNDS_PER_QUESTION tool calls)."""
    # Runs in a worker thread; re-set the thread-local video id for API stats.
    if video_id is not None:
        _current_video_id.value = video_id

    qid = q["qid"]
    question_text = q["question"]
    tool_history: List[Dict[str, Any]] = []
    total_cost = 0.0

    result = {
        "qid": qid,
        "question": question_text,
        "decision": None,
        "answer": "",
        "raw_answer": "",
        "strict_answer": "",
        "evidence": "",
        "tool_history": tool_history,
        "cost": 0.0,
    }

    def _build_planner_context():
        """Planner context: question + tool history only (no direct memory or frames)."""
        history_text = ""
        for i, entry in enumerate(tool_history):
            history_text += f"\n--- Tool call #{i+1}: {entry.get('tool', '?')} ---\n"
            history_text += f"Input: {entry.get('input_summary', '')}\n"
            history_text += f"Output: {entry.get('output_summary', '')}\n"

        parts = [
            f"Chunk [{chunk_time_range}], chunk_idx={chunk_idx}",
            f"\nYour question (qid={sanitize_qid(qid)}):\n{question_text}",
        ]
        if history_text:
            parts.append(f"\nTool call history so far:{history_text}")
        parts.append(f"\nRound {len(tool_history) + 1} of max {MAX_ROUNDS_PER_QUESTION}. Choose your next action.")
        return parts

    retriever_call_count = 0
    last_retriever_summary = ""

    for round_num in range(MAX_ROUNDS_PER_QUESTION):
        if result["decision"] is not None:
            break

        context_blocks = _build_planner_context()

        try:
            planner_parsed, planner_cost, _ = call_model_json(
                system_prompt=CORE_PLANNER_SYSTEM,
                user_text_blocks=context_blocks,
                max_output_tokens=32000,
            )
            total_cost += planner_cost
        except Exception as e:
            print(f"[Planner qid={qid}] Error in round {round_num + 1}: {e}")
            break

        tool_name = str(planner_parsed.get("tool", "")).strip().lower()


        if tool_name == "memory_retriever":
            query = str(planner_parsed.get("query", "")).strip()
            if not query:
                tool_history.append({
                    "round": round_num + 1,
                    "tool": "memory_retriever",
                    "input_summary": "(empty query)",
                    "output_summary": "Error: empty query provided.",
                })
                continue

            if retriever_call_count >= MAX_RETRIEVER_CALLS:
                tool_history.append({
                    "round": round_num + 1,
                    "tool": "memory_retriever",
                    "input_summary": f"query: {query} (BLOCKED — limit reached)",
                    "output_summary": f"Retriever limit reached ({MAX_RETRIEVER_CALLS} calls). "
                                     f"You must now call answer_proposer or defer_question. "
                                     f"Previous retrieval result: {last_retriever_summary}",
                })
                continue

            retriever_result, retriever_cost, retriever_method = tool_memory_retriever(query, memory_state, qid=qid)
            total_cost += retriever_cost
            summary = str(retriever_result.get("summary", ""))
            retriever_call_count += 1
            last_retriever_summary = summary
            tool_history.append({
                "round": round_num + 1,
                "tool": "memory_retriever",
                "input_summary": f"query: {query}",
                "output_summary": summary,
                "full_output": retriever_result,
                "retriever_method": retriever_method,
                "retriever_cost": retriever_cost,
            })

        elif tool_name == "answer_proposer":
            context = str(planner_parsed.get("context", "")).strip()
            asked_question = str(planner_parsed.get("question", "")).strip() or question_text

            try:
                proposer_result, proposer_cost, _ = tool_answer_proposer(
                    qid, asked_question, context, image_inputs,
                )
                total_cost += proposer_cost
                answer = str(proposer_result.get("answer", "")).strip()
                decision = str(proposer_result.get("decision", "")).strip() or ("answer" if answer else "need_more_frames")
                confidence = float(proposer_result.get("confidence", 0.0))
                reasoning_summary = str(proposer_result.get("reasoning_summary", ""))

                opt_line = ""
                opt = proposer_result.get("option_analysis", {})
                if isinstance(opt, dict) and opt:
                    opt_parts = []
                    for key in ("(A)", "(B)", "(C)", "(D)"):
                        entry = opt.get(key, {})
                        if isinstance(entry, dict):
                            assessment = str(entry.get("assessment", "")).strip()
                            if assessment:
                                opt_parts.append(f"{key[1]}={assessment}")
                    if opt_parts:
                        opt_line = f" Options: {', '.join(opt_parts)}."

                if decision == "answer":
                    output_summary = f"decision=answer, ans={answer}, conf={confidence:.2f}.{opt_line} {reasoning_summary}"
                else:
                    output_summary = f"decision=need_more_frames.{opt_line} {reasoning_summary}"
            except Exception as e:
                proposer_result = {}
                output_summary = f"Error: {str(e)[:200]}"

            tool_history.append({
                "round": round_num + 1,
                "tool": "answer_proposer",
                "input_summary": f'q="{asked_question}" context_len={len(context)}',
                "output_summary": output_summary,
                "full_output": proposer_result,
            })

        elif tool_name == "reflection_agent":
            proposed_answer = str(planner_parsed.get("proposed_answer", "")).strip()
            reasoning = str(planner_parsed.get("reasoning", "")).strip()

            try:
                reflection_result, reflection_cost, _ = tool_reflection_agent(
                    qid, question_text, proposed_answer, reasoning, tool_history, image_inputs
                )
                total_cost += reflection_cost
                ref_confidence = float(reflection_result.get("confidence", 0.0))
                recommended_action = str(reflection_result.get("recommended_action", "reject")).strip().lower()
                comment = str(reflection_result.get("comment", ""))
                output_summary = f"Action: {recommended_action}, Confidence: {ref_confidence:.2f}. {comment[:200]}"
            except Exception as e:
                reflection_result = {}
                output_summary = f"Error: {str(e)[:200]}"

            tool_history.append({
                "round": round_num + 1,
                "tool": "reflection_agent",
                "input_summary": f"proposed_answer={proposed_answer}; reasoning={reasoning}",
                "output_summary": output_summary,
                "full_output": reflection_result,
            })

        elif tool_name == "finalize_answer":
            answer = str(planner_parsed.get("answer", "")).strip()
            evidence = str(planner_parsed.get("evidence", "")).strip()

            strict_answer = normalize_choice_answer(answer)
            result["decision"] = "answer"
            result["answer"] = strict_answer if strict_answer else answer
            result["raw_answer"] = answer
            result["strict_answer"] = strict_answer
            result["evidence"] = evidence

            tool_history.append({
                "round": round_num + 1,
                "tool": "finalize_answer",
                "input_summary": f"answer={answer}",
                "output_summary": f"Finalized: {qid} -> {strict_answer or answer}",
            })

        elif tool_name == "defer_question":
            reason = str(planner_parsed.get("reason", "")).strip()
            result["decision"] = "need_more_frames"
            result["evidence"] = reason

            tool_history.append({
                "round": round_num + 1,
                "tool": "defer_question",
                "input_summary": f"reason: {reason}",
                "output_summary": f"Deferred: {reason}",
            })

        else:
            tool_history.append({
                "round": round_num + 1,
                "tool": tool_name or "(unknown)",
                "input_summary": json.dumps(planner_parsed)[:200],
                "output_summary": f"Unknown tool: {tool_name}. Ignoring.",
            })

    # Auto-defer when the turn budget is exhausted.
    if result["decision"] is None:
        result["decision"] = "need_more_frames"
        result["evidence"] = "Auto-deferred: max rounds per question reached."

    result["cost"] = total_cost
    return result


def run_parallel_question_pipelines(
    active_questions: List[Dict[str, Any]],
    memory_state: Dict[str, Any],
    image_inputs: List[Dict[str, Any]],
    chunk_time_range: str,
    chunk_idx: int,
    video_id: str = None,
) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """Run the planner loop for all active questions in parallel."""
    total_cost = 0.0
    all_results = []
    all_tool_history = []

    with ThreadPoolExecutor(max_workers=min(len(active_questions), 8)) as executor:
        future_to_q = {
            executor.submit(
                _process_single_question,
                q, memory_state, image_inputs, chunk_time_range, chunk_idx, video_id,
            ): q
            for q in active_questions
        }
        for future in as_completed(future_to_q):
            q = future_to_q[future]
            try:
                result = future.result()
                all_results.append(result)
                total_cost += result.get("cost", 0.0)
                for entry in result.get("tool_history", []):
                    entry["qid"] = q["qid"]
                all_tool_history.extend(result.get("tool_history", []))
            except Exception as e:
                print(f"[Question Pipeline] Error for qid={q['qid']}: {e}")
                traceback.print_exc()
                all_results.append({
                    "qid": q["qid"],
                    "question": q["question"],
                    "decision": "need_more_frames",
                    "answer": "",
                    "raw_answer": "",
                    "strict_answer": "",
                    "evidence": f"Pipeline error: {str(e)[:200]}",
                    "tool_history": [],
                    "cost": 0.0,
                })

    return all_results, total_cost, all_tool_history


# ===== Evaluation logic =====
def evaluate_one_video(
    video_id: str,
    video_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    _current_video_id.value = video_id
    with _video_stats_lock:
        _video_stats[video_id] = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0}

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

    memory_state = init_memory_state()
    mem_rng = random.Random(SEED ^ int(video_id))
    video_cost = 0.0
    trace = []
    pending_qids: List[str] = []
    qid_to_question = {q["qid"]: q for q in questions}
    next_question_idx = 0
    max_memory_tokens_seen = 0

    preloaded_terminal_qids = [q["qid"] for q in questions if question_status_is_terminal(q["status"])]

    video_start_time = time.time()
    prev_chunk_image_inputs: Optional[List[Dict[str, Any]]] = None

    for chunk_idx, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        chunk_secs = [frame_path_to_second(p) for p in chunk]
        chunk_end_sec = max(chunk_secs)
        chunk_time_range = f"{min(chunk_secs)}s-{max(chunk_secs)}s"

        # Reveal questions whose query time has been reached.
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


        chunk_image_inputs = build_image_inputs(chunk)

        # Stage 1: streaming memory maintenance (with overlap from the previous chunk).
        try:
            mm_parsed, mm_cost, mm_raw = tool_memory_manager(
                frame_paths=chunk,
                memory_state=memory_state,
                active_questions=active_questions,
                chunk_time_range=chunk_time_range,
                mem_rng=mem_rng,
                image_inputs=chunk_image_inputs,
                prev_chunk_image_inputs=prev_chunk_image_inputs,
            )
            video_cost += mm_cost
        except Exception as e:
            print(f"[Memory Manager] Error on chunk {chunk_idx}: {e}")
            mm_parsed = {}
            mm_raw = f"Error: {e}"

        current_mem_tokens = memory_token_count(memory_state)
        if current_mem_tokens > max_memory_tokens_seen:
            max_memory_tokens_seen = current_mem_tokens

        # Stage 2: agentic question processing, one planner loop per active question.
        answered_now: List[str] = []
        planner_tool_history = []

        if active_questions:
            try:
                question_results, planner_cost, planner_tool_history = run_parallel_question_pipelines(
                    active_questions=active_questions,
                    memory_state=memory_state,
                    image_inputs=chunk_image_inputs,
                    chunk_time_range=chunk_time_range,
                    chunk_idx=chunk_idx,
                    video_id=video_id,
                )
                video_cost += planner_cost
            except Exception as e:
                print(f"[Question Pipelines] Error on chunk {chunk_idx}: {e}")
                traceback.print_exc()
                question_results = [
                    {"qid": q["qid"], "decision": "need_more_frames", "answer": "",
                     "raw_answer": "", "strict_answer": "", "evidence": f"Pipeline error: {e}"}
                    for q in active_questions
                ]

            for qr in question_results:
                qid = qr["qid"]
                q = qid_to_question.get(qid)
                if not q:
                    continue

                q["decision_history"].append({
                    "chunk_idx": chunk_idx,
                    "chunk_end_sec": chunk_end_sec,
                    "decision": qr["decision"],
                    "evidence": qr.get("evidence", ""),
                })

                if qr["decision"] == "answer":
                    raw_answer = str(qr.get("raw_answer", qr.get("answer", ""))).strip()
                    strict_answer = str(qr.get("strict_answer", "")).strip()
                    if not strict_answer:
                        strict_answer = normalize_choice_answer(raw_answer)

                    q["status"] = "answered_online"
                    q["pred_answer_raw"] = raw_answer
                    q["pred_answer_strict"] = strict_answer
                    q["pred_answer"] = strict_answer if strict_answer else raw_answer
                    q["answer_time"] = chunk_end_sec
                    q["answer_source"] = "streaming_agent"
                    q["evidence"] = qr.get("evidence", "")
                    answered_now.append(qid)
                    memory_state["question_evidence"].pop(qid, None)

            pending_qids = [qid for qid in pending_qids if qid_to_question[qid]["status"] == "pending"]

        chunk_wall_time = time.time() - chunk_start_time
        trace.append({
            "chunk_idx": chunk_idx,
            "chunk_secs": chunk_secs,
            "chunk_wall_time_sec": round(chunk_wall_time, 2),
            "mode": "agent_qa" if active_questions else "memory_only",
            "newly_arrived_qids": newly_arrived_qids,
            "active_qids": [q["qid"] for q in active_questions],
            "answered_now_qids": answered_now,
            "caption": mm_parsed.get("caption", "") if isinstance(mm_parsed, dict) else "",
            "accumulated_video_cost": video_cost,
            "memory_tokens": current_mem_tokens,
            "max_memory_tokens_so_far": max_memory_tokens_seen,
            "num_characters": len(memory_state["character_registry"]),
            "memory_snapshot": memory_state_to_text(memory_state),
            "memory_manager_raw": mm_raw,
            "memory_manager_parsed": mm_parsed,
            "planner_tool_history": planner_tool_history,
        })

        # Overlap: last 8 frames (each frame is a text label + an image item).
        prev_chunk_image_inputs = list(chunk_image_inputs[-16:])


    # Questions still open at the end of the video are unanswered.
    remaining_questions = [q for q in questions if q["status"] in {"unseen", "pending"}]
    if remaining_questions:
        final_video_sec = frame_seconds[-1] if frame_seconds else None
        for q in remaining_questions:
            q["status"] = "unanswered"
            q["pred_answer"] = ""
            q["pred_answer_raw"] = ""
            q["pred_answer_strict"] = ""
            q["answer_time"] = None
            q["answer_source"] = "video_end_unanswered"
            q["evidence"] = ""
            q["decision_history"].append({
                "chunk_idx": len(chunks) - 1 if chunks else -1,
                "chunk_end_sec": final_video_sec,
                "decision": "unanswered_at_video_end",
                "evidence": "",
            })
        trace.append({
            "chunk_idx": len(chunks) - 1 if chunks else -1,
            "mode": "mark_unanswered_at_video_end",
            "remaining_qids": [q["qid"] for q in remaining_questions],
            "accumulated_video_cost": video_cost,
        })

    return _build_result(video_id, questions, frame_paths, frame_seconds,
                         video_cost, preloaded_terminal_qids, trace, memory_state,
                         max_memory_tokens_seen=max_memory_tokens_seen,
                         video_wall_time=time.time() - video_start_time)


def _build_result(
    video_id, questions, frame_paths, frame_seconds,
    video_cost, preloaded_terminal_qids, trace, memory_state,
    status="finished",
    max_memory_tokens_seen: int = 0,
    video_wall_time: float = 0.0,
):
    stats = _video_stats.get(video_id, {"api_calls": 0, "input_tokens": 0, "output_tokens": 0})
    result = {
        "video_id": video_id,
        "status": status,
        "video_wall_time_sec": round(video_wall_time, 2),
        "api_calls": stats["api_calls"],
        "total_input_tokens": stats["input_tokens"],
        "total_output_tokens": stats["output_tokens"],
        "num_questions": len(questions),
        "num_answered_online": len([q for q in questions if q["status"] == "answered_online"]),
        "num_unanswered": len([q for q in questions if q["status"] == "unanswered"]),
        "sampled_num_frames": len(frame_paths),
        "sampled_frame_seconds": frame_seconds,
        "chunk_size": CHUNK_SIZE,
        "frame_stride": FRAME_STRIDE,
        "reasoning_effort": REASONING_EFFORT,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "final_memory_state": memory_state,
        "final_memory_text": memory_state_to_text(memory_state),
        "final_memory_tokens": memory_token_count(memory_state),
        "max_memory_tokens_seen": max_memory_tokens_seen,
        "memory_token_budget": MAX_MEMORY_TOKENS,
        "video_cost": video_cost,
        "preloaded_terminal_qids": preloaded_terminal_qids,
        "questions": questions,
        "trace": trace,
    }
    return result


def make_question_output_record(question: Dict[str, Any], video_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "qid": question["qid"],
        "video_id": question["video_id"],
        "question": question["question"],
        "gt_answer": question["gt_answer"],
        "task_type": question["task_type"],
        "status": question["status"],
        "pred_answer": question["pred_answer"],
        "pred_answer_raw": question.get("pred_answer_raw", question["pred_answer"]),
        "pred_answer_strict": question.get("pred_answer_strict", normalize_choice_answer(question["pred_answer"])),
        "answer_time": question["answer_time"],
        "answer_source": question["answer_source"],
        "evidence": question["evidence"],
        "ask_time_raw": question["ask_time_raw"],
        "ask_time": question["ask_time"],
        "beg_ref": question["beg_ref"],
        "end_ref": question["end_ref"],
        "gt_answer_time_raw": question["gt_answer_time_raw"],
        "gt_answer_time": question["gt_answer_time"],
        "video_last_processed_sec": question["video_last_processed_sec"],
        "loaded_from_existing_result": question["loaded_from_existing_result"],
        "decision_history": question["decision_history"],
        "chunk_size": video_result.get("chunk_size"),
        "frame_stride": video_result.get("frame_stride"),
        "reasoning_effort": video_result.get("reasoning_effort"),
        "max_tool_rounds": video_result.get("max_tool_rounds"),
        "sampled_num_frames": video_result.get("sampled_num_frames"),
        "sampled_frame_seconds": video_result.get("sampled_frame_seconds"),
        "video_run_status": video_result.get("status"),
        "video_cost": video_result.get("video_cost"),
        "video_trace_file": os.path.basename(video_input_trace_path(question["video_id"])),
        "video_summary_file": os.path.basename(video_summary_path(question["video_id"])),
    }


def write_video_artifacts(video_result: Dict[str, Any]) -> None:
    video_id = video_result["video_id"]

    with open(video_full_trace_path(video_id), "w") as f:
        json.dump(video_result.get("trace", []), f, indent=2, ensure_ascii=False)

    trace_compact = []
    for x in video_result.get("trace", []):
        entry = {
            "chunk_idx": x.get("chunk_idx"),
            "chunk_secs": x.get("chunk_secs"),
            "chunk_wall_time_sec": x.get("chunk_wall_time_sec"),
            "mode": x.get("mode"),
            "newly_arrived_qids": x.get("newly_arrived_qids"),
            "active_qids": x.get("active_qids"),
            "answered_now_qids": x.get("answered_now_qids"),
            "remaining_qids": x.get("remaining_qids"),
            "caption": x.get("caption"),
            "num_characters": x.get("num_characters"),
            "memory_tokens": x.get("memory_tokens"),
            "max_memory_tokens_so_far": x.get("max_memory_tokens_so_far"),
        }
        if x.get("memory_snapshot"):
            entry["memory_snapshot"] = x["memory_snapshot"]
        if x.get("memory_manager_parsed"):
            entry["memory_manager_parsed"] = x["memory_manager_parsed"]
        if x.get("memory_manager_raw"):
            entry["memory_manager_raw"] = x["memory_manager_raw"]
        if x.get("planner_tool_history"):
            entry["planner_tool_history"] = x["planner_tool_history"]
        trace_compact.append(entry)

    with open(video_input_trace_path(video_id), "w") as f:
        json.dump(trace_compact, f, indent=2, ensure_ascii=False)

    video_summary = {
        "video_id": video_id,
        "status": video_result.get("status"),
        "num_questions": video_result.get("num_questions"),
        "num_answered_online": video_result.get("num_answered_online"),
        "num_unanswered": video_result.get("num_unanswered"),
        "sampled_num_frames": video_result.get("sampled_num_frames"),
        "chunk_size": video_result.get("chunk_size"),
        "frame_stride": video_result.get("frame_stride"),
        "reasoning_effort": video_result.get("reasoning_effort"),
        "max_tool_rounds": video_result.get("max_tool_rounds"),
        "video_cost": video_result.get("video_cost"),
        "video_wall_time_sec": video_result.get("video_wall_time_sec"),
        "api_calls": video_result.get("api_calls"),
        "total_input_tokens": video_result.get("total_input_tokens"),
        "total_output_tokens": video_result.get("total_output_tokens"),
        "memory_token_budget": video_result.get("memory_token_budget"),
        "max_memory_tokens_seen": video_result.get("max_memory_tokens_seen"),
        "final_memory_tokens": video_result.get("final_memory_tokens"),
        "preloaded_terminal_qids": video_result.get("preloaded_terminal_qids", []),
        "question_ids": [q["qid"] for q in video_result.get("questions", [])],
    }
    with open(video_summary_path(video_id), "w") as f:
        json.dump(video_summary, f, indent=2, ensure_ascii=False)

    for q in video_result.get("questions", []):
        q_path = question_result_path(q["qid"])
        q_payload = make_question_output_record(q, video_result)
        with open(q_path, "w") as f:
            json.dump(q_payload, f, indent=2, ensure_ascii=False)


# ===== Main =====
def main():
    with open(task_path, "r") as f:
        data = json.load(f)

    video_to_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
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
    summary_results: List[Dict[str, Any]] = []

    def process_video(idx: int, video_id: str) -> None:
        nonlocal global_total_cost


        expected_qids = [item["uid"] for item in video_to_items[video_id]]
        if video_all_questions_processed(expected_qids):
            print(f"[SKIP] video_id={video_id} all questions already processed.")
            with cost_lock:
                summary_results.append({
                    "video_id": video_id,
                    "status": "skipped_already_processed",
                    "num_questions": len(expected_qids),
                })
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
                    "video_id": res["video_id"],
                    "status": res["status"],
                    "num_questions": res.get("num_questions"),
                    "num_answered_online": res.get("num_answered_online"),
                    "num_unanswered": res.get("num_unanswered"),
                    "video_cost": res.get("video_cost"),
                    "video_wall_time_sec": res.get("video_wall_time_sec"),
                    "max_memory_tokens_seen": res.get("max_memory_tokens_seen"),
                    "final_memory_tokens": res.get("final_memory_tokens"),
                })


                global_total_cost += res["video_cost"]
                current_cost = global_total_cost

            wall_time = res.get('video_wall_time_sec', 0)
            print(f"[DONE] {video_id} | Cost: ${res['video_cost']:.4f} | Total: ${current_cost:.4f} | Time: {wall_time:.1f}s")
            print(f"  Answered: {res.get('num_answered_online',0)} | Unanswered: {res.get('num_unanswered',0)}")


        except Exception as e:
            with cost_lock:
                summary_results.append({
                    "video_id": video_id, "status": "error", "error": str(e),
                })
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
                fut = executor.submit(process_video, idx, video_id)
                futures[fut] = video_id
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
            "max_tool_rounds": MAX_TOOL_ROUNDS,
            "num_workers": num_workers,
            "max_memory_tokens": MAX_MEMORY_TOKENS,
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
