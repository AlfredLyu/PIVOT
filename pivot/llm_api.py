"""Unified LLM API wrapper (OpenAI Responses API, Google Gemini API).

    resp = call_llm(model=..., system_prompt=..., user_content=[...])
    text, (in_tok, out_tok), cost = unified_text(resp), unified_usage(resp), unified_cost(resp, model)

user_content uses the OpenAI Responses format and is converted automatically for Gemini.
"""
import base64
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# -------- Model pricing (per 1M tokens) --------
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "provider": "openai"},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50, "provider": "gemini"},
}


def provider_for(model: str) -> str:
    """Return 'openai' or 'gemini' for a model name."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]["provider"]
    return "gemini" if model.startswith("gemini-") else "openai"


# -------- Clients --------
_openai_client = None
_gemini_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG_ID"),
            timeout=300.0,
            max_retries=0,  # retries are handled in call_llm
        )
    return _openai_client



def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY env var to a working Gemini API key."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# -------- OpenAI path --------
def _call_openai(system_prompt: str, content: List[Dict[str, Any]], model: str,
                 max_output_tokens: int,
                 max_retries: int = 5, base_delay: float = 2.0):
    client = _get_openai_client()
    kwargs = dict(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
        max_output_tokens=max_output_tokens,
    )
    for attempt in range(max_retries):
        try:
            return client.responses.create(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[API openai] error (attempt {attempt + 1}): {e}. Retry in {delay:.1f}s...")
            time.sleep(delay)


# -------- Gemini path --------
def _content_to_gemini(content: List[Dict[str, Any]]):
    """Convert OpenAI-style content list to Gemini Parts."""
    from google.genai import types
    parts = []
    for item in content:
        t = item.get("type")
        if t == "input_text":
            parts.append(types.Part.from_text(text=item["text"]))
        elif t == "input_image":
            url = item.get("image_url", "")
            if url.startswith("data:"):
                header, b64data = url.split(",", 1)
                mime_match = re.search(r"data:([^;]+);", header)
                mime = mime_match.group(1) if mime_match else "image/png"
                img_bytes = base64.b64decode(b64data)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            else:
                parts.append(types.Part.from_uri(file_uri=url, mime_type="image/png"))
        else:
            parts.append(types.Part.from_text(text=str(item)))
    return parts


def _call_gemini(system_prompt: str, content: List[Dict[str, Any]], model: str,
                 max_output_tokens: int, temperature: float, seed: Optional[int],
                 thinking_budget: Optional[int],
                 max_retries: int = 5, base_delay: float = 2.0):
    from google.genai import types
    client = _get_gemini_client()
    parts = _content_to_gemini(content)
    safety_settings = [
        types.SafetySetting(category=c, threshold="BLOCK_NONE")
        for c in [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]
    ]
    cfg_dict = {
        "system_instruction": system_prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "safety_settings": safety_settings,
    }
    if seed is not None:
        cfg_dict["seed"] = seed
    if thinking_budget is not None:
        cfg_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    cfg_dict["http_options"] = types.HttpOptions(timeout=900_000)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=cfg_dict,
            )
            try:
                text = resp.text or ""
            except Exception:
                text = ""
            if not text.strip():
                cands = getattr(resp, "candidates", []) or []
                finish = [getattr(c, "finish_reason", None) for c in cands]
                pf = getattr(resp, "prompt_feedback", None)
                block_reason = getattr(pf, "block_reason", None) if pf else None
                # A blocked prompt will be blocked again; do not retry.
                if block_reason is not None:
                    print(f"[API gemini] hard block ({block_reason}) — no retry, returning empty.")
                    return resp
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[API gemini] empty response (attempt {attempt + 1}, n_cands={len(cands)}, finish={finish}, block={block_reason}). Retry in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            return resp
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[API gemini] error (attempt {attempt + 1}): {e}. Retry in {delay:.1f}s...")
            time.sleep(delay)


# -------- Unified entry --------
def call_llm(
    model: str,
    system_prompt: str,
    user_content: List[Dict[str, Any]],
    max_output_tokens: int = 32000,
    reasoning_effort: Optional[str] = None,  # unused by gpt-4o / gemini-3.1-flash-lite
    temperature: float = 0.0,
    seed: Optional[int] = 42,
    gemini_thinking_budget: Optional[int] = 0,  # 0 disables thinking
    max_retries: int = 5,
    base_delay: float = 2.0,
):
    """Unified entry point. Returns the provider-native response object (use unified_* helpers)."""
    p = provider_for(model)
    if p == "openai":
        return _call_openai(
            system_prompt, user_content, model, max_output_tokens,
            max_retries=max_retries, base_delay=base_delay,
        )
    elif p == "gemini":
        return _call_gemini(
            system_prompt, user_content, model, max_output_tokens,
            temperature=temperature, seed=seed, thinking_budget=gemini_thinking_budget,
            max_retries=max_retries, base_delay=base_delay,
        )
    raise ValueError(f"Unknown provider for model: {model}")


# -------- Unified response helpers --------
import os as _os_dbg
GEMINI_DEBUG_LOG = _os_dbg.environ.get("GEMINI_DEBUG_LOG", "")

def _log_gemini_response(resp, text):
    if not GEMINI_DEBUG_LOG: return
    try:
        with open(GEMINI_DEBUG_LOG, "a") as f:
            f.write(f"\n=== {id(resp)} text_len={len(text)} ===\n")
            f.write(text[:2000])
            f.write(f"\nfinish: {[c.finish_reason for c in getattr(resp, 'candidates', [])]}")
    except Exception:
        pass

def unified_text(resp) -> str:
    """Extract the text output from either OpenAI Responses or Gemini response.

    Also strips markdown fences (```json ... ```) that Gemini often wraps JSON in.
    """
    raw = ""
    # OpenAI Responses API
    try:
        for item in getattr(resp, "output", None) or []:
            if getattr(item, "type", "") == "message":
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        raw = c.text.strip()
                        break
                if raw:
                    break
    except Exception:
        pass
    # Gemini
    if not raw:
        try:
            t = getattr(resp, "text", None)
            if t:
                raw = t.strip()
        except Exception:
            pass
    # Fallback: candidates
    if not raw:
        try:
            raw = resp.candidates[0].content.parts[0].text.strip()
        except Exception:
            pass
    # Strip markdown code fences.
    if GEMINI_DEBUG_LOG: _log_gemini_response(resp, raw)
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def unified_usage(resp) -> Tuple[int, int]:
    """Return (input_tokens, output_tokens)."""
    # OpenAI
    u = getattr(resp, "usage", None)
    if u is not None:
        in_tok = getattr(u, "input_tokens", None)
        out_tok = getattr(u, "output_tokens", None)
        if in_tok is not None and out_tok is not None:
            return int(in_tok or 0), int(out_tok or 0)
    # Gemini
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        in_tok = getattr(um, "prompt_token_count", 0) or 0
        out_tok = getattr(um, "candidates_token_count", 0) or 0
        # Gemini thinking tokens are separate; include them as output
        think = getattr(um, "thoughts_token_count", 0) or 0
        return int(in_tok), int(out_tok + think)
    return 0, 0


def unified_cost(resp, model: str) -> float:
    """Rough cost estimate (does NOT capture image-tile surcharges precisely)."""
    in_tok, out_tok = unified_usage(resp)
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    return in_tok / 1_000_000 * pricing["input"] + out_tok / 1_000_000 * pricing["output"]
