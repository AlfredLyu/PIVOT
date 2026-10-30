# PIVOT: A Unified Agentic Framework for Streaming Long-Video Understanding

**Qiushi Lyu<sup>1</sup>, Qianlan Yang<sup>1</sup>, Ziqi Pang<sup>1</sup>, Dongxu Li<sup>2</sup>, Wentao Wu<sup>2</sup>, Zhe Gan<sup>2</sup>, Yu-Xiong Wang<sup>1</sup>, Liang-Yan Gui<sup>1</sup>**

<sup>1</sup>University of Illinois Urbana-Champaign &nbsp; <sup>2</sup>Apple

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026-6f42c1?style=flat" alt="NeurIPS 2026">
  &nbsp;
  <!-- TODO: replace #TODO with the arXiv link -->
  <a href="#TODO"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=flat&logo=arxiv&logoColor=white" alt="Paper"></a>
  &nbsp;
  <a href="https://alfredlyu.github.io/PIVOT/"><img src="https://img.shields.io/badge/Project-Page-1f72ff?style=flat&logo=googlechrome&logoColor=white" alt="Project Page"></a>
</p>

---

**PIVOT** (**P**lanning over **I**ncremental **V**ideo **O**bservations for **T**imely answering) is a
training-free agentic framework for streaming long-video question answering. Video arrives as a
continuous stream, questions are revealed at arbitrary times, and each question may concern the
**past**, **present**, or **future** relative to when it is asked — without any task-type label.
Under strict sequential access and a bounded shared memory, the system must decide for every active
question whether it is already answerable or should be **deferred** until more evidence appears.

This repository contains:

* **PIVOT** — streaming memory maintenance + a planner-controlled action loop
  (`retrieve`, `analyze`, `reflect`, `answer`, `defer`).
* **LVBench-Ref2Stream** — a streaming long-video QA benchmark built from LVBench, in which past, present,
  and future questions carry query-time and answer-time annotations.
* Baselines and evaluation code used in the paper.

## Method overview

PIVOT processes the stream in chunks (16 frames at 0.5 fps by default) and splits each step into two stages:

1. **Streaming memory maintenance.** A *Memory Manager* reads the current chunk (plus overlapping
   tail frames from the previous chunk) and updates a bounded structured memory:
   a transient **frame buffer**, a **scene log** of self-contained captions, a **character registry**
   of stable identities, and per-question **evidence notes**. A token-budgeted reservoir sampler
   keeps the total memory under a fixed budget (50K token-equivalents by default).
2. **Agentic question processing.** For every active question, a **core planner** iteratively picks
   one action — `retrieve` (LLM-reranked memory retrieval), `analyze` (visual-text reasoning over
   the current chunk + retrieved context), `reflect` (verify a candidate answer), `answer`, or
   `defer` — until it commits or the per-chunk turn budget is exhausted.

All modules use **Gemini-3.1-Flash-Lite-Preview** as the backbone in the paper. The exact system
prompts (paper Appendix D.3) live inside `pivot/run_pivot.py`.

## Repository structure

```
PIVOT/
├── pivot/
│   ├── run_pivot.py              # PIVOT on LVBench-Ref2Stream (main method, Table 1)
│   └── llm_api.py                # unified OpenAI / Gemini wrapper (+ cost accounting)
├── baselines/
│   ├── mllm_fifo.py              # generic MLLM + FIFO visual buffer (API & open-source models)
│   ├── mllm_reservoir.py         # generic MLLM + reservoir visual buffer
│   └── prior_methods/            # adapters for Dispider and TimeChat-Online (run inside their repos)
├── benchmarks/
│   └── lvbench_ref2stream/
│       ├── data/
│       │   ├── lvbench_ref2stream.json               # the benchmark (1 temporal variant per source QA)
│       │   ├── lvbench_ref2stream_all_variants.json  # all past/present/future variants
│       │   └── video_id_to_youtube_key.json          # LVBench video id ↔ YouTube id
│       └── extract_frames.py                         # video → frame folders expected by the runners
├── evaluation/
│   └── compute_metrics.py        # accuracy / latency / ARS for a result directory
└── scripts/                      # one-line launchers for every experiment
```

## Installation

```bash
git clone https://github.com/AlfredLyu/PIVOT.git && cd PIVOT
conda create -n pivot python=3.10 -y && conda activate pivot
pip install -r requirements.txt
# ffmpeg is required for frame extraction
```

Set API keys (see `.env.example`):

```bash
export GEMINI_API_KEY=...     # PIVOT, Gemini baselines
export OPENAI_API_KEY=...     # GPT-4o baselines (optional)
```

The open-source MLLM baselines additionally need `torch`, `torchvision`, `transformers` and
`accelerate` (listed in `requirements.txt`) and one GPU.

## Data preparation

### LVBench-Ref2Stream

The question/annotation files ship with this repo in `benchmarks/lvbench_ref2stream/data/`.
Videos must be obtained from [LVBench](https://github.com/THUDM/LVBench) (subject to its license).

1. Download the LVBench videos and name them by the 8-digit `video_id` used in our annotations
   (`data/lvbench/videos/00000000.mp4`, ...). `video_id_to_youtube_key.json` gives the mapping to
   the original YouTube ids; the id is the 0-based line index in LVBench's `video_info.meta.jsonl`.
2. Extract 1 fps frames:

   ```bash
   python benchmarks/lvbench_ref2stream/extract_frames.py \
       --video_dir data/lvbench/videos --out_dir data/lvbench/frames_1fps --fps 1
   ```

   All runners default to `--frames_dir data/lvbench/frames_1fps` (or `$LVBENCH_FRAMES_DIR`) and
   sub-sample with `--frame_stride 2`, i.e. the 0.5 fps stream used in the paper.

Each benchmark entry looks like:

```json
{
  "uid": "55_Past",
  "video_id": "00000000",
  "question": "What year appears in the opening caption of the video?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633",
  "answer": "D",
  "time_reference": "00:15-00:19",   // LVBench reference interval [l, r]
  "ask_time": "43:58",               // query time t
  "type": "Past",                    // Past / Present / Future — NOT shown to the model
  "question_type": ["key information retrieval"]
}
```

The ground-truth answer time is `g = t` for Past/Present questions and `g = r` for Future questions.

## Running

All launchers forward extra arguments, e.g. `--subset_path`, `--num_workers`, `--result_dir_suffix`.
Runs are resumable: questions with an existing result file are skipped. The accumulated API cost is
printed during the run and saved to `summary.json`.

```bash
# Main results (Table 1)
bash scripts/run_pivot_lvbench.sh
bash scripts/run_baselines.sh            # FIFO / reservoir MLLM wrappers
```

Prior streaming methods (Dispider, TimeChat-Online) are evaluated with the adapters in
`baselines/prior_methods/`; see the README there.

Results are written to `results/<method>/<run_name>/{question_results,video_summaries,video_traces}/`, where
`<run_name>` ends with the optional `--result_dir_suffix`:

| `<method>` | `<run_name>` |
|---|---|
| `pivot` | `<model><suffix>` |
| `fifo`, `reservoir` | `<model>_vt<max_visual_tokens>_cf<cframe>_stride<frame_stride>_chunk<chunk_size><suffix>` |

`<model>` is the model name with `-`/`.` replaced by `_`/`p` (e.g. `gemini_3p1_flash_lite_preview`).
`video_traces/` contains the full memory state and every planner action / tool input-output, which
is useful for inspecting PIVOT's decisions.

## Evaluation

```bash
python evaluation/compute_metrics.py results/pivot/gemini_3p1_flash_lite_preview
```

reports, overall and per temporal type:

* **Accuracy** — multiple-choice accuracy over all questions; unanswered questions count as wrong.
* **Latency** — `(1/n) Σ max(0, t_a − g)` in seconds of video time over answered questions (early answers get zero latency).
* **ARS** — Answer Readiness Score following StreamReady (paper Appendix C.4); unanswered questions score 0.

## Main results (LVBench-Ref2Stream)

Accuracy (%) and latency (s) on LVBench-Ref2Stream (paper Table 1).
Higher accuracy and lower latency are better.

<table>
<thead>
<tr><th rowspan="2" align="left">Method</th><th colspan="2">Overall</th><th colspan="2">Past</th><th colspan="2">Present</th><th colspan="2">Future</th></tr>
<tr><th>Acc ↑</th><th>Lat ↓</th><th>Acc ↑</th><th>Lat ↓</th><th>Acc ↑</th><th>Lat ↓</th><th>Acc ↑</th><th>Lat ↓</th></tr>
</thead>
<tbody>
<tr><td colspan="9"><i>Generic MLLMs with FIFO visual-context wrapper</i></td></tr>
<tr><td align="left">Qwen2.5-VL-7B-Instruct</td><td align="center">26.7</td><td align="center">136.2</td><td align="center">24.1</td><td align="center">233.4</td><td align="center">38.1</td><td align="center">186.8</td><td align="center">27.7</td><td align="center">35.7</td></tr>
<tr><td align="left">LLaVA-OneVision-7B</td><td align="center">34.2</td><td align="center">122.9</td><td align="center">32.3</td><td align="center">208.2</td><td align="center">57.1</td><td align="center">176.9</td><td align="center">32.9</td><td align="center">34.1</td></tr>
<tr><td align="left">InternVL-8B</td><td align="center">33.3</td><td align="center">127.7</td><td align="center">25.2</td><td align="center">217.8</td><td align="center">52.4</td><td align="center">196.4</td><td align="center">38.5</td><td align="center">33.5</td></tr>
<tr><td align="left">Gemini-3.1-flash-lite</td><td align="center">44.8</td><td align="center">97.2</td><td align="center">31.4</td><td align="center">181.4</td><td align="center">79.4</td><td align="center">42.9</td><td align="center"><b>53.2</b></td><td align="center">27.4</td></tr>
<tr><td align="left">GPT-4o</td><td align="center">46.7</td><td align="center">89.6</td><td align="center">36.1</td><td align="center">178.7</td><td align="center">77.8</td><td align="center"><b>31.3</b></td><td align="center">52.8</td><td align="center">19.8</td></tr>
<tr><td colspan="9"><i>Generic MLLMs with reservoir visual-context wrapper</i></td></tr>
<tr><td align="left">Gemini-3.1-flash-lite</td><td align="center">44.1</td><td align="center">85.3</td><td align="center">41.4</td><td align="center">83.6</td><td align="center">54.0</td><td align="center">116.4</td><td align="center">45.5</td><td align="center">82.4</td></tr>
<tr><td align="left">GPT-4o</td><td align="center">46.0</td><td align="center">89.8</td><td align="center">44.5</td><td align="center">98.4</td><td align="center">57.1</td><td align="center">101.6</td><td align="center">45.9</td><td align="center">79.7</td></tr>
<tr><td colspan="9"><i>Prior streaming video methods</i></td></tr>
<tr><td align="left">Dispider</td><td align="center">32.8</td><td align="center">45.6</td><td align="center">32.3</td><td align="center">71.9</td><td align="center">42.9</td><td align="center">72.4</td><td align="center">31.8</td><td align="center"><b>11.9</b></td></tr>
<tr><td align="left">TimeChat-Online</td><td align="center">23.5</td><td align="center">40.8</td><td align="center">24.6</td><td align="center"><b>62.4</b></td><td align="center">42.9</td><td align="center">51.5</td><td align="center">19.9</td><td align="center">18.2</td></tr>
<tr><td align="left"><b>PIVOT (ours)</b></td><td align="center"><b>56.6</b></td><td align="center"><b>37.6</b></td><td align="center"><b>56.9</b></td><td align="center">66.0</td><td align="center"><b>81.0</b></td><td align="center">36.3</td><td align="center">53.0</td><td align="center">12.2</td></tr>
</tbody>
</table>

See the paper for the full tables, ablations, and OVO-Bench / StreamingBench results.

## Citation

```bibtex
@inproceedings{lyu2026pivot,
  title     = {{PIVOT}: A Unified Agentic Framework for Streaming Long-Video Understanding},
  author    = {Lyu, Qiushi and Yang, Qianlan and Pang, Ziqi and Li, Dongxu and Wu, Wentao and
               Gan, Zhe and Wang, Yu-Xiong and Gui, Liang-Yan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

LVBench-Ref2Stream is derived from [LVBench](https://github.com/THUDM/LVBench); please also cite
LVBench and follow its license when using the benchmark. We thank the authors of
[Dispider](https://github.com/Mark12Ding/Dispider) and [TimeChat-Online](https://github.com/yaolinli/TimeChat-Online)
for releasing their code and data.
