"""Compute streaming-VideoQA metrics for an LVBench result directory.

Usage:
    python compute_metrics.py <result_dir>
    python compute_metrics.py <result_dir> --json  # machine-readable output
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

# ARS constants (paper Appendix C.4).
ARS_GAMMA_E = 0.25
ARS_GAMMA_L = 0.5
ARS_EPS = 1

QTYPES = ("Past", "Present", "Future")


def normalize_letter(s):
    """'(B)' / 'B' / 'b' / 'B.' -> 'B'. Empty/None -> ''."""
    if not s:
        return ""
    s = str(s).strip().upper().replace("(", "").replace(")", "")
    return s[:1] if s else ""


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _early_penalty(ta, ts, tau, gamma_e=ARS_GAMMA_E):
    return min(1.0, 2.0 * _sigmoid(gamma_e * (ta - ts) / (tau + ARS_EPS)))


def _late_penalty(ta, te, tau, gamma_l=ARS_GAMMA_L):
    return min(1.0, max(0.0, 1.0 - gamma_l * (ta - te) / (tau + ARS_EPS)))


def _ars_score(ta, ts, te, tau, gamma_e=ARS_GAMMA_E, gamma_l=ARS_GAMMA_L):
    return _early_penalty(ta, ts, tau, gamma_e) * _late_penalty(ta, te, tau, gamma_l)


def compute_metrics(result_dir, subset_video_ids=None):
    """Compute accuracy, latency and ARS for <result_dir>/question_results, optionally restricted to a set of video_ids."""
    qres_dir = os.path.join(result_dir, "question_results")
    if not os.path.isdir(qres_dir):
        raise FileNotFoundError(f"No question_results dir at {qres_dir}")

    correct = 0
    total = 0
    per_type = defaultdict(lambda: {"correct": 0, "total": 0, "latencies": [], "ars_inputs": []})
    all_ref_spans = []   # te - ts of all questions, for tau

    for f in glob.glob(os.path.join(qres_dir, "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue

        if subset_video_ids is not None and r.get("video_id") not in subset_video_ids:
            continue

        pred = normalize_letter(r.get("pred_letter") or r.get("pred_answer", ""))
        gt = normalize_letter(r.get("gt_letter") or r.get("gt_answer", ""))
        if not gt:
            continue

        total += 1
        qtype = r.get("task_type") or r.get("type") or "all"
        per_type[qtype]["total"] += 1
        if pred == gt:
            correct += 1
            per_type[qtype]["correct"] += 1

        # Latency: max(0, answer_time - gt_answer_time), answered questions only.
        a_t = r.get("answer_time")
        g_t = r.get("gt_answer_time")
        if a_t is not None and g_t is not None:
            per_type[qtype]["latencies"].append(max(0.0, float(a_t) - float(g_t)))

        # ARS: ts = start of the reference interval, te = ground-truth answer time.
        beg_ref = r.get("beg_ref")
        if g_t is not None and beg_ref is not None:
            all_ref_spans.append(float(g_t) - float(beg_ref))
            if a_t is not None:
                per_type[qtype]["ars_inputs"].append((float(a_t), float(beg_ref), float(g_t)))

    def acc(d):
        return d["correct"] / d["total"] if d["total"] > 0 else 0.0

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    empty = {"correct": 0, "total": 0, "latencies": [], "ars_inputs": []}
    types = {qt: per_type.get(qt, empty) for qt in QTYPES}

    # tau is computed once over all questions; unanswered questions get ARS = 0.
    tau = statistics.median(all_ref_spans) if all_ref_spans else 0.0

    def ars_sum(d):
        return sum(_ars_score(ta, ts, te, tau) for (ta, ts, te) in d["ars_inputs"])

    all_lat = [x for d in per_type.values() for x in d["latencies"]]
    all_ars_sum = sum(ars_sum(d) for d in per_type.values())

    m = {
        "result_dir": result_dir,
        "total": total,
        "correct": correct,
        "overall_acc": correct / total if total > 0 else 0.0,
    }
    for qt, d in types.items():
        k = qt.lower()
        m[f"{k}_correct"] = d["correct"]
        m[f"{k}_total"] = d["total"]
        m[f"{k}_acc"] = acc(d)
    m.update({
        "avg_latency": avg(all_lat),
        "n_latency": len(all_lat),
        "per_type_avg_latency": {qt: avg(d["latencies"]) for qt, d in types.items()},
        "ars_tau": tau,
        "avg_ars": all_ars_sum / total if total > 0 else 0.0,
        "per_type_avg_ars": {qt: (ars_sum(d) / d["total"] if d["total"] > 0 else 0.0)
                             for qt, d in types.items()},
    })
    return m


def print_human(m):
    name = os.path.basename(m["result_dir"].rstrip("/"))
    print(f"\n=== {name} ===")
    print(f"  Overall:  acc {m['correct']}/{m['total']} = {100*m['overall_acc']:.2f}%"
          f"  | latency {m['avg_latency']:.2f}s (n={m['n_latency']})  | ARS {m['avg_ars']:.3f}")
    for qt in QTYPES:
        k = qt.lower()
        print(f"  {qt + ':':9s} acc {m[k + '_correct']}/{m[k + '_total']} = {100*m[k + '_acc']:.2f}%"
              f"  | latency {m['per_type_avg_latency'][qt]:.1f}s  | ARS {m['per_type_avg_ars'][qt]:.3f}")
    print(f"  ARS tau = {m['ars_tau']:.1f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result_dir", nargs="?", help="Path to result directory containing question_results/")
    p.add_argument("--json", action="store_true", help="Output JSON instead of human-readable.")
    p.add_argument("--subset_path", type=str, default=None,
                   help="Optional JSON file with 'video_ids' key; restrict scoring to these videos.")
    args = p.parse_args()

    subset_video_ids = None
    if args.subset_path:
        with open(args.subset_path) as f:
            subset_video_ids = set(json.load(f)["video_ids"])
        print(f"[info] restricting to {len(subset_video_ids)} videos from {args.subset_path}", file=sys.stderr)

    if not args.result_dir:
        p.print_help()
        sys.exit(1)

    m = compute_metrics(args.result_dir, subset_video_ids=subset_video_ids)
    if args.json:
        print(json.dumps(m, indent=2, default=str))
    else:
        print_human(m)


if __name__ == "__main__":
    main()
