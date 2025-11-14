#!/usr/bin/env python3
"""
Evaluate Assessor - Compare our LLM assessor scores against human ratings

Loads one or more labeled datasets (JSON/JSONL) containing jokes and
human ratings, runs the existing JokeAssessor to score those jokes, and
reports agreement metrics:

- Pearson correlation (linear correlation)
- Spearman correlation (rank correlation)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean bias (assessor - human)

Notes:
- Uses assessor's caching so repeated runs are cheap
- Attempts to auto-detect text and rating fields; can be overridden via CLI
- Supports quick normalization of human scores to a 1–10 scale
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Any

import numpy as np


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load JSON or JSONL into a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Some datasets may nest under a key
            for key in ["data", "rows", "items", "examples", "results"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Fallback: single record
            return [data]
        raise ValueError("Unsupported JSON structure")


def load_merged_data(path: str) -> Dict[str, Dict[str, Any]]:
    """Specialized loader for data/merged_data_combined.json.

    Returns mapping ID -> {"text": str, "score": float}
    where score is the provided "Funniness (1-5)" numeric value (0–5 range).
    """
    rows = load_records(path)
    result: Dict[str, Dict[str, Any]] = {}
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("ID", "")).strip()
        text = str(rec.get("text", "")).strip()
        score = rec.get("Funniness (1-5)")
        if not rid or not text or not isinstance(score, (int, float)):
            continue
        result[rid] = {"text": text, "score": float(score)}
    return result


def load_expunations_scores(path: str) -> Dict[str, float]:
    """Specialized loader for data/expunations_annotated_full.json.

    The file contains per-annotator arrays for "Funniness (1-5)"; we compute
    the mean over numeric entries (treat missing/empty as non-numeric and drop).

    Returns mapping ID -> avg_score (0–5 range).
    """
    rows = load_records(path)
    result: Dict[str, float] = {}
    for rec in rows:
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("ID", "")).strip()
        scores = rec.get("Funniness (1-5)")
        if not rid or not isinstance(scores, list):
            continue
        numeric = [float(s) for s in scores if isinstance(s, (int, float))]
        if not numeric:
            continue
        avg = float(sum(numeric) / len(numeric))
        result[rid] = avg
    return result


def guess_text_key(sample: Dict[str, Any]) -> Optional[str]:
    candidates = [
        "joke",
        "text",
        "content",
        "setup",
        "punchline",
        "sentence",
    ]
    for key in candidates:
        if key in sample and isinstance(sample[key], str) and sample[key].strip():
            return key
    # Fabricio data might have different fields; try nested structure
    for key in sample.keys():
        val = sample[key]
        if isinstance(val, str) and len(val) > 10:
            return key
    return None


def guess_score_key(sample: Dict[str, Any]) -> Optional[str]:
    candidates = [
        "rating",
        "score",
        "human_score",
        "funny",
        "humour",
        "humor",
        "label",
        "gold",
        "annotation",
    ]
    for key in candidates:
        if key in sample and isinstance(sample[key], (int, float)):
            return key
    # Try to find numeric-looking value
    for key, val in sample.items():
        if isinstance(val, (int, float)):
            return key
    return None


def normalize_human_score(value: float, assume_min: float, assume_max: float) -> float:
    """Map [assume_min, assume_max] to [1, 10]."""
    if assume_max == assume_min:
        return float(value)
    scaled = 1.0 + 9.0 * (float(value) - assume_min) / (assume_max - assume_min)
    return max(1.0, min(10.0, scaled))


def compute_spearman(x: List[float], y: List[float]) -> float:
    """Compute Spearman's rho with average ranks for ties."""

    def average_ranks(values: List[float]) -> List[float]:
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1)
        # Handle ties: average the ranks of equal values
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j]] == values[order[j + 1]]:
                j += 1
            if j > i:
                avg = ranks[order[i : j + 1]].mean()
                ranks[order[i : j + 1]] = avg
            i = j + 1
        return ranks.tolist()

    rx = average_ranks(x)
    ry = average_ranks(y)
    rx = np.array(rx, dtype=float)
    ry = np.array(ry, dtype=float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JokeAssessor against human-rated datasets"
    )
    parser.add_argument(
        "--merged",
        type=str,
        default=os.path.join("data", "merged_data_combined.json"),
        help="Path to merged_data_combined.json (ID, text, Funniness (1-5) numeric)",
    )
    parser.add_argument(
        "--expunations",
        type=str,
        default=os.path.join("data", "expunations_annotated_full.json"),
        help="Path to expunations_annotated_full.json (arrays of annotator ratings)",
    )
    # Text/score keys no longer needed for the two known formats
    parser.add_argument(
        "--assume-scale",
        type=str,
        default="auto",
        help="Assumed human rating scale (auto|1-10|1-5|0-1). Values will be normalized to 1–10",
    )
    parser.add_argument(
        "--limit", type=int, help="Only evaluate first N ids (join set)"
    )
    parser.add_argument(
        "--merged-only",
        action="store_true",
        default=True,
        help="Ignore expunations; evaluate using merged file only",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default="median",
        help="Binary split for classification metrics: 'median' or a numeric threshold on human 1–10 scale",
    )

    args = parser.parse_args()

    # Parse assumed scale (may be auto)
    assume_min: Optional[float] = None
    assume_max: Optional[float] = None
    if args.assume_scale != "auto":
        try:
            parts = args.assume_scale.split("-")
            assume_min = float(parts[0])
            assume_max = float(parts[1])
        except (ValueError, IndexError, AttributeError):
            print("[ERROR] --assume-scale must be 'auto' or like '1-10' or '0-1'")
            sys.exit(1)

    # Load assessor
    try:
        from evaluation.assessor import JokeAssessor

        assessor = JokeAssessor()
    except (ImportError, RuntimeError) as e:
        print(f"[ERROR] Failed to load assessor: {e}")
        sys.exit(1)

    # Load known formats and join on ID
    merged = load_merged_data(args.merged) if args.merged else {}
    exp_scores = (
        {}
        if args.merged_only
        else (load_expunations_scores(args.expunations) if args.expunations else {})
    )

    # Build join set (intersection) to ensure we compare same IDs when both given
    ids = set(merged.keys())
    if exp_scores:
        ids &= set(exp_scores.keys())
    if args.limit:
        ids = set(list(ids)[: args.limit])

    if not ids:
        # Fallback: if only merged present, use it directly
        ids = set(merged.keys()) if merged else set()
        if args.limit:
            ids = set(list(ids)[: args.limit])

    jokes: List[str] = []
    human_raw_scores: List[float] = []
    for rid in ids:
        row = merged.get(rid)
        if not row:
            continue
        text = str(row.get("text", "")).strip()
        score = exp_scores.get(rid, row.get("score"))
        if not text or not isinstance(score, (int, float)):
            continue
        jokes.append(text)
        human_raw_scores.append(float(score))

    if not jokes:
        print("[ERROR] No jokes found after parsing inputs")
        sys.exit(1)

    print(f"[INFO] Loaded {len(jokes)} labeled jokes from merged/expunations datasets")

    # Determine normalization scale if auto
    if assume_min is None or assume_max is None:
        if not human_raw_scores:
            print("[ERROR] No human scores found to infer scale")
            sys.exit(1)
        observed_min = float(min(human_raw_scores))
        observed_max = float(max(human_raw_scores))

        # Heuristics for common scales
        if observed_min >= 0.0 and observed_max <= 1.0:
            assume_min, assume_max = 0.0, 1.0
            print("[INFO] Detected rating scale: 0–1 (auto)")
        elif observed_min >= 1.0 and observed_max <= 5.0:
            assume_min, assume_max = 1.0, 5.0
            print("[INFO] Detected rating scale: 1–5 (auto)")
        elif observed_min >= 1.0 and observed_max <= 10.0:
            assume_min, assume_max = 1.0, 10.0
            print("[INFO] Detected rating scale: 1–10 (auto)")
        else:
            assume_min, assume_max = observed_min, observed_max
            print(
                f"[INFO] Detected rating scale: {assume_min:.3g}–{assume_max:.3g} (auto)"
            )

    # Normalize human scores now
    human_scores: List[float] = [
        normalize_human_score(s, assume_min, assume_max) for s in human_raw_scores
    ]

    # Run assessor with caching; compute metrics
    predicted_scores: List[float] = []
    for i, joke in enumerate(jokes, 1):
        if i % 25 == 1 or i == len(jokes):
            print(f"  Scoring {i}/{len(jokes)}…")
        assessment = assessor.evaluate_joke(joke)
        predicted_scores.append(float(assessment.get("score", 0)))

    human_np = np.array(human_scores, dtype=float)
    pred_np = np.array(predicted_scores, dtype=float)

    # Metrics
    pearson = (
        float(np.corrcoef(human_np, pred_np)[0, 1])
        if human_np.std() > 0 and pred_np.std() > 0
        else 0.0
    )
    spearman = compute_spearman(human_scores, predicted_scores)
    mae = float(np.mean(np.abs(pred_np - human_np)))
    rmse = float(math.sqrt(np.mean((pred_np - human_np) ** 2)))
    bias = float(np.mean(pred_np - human_np))

    print("\nAgreement Metrics (human vs assessor, all on 1–10 scale):")
    print(f"  Pearson r: {pearson:.3f}")
    print(f"  Spearman ρ: {spearman:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Mean bias (assessor - human): {bias:+.3f}")

    # Simple calibration bins (optional quick view)
    try:
        bins = [1, 3, 5, 7, 9, 10.0001]
        bin_indices = np.digitize(human_np, bins)
        print("\nCalibration by human-score bins:")
        for b in sorted(set(bin_indices)):
            mask = bin_indices == b
            if mask.sum() < 5:
                continue
            avg_h = float(human_np[mask].mean())
            avg_p = float(pred_np[mask].mean())
            print(
                f"  Bin {b} (n={int(mask.sum())}): human {avg_h:.2f} → assessor {avg_p:.2f}"
            )
    except (ValueError, ArithmeticError):
        pass

    # Binary split evaluation (top vs bottom by human score)
    try:
        if isinstance(args.binary, str) and args.binary.lower() == "median":
            threshold = float(np.median(human_np))
            label = "median"
        else:
            threshold = float(args.binary)
            label = f"{threshold:.2f}"

        human_bin = human_np > threshold
        pred_bin = pred_np > threshold

        accuracy = float(np.mean(human_bin == pred_bin))

        # Precision/recall/F1 for positive class (above threshold)
        tp = float(np.sum((pred_bin == 1) & (human_bin == 1)))
        fp = float(np.sum((pred_bin == 1) & (human_bin == 0)))
        fn = float(np.sum((pred_bin == 0) & (human_bin == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print(f"\nBinary split vs human (threshold={label}):")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision (pos): {precision:.3f}")
        print(f"  Recall (pos): {recall:.3f}")
        print(f"  F1 (pos): {f1:.3f}")

        # Rank overlap: top half overlap
        n = len(human_np)
        k = n // 2
        human_top_idx = set(np.argsort(-human_np)[:k].tolist())
        pred_top_idx = set(np.argsort(-pred_np)[:k].tolist())
        overlap = len(human_top_idx & pred_top_idx)
        overlap_rate = overlap / max(1, k)
        print(f"  Top-half overlap: {overlap}/{k} ({overlap_rate:.3f})")
    except (ValueError, ArithmeticError):
        pass


if __name__ == "__main__":
    main()
