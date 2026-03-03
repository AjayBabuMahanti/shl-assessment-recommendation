"""
==============================================================
 SHL Assessment Recommendation System — Evaluation Module
 Metric: Mean Recall@K  (default K = 10)
==============================================================

Usage
-----
  # Run with default settings (K=10, CSV auto-detected):
  python evaluation/evaluate.py

  # Custom CSV and K:
  python evaluation/evaluate.py --csv data/eval_labels.csv --k 5

  # Import as library:
  from evaluation.evaluate import evaluate_model
  report = evaluate_model("data/eval_labels.csv", k=10)

CSV format expected
-------------------
  query,assessment_url
  "Looking for Java developer",https://www.shl.com/.../java-new
  "Looking for Java developer",https://www.shl.com/.../personality-test
  "Python analyst role",https://www.shl.com/.../python-new
==============================================================
"""

import argparse
import logging
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

# ── Add project root to sys.path so embeddings package is importable ──────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Import the retrieval layer (does NOT call Mistral — pure vector search) ───
from embeddings.query_engine import _retrieve  # noqa: E402  (internal but stable)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(message)s",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public adapter — recommend_assessments
# ──────────────────────────────────────────────────────────────────────────────

def recommend_assessments(query: str, k: int = 10) -> List[dict]:
    """
    Return up to *k* assessment dicts for *query*.

    Each dict contains at minimum:
        { "name": str, "url": str, "test_type": str, ... }

    This calls the vector-search layer directly (no Mistral LLM round-trip),
    which is both faster and deterministic for evaluation purposes.
    """
    candidates = _retrieve(query, n_results=k)
    # _retrieve returns dicts that already contain name, url, test_type, etc.
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the evaluation CSV file.

    Expected columns: query, assessment_url
    Returns a cleaned DataFrame with those two columns.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Evaluation CSV not found: '{csv_path}'\n"
            "Please provide a CSV with columns: query, assessment_url"
        )

    df = pd.read_csv(csv_path)

    required = {"query", "assessment_url"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Strip whitespace, drop rows with nulls in either key column
    df["query"]          = df["query"].astype(str).str.strip()
    df["assessment_url"] = df["assessment_url"].astype(str).str.strip()
    df.dropna(subset=["query", "assessment_url"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info("📂  Loaded %d rows from '%s'", len(df), csv_path)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Group relevant URLs by query
# ──────────────────────────────────────────────────────────────────────────────

def group_by_query(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Aggregate all relevant assessment URLs per unique query.

    Returns:
        { query_string: [url1, url2, ...], ... }
    """
    grouped: Dict[str, List[str]] = defaultdict(list)

    for _, row in df.iterrows():
        url = row["assessment_url"]
        if url and url not in grouped[row["query"]]:
            grouped[row["query"]].append(url)

    log.info("🗂️   Grouped into %d unique queries", len(grouped))
    return dict(grouped)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Recall@K for a single query
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    relevant:  List[str],
    predicted: List[str],
    k:         int = 10,
) -> float:
    """
    Compute Recall@K for a single query using exact URL matching.

    Recall@K = |relevant ∩ top-k predicted| / |relevant|

    Edge cases handled:
    - No relevant URLs       → returns 0.0
    - No predictions         → returns 0.0
    - Fewer than K predicted → uses however many are available
    - Duplicate predictions  → deduplicated before evaluation
    """
    if not relevant:
        return 0.0

    # Deduplicate while preserving order
    seen: set         = set()
    unique_predicted  = []
    for url in predicted:
        if url not in seen:
            seen.add(url)
            unique_predicted.append(url)

    top_k     = unique_predicted[:k]
    relevant_set = set(relevant)
    hits      = sum(1 for url in top_k if url in relevant_set)
    return hits / len(relevant_set)


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Full evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    csv_path:    str,
    k:           int  = 10,
    verbose:     bool = False,
) -> dict:
    """
    Orchestrate the full Recall@K evaluation pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the labeled evaluation CSV.
    k : int
        Cut-off rank (default 10).
    verbose : bool
        Print predicted URLs for each query.

    Returns
    -------
    dict with keys:
        mean_recall, std_recall, min_recall, max_recall,
        per_query  (list of dicts with query / recall / hits / n_relevant),
        k          (the K used),
        n_queries  (total evaluated),
        n_errors   (queries that threw an exception),
    """
    df          = load_dataset(csv_path)
    query_map   = group_by_query(df)
    queries     = list(query_map.keys())
    n_total     = len(queries)

    per_query_results: List[dict] = []
    recall_scores:     List[float] = []
    n_errors = 0

    log.info("\n%s", "=" * 65)
    log.info("🚀  Starting evaluation  |  K=%d  |  %d queries", k, n_total)
    log.info("%s\n", "=" * 65)

    for idx, query in enumerate(queries, start=1):
        relevant = query_map[query]
        log.info("[Query %d/%d]  Evaluating: \"%s\"", idx, n_total, query[:70])

        try:
            results        = recommend_assessments(query, k=k)
            predicted_urls = [r.get("url", "") for r in results]

            if verbose:
                log.info("  Predicted URLs:")
                for u in predicted_urls[:k]:
                    log.info("    • %s", u)

            score = recall_at_k(relevant, predicted_urls, k=k)
            hits  = int(round(score * len(relevant)))

        except KeyboardInterrupt:
            raise
        except Exception as exc:          # noqa: BLE001
            log.warning(
                "  ⚠️  Error for query '%s': %s — skipping.", query[:60], exc
            )
            n_errors += 1
            per_query_results.append({
                "query":      query,
                "recall":     None,
                "hits":       None,
                "n_relevant": len(relevant),
                "error":      str(exc),
            })
            continue

        recall_scores.append(score)
        per_query_results.append({
            "query":      query,
            "recall":     round(score, 4),
            "hits":       hits,
            "n_relevant": len(relevant),
            "error":      None,
        })

        log.info(
            "  ✅  Recall@%d = %.4f  (%d/%d relevant found)\n",
            k, score, hits, len(relevant),
        )

    # ── Summary statistics ───────────────────────────────────────────────────
    if recall_scores:
        mean_recall = statistics.mean(recall_scores)
        std_recall  = statistics.pstdev(recall_scores)   # population std dev
        min_recall  = min(recall_scores)
        max_recall  = max(recall_scores)
    else:
        mean_recall = std_recall = min_recall = max_recall = 0.0

    log.info("\n%s", "=" * 65)
    log.info("📊  EVALUATION SUMMARY")
    log.info("%s", "=" * 65)
    log.info("  Queries evaluated   : %d / %d", len(recall_scores), n_total)
    log.info("  Queries with errors : %d", n_errors)
    log.info("  K (cut-off rank)    : %d", k)
    log.info("  ──────────────────────────────────────")
    log.info("  Mean   Recall@%d    : %.4f", k, mean_recall)
    log.info("  Std    Recall@%d    : %.4f", k, std_recall)
    log.info("  Min    Recall@%d    : %.4f", k, min_recall)
    log.info("  Max    Recall@%d    : %.4f", k, max_recall)
    log.info("%s\n", "=" * 65)

    return {
        "mean_recall":    round(mean_recall, 4),
        "std_recall":     round(std_recall,  4),
        "min_recall":     round(min_recall,  4),
        "max_recall":     round(max_recall,  4),
        "per_query":      per_query_results,
        "k":              k,
        "n_queries":      n_total,
        "n_errors":       n_errors,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SHL Recommender using Mean Recall@K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type    = str,
        default = os.path.join(_ROOT, "data", "eval_labels.csv"),
        help    = "Path to the labeled evaluation CSV file.",
    )
    parser.add_argument(
        "--k",
        type    = int,
        default = 10,
        help    = "Rank cut-off for Recall@K.",
    )
    parser.add_argument(
        "--verbose",
        action  = "store_true",
        help    = "Print predicted URLs for each query.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        report = evaluate_model(
            csv_path = args.csv,
            k        = args.k,
            verbose  = args.verbose,
        )
    except FileNotFoundError as exc:
        log.error("\n❌  %s", exc)
        sys.exit(1)
    except ValueError as exc:
        log.error("\n❌  Invalid dataset: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("\n⚠️   Evaluation interrupted by user.")
        sys.exit(0)
