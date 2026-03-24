"""Postprocess PERSONACHAT benchmark outputs into final CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _f4(value: Any) -> str:
    return f"{float(value):.4f}"


def _f6(value: Any) -> str:
    return f"{float(value):.6f}"


def save_results_tables(
    table1_results: List[Dict[str, Any]],
    table2_result: Optional[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save Table 1 and Table 2 CSV files with fixed output names."""
    output_dir.mkdir(parents=True, exist_ok=True)

    table1_path = output_dir / "results_table_1.csv"
    with open(table1_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Method",
                "Long-Horizon QA",
                "Multi-Session Continuity",
                "Hallucination Rate",
                "Answer Utility",
                "Retrieval Recall@3",
                "Retrieval MRR",
                "Retrieval nDCG@5",
                "Evidence Hit Rate",
                "Fact Retention",
                "Preference Retention",
                "Contradiction Rate",
                "Unsupported Claim Proportion",
                "High-Risk Factual Hallucinations",
                "Avg Runtime/Turn (ms)",
            ]
        )
        for row in table1_results:
            writer.writerow(
                [
                    row.get("method", ""),
                    _f4(row.get("long_horizon_qa", 0.0)),
                    _f4(row.get("multi_session_continuity", 0.0)),
                    _f6(row.get("hallucination_rate", 0.0)),
                    _f4(row.get("answer_utility", 0.0)),
                    _f4(row.get("retrieval_recall_at_3", 0.0)),
                    _f4(row.get("retrieval_mrr", 0.0)),
                    _f4(row.get("retrieval_ndcg_at_5", 0.0)),
                    _f4(row.get("evidence_hit_rate", 0.0)),
                    _f4(row.get("fact_retention_rate", 0.0)),
                    _f4(row.get("preference_retention_rate", 0.0)),
                    _f4(row.get("contradiction_rate", 0.0)),
                    _f4(row.get("unsupported_claim_proportion", 0.0)),
                    _f4(row.get("high_risk_factual_hallucinations", 0.0)),
                    _f4(row.get("avg_runtime_per_turn_ms", 0.0)),
                ]
            )

    if table2_result and table2_result.get("applicable", False):
        table2_path = output_dir / "results_table_2.csv"
        with open(table2_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Probe", "Pre-Consolidation", "Post-Consolidation", "Delta Improvement"])
            writer.writerow(
                [
                    "Delayed Recall Accuracy",
                    _f4(table2_result.get("delayed_recall_pre", 0.0)),
                    _f4(table2_result.get("delayed_recall_post", 0.0)),
                    _f4(table2_result.get("delayed_recall_improvement", 0.0)),
                ]
            )
            writer.writerow(
                [
                    "Cue-Based Recall",
                    _f4(table2_result.get("cue_based_pre", 0.0)),
                    _f4(table2_result.get("cue_based_post", 0.0)),
                    _f4(table2_result.get("cue_based_improvement", 0.0)),
                ]
            )
            writer.writerow(
                [
                    "Cross-Episode Integration",
                    _f4(table2_result.get("integration_pre", 0.0)),
                    _f4(table2_result.get("integration_post", 0.0)),
                    _f4(table2_result.get("integration_improvement", 0.0)),
                ]
            )
            writer.writerow(
                [
                    "Schema Utilization Rate",
                    _f4(table2_result.get("schema_util_pre", 0.0)),
                    _f4(table2_result.get("schema_util_post", 0.0)),
                    _f4(table2_result.get("schema_util_improvement", 0.0)),
                ]
            )


def _load_latest_results_json(output_dir: Path) -> Dict[str, Any]:
    """Load latest personachat raw results JSON from output directory."""
    candidates = sorted(output_dir.glob("personachat_results_*.json"))
    if not candidates:
        raise FileNotFoundError("No personachat_results_*.json file found.")
    latest = candidates[-1]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess PERSONACHAT benchmark outputs")
    parser.add_argument("--output_dir", type=str, default="personachat_results", help="Directory with result JSON")
    parser.add_argument("--results_json", type=str, default=None, help="Path to specific raw result JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.results_json:
        with open(args.results_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = _load_latest_results_json(output_dir)

    table1 = raw.get("table1", [])
    table2_list = raw.get("table2", [])
    table2 = table2_list[0] if isinstance(table2_list, list) and table2_list else None

    save_results_tables(table1, table2, output_dir)
    print(f"Saved {output_dir / 'results_table_1.csv'}")
    if table2 and table2.get("applicable", False):
        print(f"Saved {output_dir / 'results_table_2.csv'}")


if __name__ == "__main__":
    main()
