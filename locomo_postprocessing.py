"""Postprocess LOCOMO benchmark outputs into final CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _f4(value: Any) -> str:
    return f"{float(value):.4f}"


def _f6(value: Any) -> str:
    return f"{float(value):.6f}"


def save_research_graphs(
    table1_results: List[Dict[str, Any]],
    table2_result: Optional[Dict[str, Any]],
    research_dir: Path,
) -> None:
    """Save research-facing line/bar/column PNG charts in RESEARCH/ folder."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Skipping graph generation (matplotlib not installed).")
        return

    research_dir.mkdir(parents=True, exist_ok=True)

    # Graph 1 (BAR): Table 1 primary quality metrics by method.
    if table1_results:
        methods = [str(r.get("method", "")) for r in table1_results]
        qa = [_safe_float(r.get("long_horizon_qa", 0.0)) for r in table1_results]
        continuity = [_safe_float(r.get("multi_session_continuity", 0.0)) for r in table1_results]
        utility = [_safe_float(r.get("answer_utility", 0.0)) for r in table1_results]

        x = list(range(len(methods)))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar([i - width for i in x], qa, width=width, label="Long-Horizon QA")
        plt.bar(x, continuity, width=width, label="Multi-Session Continuity")
        plt.bar([i + width for i in x], utility, width=width, label="Answer Utility")
        plt.xticks(x, methods, rotation=20)
        plt.ylabel("Score")
        plt.title("LOCOMO Table 1: Quality Metrics by Method")
        plt.legend()
        plt.tight_layout()
        plt.savefig(research_dir / "locomo_table1_quality_bar.png", dpi=160)
        plt.close()

    # Graph 2 (LINE): Table 2 pre vs post across probes.
    if table2_result and table2_result.get("applicable", False):
        probes = [
            "Delayed Recall",
            "Cue-Based Recall",
            "Cross-Episode Integration",
            "Schema Utilization",
        ]
        pre = [
            _safe_float(table2_result.get("delayed_recall_pre", 0.0)),
            _safe_float(table2_result.get("cue_based_pre", 0.0)),
            _safe_float(table2_result.get("integration_pre", 0.0)),
            _safe_float(table2_result.get("schema_util_pre", 0.0)),
        ]
        post = [
            _safe_float(table2_result.get("delayed_recall_post", 0.0)),
            _safe_float(table2_result.get("cue_based_post", 0.0)),
            _safe_float(table2_result.get("integration_post", 0.0)),
            _safe_float(table2_result.get("schema_util_post", 0.0)),
        ]

        x = list(range(len(probes)))
        plt.figure(figsize=(10, 5))
        plt.plot(x, pre, marker="o", linewidth=2, label="Pre-Consolidation")
        plt.plot(x, post, marker="o", linewidth=2, label="Post-Consolidation")
        plt.xticks(x, probes, rotation=20)
        plt.ylabel("Score")
        plt.title("LOCOMO Table 2: Pre vs Post Consolidation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(research_dir / "locomo_table2_pre_post_line.png", dpi=160)
        plt.close()

    # Graph 3 (COLUMN): Efficiency and hallucination diagnostics by method.
    if table1_results:
        methods = [str(r.get("method", "")) for r in table1_results]
        runtime_ms = [_safe_float(r.get("avg_runtime_per_turn_ms", 0.0)) for r in table1_results]
        halluc = [_safe_float(r.get("hallucination_rate", 0.0)) for r in table1_results]

        x = list(range(len(methods)))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar([i - width / 2 for i in x], runtime_ms, width=width, label="Runtime/Turn (ms)")
        ax1.set_ylabel("Runtime/Turn (ms)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=20)

        ax2 = ax1.twinx()
        ax2.bar([i + width / 2 for i in x], halluc, width=width, label="Hallucination Rate", alpha=0.75)
        ax2.set_ylabel("Hallucination Rate")

        ax1.set_title("LOCOMO Diagnostics: Runtime vs Hallucination")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(research_dir / "locomo_runtime_hallucination_column.png", dpi=160)
        plt.close(fig)


def save_results_tables(
    table1_results: List[Dict[str, Any]],
    table2_result: Optional[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save Table 1 and Table 2 CSV with fixed filenames."""
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

    # Save research graphs (line/bar/column) in workspace RESEARCH folder.
    research_dir = Path(__file__).parent / "RESEARCH"
    save_research_graphs(table1_results, table2_result, research_dir)


def _load_latest_results_json(output_dir: Path) -> Dict[str, Any]:
    """Load latest raw results JSON from output directory."""
    candidates = sorted(output_dir.glob("locomo_results_*.json"))
    if not candidates:
        raise FileNotFoundError("No locomo_results_*.json file found.")
    latest = candidates[-1]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess LOCOMO benchmark outputs")
    parser.add_argument("--output_dir", type=str, default="locomo_results", help="Directory with result JSON")
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
