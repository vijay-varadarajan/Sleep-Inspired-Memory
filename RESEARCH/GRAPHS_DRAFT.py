"""
Graphs for RESEARCH results draft.

- Edit values only inside RESULTS_DATA.
- Run: python RESEARCH/GRAPHS_DRAFT.py
- Output figures: RESEARCH/figures/
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Editable data structure (copied from RESULTS_DRAFT.md, n=200)
# =========================================================
RESULTS_DATA: Dict[str, Dict] = {
    "PersonaMem": {
        "split": "benchmark",
        "n": 200,
        "methods": {
            "vanilla": {
                "long_horizon_qa": 14.20,
                "multi_session_continuity": 42.80,
                "hallucination_rate": 0.8840,
                "answer_utility": 8.02,
                "fact_retention": 0.6480,
                "high_risk_hallucinations": 0.0580,
                "avg_runtime_ms": 8920.40,
                "estimated_storage_mb": 520,
            },
            "rag": {
                "long_horizon_qa": 20.90,
                "multi_session_continuity": 49.60,
                "hallucination_rate": 0.7440,
                "answer_utility": 8.41,
                "fact_retention": 0.6760,
                "high_risk_hallucinations": 0.0460,
                "avg_runtime_ms": 10148.25,
                "estimated_storage_mb": 610,
            },
            "episodic": {
                "long_horizon_qa": 19.60,
                "multi_session_continuity": 57.20,
                "hallucination_rate": 1.1260,
                "answer_utility": 8.58,
                "fact_retention": 0.6690,
                "high_risk_hallucinations": 0.0830,
                "avg_runtime_ms": 9304.12,
                "estimated_storage_mb": 700,
            },
            "summarization": {
                "long_horizon_qa": 17.40,
                "multi_session_continuity": 54.80,
                "hallucination_rate": 0.8120,
                "answer_utility": 8.29,
                "fact_retention": 0.6570,
                "high_risk_hallucinations": 0.0520,
                "avg_runtime_ms": 9611.83,
                "estimated_storage_mb": 640,
            },
            "sleep": {
                "long_horizon_qa": 26.10,
                "multi_session_continuity": 64.70,
                "hallucination_rate": 0.5980,
                "answer_utility": 9.62,
                "fact_retention": 0.7290,
                "high_risk_hallucinations": 0.0320,
                "avg_runtime_ms": 10872.44,
                "estimated_storage_mb": 370,
            },
        },
    },
    "PersonaChat": {
        "split": "validation",
        "n": 200,
        "methods": {
            "vanilla": {
                "long_horizon_qa": 45.20,
                "multi_session_continuity": 56.40,
                "hallucination_rate": 2.0810,
                "answer_utility": 8.05,
                "fact_retention": 0.6390,
                "high_risk_hallucinations": 0.2040,
                "avg_runtime_ms": 4381.55,
                "estimated_storage_mb": 505,
            },
            "rag": {
                "long_horizon_qa": 41.30,
                "multi_session_continuity": 55.10,
                "hallucination_rate": 1.9320,
                "answer_utility": 7.92,
                "fact_retention": 0.6280,
                "high_risk_hallucinations": 0.1710,
                "avg_runtime_ms": 5128.43,
                "estimated_storage_mb": 595,
            },
            "episodic": {
                "long_horizon_qa": 47.10,
                "multi_session_continuity": 60.80,
                "hallucination_rate": 2.2210,
                "answer_utility": 8.11,
                "fact_retention": 0.6680,
                "high_risk_hallucinations": 0.1760,
                "avg_runtime_ms": 4726.90,
                "estimated_storage_mb": 680,
            },
            "summarization": {
                "long_horizon_qa": 33.50,
                "multi_session_continuity": 43.70,
                "hallucination_rate": 1.9480,
                "answer_utility": 6.44,
                "fact_retention": 0.5310,
                "high_risk_hallucinations": 0.1180,
                "avg_runtime_ms": 4894.22,
                "estimated_storage_mb": 625,
            },
            "sleep": {
                "long_horizon_qa": 53.80,
                "multi_session_continuity": 67.20,
                "hallucination_rate": 1.6830,
                "answer_utility": 8.97,
                "fact_retention": 0.7140,
                "high_risk_hallucinations": 0.1240,
                "avg_runtime_ms": 10921.37,
                "estimated_storage_mb": 355,
            },
        },
    },
    "LOCOMO": {
        "split": "validation",
        "n": 200,
        "methods": {
            "vanilla": {
                "long_horizon_qa": 62.40,
                "multi_session_continuity": 39.10,
                "hallucination_rate": 2.7440,
                "answer_utility": 1.51,
                "fact_retention": 0.6030,
                "high_risk_hallucinations": 0.2280,
                "avg_runtime_ms": 10284.71,
                "estimated_storage_mb": 540,
            },
            "rag": {
                "long_horizon_qa": 50.20,
                "multi_session_continuity": 31.00,
                "hallucination_rate": 2.1140,
                "answer_utility": 1.29,
                "fact_retention": 0.5630,
                "high_risk_hallucinations": 0.1760,
                "avg_runtime_ms": 10976.36,
                "estimated_storage_mb": 632,
            },
            "episodic": {
                "long_horizon_qa": 55.10,
                "multi_session_continuity": 41.90,
                "hallucination_rate": 3.5110,
                "answer_utility": 1.78,
                "fact_retention": 0.6220,
                "high_risk_hallucinations": 0.2480,
                "avg_runtime_ms": 6688.18,
                "estimated_storage_mb": 712,
            },
            "summarization": {
                "long_horizon_qa": 48.60,
                "multi_session_continuity": 35.20,
                "hallucination_rate": 2.4630,
                "answer_utility": 1.44,
                "fact_retention": 0.5780,
                "high_risk_hallucinations": 0.1950,
                "avg_runtime_ms": 9315.02,
                "estimated_storage_mb": 648,
            },
            "sleep": {
                "long_horizon_qa": 60.80,
                "multi_session_continuity": 46.10,
                "hallucination_rate": 2.1930,
                "answer_utility": 2.23,
                "fact_retention": 0.6560,
                "high_risk_hallucinations": 0.1680,
                "avg_runtime_ms": 8460.55,
                "estimated_storage_mb": 382,
            },
        },
    },
    "OK-VQA": {
        "split": "validation",
        "n": 200,
        "methods": {
            "vanilla": {
                "long_horizon_qa": 57.10,
                "multi_session_continuity": 43.90,
                "hallucination_rate": 6.8120,
                "answer_utility": 4.83,
                "fact_retention": 0.4860,
                "high_risk_hallucinations": 0.0830,
                "avg_runtime_ms": 6844.30,
                "estimated_storage_mb": 498,
            },
            "rag": {
                "long_horizon_qa": 52.40,
                "multi_session_continuity": 42.20,
                "hallucination_rate": 5.7310,
                "answer_utility": 4.46,
                "fact_retention": 0.4610,
                "high_risk_hallucinations": 0.0580,
                "avg_runtime_ms": 7391.62,
                "estimated_storage_mb": 584,
            },
            "episodic": {
                "long_horizon_qa": 58.20,
                "multi_session_continuity": 47.10,
                "hallucination_rate": 7.4380,
                "answer_utility": 4.24,
                "fact_retention": 0.4970,
                "high_risk_hallucinations": 0.0910,
                "avg_runtime_ms": 5562.71,
                "estimated_storage_mb": 673,
            },
            "summarization": {
                "long_horizon_qa": 45.30,
                "multi_session_continuity": 36.40,
                "hallucination_rate": 6.1420,
                "answer_utility": 3.71,
                "fact_retention": 0.4210,
                "high_risk_hallucinations": 0.0760,
                "avg_runtime_ms": 6128.05,
                "estimated_storage_mb": 618,
            },
            "sleep": {
                "long_horizon_qa": 61.70,
                "multi_session_continuity": 50.40,
                "hallucination_rate": 5.2860,
                "answer_utility": 5.74,
                "fact_retention": 0.5410,
                "high_risk_hallucinations": 0.0600,
                "avg_runtime_ms": 10410.66,
                "estimated_storage_mb": 348,
            },
        },
    },
}

DATASETS: List[str] = ["PersonaMem", "PersonaChat", "LOCOMO", "OK-VQA"]
METHODS: List[str] = ["vanilla", "rag", "episodic", "summarization", "sleep"]

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Expanded cognitive probe deltas (Post - Pre)
# Source: RESULTS_DRAFT.md, "Expanded Cognitive probe results"
# =========================================================
COGNITIVE_PROBE_DELTAS: Dict[str, Dict[str, float]] = {
    "PersonaMem": {
        "delayed_recall": 16.00,
        "cue_based_recall": 16.00,
        "cross_episode_integration": 11.00,
        "schema_utilization": 17.00,
    },
    "PersonaChat": {
        "delayed_recall": 21.00,
        "cue_based_recall": 12.00,
        "cross_episode_integration": 7.00,
        "schema_utilization": -3.00,
    },
    "LOCOMO": {
        "delayed_recall": 13.00,
        "cue_based_recall": 12.00,
        "cross_episode_integration": 9.00,
        "schema_utilization": 8.00,
    },
    "OK-VQA": {
        "delayed_recall": 15.00,
        "cue_based_recall": 12.00,
        "cross_episode_integration": 4.00,
        "schema_utilization": -2.00,
    },
}


# =========================================================
# Expanded cognitive probe raw values (Pre/Post)
# Source: RESULTS_DRAFT.md, "Expanded Cognitive probe results"
# =========================================================
COGNITIVE_PROBE_PRE_POST: Dict[str, Dict[str, Dict[str, float]]] = {
    "PersonaMem": {
        "delayed_recall": {"pre": 52.00, "post": 68.00},
        "cue_based_recall": {"pre": 41.00, "post": 57.00},
        "cross_episode_integration": {"pre": 71.00, "post": 82.00},
        "schema_utilization": {"pre": 38.00, "post": 55.00},
    },
    "PersonaChat": {
        "delayed_recall": {"pre": 58.00, "post": 79.00},
        "cue_based_recall": {"pre": 62.00, "post": 74.00},
        "cross_episode_integration": {"pre": 76.00, "post": 83.00},
        "schema_utilization": {"pre": 49.00, "post": 46.00},
    },
    "LOCOMO": {
        "delayed_recall": {"pre": 34.00, "post": 47.00},
        "cue_based_recall": {"pre": 29.00, "post": 41.00},
        "cross_episode_integration": {"pre": 63.00, "post": 72.00},
        "schema_utilization": {"pre": 27.00, "post": 35.00},
    },
    "OK-VQA": {
        "delayed_recall": {"pre": 46.00, "post": 61.00},
        "cue_based_recall": {"pre": 38.00, "post": 50.00},
        "cross_episode_integration": {"pre": 69.00, "post": 73.00},
        "schema_utilization": {"pre": 44.00, "post": 42.00},
    },
}


def _metric_matrix(metric: str) -> np.ndarray:
    """Returns matrix with shape [len(METHODS), len(DATASETS)]."""
    rows = []
    for method in METHODS:
        row = [RESULTS_DATA[d]["methods"][method][metric] for d in DATASETS]
        rows.append(row)
    return np.array(rows, dtype=float)


def plot_grouped_bars(metric: str, title: str, y_label: str, filename: str) -> None:
    values = _metric_matrix(metric)
    x = np.arange(len(DATASETS))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(METHODS):
        ax.bar(x + (i - 2) * width, values[i], width=width, label=method)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend(ncols=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220)
    plt.close(fig)


def _mean_by_method(metric: str) -> np.ndarray:
    """Returns mean metric over datasets, one value per method."""
    values = []
    for method in METHODS:
        m_vals = [RESULTS_DATA[d]["methods"][method][metric] for d in DATASETS]
        values.append(float(np.mean(m_vals)))
    return np.array(values, dtype=float)


def plot_runtime_lines() -> None:
    runtime = _metric_matrix("avg_runtime_ms")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(METHODS):
        ax.plot(DATASETS, runtime[i], marker="o", linewidth=2, label=method)

    ax.set_title("Average Runtime per Turn by Method Across Datasets")
    ax.set_ylabel("Runtime (ms)")
    ax.legend(ncols=3, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_4_runtime_lines.png", dpi=220)
    plt.close(fig)


def plot_utility_vs_hallucination() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    marker_map = {
        "vanilla": "o",
        "rag": "s",
        "episodic": "^",
        "summarization": "D",
        "sleep": "P",
    }
    color_map = {
        "PersonaMem": "tab:blue",
        "PersonaChat": "tab:orange",
        "LOCOMO": "tab:green",
        "OK-VQA": "tab:red",
    }

    for dataset in DATASETS:
        for method in METHODS:
            m = RESULTS_DATA[dataset]["methods"][method]
            x = m["hallucination_rate"]
            y = m["answer_utility"]
            ax.scatter(
                x,
                y,
                marker=marker_map[method],
                color=color_map[dataset],
                s=80,
                alpha=0.85,
            )

    ax.set_title("Answer Utility vs Hallucination Rate")
    ax.set_xlabel("Hallucination Rate")
    ax.set_ylabel("Answer Utility")
    ax.grid(alpha=0.3)

    # Compact legend entries
    dataset_legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[d], label=d, markersize=8)
        for d in DATASETS
    ]
    method_legend = [
        plt.Line2D([0], [0], marker=marker_map[m], color="gray", linestyle="", label=m, markersize=8)
        for m in METHODS
    ]

    leg1 = ax.legend(handles=dataset_legend, title="Dataset", loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=method_legend, title="Method", loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_5_utility_vs_hallucination.png", dpi=220)
    plt.close(fig)


def plot_method_efficiency_tradeoff() -> None:
    """Method-level mean runtime vs mean storage with utility as point labels."""
    mean_runtime = _mean_by_method("avg_runtime_ms")
    mean_storage = _mean_by_method("estimated_storage_mb")
    mean_utility = _mean_by_method("answer_utility")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(mean_runtime, mean_storage, s=120, alpha=0.9)

    for i, method in enumerate(METHODS):
        ax.annotate(
            f"{method} (U={mean_utility[i]:.2f})",
            (mean_runtime[i], mean_storage[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax.set_title("Method Trade-off: Mean Runtime vs Mean Storage")
    ax.set_xlabel("Mean Runtime per Turn (ms)")
    ax.set_ylabel("Mean Estimated Storage (MB)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_6_runtime_vs_storage_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_cognitive_probe_heatmap() -> None:
    """Heatmap for cognitive probe deltas to highlight significant gains and anomalies."""
    probe_keys = [
        "delayed_recall",
        "cue_based_recall",
        "cross_episode_integration",
        "schema_utilization",
    ]
    probe_labels = [
        "Delayed Recall Δ",
        "Cue-Based Recall Δ",
        "Cross-Episode Integration Δ",
        "Schema Utilization Δ",
    ]

    mat = np.array(
        [[COGNITIVE_PROBE_DELTAS[d][k] for k in probe_keys] for d in DATASETS],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9, 5.3))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(probe_labels)))
    ax.set_xticklabels(probe_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_title("Cognitive Probe Deltas by Dataset (Post - Pre)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_8_cognitive_probe_heatmap.png", dpi=220)
    plt.close(fig)


def plot_cognitive_net_mean_delta() -> None:
    """Bar chart of net mean cognitive improvement per dataset."""
    net_means = [
        float(np.mean(list(COGNITIVE_PROBE_DELTAS[d].values()))) for d in DATASETS
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(DATASETS, net_means, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])

    for bar, val in zip(bars, net_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:+.2f}", ha="center", fontsize=9)

    ax.set_title("Net Mean Cognitive Probe Delta by Dataset")
    ax.set_ylabel("Mean Delta (Post - Pre)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_9_cognitive_net_mean_delta.png", dpi=220)
    plt.close(fig)


def plot_schema_anomaly_focus() -> None:
    """Focused chart for schema-utilization anomalies."""
    schema_vals = [COGNITIVE_PROBE_DELTAS[d]["schema_utilization"] for d in DATASETS]
    colors = ["tab:green" if v >= 0 else "tab:red" for v in schema_vals]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(DATASETS, schema_vals, color=colors)
    ax.axhline(0, color="black", linewidth=1)

    for bar, val in zip(bars, schema_vals):
        offset = 0.25 if val >= 0 else -0.55
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.2f}", ha="center", fontsize=9)

    ax.set_title("Schema Utilization Delta (Anomaly Focus)")
    ax.set_ylabel("Delta (Post - Pre)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "graph_10_schema_utilization_anomaly.png", dpi=220)
    plt.close(fig)


def plot_cognitive_pre_post_comparison() -> None:
    """2x2 panel comparing pre vs post consolidation values without delta transformation."""
    probe_keys = [
        "delayed_recall",
        "cue_based_recall",
        "cross_episode_integration",
        "schema_utilization",
    ]
    probe_labels = [
        "Delayed Recall",
        "Cue-Based Recall",
        "Cross-Episode Integration",
        "Schema Utilization",
    ]

    x = np.arange(len(DATASETS))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for i, (probe_key, probe_label) in enumerate(zip(probe_keys, probe_labels)):
        pre_vals = [COGNITIVE_PROBE_PRE_POST[d][probe_key]["pre"] for d in DATASETS]
        post_vals = [COGNITIVE_PROBE_PRE_POST[d][probe_key]["post"] for d in DATASETS]

        ax = axes[i]
        ax.bar(x - width / 2, pre_vals, width=width, label="Pre", color="#9E9E9E")
        ax.bar(x + width / 2, post_vals, width=width, label="Post", color="#2E7D32")
        ax.set_title(probe_label)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=15)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    axes[0].legend(loc="upper left")

    fig.suptitle("Expanded Cognitive Probe Results: Pre vs Post Consolidation", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "graph_11_cognitive_pre_post_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    # 1) Answer utility comparison
    plot_grouped_bars(
        metric="answer_utility",
        title="Answer Utility by Method Across Datasets",
        y_label="Answer Utility",
        filename="graph_1_answer_utility.png",
    )

    # 2) Multi-session continuity comparison
    plot_grouped_bars(
        metric="multi_session_continuity",
        title="Multi-Session Continuity by Method Across Datasets",
        y_label="Continuity (%)",
        filename="graph_2_continuity.png",
    )

    # 3) Hallucination rate comparison
    plot_grouped_bars(
        metric="hallucination_rate",
        title="Hallucination Rate by Method Across Datasets",
        y_label="Hallucination Rate",
        filename="graph_3_hallucination_rate.png",
    )

    # 4) Fact retention comparison
    plot_grouped_bars(
        metric="fact_retention",
        title="Fact Retention by Method Across Datasets",
        y_label="Fact Retention",
        filename="graph_4_fact_retention.png",
    )

    # 5) Runtime trend across datasets
    plot_runtime_lines()

    # 6) Utility-Hallucination trade-off view
    plot_utility_vs_hallucination()

    # 7) Method-level efficiency trade-off
    plot_method_efficiency_tradeoff()

    # 8) Cognitive probe deltas heatmap
    plot_cognitive_probe_heatmap()

    # 9) Net cognitive improvement by dataset
    plot_cognitive_net_mean_delta()

    # 10) Schema-utilization anomaly focus
    plot_schema_anomaly_focus()

    # 11) Pre vs post cognitive probe comparison (raw values)
    plot_cognitive_pre_post_comparison()

    print(f"Saved 11 graphs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
