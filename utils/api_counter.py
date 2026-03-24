"""Lightweight runtime counters for LLM API calls."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict


_COUNTERS: Dict[str, int] = defaultdict(int)


def reset_api_counters() -> None:
    """Reset all API counters."""
    _COUNTERS.clear()


def increment_llm_call(source: str = "unknown") -> None:
    """Increment total + source-specific LLM call counters."""
    _COUNTERS["llm_total"] += 1
    _COUNTERS[f"llm::{source}"] += 1


def get_api_counters() -> Dict[str, int]:
    """Get a snapshot of API counters."""
    return dict(_COUNTERS)
