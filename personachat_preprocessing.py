"""Preprocess PERSONACHAT dataset into benchmark-ready JSON files.

This script converts PERSONACHAT train/validation splits into the same
high-level sample format used by the existing memory benchmark backend.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_from_disk


def _safe_text(value: Any) -> str:
    """Convert any value to a clean string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def preprocess_dialog_row(row: Dict[str, Any], row_idx: int, split: str) -> List[Dict[str, Any]]:
    """Convert a PERSONACHAT row into multiple turn-level samples."""
    personality = row.get("personality", []) or []
    utterances = row.get("utterances", []) or []

    persona_lines = [_safe_text(p) for p in personality if _safe_text(p)]
    persona_text = "\n".join(f"- {line}" for line in persona_lines)

    samples: List[Dict[str, Any]] = []

    for turn_idx, utt in enumerate(utterances):
        history = utt.get("history", []) or []
        candidates = utt.get("candidates", []) or []

        if not history or not candidates:
            continue

        query = _safe_text(history[-1])
        if not query:
            continue

        # PersonaChat candidate ordering typically has true response at the end.
        correct_answer = _safe_text(candidates[-1])
        incorrect_answers = [_safe_text(c) for c in candidates[:-1] if _safe_text(c)][:5]

        related_snippet = "\n".join(_safe_text(h) for h in history[:-1] if _safe_text(h))
        topic_query = query[:80]

        samples.append(
            {
                "id": f"{split}_p{row_idx}_t{turn_idx}",
                "persona_id": row_idx,
                "query": query,
                "correct_answer": correct_answer,
                "incorrect_answers": incorrect_answers,
                "persona": persona_text,
                "expanded_persona": persona_lines,
                "related_conversation_snippet": related_snippet,
                "topic_query": topic_query,
            }
        )

    return samples


def group_by_persona(samples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group processed samples by persona id."""
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["persona_id"]].append(sample)
    return dict(grouped)


def preprocess_split(dataset_path: Path, split: str) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Load and preprocess one PERSONACHAT split."""
    dataset = load_from_disk(str(dataset_path))

    processed: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(dataset):
        processed.extend(preprocess_dialog_row(row, row_idx=row_idx, split=split))

    grouped = group_by_persona(processed)
    return processed, grouped


def save_split_outputs(
    output_dir: Path,
    split: str,
    samples: List[Dict[str, Any]],
    persona_groups: Dict[int, List[Dict[str, Any]]],
) -> None:
    """Write split outputs to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = output_dir / f"{split}_processed.json"
    persona_path = output_dir / f"{split}_persona_sessions.json"

    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    with open(persona_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in persona_groups.items()}, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Run preprocessing for PERSONACHAT train and validation splits."""
    base_dir = Path(__file__).parent / "PERSONACHAT"
    output_dir = base_dir / "preprocessed"

    split_map = {
        "train": base_dir / "train",
        "validation": base_dir / "validation",
    }

    summary: Dict[str, Any] = {}

    for split, path in split_map.items():
        if not path.exists():
            print(f"Skipping missing split: {path}")
            continue

        print(f"Preprocessing {split} from {path} ...")
        samples, groups = preprocess_split(path, split)
        save_split_outputs(output_dir, split, samples, groups)

        summary[split] = {
            "num_samples": len(samples),
            "num_personas": len(groups),
            "output_processed": str((output_dir / f"{split}_processed.json").name),
            "output_persona_sessions": str((output_dir / f"{split}_persona_sessions.json").name),
        }

    # Optional alias so runner can use --split benchmark mapped to validation.
    if (output_dir / "validation_processed.json").exists():
        with open(output_dir / "validation_processed.json", "r", encoding="utf-8") as f:
            validation_samples = json.load(f)
        with open(output_dir / "validation_persona_sessions.json", "r", encoding="utf-8") as f:
            validation_groups = json.load(f)

        with open(output_dir / "benchmark_processed.json", "w", encoding="utf-8") as f:
            json.dump(validation_samples, f, indent=2, ensure_ascii=False)
        with open(output_dir / "benchmark_persona_sessions.json", "w", encoding="utf-8") as f:
            json.dump(validation_groups, f, indent=2, ensure_ascii=False)

        summary["benchmark"] = {
            "source": "validation",
            "num_samples": len(validation_samples),
            "num_personas": len(validation_groups),
        }

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done. Preprocessed files are in PERSONACHAT/preprocessed/")


if __name__ == "__main__":
    main()
