"""Preprocess LOCOMO benchmark into backend-ready JSON files.

Input:
- LOCOMO/locomo10.json

Output:
- LOCOMO/preprocessed/benchmark_processed.json
- LOCOMO/preprocessed/benchmark_persona_sessions.json
- LOCOMO/preprocessed/preprocessing_summary.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_text(value: Any) -> str:
    """Convert value to clean text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _build_dialog_index(conversation: Dict[str, Any]) -> Dict[str, str]:
    """Create a dia_id -> text lookup across all sessions."""
    dialog_index: Dict[str, str] = {}
    for key, turns in conversation.items():
        if not key.startswith("session_"):
            continue
        if key.endswith("_date_time") or key.endswith("_summary"):
            continue
        if not isinstance(turns, list):
            continue

        for turn in turns:
            dia_id = _safe_text(turn.get("dia_id"))
            text = _safe_text(turn.get("text"))
            if dia_id and text:
                dialog_index[dia_id] = text
    return dialog_index


def _build_persona_text(conversation: Dict[str, Any]) -> str:
    """Create a lightweight persona/context string from conversation metadata."""
    speaker_a = _safe_text(conversation.get("speaker_a", "Speaker A"))
    speaker_b = _safe_text(conversation.get("speaker_b", "Speaker B"))

    summaries: List[str] = []
    session_summary = conversation.get("session_summary", {}) or {}
    if isinstance(session_summary, dict):
        for k in sorted(session_summary.keys())[:3]:
            text = _safe_text(session_summary.get(k))
            if text:
                summaries.append(text)

    persona_parts = [f"Primary speakers: {speaker_a} and {speaker_b}."]
    if summaries:
        persona_parts.append("Context summaries:")
        persona_parts.extend(f"- {s}" for s in summaries)
    return "\n".join(persona_parts)


def _build_incorrect_answers(qa_item: Dict[str, Any], qa_items: List[Dict[str, Any]], idx: int) -> List[str]:
    """Build negative answers using adversarial answer first, then other answers."""
    negatives: List[str] = []

    adv = _safe_text(qa_item.get("adversarial_answer"))
    if adv:
        negatives.append(adv)

    for j, other in enumerate(qa_items):
        if j == idx:
            continue
        ans = _safe_text(other.get("answer"))
        if ans and ans not in negatives:
            negatives.append(ans)
        if len(negatives) >= 5:
            break

    return negatives


def preprocess_record(record: Dict[str, Any], persona_id: int) -> List[Dict[str, Any]]:
    """Convert one LOCOMO conversation record into benchmark samples."""
    qa_items = record.get("qa", []) or []
    conversation = record.get("conversation", {}) or {}

    sample_id = _safe_text(record.get("sample_id")) or f"conv-{persona_id}"
    persona_text = _build_persona_text(conversation)
    dialog_index = _build_dialog_index(conversation)

    processed: List[Dict[str, Any]] = []

    for i, qa in enumerate(qa_items):
        question = _safe_text(qa.get("question"))
        if not question:
            continue

        answer = _safe_text(qa.get("answer"))
        evidence_ids = qa.get("evidence", []) or []
        evidence_texts = []
        for eid in evidence_ids:
            eid_str = _safe_text(eid)
            if eid_str and eid_str in dialog_index:
                evidence_texts.append(f"{eid_str}: {dialog_index[eid_str]}")

        related_snippet = "\n".join(evidence_texts)
        incorrect_answers = _build_incorrect_answers(qa, qa_items, i)

        processed.append(
            {
                "id": f"{sample_id}_q{i}",
                "persona_id": persona_id,
                "query": question,
                "correct_answer": answer,
                "incorrect_answers": incorrect_answers,
                "persona": persona_text,
                "expanded_persona": [conversation.get("speaker_a"), conversation.get("speaker_b")],
                "related_conversation_snippet": related_snippet,
                "topic_query": question[:80],
                "category": qa.get("category"),
                "evidence": evidence_ids,
                "source_sample_id": sample_id,
            }
        )

    return processed


def group_by_persona(samples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group processed samples by persona id."""
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["persona_id"]].append(sample)
    return dict(grouped)


def main() -> None:
    """Run LOCOMO preprocessing."""
    base_dir = Path(__file__).parent / "LOCOMO"
    input_path = base_dir / "locomo10.json"
    output_dir = base_dir / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    all_samples: List[Dict[str, Any]] = []
    for persona_id, record in enumerate(raw_records):
        all_samples.extend(preprocess_record(record, persona_id=persona_id))

    persona_groups = group_by_persona(all_samples)

    with open(output_dir / "benchmark_processed.json", "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    with open(output_dir / "benchmark_persona_sessions.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in persona_groups.items()}, f, indent=2, ensure_ascii=False)

    summary = {
        "input_file": str(input_path.name),
        "num_conversations": len(raw_records),
        "num_samples": len(all_samples),
        "num_personas": len(persona_groups),
        "outputs": [
            "benchmark_processed.json",
            "benchmark_persona_sessions.json",
        ],
    }

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done. Preprocessed files are in LOCOMO/preprocessed/")


if __name__ == "__main__":
    main()
