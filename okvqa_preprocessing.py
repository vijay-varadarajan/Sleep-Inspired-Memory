"""Preprocess OKVQA into benchmark-ready JSON files for the memory runners."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_from_disk


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _majority_answer(answers: List[str]) -> str:
    cleaned = [_safe_text(a).lower() for a in answers if _safe_text(a)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def preprocess_okvqa(dataset_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    ds = load_from_disk(str(dataset_dir))
    split = ds["val2014"]

    # Build answer pools by question_type for stronger negatives.
    answer_pool_by_type: Dict[str, List[str]] = defaultdict(list)
    for row in split:
        qtype = _safe_text(row.get("question_type", "unknown")) or "unknown"
        answers = row.get("answers", []) or []
        answer_pool_by_type[qtype].extend(_safe_text(a).lower() for a in answers if _safe_text(a))

    answer_pool_by_type = {
        k: [a for a, _ in Counter(v).most_common(300)] for k, v in answer_pool_by_type.items()
    }

    qtype_to_persona_id: Dict[str, int] = {}
    processed: List[Dict[str, Any]] = []
    last_question_by_qtype: Dict[str, str] = {}

    for idx, row in enumerate(split):
        qid = row.get("question_id", idx)
        question = _safe_text(row.get("question", ""))
        qtype = _safe_text(row.get("question_type", "unknown")) or "unknown"
        atype = _safe_text(row.get("answer_type", "other")) or "other"

        answers = row.get("answers", []) or []
        correct_answer = _majority_answer(answers)
        if not question or not correct_answer:
            continue

        if qtype not in qtype_to_persona_id:
            qtype_to_persona_id[qtype] = len(qtype_to_persona_id)
        persona_id = qtype_to_persona_id[qtype]

        img = row.get("image")
        img_meta = ""
        if img is not None:
            mode = getattr(img, "mode", "")
            size = getattr(img, "size", None)
            if size:
                img_meta = f"Image mode={mode}, size={size[0]}x{size[1]}"

        pool = [a for a in answer_pool_by_type.get(qtype, []) if a and a != correct_answer]
        incorrect_answers = pool[:5]

        related = last_question_by_qtype.get(qtype, "")
        if related and img_meta:
            related = f"{related}\n{img_meta}"
        elif img_meta:
            related = img_meta

        persona_text = (
            f"OKVQA category: {qtype}. "
            f"Answer style: concise, grounded, avoid hallucination. "
            f"Answer type expected: {atype}."
        )

        processed.append(
            {
                "id": f"okvqa_{qid}",
                "persona_id": persona_id,
                "query": question,
                "correct_answer": correct_answer,
                "incorrect_answers": incorrect_answers,
                "persona": persona_text,
                "expanded_persona": [qtype, atype],
                "related_conversation_snippet": related,
                "topic_query": qtype,
                "question_id": qid,
                "question_type": qtype,
                "answer_type": atype,
                "image_meta": img_meta,
            }
        )

        last_question_by_qtype[qtype] = question

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in processed:
        grouped[sample["persona_id"]].append(sample)

    return processed, dict(grouped)


def main() -> None:
    base_dir = Path(__file__).parent / "OKVQA"
    dataset_dir = base_dir / "dataset"
    output_dir = base_dir / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed, grouped = preprocess_okvqa(dataset_dir)

    with open(output_dir / "benchmark_processed.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    with open(output_dir / "benchmark_persona_sessions.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in grouped.items()}, f, indent=2, ensure_ascii=False)

    # Alias for runner consistency.
    with open(output_dir / "val_processed.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    with open(output_dir / "val_persona_sessions.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in grouped.items()}, f, indent=2, ensure_ascii=False)

    summary = {
        "dataset": "OKVQA",
        "split": "val2014",
        "num_samples": len(processed),
        "num_persona_groups": len(grouped),
        "outputs": [
            "benchmark_processed.json",
            "benchmark_persona_sessions.json",
            "val_processed.json",
            "val_persona_sessions.json",
        ],
    }

    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done. Preprocessed files are in OKVQA/preprocessed/")


if __name__ == "__main__":
    main()
