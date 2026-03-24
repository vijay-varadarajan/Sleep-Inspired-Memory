"""Run LOCOMO evaluation with existing memory/agent/sleep backend."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.baselines import create_agent
from evaluation.personamem_benchmark import PersonaMemEvaluator
from locomo_postprocessing import save_results_tables
from utils.api_counter import get_api_counters, reset_api_counters


class LocomoRunner:
    """Simple benchmark runner for LOCOMO data."""

    def __init__(
        self,
        split: str = "benchmark",
        num_samples: int = 200,
        methods: Optional[List[str]] = None,
        output_dir: str = "locomo_results",
        ablation_flags: Optional[Dict[str, bool]] = None,
    ) -> None:
        load_dotenv()

        self.split = split
        self.num_samples = num_samples
        self.methods = methods or ["vanilla", "rag", "episodic", "summarization", "sleep"]

        self.data_dir = Path("LOCOMO/preprocessed")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.samples = self._load_data()
        self.persona_groups = self._load_persona_groups()
        self.evaluator = PersonaMemEvaluator()
        self.ablation_flags: Dict[str, bool] = ablation_flags or {}

    def _load_data(self) -> List[Dict[str, Any]]:
        path = self.data_dir / f"{self.split}_processed.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_persona_groups(self) -> Dict[int, List[Dict[str, Any]]]:
        path = self.data_dir / f"{self.split}_persona_sessions.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def run_table1_evaluation(self, method: str) -> Dict[str, Any]:
        """Run Table 1 metrics for one method."""
        agent = create_agent(method, dataset_name="locomo", ablation_flags=self.ablation_flags)

        samples_to_use = self.samples[: self.num_samples]
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples_to_use:
            grouped[sample["persona_id"]].append(sample)

        qa_results: List[Dict[str, Any]] = []
        continuity_results: List[Dict[str, Any]] = []
        hallucination_results: List[Dict[str, Any]] = []
        utility_results: List[Dict[str, Any]] = []
        retrieval_results: List[Dict[str, Any]] = []
        fidelity_results: List[Dict[str, Any]] = []
        unit_hallu_results: List[Dict[str, Any]] = []
        latency_ms: List[float] = []

        for _, persona_samples in tqdm(grouped.items(), desc=f"Table1 {method}"):
            persona_samples = persona_samples[:3]
            if not persona_samples:
                continue

            persona_info = persona_samples[0].get("persona", "")

            for idx, sample in enumerate(persona_samples):
                query = sample.get("query", "")
                correct_answer = sample.get("correct_answer", "")
                incorrect_answers = sample.get("incorrect_answers", [])
                related_snippet = sample.get("related_conversation_snippet", "")

                try:
                    t0 = time.perf_counter()
                    response = agent.interact(
                        user_input=query,
                        persona=persona_info,
                        importance=0.7,
                        tags=["locomo"],
                        use_memory=(idx > 0),
                    )
                    latency_ms.append((time.perf_counter() - t0) * 1000.0)
                except Exception:
                    response = ""

                try:
                    qa_results.append(
                        self.evaluator.evaluate_long_horizon_qa(response, correct_answer, incorrect_answers)
                    )
                except Exception:
                    qa_results.append({"correct": False})

                utility_results.append(
                    self.evaluator.evaluate_answer_utility(response, correct_answer, incorrect_answers)
                )

                bundles = getattr(agent, "last_retrieval_bundles", []) or []
                retrieval_results.append(self.evaluator.evaluate_retrieval_success(bundles, correct_answer))

                fidelity_results.append(
                    self.evaluator.evaluate_memory_fidelity(
                        original_text=f"{query} {correct_answer}",
                        consolidated_text=response,
                        memory_type="fact",
                        contradiction_flags=[b.get("memory_id", "") for b in bundles if b.get("is_contradictory")],
                    )
                )

                unit_hallu_results.append(
                    self.evaluator.evaluate_hallucination_units(
                        response=response,
                        supported_context=f"{correct_answer}\n{related_snippet}\n{persona_info}",
                    )
                )

                if related_snippet:
                    try:
                        continuity_results.append(
                            self.evaluator.evaluate_multi_session_continuity(
                                response,
                                related_snippet,
                                correct_answer,
                            )
                        )
                    except Exception:
                        continuity_results.append({"score": 0.0})

                try:
                    context = f"Persona: {persona_info}\nRelated: {related_snippet}"
                    hallucination_results.append(
                        self.evaluator.evaluate_hallucination_rate(response, correct_answer, context)
                    )
                except Exception:
                    hallucination_results.append(
                        {
                            "hallucination_count": 0,
                            "response_length": max(1, len(str(response).split())),
                        }
                    )

            if method == "sleep" and hasattr(agent, "sleep"):
                try:
                    agent.sleep(verbose=False)
                except Exception:
                    pass

        qa_accuracy = (
            sum(1.0 for r in qa_results if r.get("correct", False)) / len(qa_results)
            if qa_results
            else 0.0
        )
        continuity_score = (
            sum(float(r.get("score", 0.0)) for r in continuity_results) / len(continuity_results)
            if continuity_results
            else 0.0
        )
        total_hallucinations = sum(int(r.get("hallucination_count", 0)) for r in hallucination_results)
        total_words = sum(max(1, int(r.get("response_length", 1))) for r in hallucination_results)
        hallucination_rate = (total_hallucinations / (total_words / 100.0)) if total_words > 0 else 0.0

        utility_score = float(np.mean([r.get("utility_score", 0.0) for r in utility_results])) * 100.0 if utility_results else 0.0
        recall_at_3 = float(np.mean([r.get("recall@3", 0.0) for r in retrieval_results])) if retrieval_results else 0.0
        mrr = float(np.mean([r.get("mrr", 0.0) for r in retrieval_results])) if retrieval_results else 0.0
        ndcg = float(np.mean([r.get("ndcg@5", 0.0) for r in retrieval_results])) if retrieval_results else 0.0
        evidence_hit_rate = float(np.mean([r.get("evidence_hit_rate", 0.0) for r in retrieval_results])) if retrieval_results else 0.0

        fact_retention = float(np.mean([r.get("fact_retention_rate", 0.0) for r in fidelity_results])) if fidelity_results else 0.0
        pref_retention = float(np.mean([r.get("preference_retention_rate", 0.0) for r in fidelity_results])) if fidelity_results else 0.0
        contradiction_rate = float(np.mean([r.get("contradiction_rate", 0.0) for r in fidelity_results])) if fidelity_results else 0.0

        unsupported_prop = float(np.mean([r.get("unsupported_claim_proportion", 0.0) for r in unit_hallu_results])) if unit_hallu_results else 0.0
        unsupported_cnt = float(np.mean([r.get("unsupported_claim_count", 0.0) for r in unit_hallu_results])) if unit_hallu_results else 0.0
        high_risk_hallu = float(np.mean([r.get("high_risk_factual_hallucinations", 0.0) for r in unit_hallu_results])) if unit_hallu_results else 0.0
        avg_latency = float(np.mean(latency_ms)) if latency_ms else 0.0
        avg_resp_len = float(np.mean([max(1, len(str(r).split())) for r in [x.get("response", "") for x in qa_results]])) if qa_results else 0.0

        return {
            "method": method,
            "long_horizon_qa": qa_accuracy * 100.0,
            "multi_session_continuity": continuity_score * 100.0,
            "hallucination_rate": hallucination_rate,
            "num_qa_samples": len(qa_results),
            "num_continuity_samples": len(continuity_results),
            "num_hallucination_samples": len(hallucination_results),
            "answer_utility": utility_score,
            "retrieval_recall_at_3": recall_at_3,
            "retrieval_mrr": mrr,
            "retrieval_ndcg_at_5": ndcg,
            "evidence_hit_rate": evidence_hit_rate,
            "fact_retention_rate": fact_retention,
            "preference_retention_rate": pref_retention,
            "contradiction_rate": contradiction_rate,
            "unsupported_claim_count": unsupported_cnt,
            "unsupported_claim_proportion": unsupported_prop,
            "high_risk_factual_hallucinations": high_risk_hallu,
            "avg_runtime_per_turn_ms": avg_latency,
            "avg_response_length_words": avg_resp_len,
        }

    def run_table2_evaluation(self) -> Dict[str, Any]:
        """Run Table 2 probes with sleep method (pre vs post consolidation)."""
        agent = create_agent("sleep", dataset_name="locomo", ablation_flags=self.ablation_flags)

        samples_to_use = self.samples[: min(self.num_samples, 300)]
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples_to_use:
            grouped[sample["persona_id"]].append(sample)

        delayed_recall_pre: List[float] = []
        delayed_recall_post: List[float] = []
        cue_based_pre: List[float] = []
        cue_based_post: List[float] = []
        integration_pre: List[float] = []
        integration_post: List[float] = []
        schema_util_pre: List[float] = []
        schema_util_post: List[float] = []

        for _, persona_samples in tqdm(grouped.items(), desc="Table2 sleep"):
            persona_samples = persona_samples[:2]
            if len(persona_samples) < 2:
                continue

            persona_info = persona_samples[0].get("persona", "")

            for sample in persona_samples:
                try:
                    agent.interact(
                        user_input=sample.get("query", ""),
                        persona=persona_info,
                        importance=0.8,
                        tags=["locomo"],
                        use_memory=False,
                    )
                except Exception:
                    pass

            s0 = persona_samples[0]
            s1 = persona_samples[1]

            topic0 = s0.get("topic_query", "this topic")
            topic1 = s1.get("topic_query", "that topic")
            recall_query = f"What did we discuss about {topic0}?"
            cue_query = "Based on our previous conversations, what would you recommend?"
            cue_text = str(s0.get("related_conversation_snippet", ""))[:500]
            cue_query_with = f"Given this context: {cue_text}\n\n{cue_query}"
            integration_query = f"How does {topic0} relate to {topic1}?"
            schema_query = s1.get("query", "")

            # Pre-consolidation
            try:
                recall_pre = agent.interact(recall_query, persona=persona_info, use_memory=True)
                delayed_pre = self.evaluator.evaluate_delayed_recall(
                    recall_pre,
                    s0.get("query", "") + " " + s0.get("correct_answer", ""),
                    [topic0],
                )
                delayed_recall_pre.append(float(delayed_pre.get("accuracy", 0.0)))
            except Exception:
                delayed_recall_pre.append(0.0)

            try:
                wo = agent.interact(cue_query, persona=persona_info, use_memory=True)
                w = agent.interact(cue_query_with, persona=persona_info, use_memory=True)
                cue_pre = self.evaluator.evaluate_cue_based_recall(
                    w,
                    wo,
                    cue_text,
                    s0.get("correct_answer", ""),
                )
                cue_based_pre.append(float(cue_pre.get("with_cue_accuracy", 0.0)))
            except Exception:
                cue_based_pre.append(0.0)

            try:
                integration_resp_pre = agent.interact(integration_query, persona=persona_info, use_memory=True)
                integration_pre_result = self.evaluator.evaluate_cross_episode_integration(
                    integration_resp_pre,
                    s0.get("query", ""),
                    s1.get("query", ""),
                    integration_query,
                )
                integration_pre.append(float(integration_pre_result.get("integration_score", 0.0)))
            except Exception:
                integration_pre.append(0.0)

            try:
                schema_resp_pre = agent.interact(schema_query, persona=persona_info, use_memory=True)
                schema_pre_result = self.evaluator.evaluate_schema_utilization(
                    schema_resp_pre,
                    persona_info,
                    schema_query,
                )
                schema_util_pre.append(float(schema_pre_result.get("schema_utilization_score", 0.0)))
            except Exception:
                schema_util_pre.append(0.0)

            # Consolidation
            try:
                agent.sleep(verbose=False)
            except Exception:
                pass

            # Post-consolidation
            try:
                recall_post = agent.interact(recall_query, persona=persona_info, use_memory=True)
                delayed_post = self.evaluator.evaluate_delayed_recall(
                    recall_post,
                    s0.get("query", "") + " " + s0.get("correct_answer", ""),
                    [topic0],
                )
                delayed_recall_post.append(float(delayed_post.get("accuracy", 0.0)))
            except Exception:
                delayed_recall_post.append(0.0)

            try:
                wo = agent.interact(cue_query, persona=persona_info, use_memory=True)
                w = agent.interact(cue_query_with, persona=persona_info, use_memory=True)
                cue_post = self.evaluator.evaluate_cue_based_recall(
                    w,
                    wo,
                    cue_text,
                    s0.get("correct_answer", ""),
                )
                cue_based_post.append(float(cue_post.get("with_cue_accuracy", 0.0)))
            except Exception:
                cue_based_post.append(0.0)

            try:
                integration_resp_post = agent.interact(integration_query, persona=persona_info, use_memory=True)
                integration_post_result = self.evaluator.evaluate_cross_episode_integration(
                    integration_resp_post,
                    s0.get("query", ""),
                    s1.get("query", ""),
                    integration_query,
                )
                integration_post.append(float(integration_post_result.get("integration_score", 0.0)))
            except Exception:
                integration_post.append(0.0)

            try:
                schema_resp_post = agent.interact(schema_query, persona=persona_info, use_memory=True)
                schema_post_result = self.evaluator.evaluate_schema_utilization(
                    schema_resp_post,
                    persona_info,
                    schema_query,
                )
                schema_util_post.append(float(schema_post_result.get("schema_utilization_score", 0.0)))
            except Exception:
                schema_util_post.append(0.0)

        def pct_mean(values: List[float]) -> float:
            return float(np.mean(values) * 100.0) if values else 0.0

        delayed_pre = pct_mean(delayed_recall_pre)
        delayed_post = pct_mean(delayed_recall_post)
        cue_pre = pct_mean(cue_based_pre)
        cue_post = pct_mean(cue_based_post)
        integration_pre_pct = pct_mean(integration_pre)
        integration_post_pct = pct_mean(integration_post)
        schema_pre_pct = pct_mean(schema_util_pre)
        schema_post_pct = pct_mean(schema_util_post)

        return {
            "method": "sleep",
            "applicable": True,
            "delayed_recall_pre": delayed_pre,
            "delayed_recall_post": delayed_post,
            "delayed_recall_improvement": delayed_post - delayed_pre,
            "cue_based_pre": cue_pre,
            "cue_based_post": cue_post,
            "cue_based_improvement": cue_post - cue_pre,
            "integration_pre": integration_pre_pct,
            "integration_post": integration_post_pct,
            "integration_improvement": integration_post_pct - integration_pre_pct,
            "schema_util_pre": schema_pre_pct,
            "schema_util_post": schema_post_pct,
            "schema_util_improvement": schema_post_pct - schema_pre_pct,
        }

    def run(self) -> Dict[str, Any]:
        """Run all configured evaluations and save outputs."""
        reset_api_counters()
        table1_results = [self.run_table1_evaluation(method) for method in self.methods]
        table2_result = self.run_table2_evaluation()
        api_counters = get_api_counters()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_results = {
            "timestamp": timestamp,
            "dataset": "LOCOMO",
            "split": self.split,
            "num_samples": self.num_samples,
            "methods": self.methods,
            "ablation_flags": self.ablation_flags,
            "prompt_version": "v2_structured_dataset_aware",
            "sleep_configuration": {
                "dataset_name": "locomo",
            },
            "retrieval_configuration": {
                "type": "hybrid",
            },
            "api_counters": api_counters,
            "table1": table1_results,
            "table2": [table2_result],
        }

        raw_path = self.output_dir / f"locomo_results_{timestamp}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)

        save_results_tables(table1_results, table2_result, self.output_dir)
        print(f"LLM API calls this run: {api_counters.get('llm_total', 0)}")
        return raw_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOCOMO benchmark")
    parser.add_argument(
        "--split",
        type=str,
        default="benchmark",
        choices=["benchmark"],
        help="Preprocessed split to use",
    )
    parser.add_argument("--num_samples", type=int, default=200, help="Number of preprocessed samples to evaluate")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to run: vanilla rag episodic summarization sleep (or all)",
    )
    parser.add_argument("--output_dir", type=str, default="locomo_results", help="Result output folder")
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "no_sleep",
            "episodic_only",
            "summarization_only",
            "disable_schema",
            "disable_replay_selection",
            "disable_conflict_handling",
            "disable_evidence_priority",
        ],
        help="Run a single ablation mode",
    )

    args = parser.parse_args()

    methods = args.methods
    if "all" in methods:
        methods = ["vanilla", "rag", "episodic", "summarization", "sleep"]

    runner = LocomoRunner(
        split=args.split,
        num_samples=args.num_samples,
        methods=methods,
        output_dir=args.output_dir,
        ablation_flags={} if args.ablation == "none" else {args.ablation: True},
    )
    runner.run()


if __name__ == "__main__":
    main()
