"""
PersonaMem Benchmark Evaluation

Implements evaluation metrics for Tables 1 and 2:

TABLE 1: Task-Based Memory Performance
- Long-Horizon QA: Accuracy on multi-turn questions
- Multi-Session Continuity: Correct references to prior sessions
- Hallucination Rate: Unsupported claims per 100 responses

TABLE 2: Cognitive-Style Probes (Before vs After Sleep)
- Delayed Recall Accuracy: Recall after time delay
- Cue-Based Recall: Recall using conversation snippets as cues
- Cross-Episode Integration: Connect information across sessions
- Schema Utilization Rate: Use of persona schemas
"""

import json
import os
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.api_counter import increment_llm_call


class PersonaMemEvaluator:
    """Evaluator for PersonaMem benchmarking."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        """
        Initialize evaluator.
        
        Args:
            api_key: Google API key
            model_name: Model to use for evaluation
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.0,  # Deterministic for evaluation
            request_timeout=45,  # 45 second timeout per request
            max_retries=2  # Retry failed requests
        )

    @staticmethod
    def _ensure_text(value: Any) -> str:
        """Ensure a value is converted to a safe string for prompting."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        """Ensure a value is a list of strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
        return [str(value)]

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def evaluate_answer_utility(
        self,
        response: str,
        correct_answer: str,
        incorrect_answers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Automatic primary utility metric: exact + token-F1 + semantic fallback flag."""
        r = self._normalize(self._ensure_text(response))
        c = self._normalize(self._ensure_text(correct_answer))
        incorrect = [self._normalize(x) for x in self._ensure_list(incorrect_answers)]

        exact = 1.0 if r == c and r else 0.0

        r_toks = r.split()
        c_toks = c.split()
        common = len(set(r_toks) & set(c_toks))
        precision = common / max(1, len(set(r_toks)))
        recall = common / max(1, len(set(c_toks)))
        token_f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

        # Utility score primarily automatic.
        utility = 0.65 * token_f1 + 0.35 * exact

        # Negative overlap penalty against distractors.
        if incorrect:
            neg_overlap = max(
                (len(set(r_toks) & set(inc.split())) / max(1, len(set(r_toks))))
                for inc in incorrect
            )
            utility = max(0.0, utility - 0.2 * neg_overlap)

        return {
            "exact_match": exact,
            "token_f1": token_f1,
            "utility_score": utility,
            "primary_metric": "automatic",
        }

    def evaluate_retrieval_success(
        self,
        retrieval_bundles: List[Dict[str, Any]],
        target_text: str,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Compute Recall@k, MRR, nDCG, and evidence hit rate from retrieval bundles."""
        k_values = k_values or [1, 3, 5]
        target_norm = self._normalize(target_text)
        target_tokens = set(target_norm.split())

        graded = []
        first_rel_rank = None
        evidence_hit = 0
        for rank, b in enumerate(retrieval_bundles, start=1):
            text = self._normalize(str(b.get("text", "")) + " " + str(b.get("core_fact", "")))
            text_tokens = set(text.split())
            overlap = len(target_tokens & text_tokens) / max(1, len(target_tokens))
            rel = 1 if overlap >= 0.25 else 0
            graded.append((rank, rel, overlap))
            if rel and first_rel_rank is None:
                first_rel_rank = rank
            if b.get("is_evidence_grounded") and rel:
                evidence_hit = 1

        recall = {}
        for k in k_values:
            recall[f"recall@{k}"] = 1.0 if any(rel for rank, rel, _ in graded if rank <= k) else 0.0

        mrr = 1.0 / first_rel_rank if first_rel_rank else 0.0
        # Binary nDCG@5.
        dcg = sum((rel / (1.0 if rank == 1 else (1.0 + math.log2(rank)))) for rank, rel, _ in graded[:5])
        ideal_rels = sorted([rel for _, rel, _ in graded], reverse=True)[:5]
        idcg = sum((rel / (1.0 if i == 0 else (1.0 + math.log2(i + 1)))) for i, rel in enumerate(ideal_rels))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        out = {
            **recall,
            "mrr": mrr,
            "ndcg@5": ndcg,
            "evidence_hit_rate": float(evidence_hit),
        }
        return out

    def evaluate_memory_fidelity(
        self,
        original_text: str,
        consolidated_text: str,
        memory_type: str = "mixed",
        contradiction_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Measure retention and contradiction risk after consolidation."""
        o = self._normalize(original_text)
        c = self._normalize(consolidated_text)
        o_set = set(o.split())
        c_set = set(c.split())

        retention = len(o_set & c_set) / max(1, len(o_set))

        entities_orig = [t for t in original_text.split() if t[:1].isupper() and len(t) > 2]
        entities_cons = set(t for t in consolidated_text.split() if t[:1].isupper() and len(t) > 2)
        entity_retention = (
            sum(1 for e in entities_orig if e in entities_cons) / max(1, len(entities_orig))
            if entities_orig
            else retention
        )

        preference_terms = ["like", "dislike", "prefer", "favorite", "hate"]
        pref_orig = sum(1 for t in preference_terms if t in o)
        pref_cons = sum(1 for t in preference_terms if t in c)
        preference_retention = min(1.0, pref_cons / max(1, pref_orig)) if pref_orig else 1.0

        contradiction_rate = 1.0 if contradiction_flags else 0.0
        return {
            "fact_retention_rate": retention,
            "entity_retention_rate": entity_retention,
            "preference_retention_rate": preference_retention,
            "contradiction_rate": contradiction_rate,
            "memory_type": memory_type,
        }

    def evaluate_efficiency_metrics(
        self,
        runtime_ms: float,
        retrieval_latency_ms: float,
        sleep_time_ms: float,
        tokens_used: int,
        memory_count: int,
        replayed_episodes: int,
    ) -> Dict[str, Any]:
        """Runtime and deployment tradeoff metrics."""
        return {
            "runtime_per_turn_ms": float(runtime_ms),
            "retrieval_latency_ms": float(retrieval_latency_ms),
            "sleep_consolidation_time_ms": float(sleep_time_ms),
            "tokens_per_query": int(tokens_used),
            "consolidated_memory_count": int(memory_count),
            "replayed_episodes_per_cycle": int(replayed_episodes),
        }

    def evaluate_hallucination_units(
        self,
        response: str,
        supported_context: str,
    ) -> Dict[str, Any]:
        """Unit-normalized unsupported claim metrics (sentence-level)."""
        resp = self._ensure_text(response)
        ctx = self._normalize(supported_context)
        sentences = [s.strip() for s in re.split(r"[.!?]+", resp) if s.strip()]
        unsupported = 0
        high_risk = 0
        for s in sentences:
            s_norm = self._normalize(s)
            if not s_norm:
                continue
            overlap = len(set(s_norm.split()) & set(ctx.split())) / max(1, len(set(s_norm.split())))
            if overlap < 0.2:
                unsupported += 1
                if any(k in s_norm for k in ["date", "time", "when", "where", "exact", "evidence"]):
                    high_risk += 1
        total = max(1, len(sentences))
        return {
            "unsupported_claim_count": unsupported,
            "unsupported_claim_proportion": unsupported / total,
            "high_risk_factual_hallucinations": high_risk,
            "content_units": total,
        }
    
    def evaluate_long_horizon_qa(
        self,
        response: str,
        correct_answer: str,
        incorrect_answers: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate Long-Horizon QA accuracy.
        
        Uses semantic similarity to determine if the response matches the
        correct answer better than incorrect answers.
        
        Args:
            response: Agent's response
            correct_answer: Ground truth answer
            incorrect_answers: List of incorrect answers
            
        Returns:
            Dictionary with accuracy score and details
        """
        response_text = self._ensure_text(response)
        correct_text = self._ensure_text(correct_answer)
        incorrect_list = self._ensure_list(incorrect_answers)

        # Use LLM to judge if response matches correct answer - simplified prompt
        prompt = f"""Evaluate if Response matches Correct Answer (ignore incorrect answers).

Response: {response_text[:500]}
Correct Answer: {correct_text[:500]}

Answer: MATCH: YES or NO
Reason: [one sentence]
"""
        
        try:
            increment_llm_call("evaluator_long_horizon_qa")
            result = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = self._ensure_text(result.content)
            
            # Parse result - more robust
            match = "YES" in result_text.upper() and "MATCH" in result_text.upper()
            reason = result_text.split("Reason:")[-1].strip()[:100] if "Reason:" in result_text else result_text[:100]
            
            return {
                'correct': match,
                'confidence': 1.0 if match else 0.0,
                'reason': reason,
                'response': response_text[:200],
                'correct_answer': correct_text[:200]
            }
        except Exception as e:
            print(f"QA eval timeout/error: {str(e)[:50]}")
            return {
                'correct': False,
                'confidence': 0.0,
                'reason': f"Timeout",
                'response': response_text[:200],
                'correct_answer': correct_text[:200]
            }
    
    def evaluate_multi_session_continuity(
        self,
        response: str,
        related_conversation_snippet: Any,
        correct_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate Multi-Session Continuity.
        
        Checks if the agent correctly references information from prior
        conversation snippets.
        
        Args:
            response: Agent's response
            related_conversation_snippet: Prior conversation context
            correct_answer: Ground truth that should reference the snippet
            
        Returns:
            Dictionary with continuity score
        """
        if not related_conversation_snippet:
            return {
                'has_prior_context': False,
                'correctly_referenced': False,
                'score': 0.0
            }
        
        # Extract key information from conversation snippet
        snippet_text = self._ensure_text(related_conversation_snippet)
        response_text = self._ensure_text(response)
        correct_text = self._ensure_text(correct_answer)
        
        prompt = f"""You are evaluating an AI agent's ability to maintain continuity across conversation sessions.

Prior Conversation Context:
{snippet_text[:1000]}

Current Response:
{response_text}

Ground Truth (should incorporate prior context):
{correct_text}

Does the Current Response correctly reference or incorporate information from the Prior Conversation Context, similar to how the Ground Truth does?

Answer with:
CONTINUITY: YES/NO
EVIDENCE: [quote relevant parts from response that show continuity]
SCORE: [0.0-1.0, how well continuity is maintained]
"""
        
        try:
            increment_llm_call("evaluator_continuity")
            result = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = self._ensure_text(result.content)
            
            # Parse result
            has_continuity = "YES" in result_text.split("CONTINUITY:")[1].split("\n")[0].upper() if "CONTINUITY:" in result_text else False
            
            # Extract score
            score_match = re.search(r"SCORE:\s*([\d.]+)", result_text)
            score = float(score_match.group(1)) if score_match else (1.0 if has_continuity else 0.0)
            
            evidence = result_text.split("EVIDENCE:")[1].split("SCORE:")[0].strip() if "EVIDENCE:" in result_text else ""
            
            return {
                'has_prior_context': True,
                'correctly_referenced': has_continuity,
                'score': score,
                'evidence': evidence
            }
        except Exception as e:
            print(f"Error in continuity evaluation: {e}")
            return {
                'has_prior_context': True,
                'correctly_referenced': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def evaluate_hallucination_rate(
        self,
        response: str,
        ground_truth: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluate Hallucination Rate.
        
        Counts unsupported claims in the response that aren't in the
        ground truth or provided context.
        
        Args:
            response: Agent's response
            ground_truth: Ground truth answer
            context: Additional context (persona, snippets)
            
        Returns:
            Dictionary with hallucination count and rate
        """
        response_text = self._ensure_text(response)
        ground_truth_text = self._ensure_text(ground_truth)
        context_text = self._ensure_text(context)

        prompt = f"""You are evaluating an AI response for hallucinations (unsupported claims).

Ground Truth Information:
{ground_truth_text}

Additional Context:
{context_text[:500] if context_text else "None"}

AI Response to Evaluate:
{response_text}

Identify any claims in the AI Response that are:
1. Not supported by the Ground Truth or Additional Context
2. Factually incorrect based on the provided information
3. Fabricated details not present in the source material

List each hallucination found and provide a count.

Format:
HALLUCINATION_COUNT: [number]
HALLUCINATIONS:
1. [description of claim] - [why it's unsupported]
2. ...
"""
        
        try:
            increment_llm_call("evaluator_hallucination")
            result = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = self._ensure_text(result.content)
            
            # Extract hallucination count
            count_match = re.search(r"HALLUCINATION_COUNT:\s*(\d+)", result_text)
            hallucination_count = int(count_match.group(1)) if count_match else 0
            
            # Extract hallucinations list
            hallucinations = []
            if "HALLUCINATIONS:" in result_text:
                hallu_section = result_text.split("HALLUCINATIONS:")[1].strip()
                hallucinations = [line.strip() for line in hallu_section.split("\n") if line.strip() and line.strip()[0].isdigit()]
            
            return {
                'hallucination_count': hallucination_count,
                'hallucinations': hallucinations,
                'response_length': len(response_text.split()),
                'rate_per_100_words': (hallucination_count / max(len(response_text.split()), 1)) * 100
            }
        except Exception as e:
            print(f"Error in hallucination evaluation: {e}")
            return {
                'hallucination_count': 0,
                'hallucinations': [],
                'response_length': len(response_text.split()),
                'rate_per_100_words': 0.0,
                'error': str(e)
            }
    
    def evaluate_delayed_recall(
        self,
        recall_response: str,
        original_content: str,
        key_facts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate Delayed Recall Accuracy.
        
        Measures how well the agent recalls information after a delay
        (simulated by consolidation).
        
        Args:
            recall_response: Agent's recall attempt
            original_content: Original information stored
            key_facts: Key facts that should be recalled
            
        Returns:
            Dictionary with recall accuracy
        """
        recall_text = self._ensure_text(recall_response)
        original_text = self._ensure_text(original_content)
        key_fact_list = self._ensure_list(key_facts)

        prompt = f"""You are evaluating memory recall accuracy.

Original Information:
{original_text}

Key Facts to Recall:
{chr(10).join(f"- {fact}" for fact in key_fact_list)}

Recall Attempt:
{recall_text}

For each key fact, determine if it was correctly recalled (fully, partially, or not at all).

Format:
OVERALL_ACCURACY: [0.0-1.0]
DETAILS:
- Fact 1: [FULL/PARTIAL/NONE] - [explanation]
- Fact 2: [FULL/PARTIAL/NONE] - [explanation]
...
"""
        
        try:
            increment_llm_call("evaluator_delayed_recall")
            result = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = self._ensure_text(result.content)
            
            # Extract overall accuracy
            acc_match = re.search(r"OVERALL_ACCURACY:\s*([\d.]+)", result_text)
            accuracy = float(acc_match.group(1)) if acc_match else 0.0
            
            # Count recall quality
            full_recalls = len(re.findall(r"\bFULL\b", result_text))
            partial_recalls = len(re.findall(r"\bPARTIAL\b", result_text))
            none_recalls = len(re.findall(r"\bNONE\b", result_text))
            
            return {
                'accuracy': accuracy,
                'full_recalls': full_recalls,
                'partial_recalls': partial_recalls,
                'none_recalls': none_recalls,
                'total_facts': len(key_fact_list)
            }
        except Exception as e:
            print(f"Error in delayed recall evaluation: {e}")
            return {
                'accuracy': 0.0,
                'full_recalls': 0,
                'partial_recalls': 0,
                'none_recalls': len(key_fact_list),
                'total_facts': len(key_fact_list),
                'error': str(e)
            }
    
    def evaluate_cue_based_recall(
        self,
        response_with_cue: str,
        response_without_cue: str,
        cue_content: str,
        target_information: str
    ) -> Dict[str, Any]:
        """
        Evaluate Cue-Based Recall.
        
        Measures improvement in recall when provided with a cue (conversation snippet).
        
        Args:
            response_with_cue: Response when cue is provided
            response_without_cue: Response without cue
            cue_content: The cue provided
            target_information: Information that should be recalled
            
        Returns:
            Dictionary with cue effectiveness score
        """
        response_with = self._ensure_text(response_with_cue)
        response_without = self._ensure_text(response_without_cue)
        cue_text = self._ensure_text(cue_content)
        target_text = self._ensure_text(target_information)

        prompt = f"""You are evaluating the effectiveness of memory cues.

Target Information (what should be recalled):
{target_text}

Cue Provided:
{cue_text[:500]}

Response WITHOUT Cue:
{response_without}

Response WITH Cue:
{response_with}

Rate how much the cue improved recall:
1. Did the response WITH cue contain more target information?
2. Was the response WITH cue more accurate?
3. Did the cue effectively trigger relevant memories?

Provide:
IMPROVEMENT_SCORE: [0.0-1.0, where 0=no improvement, 1=significant improvement]
WITHOUT_CUE_ACCURACY: [0.0-1.0]
WITH_CUE_ACCURACY: [0.0-1.0]
EXPLANATION: [brief explanation]
"""
        
        try:
            increment_llm_call("evaluator_cue_recall")
            result = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = self._ensure_text(result.content)
            
            # Extract scores
            improvement_match = re.search(r"IMPROVEMENT_SCORE:\s*([\d.]+)", result_text)
            without_match = re.search(r"WITHOUT_CUE_ACCURACY:\s*([\d.]+)", result_text)
            with_match = re.search(r"WITH_CUE_ACCURACY:\s*([\d.]+)", result_text)
            
            improvement_score = float(improvement_match.group(1)) if improvement_match else 0.0
            without_cue_accuracy = float(without_match.group(1)) if without_match else 0.0
            with_cue_accuracy = float(with_match.group(1)) if with_match else 0.0
            
            explanation = result_text.split("EXPLANATION:")[1].strip() if "EXPLANATION:" in result_text else ""
            
            return {
                'improvement_score': improvement_score,
                'without_cue_accuracy': without_cue_accuracy,
                'with_cue_accuracy': with_cue_accuracy,
                'cue_effectiveness': with_cue_accuracy - without_cue_accuracy,
                'explanation': explanation
            }
        except Exception as e:
            print(f"Error in cue-based recall evaluation: {e}")
            return {
                'improvement_score': 0.0,
                'without_cue_accuracy': 0.0,
                'with_cue_accuracy': 0.0,
                'cue_effectiveness': 0.0,
                'error': str(e)
            }
    
    def evaluate_cross_episode_integration(
        self,
        response: str,
        episode_1: str,
        episode_2: str,
        integration_question: str
    ) -> Dict[str, Any]:
        """
        Evaluate Cross-Episode Integration.
        
        Measures ability to connect information across multiple episodes.
        
        Args:
            response: Agent's response
            episode_1: First episode content
            episode_2: Second episode content
            integration_question: Question requiring both episodes
            
        Returns:
            Dictionary with integration score
        """
        response_text = self._ensure_text(response)
        episode_1_text = self._ensure_text(episode_1)
        episode_2_text = self._ensure_text(episode_2)
        integration_q = self._ensure_text(integration_question)

        prompt = f"""You are evaluating cross-episode memory integration.

Episode 1:
{episode_1_text[:500]}

Episode 2:
{episode_2_text[:500]}

Integration Question (requires both episodes):
{integration_q}

Agent's Response:
{response_text}

Evaluate:
1. Does the response incorporate information from BOTH episodes?
2. Does the response show understanding of connections between episodes?
3. Is the integration coherent and accurate?

Provide:
INTEGRATION_SCORE: [0.0-1.0]
USES_EPISODE_1: YES/NO
USES_EPISODE_2: YES/NO
CONNECTION_QUALITY: [LOW/MEDIUM/HIGH]
EXPLANATION: [brief explanation]
"""
        
        try:
            increment_llm_call("evaluator_cross_episode")
            result = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = self._ensure_text(result.content)
            
            # Extract scores
            integration_match = re.search(r"INTEGRATION_SCORE:\s*([\d.]+)", result_text)
            integration_score = float(integration_match.group(1)) if integration_match else 0.0
            
            uses_ep1 = "YES" in result_text.split("USES_EPISODE_1:")[1].split("\n")[0].upper() if "USES_EPISODE_1:" in result_text else False
            uses_ep2 = "YES" in result_text.split("USES_EPISODE_2:")[1].split("\n")[0].upper() if "USES_EPISODE_2:" in result_text else False
            
            quality = "LOW"
            if "HIGH" in result_text:
                quality = "HIGH"
            elif "MEDIUM" in result_text:
                quality = "MEDIUM"
            
            explanation = result_text.split("EXPLANATION:")[1].strip() if "EXPLANATION:" in result_text else ""
            
            return {
                'integration_score': integration_score,
                'uses_episode_1': uses_ep1,
                'uses_episode_2': uses_ep2,
                'both_episodes_used': uses_ep1 and uses_ep2,
                'connection_quality': quality,
                'explanation': explanation
            }
        except Exception as e:
            print(f"Error in cross-episode integration evaluation: {e}")
            return {
                'integration_score': 0.0,
                'uses_episode_1': False,
                'uses_episode_2': False,
                'both_episodes_used': False,
                'connection_quality': 'LOW',
                'error': str(e)
            }
    
    def evaluate_schema_utilization(
        self,
        response: str,
        persona_schema: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Evaluate Schema Utilization Rate.
        
        Measures how well the agent uses persona schemas in responses.
        
        Args:
            response: Agent's response
            persona_schema: Persona schema information
            query: User query
            
        Returns:
            Dictionary with schema utilization score
        """
        response_text = self._ensure_text(response)
        persona_text = self._ensure_text(persona_schema)
        query_text = self._ensure_text(query)

        prompt = f"""You are evaluating schema utilization in AI responses.

Persona Schema (knowledge about the user):
{persona_text}

User Query:
{query_text}

Agent's Response:
{response_text}

Evaluate:
1. Does the response appropriately use information from the persona schema?
2. Is the schema application relevant to the query?
3. Does it show understanding of the persona?

Provide:
SCHEMA_UTILIZATION_SCORE: [0.0-1.0]
SCHEMA_ELEMENTS_USED: [count of schema elements referenced]
RELEVANCE: [LOW/MEDIUM/HIGH]
EXAMPLES: [specific examples of schema use from the response]
"""
        
        try:
            increment_llm_call("evaluator_schema_utilization")
            result = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = self._ensure_text(result.content)
            
            # Extract scores
            util_match = re.search(r"SCHEMA_UTILIZATION_SCORE:\s*([\d.]+)", result_text)
            utilization_score = float(util_match.group(1)) if util_match else 0.0
            
            elements_match = re.search(r"SCHEMA_ELEMENTS_USED:\s*(\d+)", result_text)
            elements_used = int(elements_match.group(1)) if elements_match else 0
            
            relevance = "LOW"
            if "HIGH" in result_text:
                relevance = "HIGH"
            elif "MEDIUM" in result_text:
                relevance = "MEDIUM"
            
            examples = result_text.split("EXAMPLES:")[1].strip() if "EXAMPLES:" in result_text else ""
            
            return {
                'schema_utilization_score': utilization_score,
                'schema_elements_used': elements_used,
                'relevance': relevance,
                'examples': examples[:200]
            }
        except Exception as e:
            print(f"Error in schema utilization evaluation: {e}")
            return {
                'schema_utilization_score': 0.0,
                'schema_elements_used': 0,
                'relevance': 'LOW',
                'error': str(e)
            }


def aggregate_results(results: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, float]:
    """
    Aggregate evaluation results across multiple samples.
    
    Args:
        results: List of individual evaluation results
        metric_keys: Keys to aggregate
        
    Returns:
        Dictionary of aggregated metrics
    """
    aggregated = {}
    
    for key in metric_keys:
        values = [r.get(key, 0) for r in results if key in r]
        if values:
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_count"] = len(values)
        else:
            aggregated[f"{key}_mean"] = 0.0
            aggregated[f"{key}_count"] = 0
    
    return aggregated
