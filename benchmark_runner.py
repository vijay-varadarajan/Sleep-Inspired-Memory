"""
PersonaMem Benchmark Runner

Orchestrates running all experiments and generating Tables 1 and 2.

Usage:
    python benchmark_runner.py --split benchmark --num_samples 100 --methods all

This will:
1. Load preprocessed PersonaMem data
2. Run each baseline method
3. Evaluate using PersonaMemEvaluator
4. Generate formatted tables
5. Save results
"""

import os
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from dotenv import load_dotenv

from evaluation.baselines import create_agent
from evaluation.personamem_benchmark import PersonaMemEvaluator, aggregate_results


class BenchmarkRunner:
    """Main benchmark runner for PersonaMem experiments."""
    
    def __init__(
        self,
        split: str = "benchmark",
        num_samples: int = 100,
        methods: List[str] = None,
        output_dir: str = "results"
    ):
        """
        Initialize benchmark runner.
        
        Args:
            split: Dataset split to use ('benchmark', 'val', 'train')
            num_samples: Number of samples to evaluate
            methods: List of methods to evaluate
            output_dir: Directory to save results
        """
        load_dotenv()
        
        self.split = split
        self.num_samples = num_samples
        self.methods = methods or ['vanilla', 'rag', 'episodic', 'summarization', 'sleep']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load preprocessed data
        self.data_dir = Path("PERSONAMEM/preprocessed")
        self.samples = self._load_data()
        self.persona_groups = self._load_persona_groups()
        
        # Initialize evaluator
        self.evaluator = PersonaMemEvaluator()
        
        print(f"\n{'='*70}")
        print(f"Benchmark Runner Initialized")
        print(f"{'='*70}")
        print(f"Split: {split}")
        print(f"Samples: {len(self.samples)} (will use {num_samples})")
        print(f"Methods: {', '.join(self.methods)}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load preprocessed data."""
        data_file = self.data_dir / f"{self.split}_processed.json"
        with open(data_file, 'r') as f:
            return json.load(f)
    
    def _load_persona_groups(self) -> Dict[int, List[Dict[str, Any]]]:
        """Load persona-grouped data."""
        persona_file = self.data_dir / f"{self.split}_persona_sessions.json"
        with open(persona_file, 'r') as f:
            data = json.load(f)
            # Convert string keys to int
            return {int(k): v for k, v in data.items()}
    
    def run_table1_evaluation(self, method: str) -> Dict[str, Any]:
        """
        Run Table 1 evaluation: Task-Based Memory Performance.
        
        Metrics:
        - Long-Horizon QA
        - Multi-Session Continuity
        - Hallucination Rate
        
        Args:
            method: Method to evaluate
            
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*70}")
        print(f"TABLE 1 Evaluation: {method.upper()}")
        print(f"{'='*70}\n")
        
        # Create agent
        agent = create_agent(method)
        
        # Select samples
        samples_to_use = self.samples[:self.num_samples]
        
        # Group samples by persona for multi-session scenarios
        persona_samples = defaultdict(list)
        for sample in samples_to_use:
            persona_samples[sample['persona_id']].append(sample)
        
        # Results storage
        qa_results = []
        continuity_results = []
        hallucination_results = []
        
        # Process each persona (multi-session)
        for persona_id, persona_sample_list in tqdm(list(persona_samples.items())[:20], desc=f"Evaluating {method}"):
            # Sort by any available timestamp or use order
            persona_sample_list = persona_sample_list[:5]  # Limit to 5 interactions per persona
            
            persona_info = persona_sample_list[0]['persona']
            
            # Simulate multi-session interactions
            for i, sample in enumerate(persona_sample_list):
                query = sample['query']
                correct_answer = sample['correct_answer']
                incorrect_answers = sample['incorrect_answers']
                related_snippet = sample['related_conversation_snippet']
                
                # Agent interaction
                use_memory = (i > 0)  # Use memory after first interaction
                
                try:
                    response = agent.interact(
                        user_input=query,
                        persona=persona_info,
                        importance=0.7,
                        tags=['persona_session'],
                        use_memory=use_memory
                    )
                except Exception as e:
                    print(f"Error in agent interaction: {e}")
                    response = "Error generating response"
                
                # Evaluate Long-Horizon QA
                qa_result = self.evaluator.evaluate_long_horizon_qa(
                    response, correct_answer, incorrect_answers
                )
                qa_results.append(qa_result)
                
                # Evaluate Multi-Session Continuity (if there's prior context)
                if related_snippet:
                    continuity_result = self.evaluator.evaluate_multi_session_continuity(
                        response, related_snippet, correct_answer
                    )
                    continuity_results.append(continuity_result)
                
                # Evaluate Hallucination Rate
                context = f"Persona: {persona_info}\nRelated Context: {str(related_snippet)[:500]}"
                hallucination_result = self.evaluator.evaluate_hallucination_rate(
                    response, correct_answer, context
                )
                hallucination_results.append(hallucination_result)
            
            # Sleep consolidation (if applicable)
            if method == 'sleep' and hasattr(agent, 'sleep'):
                try:
                    agent.sleep(verbose=False)
                except Exception as e:
                    print(f"Error in sleep consolidation: {e}")
        
        # Aggregate results
        qa_accuracy = sum(r['correct'] for r in qa_results) / len(qa_results) if qa_results else 0.0
        
        continuity_score = sum(r['score'] for r in continuity_results) / len(continuity_results) if continuity_results else 0.0
        
        avg_hallucinations = sum(r['hallucination_count'] for r in hallucination_results) / len(hallucination_results) if hallucination_results else 0.0
        total_words = sum(r['response_length'] for r in hallucination_results)
        hallucination_rate = (sum(r['hallucination_count'] for r in hallucination_results) / (total_words / 100)) if total_words > 0 else 0.0
        
        table1_results = {
            'method': method,
            'long_horizon_qa': qa_accuracy * 100,  # Convert to percentage
            'multi_session_continuity': continuity_score * 100,  # Convert to percentage
            'hallucination_rate': hallucination_rate,
            'num_qa_samples': len(qa_results),
            'num_continuity_samples': len(continuity_results),
            'num_hallucination_samples': len(hallucination_results)
        }
        
        print(f"\nResults for {method}:")
        print(f"  Long-Horizon QA: {table1_results['long_horizon_qa']:.2f}%")
        print(f"  Multi-Session Continuity: {table1_results['multi_session_continuity']:.2f}%")
        print(f"  Hallucination Rate: {table1_results['hallucination_rate']:.2f} per 100 words")
        
        return table1_results
    
    def run_table2_evaluation(self, method: str) -> Dict[str, Any]:
        """
        Run Table 2 evaluation: Cognitive-Style Probes (Before vs After Sleep).
        
        Metrics:
        - Delayed Recall Accuracy
        - Cue-Based Recall
        - Cross-Episode Integration
        - Schema Utilization Rate
        
        Args:
            method: Method to evaluate
            
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*70}")
        print(f"TABLE 2 Evaluation: {method.upper()}")
        print(f"{'='*70}\n")
        
        # Only evaluate sleep method for Table 2 (before/after comparison)
        if method != 'sleep':
            print(f"Skipping Table 2 for {method} (only applicable to sleep method)")
            return {
                'method': method,
                'applicable': False
            }
        
        # Create agent
        agent = create_agent(method)
        
        # Select samples
        samples_to_use = self.samples[:min(50, self.num_samples)]  # Smaller sample for Table 2
        
        # Group by persona
        persona_samples = defaultdict(list)
        for sample in samples_to_use:
            persona_samples[sample['persona_id']].append(sample)
        
        # Results storage
        delayed_recall_pre = []
        delayed_recall_post = []
        cue_based_pre = []
        cue_based_post = []
        integration_pre = []
        integration_post = []
        schema_util_pre = []
        schema_util_post = []
        
        # Process personas
        for persona_id, persona_sample_list in tqdm(list(persona_samples.items())[:10], desc="Table 2 Evaluation"):
            persona_sample_list = persona_sample_list[:3]
            persona_info = persona_sample_list[0]['persona']
            
            # Store initial interactions
            for sample in persona_sample_list:
                agent.interact(
                    user_input=sample['query'],
                    persona=persona_info,
                    importance=0.8,
                    tags=['test'],
                    use_memory=False
                )
            
            # PRE-CONSOLIDATION TESTS
            if len(persona_sample_list) >= 2:
                test_sample = persona_sample_list[0]
                
                # Delayed Recall (pre)
                recall_query = f"What did we discuss about {test_sample['topic_query']}?"
                recall_response_pre = agent.interact(recall_query, persona=persona_info, use_memory=True)
                
                delayed_recall_result = self.evaluator.evaluate_delayed_recall(
                    recall_response_pre,
                    test_sample['query'] + " " + test_sample['correct_answer'],
                    [test_sample['topic_query']]
                )
                delayed_recall_pre.append(delayed_recall_result['accuracy'])
                
                # Cue-Based Recall (pre)
                if test_sample['related_conversation_snippet']:
                    cue_query = "Based on our previous conversations, what would you recommend?"
                    response_without_cue = agent.interact(cue_query, persona=persona_info, use_memory=True)
                    
                    # Provide cue
                    cue_text = str(test_sample['related_conversation_snippet'])[:500]
                    cue_query_with = f"Given this context: {cue_text}\n\n{cue_query}"
                    response_with_cue = agent.interact(cue_query_with, persona=persona_info, use_memory=True)
                    
                    cue_result = self.evaluator.evaluate_cue_based_recall(
                        response_with_cue, response_without_cue,
                        cue_text, test_sample['correct_answer']
                    )
                    cue_based_pre.append(cue_result['with_cue_accuracy'])
                
                # Cross-Episode Integration (pre)
                if len(persona_sample_list) >= 2:
                    integration_query = f"How does {persona_sample_list[0]['topic_query']} relate to {persona_sample_list[1]['topic_query']}?"
                    integration_response = agent.interact(integration_query, persona=persona_info, use_memory=True)
                    
                    integration_result = self.evaluator.evaluate_cross_episode_integration(
                        integration_response,
                        persona_sample_list[0]['query'],
                        persona_sample_list[1]['query'],
                        integration_query
                    )
                    integration_pre.append(integration_result['integration_score'])
                
                # Schema Utilization (pre)
                schema_query = persona_sample_list[-1]['query']
                schema_response = agent.interact(schema_query, persona=persona_info, use_memory=True)
                
                schema_result = self.evaluator.evaluate_schema_utilization(
                    schema_response, persona_info, schema_query
                )
                schema_util_pre.append(schema_result['schema_utilization_score'])
            
            # SLEEP CONSOLIDATION
            try:
                agent.sleep(verbose=False)
            except Exception as e:
                print(f"Error during sleep: {e}")
            
            # POST-CONSOLIDATION TESTS (repeat same tests)
            if len(persona_sample_list) >= 2:
                test_sample = persona_sample_list[0]
                
                # Delayed Recall (post)
                recall_query = f"What did we discuss about {test_sample['topic_query']}?"
                recall_response_post = agent.interact(recall_query, persona=persona_info, use_memory=True)
                
                delayed_recall_result = self.evaluator.evaluate_delayed_recall(
                    recall_response_post,
                    test_sample['query'] + " " + test_sample['correct_answer'],
                    [test_sample['topic_query']]
                )
                delayed_recall_post.append(delayed_recall_result['accuracy'])
                
                # Cue-Based Recall (post)
                if test_sample['related_conversation_snippet']:
                    response_without_cue = agent.interact(cue_query, persona=persona_info, use_memory=True)
                    response_with_cue = agent.interact(cue_query_with, persona=persona_info, use_memory=True)
                    
                    cue_result = self.evaluator.evaluate_cue_based_recall(
                        response_with_cue, response_without_cue,
                        cue_text, test_sample['correct_answer']
                    )
                    cue_based_post.append(cue_result['with_cue_accuracy'])
                
                # Cross-Episode Integration (post)
                if len(persona_sample_list) >= 2:
                    integration_response = agent.interact(integration_query, persona=persona_info, use_memory=True)
                    
                    integration_result = self.evaluator.evaluate_cross_episode_integration(
                        integration_response,
                        persona_sample_list[0]['query'],
                        persona_sample_list[1]['query'],
                        integration_query
                    )
                    integration_post.append(integration_result['integration_score'])
                
                # Schema Utilization (post)
                schema_response = agent.interact(schema_query, persona=persona_info, use_memory=True)
                
                schema_result = self.evaluator.evaluate_schema_utilization(
                    schema_response, persona_info, schema_query
                )
                schema_util_post.append(schema_result['schema_utilization_score'])
        
        # Aggregate results
        def safe_mean(lst):
            return np.mean(lst) * 100 if lst else 0.0
        
        table2_results = {
            'method': method,
            'applicable': True,
            'delayed_recall_pre': safe_mean(delayed_recall_pre),
            'delayed_recall_post': safe_mean(delayed_recall_post),
            'delayed_recall_improvement': safe_mean(delayed_recall_post) - safe_mean(delayed_recall_pre),
            'cue_based_pre': safe_mean(cue_based_pre),
            'cue_based_post': safe_mean(cue_based_post),
            'cue_based_improvement': safe_mean(cue_based_post) - safe_mean(cue_based_pre),
            'integration_pre': safe_mean(integration_pre),
            'integration_post': safe_mean(integration_post),
            'integration_improvement': safe_mean(integration_post) - safe_mean(integration_pre),
            'schema_util_pre': safe_mean(schema_util_pre),
            'schema_util_post': safe_mean(schema_util_post),
            'schema_util_improvement': safe_mean(schema_util_post) - safe_mean(schema_util_pre)
        }
        
        print(f"\nTable 2 Results (Before → After Sleep):")
        print(f"  Delayed Recall: {table2_results['delayed_recall_pre']:.2f}% → {table2_results['delayed_recall_post']:.2f}% (Δ{table2_results['delayed_recall_improvement']:+.2f}%)")
        print(f"  Cue-Based Recall: {table2_results['cue_based_pre']:.2f}% → {table2_results['cue_based_post']:.2f}% (Δ{table2_results['cue_based_improvement']:+.2f}%)")
        print(f"  Integration: {table2_results['integration_pre']:.2f}% → {table2_results['integration_post']:.2f}% (Δ{table2_results['integration_improvement']:+.2f}%)")
        print(f"  Schema Util: {table2_results['schema_util_pre']:.2f}% → {table2_results['schema_util_post']:.2f}% (Δ{table2_results['schema_util_improvement']:+.2f}%)")
        
        return table2_results
    
    def run_all_experiments(self):
        """Run all experiments and generate tables."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Table 1: All methods
        print("\n" + "="*70)
        print("RUNNING TABLE 1 EXPERIMENTS")
        print("="*70)
        
        table1_results = []
        for method in self.methods:
            try:
                result = self.run_table1_evaluation(method)
                table1_results.append(result)
            except Exception as e:
                print(f"Error evaluating {method}: {e}")
                import traceback
                traceback.print_exc()
        
        # Table 2: Only sleep method
        print("\n" + "="*70)
        print("RUNNING TABLE 2 EXPERIMENTS")
        print("="*70)
        
        table2_results = []
        try:
            result = self.run_table2_evaluation('sleep')
            table2_results.append(result)
        except Exception as e:
            print(f"Error in Table 2 evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Format and display tables
        self.display_table1(table1_results)
        if table2_results and table2_results[0].get('applicable'):
            self.display_table2(table2_results[0])
        
        # Save results
        results = {
            'timestamp': timestamp,
            'split': self.split,
            'num_samples': self.num_samples,
            'methods': self.methods,
            'table1': table1_results,
            'table2': table2_results
        }
        
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save Table 1 as CSV
        table1_csv = self.output_dir / f"table1_{timestamp}.csv"
        with open(table1_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method",
                "Long-Horizon QA",
                "Multi-Session Continuity",
                "Hallucination Rate"
            ])
            for row in table1_results:
                writer.writerow([
                    row.get('method', ''),
                    f"{row.get('long_horizon_qa', 0.0):.4f}",
                    f"{row.get('multi_session_continuity', 0.0):.4f}",
                    f"{row.get('hallucination_rate', 0.0):.6f}"
                ])

        # Save Table 2 as CSV (if applicable)
        if table2_results and table2_results[0].get('applicable'):
            table2 = table2_results[0]
            table2_csv = self.output_dir / f"table2_{timestamp}.csv"
            with open(table2_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Probe",
                    "Pre-Consolidation",
                    "Post-Consolidation",
                    "Delta Improvement"
                ])
                writer.writerow([
                    "Delayed Recall Accuracy",
                    f"{table2.get('delayed_recall_pre', 0.0):.4f}",
                    f"{table2.get('delayed_recall_post', 0.0):.4f}",
                    f"{table2.get('delayed_recall_improvement', 0.0):.4f}"
                ])
                writer.writerow([
                    "Cue-Based Recall",
                    f"{table2.get('cue_based_pre', 0.0):.4f}",
                    f"{table2.get('cue_based_post', 0.0):.4f}",
                    f"{table2.get('cue_based_improvement', 0.0):.4f}"
                ])
                writer.writerow([
                    "Cross-Episode Integration",
                    f"{table2.get('integration_pre', 0.0):.4f}",
                    f"{table2.get('integration_post', 0.0):.4f}",
                    f"{table2.get('integration_improvement', 0.0):.4f}"
                ])
                writer.writerow([
                    "Schema Utilization Rate",
                    f"{table2.get('schema_util_pre', 0.0):.4f}",
                    f"{table2.get('schema_util_post', 0.0):.4f}",
                    f"{table2.get('schema_util_improvement', 0.0):.4f}"
                ])
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {results_file}")
        print(f"Table 1 CSV saved to: {table1_csv}")
        if table2_results and table2_results[0].get('applicable'):
            print(f"Table 2 CSV saved to: {table2_csv}")
        print(f"{'='*70}\n")
        
        return results
    
    def display_table1(self, results: List[Dict[str, Any]]):
        """Display Table 1 in formatted output."""
        print("\n" + "="*70)
        print("TABLE 1: Task-Based Memory Performance")
        print("="*70)
        print(f"{'Method':<25} {'Long-Horizon QA ↑':<20} {'Multi-Session Continuity ↑':<30} {'Hallucination Rate ↓':<25}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['method']:<25} {result['long_horizon_qa']:>6.2f}%{'':<13} {result['multi_session_continuity']:>6.2f}%{'':<23} {result['hallucination_rate']:>6.2f}")
        
        print("="*70 + "\n")
    
    def display_table2(self, result: Dict[str, Any]):
        """Display Table 2 in formatted output."""
        print("\n" + "="*70)
        print("TABLE 2: Cognitive-Style Probes (Before vs After Sleep)")
        print("="*70)
        print(f"{'Probe':<30} {'Pre-Consolidation':<20} {'Post-Consolidation':<20} {'Δ Improvement':<15}")
        print("-" * 70)
        
        probes = [
            ('Delayed Recall Accuracy', 'delayed_recall'),
            ('Cue-Based Recall', 'cue_based'),
            ('Cross-Episode Integration', 'integration'),
            ('Schema Utilization Rate', 'schema_util')
        ]
        
        for probe_name, probe_key in probes:
            pre = result[f'{probe_key}_pre']
            post = result[f'{probe_key}_post']
            improvement = result[f'{probe_key}_improvement']
            print(f"{probe_name:<30} {pre:>6.2f}%{'':<13} {post:>6.2f}%{'':<13} {improvement:>+6.2f}%")
        
        print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PersonaMem benchmarks")
    parser.add_argument('--split', type=str, default='benchmark', choices=['benchmark', 'val', 'train'],
                        help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to evaluate (default: all)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Default to all methods if not specified
    if args.methods is None or 'all' in args.methods:
        args.methods = ['vanilla', 'rag', 'episodic', 'summarization', 'sleep']
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        split=args.split,
        num_samples=args.num_samples,
        methods=args.methods,
        output_dir=args.output_dir
    )
    
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
