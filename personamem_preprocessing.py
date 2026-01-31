"""
PersonaMem Dataset Preprocessing

Preprocesses the PersonaMem-v2 text datasets for benchmarking sleep-inspired
memory consolidation. Creates a standardized format for evaluation.

The preprocessing includes:
1. Loading benchmark_text, train_text, and val_text splits
2. Parsing JSON fields (user_query, persona, etc.)
3. Organizing data by persona_id for multi-session scenarios
4. Creating temporal sequences for delayed recall testing
5. Extracting conversation history snippets
6. Saving preprocessed data in a standardized format

Output Structure:
    PERSONAMEM/preprocessed/
        benchmark_processed.json
        train_processed.json
        val_processed.json
        persona_sessions.json (grouped by persona_id)
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path
from datasets import load_from_disk
from collections import defaultdict
import ast


def parse_json_field(field_value: str) -> Any:
    """
    Parse JSON-like string fields from the dataset.
    
    Args:
        field_value: String representation of JSON/dict
        
    Returns:
        Parsed Python object
    """
    if not field_value or field_value == "":
        return None
    
    try:
        # Try standard JSON parsing
        return json.loads(field_value)
    except json.JSONDecodeError:
        try:
            # Try Python literal evaluation (for single quotes)
            return ast.literal_eval(field_value)
        except (ValueError, SyntaxError):
            # Return as-is if parsing fails
            return field_value


def preprocess_sample(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Preprocess a single sample from the dataset.
    
    Args:
        sample: Raw sample from dataset
        index: Sample index
        
    Returns:
        Preprocessed sample with parsed fields
    """
    # Parse JSON fields
    user_query = parse_json_field(sample['user_query'])
    short_persona = parse_json_field(sample['short_persona'])
    expanded_persona = parse_json_field(sample['expanded_persona'])
    related_snippet = parse_json_field(sample['related_conversation_snippet'])
    
    # Extract user query text
    query_text = ""
    if isinstance(user_query, dict) and 'content' in user_query:
        query_text = user_query['content']
    elif isinstance(user_query, str):
        query_text = user_query
    
    # Extract persona text
    persona_text = ""
    if isinstance(short_persona, dict) and 'persona' in short_persona:
        persona_text = short_persona['persona']
    elif isinstance(short_persona, str):
        persona_text = short_persona
    
    # Parse incorrect answers (usually a list)
    incorrect_answers = []
    try:
        if isinstance(sample['incorrect_answers'], str):
            incorrect_answers = json.loads(sample['incorrect_answers'])
        else:
            incorrect_answers = sample['incorrect_answers']
    except:
        incorrect_answers = []
    
    # Build processed sample
    processed = {
        'id': f"sample_{index}",
        'persona_id': int(sample['persona_id']),
        'query': query_text,
        'correct_answer': sample['correct_answer'],
        'incorrect_answers': incorrect_answers,
        'persona': persona_text,
        'expanded_persona': expanded_persona,
        'related_conversation_snippet': related_snippet,
        'topic_query': sample.get('topic_query', ''),
        'preference': sample.get('preference', ''),
        'topic_preference': sample.get('topic_preference', ''),
        'conversation_scenario': sample.get('conversation_scenario', ''),
        'pref_type': sample.get('pref_type', ''),
        'who': sample.get('who', ''),
        'updated': sample.get('updated', ''),
        'prev_pref': sample.get('prev_pref', ''),
        # Metadata
        'distance_from_related_snippet_to_query_32k': sample.get('distance_from_related_snippet_to_query_32k', 0),
        'num_persona_relevant_tokens_32k': sample.get('num_persona_relevant_tokens_32k', 0),
        'num_persona_irrelevant_tokens_32k': sample.get('num_persona_irrelevant_tokens_32k', 0),
    }
    
    return processed


def group_by_persona(samples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group samples by persona_id for multi-session testing.
    
    Args:
        samples: List of preprocessed samples
        
    Returns:
        Dictionary mapping persona_id to list of samples
    """
    persona_groups = defaultdict(list)
    for sample in samples:
        persona_groups[sample['persona_id']].append(sample)
    
    return dict(persona_groups)


def preprocess_split(
    dataset_path: str,
    split_name: str,
    output_dir: str
) -> tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """
    Preprocess a single dataset split.
    
    Args:
        dataset_path: Path to the dataset split directory
        split_name: Name of the split (benchmark, train, val)
        output_dir: Output directory for preprocessed data
        
    Returns:
        Tuple of (processed_samples, persona_grouped_samples)
    """
    print(f"\n{'='*70}")
    print(f"Preprocessing {split_name} split")
    print(f"{'='*70}")
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} samples")
    
    # Preprocess all samples
    print("Processing samples...")
    processed_samples = []
    for idx, sample in enumerate(dataset):
        processed = preprocess_sample(sample, idx)
        processed_samples.append(processed)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples")
    
    print(f"Completed processing {len(processed_samples)} samples")
    
    # Group by persona
    print("Grouping by persona_id...")
    persona_groups = group_by_persona(processed_samples)
    print(f"Found {len(persona_groups)} unique personas")
    
    # Calculate statistics
    persona_counts = [len(samples) for samples in persona_groups.values()]
    print(f"  Samples per persona: min={min(persona_counts)}, max={max(persona_counts)}, avg={sum(persona_counts)/len(persona_counts):.2f}")
    
    # Save processed data
    output_file = os.path.join(output_dir, f"{split_name}_processed.json")
    print(f"Saving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(processed_samples, f, indent=2)
    
    persona_file = os.path.join(output_dir, f"{split_name}_persona_sessions.json")
    print(f"Saving persona groups to: {persona_file}")
    with open(persona_file, 'w') as f:
        json.dump(persona_groups, f, indent=2)
    
    return processed_samples, persona_groups


def create_evaluation_subsets(
    processed_samples: List[Dict[str, Any]],
    persona_groups: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    split_name: str
):
    """
    Create specialized subsets for different evaluation tasks.
    
    Args:
        processed_samples: All processed samples
        persona_groups: Samples grouped by persona
        output_dir: Output directory
        split_name: Name of the split
    """
    print("\nCreating evaluation subsets...")
    
    # Subset 1: Long-horizon QA (personas with multiple interactions)
    long_horizon_personas = {
        pid: samples for pid, samples in persona_groups.items()
        if len(samples) >= 3
    }
    long_horizon_samples = []
    for samples in long_horizon_personas.values():
        long_horizon_samples.extend(samples)
    
    print(f"  Long-horizon QA subset: {len(long_horizon_samples)} samples from {len(long_horizon_personas)} personas")
    
    # Subset 2: Multi-session continuity (samples with related_conversation_snippet)
    multi_session_samples = [
        s for s in processed_samples
        if s['related_conversation_snippet'] is not None
    ]
    print(f"  Multi-session continuity subset: {len(multi_session_samples)} samples")
    
    # Subset 3: Cross-episode integration (personas with related snippets)
    cross_episode_personas = defaultdict(list)
    for sample in multi_session_samples:
        cross_episode_personas[sample['persona_id']].append(sample)
    
    print(f"  Cross-episode integration subset: {len(cross_episode_personas)} personas")
    
    # Save subsets
    subsets = {
        'long_horizon': {
            'samples': long_horizon_samples,
            'persona_groups': long_horizon_personas,
            'description': 'Personas with 3+ interactions for long-horizon QA testing'
        },
        'multi_session': {
            'samples': multi_session_samples,
            'description': 'Samples with related conversation snippets for continuity testing'
        },
        'cross_episode': {
            'persona_groups': cross_episode_personas,
            'description': 'Personas with related snippets for integration testing'
        }
    }
    
    subsets_file = os.path.join(output_dir, f"{split_name}_evaluation_subsets.json")
    print(f"Saving evaluation subsets to: {subsets_file}")
    
    # Convert defaultdict to regular dict for JSON serialization
    subsets_serializable = {
        'long_horizon': {
            'samples': subsets['long_horizon']['samples'],
            'persona_groups': {str(k): v for k, v in subsets['long_horizon']['persona_groups'].items()},
            'description': subsets['long_horizon']['description']
        },
        'multi_session': subsets['multi_session'],
        'cross_episode': {
            'persona_groups': {str(k): v for k, v in cross_episode_personas.items()},
            'description': subsets['cross_episode']['description']
        }
    }
    
    with open(subsets_file, 'w') as f:
        json.dump(subsets_serializable, f, indent=2)


def main():
    """Main preprocessing pipeline."""
    print("\n" + "="*70)
    print("PersonaMem Dataset Preprocessing")
    print("="*70)
    
    # Paths
    base_path = Path(__file__).parent / "PERSONAMEM"
    output_dir = base_path / "preprocessed"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Preprocess each split
    splits_to_process = [
        ('benchmark_text', 'benchmark'),
        ('train_text', 'train'),
        ('val_text', 'val')
    ]
    
    all_results = {}
    
    for dataset_name, split_name in splits_to_process:
        dataset_path = base_path / dataset_name
        
        if not dataset_path.exists():
            print(f"\nWarning: {dataset_path} not found, skipping...")
            continue
        
        try:
            processed_samples, persona_groups = preprocess_split(
                str(dataset_path),
                split_name,
                str(output_dir)
            )
            
            # Create evaluation subsets
            create_evaluation_subsets(
                processed_samples,
                persona_groups,
                str(output_dir),
                split_name
            )
            
            all_results[split_name] = {
                'num_samples': len(processed_samples),
                'num_personas': len(persona_groups)
            }
            
        except Exception as e:
            print(f"Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save preprocessing summary
    summary = {
        'preprocessing_date': str(Path(__file__).stat().st_mtime),
        'splits': all_results,
        'output_directory': str(output_dir),
        'description': 'Preprocessed PersonaMem-v2 text datasets for memory consolidation benchmarking'
    }
    
    summary_file = output_dir / "preprocessing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print(f"\nSummary:")
    for split_name, results in all_results.items():
        print(f"  {split_name}: {results['num_samples']} samples, {results['num_personas']} personas")
    print(f"\nAll files saved to: {output_dir}")


if __name__ == "__main__":
    main()
