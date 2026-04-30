"""
Sleep-Based Memory Consolidation

Orchestrates the offline consolidation process that transforms
episodic memories into consolidated long-term memories and schemas.

Inspired by:
- Sleep-dependent memory consolidation in humans
- Systems consolidation theory
- Active systems consolidation hypothesis
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from memory.episodic import EpisodicMemoryStore, Episode
from memory.consolidated import ConsolidatedMemoryStore, ConsolidatedMemory
from memory.schema import SchemaStore, Schema
from memory.config import DatasetMemoryConfig, resolve_dataset_config
from sleep.replay import (
    select_episodes_for_replay,
    select_diverse_batch,
    select_episodes_for_replay_weighted,
)

# Simplified imports and removed unused code
from sleep.compression import MemoryCompressor, CompressionResult


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SleepCycle:
    
    def __init__(
        self,
        episodic_store: EpisodicMemoryStore,
        consolidated_store: ConsolidatedMemoryStore,
        schema_store: SchemaStore,
        compressor: MemoryCompressor,
        dataset_name: str = "personamem",
        policy: Optional[DatasetMemoryConfig] = None,
        replay_batch_size: int = 10,
        consolidation_batch_size: int = 3,
        schema_min_memories: int = 3
    ):
        """
        Initialize the sleep consolidation system.
        
        Args:
            episodic_store: Episodic memory store
            consolidated_store: Consolidated memory store
            schema_store: Schema store
            compressor: LLM-based memory compressor
            replay_batch_size: Number of episodes to replay per cycle
            consolidation_batch_size: Number of episodes to consolidate together
            schema_min_memories: Minimum memories needed to induce a schema
        """
        self.episodic_store = episodic_store
        self.consolidated_store = consolidated_store
        self.schema_store = schema_store
        self.compressor = compressor
        self.replay_batch_size = replay_batch_size
        self.consolidation_batch_size = consolidation_batch_size
        self.schema_min_memories = schema_min_memories
        self.policy = policy or resolve_dataset_config(dataset_name)
        
        self.cycle_count = 0
        self.total_consolidated = 0
        self.total_schemas_formed = 0
    
    def run_sleep_cycle(
        self,
        current_time: Optional[datetime] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a complete sleep cycle.
        
        Args:
            current_time: Current time (defaults to now)
            verbose: Whether to log progress
            
        Returns:
            Dictionary with cycle statistics
        """
        if current_time is None:
            current_time = datetime.now()
        
        self.cycle_count += 1
        
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Sleep Cycle #{self.cycle_count}")
            logger.info(f"{'='*60}")
        
        stats = {
            'cycle_number': self.cycle_count,
            'timestamp': current_time.isoformat(),
            'episodes_replayed': 0,
            'memories_consolidated': 0,
            'schemas_formed': 0,
            'episodes_forgotten': 0,
            'conflicts_detected': 0,
            'replay_modes': {},
        }
        
        # Phase 1: Prioritized Replay
        if verbose:
            logger.info("\n[Phase 1] Prioritized Replay")
        
        replay_results = self._phase_1_replay(current_time, verbose)
        stats['episodes_replayed'] = replay_results['replayed']
        stats['replay_modes'] = replay_results.get('replay_modes', {})
        
        # Phase 2: Generative Compression & Consolidation
        if verbose:
            logger.info("\n[Phase 2] Generative Compression & Consolidation")
        
        consolidation_results = self._phase_2_consolidation(
            replay_results['selected_episodes'],
            replay_results.get('replay_annotations', []),
            verbose
        )
        stats['memories_consolidated'] = consolidation_results['consolidated']
        stats['conflicts_detected'] = consolidation_results.get('conflicts_detected', 0)
        
        # Phase 3: Schema Formation
        if verbose:
            logger.info("\n[Phase 3] Schema Formation")
        
        schema_results = self._phase_3_schema_formation(verbose)
        stats['schemas_formed'] = schema_results['schemas_formed']
        
        # Phase 4: Forgetting & Decay
        if verbose:
            logger.info("\n[Phase 4] Forgetting & Decay")
        
        decay_results = self._phase_4_decay(verbose)
        stats['episodes_forgotten'] = decay_results['forgotten']
        
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Sleep Cycle #{self.cycle_count} Complete")
            logger.info(f"  Replayed: {stats['episodes_replayed']} episodes")
            logger.info(f"  Consolidated: {stats['memories_consolidated']} memories")
            logger.info(f"  Schemas: {stats['schemas_formed']} new schemas")
            logger.info(f"  Forgotten: {stats['episodes_forgotten']} episodes")
            logger.info(f"{'='*60}\n")
        
        return stats
    
    def _phase_1_replay(
        self,
        current_time: datetime,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Phase 1: Select and replay important episodes.
        
        Returns:
            Dictionary with replay results
        """
        # Get unconsolidated episodes
        unconsolidated = self.episodic_store.get_unconsolidated()
        
        if not unconsolidated:
            if verbose:
                logger.info("  No unconsolidated episodes to replay")
            return {'replayed': 0, 'selected_episodes': []}
        
        if verbose:
            logger.info(f"  Found {len(unconsolidated)} unconsolidated episodes")
        
        # Dataset-aware weighted replay selection with adaptive top-k and replay modes.
        selected_weighted = select_episodes_for_replay_weighted(
            episodes=unconsolidated,
            config=self.policy,
            current_time=current_time,
            exclude_consolidated=True,
        )

        # Backwards fallback in case weighted selection returns empty.
        if not selected_weighted:
            selected_fallback = select_episodes_for_replay(
                episodes=unconsolidated,
                n_replay=self.replay_batch_size,
                current_time=current_time,
            )
            selected_weighted = [(ep, p, "lossy") for ep, p in selected_fallback]
        
        # Mark episodes as accessed (simulates replay)
        selected_episodes = []
        replay_annotations = []
        replay_modes: Dict[str, int] = {}
        for episode, priority, mode in selected_weighted:
            self.episodic_store.mark_accessed(episode.id)
            selected_episodes.append(episode)
            replay_annotations.append({"episode_id": episode.id, "priority": priority, "mode": mode})
            replay_modes[mode] = replay_modes.get(mode, 0) + 1
            if verbose:
                logger.info(
                    f"  Replaying[{mode}]: {episode.content[:60]}... "
                    f"(priority: {priority:.3f})"
                )
        
        return {
            'replayed': len(selected_episodes),
            'selected_episodes': selected_episodes,
            'replay_annotations': replay_annotations,
            'replay_modes': replay_modes,
        }
    
    def _phase_2_consolidation(
        self,
        replayed_episodes: List[Episode],
        replay_annotations: List[Dict[str, Any]],
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Phase 2: Compress and consolidate replayed episodes using batch processing.
        
        Groups episodes into batches and compresses each batch together to:
        - Extract better cross-episode patterns
        - Identify shared concepts more effectively
        - Create richer consolidated memories
        
        Returns:
            Dictionary with consolidation results
        """
        if not replayed_episodes:
            return {'consolidated': 0}
        
        consolidated_count = 0
        conflicts_detected = 0
        mode_by_episode = {a.get("episode_id"): a.get("mode", "lossy") for a in replay_annotations}
        
        # Group episodes into batches for compression
        batches = self._create_episode_batches(replayed_episodes)
        
        for batch_idx, batch in enumerate(batches):
            if verbose:
                logger.info(f"  Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} episodes)")
            
            # Compress entire batch together to find relationships
            batch_compression = self.compressor.compress_batch_episodes(batch)
            batch_structured = self.compressor.extract_structured_record_batch(batch, batch_compression)
            
            if verbose:
                logger.info(f"    Batch summary: {batch_compression.summary[:80]}...")
                logger.info(f"    Shared concepts: {', '.join(batch_compression.key_concepts[:5])}")
            
            # Create individual consolidated memories from batch
            # Each episode in the batch gets its own consolidated memory,
            # but enriched with the batch-level concepts and themes
            for episode in batch:
                # Compress individual episode for its specific summary
                episode_compression = self.compressor.compress_single_episode(episode)
                structured = self.compressor.extract_structured_record(episode, episode_compression)
                
                # Merge batch-level concepts with episode-specific ones
                # Batch concepts are more reliable (cross-validated across multiple memories)
                merged_concepts = list(set(episode_compression.key_concepts + batch_compression.key_concepts))
                merged_themes = list(set(episode_compression.themes + batch_compression.themes))
                
                # Create consolidated memory
                consolidated = self.consolidated_store.add_memory(
                    summary=episode_compression.summary,
                    source_episode_ids=[episode.id],
                    key_concepts=merged_concepts,
                    importance=episode.importance,
                    tags=episode.tags + merged_themes,
                    core_fact=structured.core_fact,
                    supporting_context=structured.supporting_context,
                    confidence=structured.confidence,
                    time_span=structured.time_span,
                    persona_link=structured.persona_link,
                    evidence_link=structured.evidence_link,
                    evidence_strength=episode.evidence_strength,
                    contradiction_flags=structured.contradiction_flags,
                    schema_label=structured.schema_label,
                    memory_type=mode_by_episode.get(episode.id, structured.memory_type),
                    stability_score=max(0.0, min(1.0, 0.5 * episode.memory_consistency_score + 0.5 * batch_structured.confidence)),
                    memory_consistency_score=episode.memory_consistency_score,
                )

                if consolidated.contradiction_flags:
                    conflicts_detected += 1
                
                # Mark episode as consolidated
                self.episodic_store.mark_consolidated(episode.id)
                
                consolidated_count += 1
                
                if verbose:
                    logger.info(f"    ✓ Consolidated episode: {episode.content[:50]}...")
                    logger.info(f"      Concepts: {', '.join(merged_concepts[:5])}")
        
        self.total_consolidated += consolidated_count
        
        if verbose:
            logger.info(f"  Batch consolidation complete: {consolidated_count} memories created")
        
        return {'consolidated': consolidated_count, 'conflicts_detected': conflicts_detected}
    
    def _create_episode_batches(
        self,
        episodes: List[Episode],
        batch_size: Optional[int] = None
    ) -> List[List[Episode]]:
        """
        Create batches of episodes for group compression.
        
        Groups episodes by similarity/relatedness when possible,
        falls back to simple binning if similarity info unavailable.
        
        Args:
            episodes: List of episodes to batch
            batch_size: Size of each batch (defaults to consolidation_batch_size)
            
        Returns:
            List of episode batches
        """
        if batch_size is None:
            batch_size = self.consolidation_batch_size
        
        if len(episodes) <= batch_size:
            return [episodes]
        
        # Try to group by similarity using concepts
        batches = []
        current_batch = []
        
        for episode in episodes:
            current_batch.append(episode)
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining episodes
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _phase_3_schema_formation(self, verbose: bool) -> Dict[str, Any]:
        """
        Phase 3: Induce schemas from consolidated memories.
        
        Looks for patterns across multiple consolidated memories
        and creates abstract knowledge schemas.
        
        Returns:
            Dictionary with schema formation results
        """
        # Get all consolidated memories
        all_memories = self.consolidated_store.get_all_memories()
        
        if len(all_memories) < self.schema_min_memories:
            if verbose:
                logger.info(f"  Not enough memories for schema induction (need {self.schema_min_memories})")
            return {'schemas_formed': 0}
        
        if verbose:
            logger.info(f"  Analyzing {len(all_memories)} consolidated memories for patterns...")
            for mem in all_memories:
                logger.info(f"    Memory: {mem.summary[:60]}... | Concepts: {mem.key_concepts}")
        
        schemas_formed = 0
        
        # Better schema induction: group memories by shared concepts
        concept_groups = self._group_memories_by_concepts_improved(all_memories)
        
        if verbose:
            logger.info(f"  Found {len(concept_groups)} concept groups")
        
        for concepts, memories in concept_groups.items():
            if len(memories) < self.schema_min_memories:
                if verbose:
                    logger.info(f"  Skipping group '{', '.join(concepts[:2])}' - only {len(memories)} memories (need {self.schema_min_memories})")
                continue
            
            # Check if schema already exists for these concepts
            existing = self.schema_store.find_by_concepts(list(concepts), min_overlap=len(concepts))
            
            if existing:
                # Update existing schema
                schema = existing[0]
                new_memory_ids = [m.id for m in memories if m.id not in schema.related_memory_ids]
                if new_memory_ids:
                    confidence_boost = 0.05 + 0.1 * self.policy.schema_abstraction_strength
                    self.schema_store.update_schema(
                        schema.id,
                        new_memory_ids=new_memory_ids,
                        confidence_boost=confidence_boost
                    )

                    # If supporting memories contain conflicts, keep schema explicitly conflicted.
                    if any(m.contradiction_flags for m in memories):
                        schema.status = "conflicted"
                    if verbose:
                        logger.info(f"  ↑ Updated schema: {schema.name}")
            else:
                # Create new schema
                schema_name = f"Schema: {', '.join(list(concepts)[:3])}"
                schema_description = self._generate_schema_description(concepts, memories)
                has_conflict = any(m.contradiction_flags for m in memories)
                schema_type = "user_specific" if any(m.persona_link for m in memories) else "general_conversational"
                status = "conflicted" if has_conflict else "emergent"
                
                schema = self.schema_store.add_schema(
                    name=schema_name,
                    description=schema_description,
                    core_concepts=list(concepts),
                    related_memory_ids=[m.id for m in memories],
                    examples=[m.summary for m in memories[:3]],
                    confidence=min(1.0, 0.5 + 0.2 * self.policy.schema_abstraction_strength),
                    schema_type=schema_type,
                    status=status,
                    version=1,
                )
                
                schemas_formed += 1
                
                if verbose:
                    logger.info(f"  ✓ New schema: {schema_name}")
                    logger.info(f"    Based on {len(memories)} memories")
        
        self.total_schemas_formed += schemas_formed
        
        return {'schemas_formed': schemas_formed}
    
    def _phase_4_decay(self, verbose: bool) -> Dict[str, Any]:
        """
        Phase 4: Forget low-value, unconsolidated episodes.
        
        Simulates natural forgetting of unimportant memories.
        
        Returns:
            Dictionary with decay results
        """
        # Controlled forgetting policy:
        # - faster decay for stale/low-salience redundant episodes
        # - preserve evidence-linked memories for LOCOMO unless superseded
        threshold_days = max(7, int(30 * self.policy.decay_rate))
        forgotten_ids = []

        for ep_id, ep in list(self.episodic_store.episodes.items()):
            if ep.consolidated:
                continue

            age_days = (datetime.now() - ep.timestamp).days
            if age_days < threshold_days:
                continue

            if self.policy.dataset_name == "locomo" and ep.evidence_strength >= 0.4:
                continue

            redundancy = min(1.0, max(0.0, (ep.repetition_count - 1) / 4.0))
            forget_score = (
                0.35 * (1.0 - ep.salience_score)
                + 0.25 * (1.0 - ep.confidence)
                + 0.20 * redundancy
                + 0.20 * min(1.0, age_days / max(1, threshold_days * 2))
            )

            if forget_score >= 0.62:
                del self.episodic_store.episodes[ep_id]
                forgotten_ids.append(ep_id)
        
        if verbose and forgotten_ids:
            logger.info(f"  Forgotten {len(forgotten_ids)} old, low-importance episodes")
        
        return {'forgotten': len(forgotten_ids)}
    
    def _group_memories_by_concepts(
        self,
        memories: List[ConsolidatedMemory],
        min_shared: int = 2
    ) -> Dict[tuple, List[ConsolidatedMemory]]:
        """
        Group memories by shared concepts for schema induction.
        
        Args:
            memories: List of consolidated memories
            min_shared: Minimum number of shared concepts
            
        Returns:
            Dictionary mapping concept tuples to memory lists
        """
        from itertools import combinations
        
        groups = {}
        
        # For each pair of memories, find shared concepts
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                shared = set(mem1.key_concepts) & set(mem2.key_concepts)
                
                if len(shared) >= min_shared:
                    # Create a key from sorted shared concepts
                    key = tuple(sorted(shared)[:5])  # Limit to 5 concepts
                    
                    if key not in groups:
                        groups[key] = []
                    
                    if mem1 not in groups[key]:
                        groups[key].append(mem1)
                    if mem2 not in groups[key]:
                        groups[key].append(mem2)
        
        return groups
    
    def _group_memories_by_concepts_improved(
        self,
        memories: List[ConsolidatedMemory],
        min_shared: int = 1
    ) -> Dict[tuple, List[ConsolidatedMemory]]:
        """
        Improved schema grouping that is more lenient with concept matching.
        
        Groups memories by ANY shared concepts (not just pairwise overlaps).
        This helps create schemas from diverse but related memories.
        
        Args:
            memories: List of consolidated memories
            min_shared: Minimum number of shared concepts (default 1)
            
        Returns:
            Dictionary mapping concept tuples to memory lists
        """
        if not memories:
            return {}
        
        # Collect all unique concepts
        all_concepts = set()
        for mem in memories:
            all_concepts.update(mem.key_concepts)
        
        groups = {}
        used_memories = set()
        
        # For each concept, group all memories that mention it
        for concept in all_concepts:
            concept_memories = [m for m in memories if concept in m.key_concepts]
            
            if len(concept_memories) >= self.schema_min_memories:
                # Create a group for this concept
                key = (concept,)
                if key not in groups:
                    groups[key] = concept_memories
                    for m in concept_memories:
                        used_memories.add(m.id)
        
        # Also create groups from shared concept combinations
        from itertools import combinations
        
        for mem1 in memories:
            for mem2 in memories:
                if mem1.id >= mem2.id:  # Avoid duplicates
                    continue
                
                shared = set(mem1.key_concepts) & set(mem2.key_concepts)
                
                if len(shared) >= min_shared:
                    # Create a group from these shared concepts
                    key = tuple(sorted(shared)[:3])  # Limit to 3 concepts per key
                    
                    if key not in groups:
                        groups[key] = []
                    
                    # Add all memories that share these concepts
                    for mem in memories:
                        if all(c in mem.key_concepts for c in key) and mem not in groups[key]:
                            groups[key].append(mem)
        
        return groups
    
    def _generate_schema_description(
        self,
        concepts: tuple,
        memories: List[ConsolidatedMemory]
    ) -> str:
        """
        Generate a description for a schema based on its concepts and supporting memories.
        
        Args:
            concepts: Core concepts
            memories: Supporting memories
            
        Returns:
            Schema description
        """
        # Simple template-based description
        concept_str = ", ".join(concepts)
        memory_count = len(memories)
        
        description = (
            f"This schema represents knowledge about {concept_str}. "
            f"It is based on {memory_count} consolidated memories and captures "
            f"recurring patterns and relationships in this domain."
        )
        
        return description
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get overall consolidation statistics."""
        return {
            'total_cycles': self.cycle_count,
            'total_consolidated': self.total_consolidated,
            'total_schemas': self.total_schemas_formed,
            'episodic_stats': self.episodic_store.get_stats(),
            'consolidated_stats': self.consolidated_store.get_stats(),
            'schema_stats': self.schema_store.get_stats()
        }
