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
from sleep.replay import select_episodes_for_replay, select_diverse_batch
from sleep.compression import MemoryCompressor, CompressionResult


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SleepCycle:
    """
    Manages the sleep-based memory consolidation process.
    
    A sleep cycle involves:
    1. Prioritized replay of important/novel episodes
    2. Generative compression using LLM
    3. Creation of consolidated memories
    4. Schema induction from patterns
    5. Forgetting/decay of low-value memories
    
    This is the core of the biologically-inspired consolidation system.
    """
    
    def __init__(
        self,
        episodic_store: EpisodicMemoryStore,
        consolidated_store: ConsolidatedMemoryStore,
        schema_store: SchemaStore,
        compressor: MemoryCompressor,
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
            'episodes_forgotten': 0
        }
        
        # Phase 1: Prioritized Replay
        if verbose:
            logger.info("\n[Phase 1] Prioritized Replay")
        
        replay_results = self._phase_1_replay(current_time, verbose)
        stats['episodes_replayed'] = replay_results['replayed']
        
        # Phase 2: Generative Compression & Consolidation
        if verbose:
            logger.info("\n[Phase 2] Generative Compression & Consolidation")
        
        consolidation_results = self._phase_2_consolidation(
            replay_results['selected_episodes'],
            verbose
        )
        stats['memories_consolidated'] = consolidation_results['consolidated']
        
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
        
        # Select episodes for replay using priority-based selection
        selected = select_episodes_for_replay(
            episodes=unconsolidated,
            n_replay=self.replay_batch_size,
            current_time=current_time
        )
        
        # Mark episodes as accessed (simulates replay)
        selected_episodes = []
        for episode, priority in selected:
            self.episodic_store.mark_accessed(episode.id)
            selected_episodes.append(episode)
            if verbose:
                logger.info(f"  Replaying: {episode.content[:60]}... (priority: {priority:.3f})")
        
        return {
            'replayed': len(selected_episodes),
            'selected_episodes': selected_episodes
        }
    
    def _phase_2_consolidation(
        self,
        replayed_episodes: List[Episode],
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Phase 2: Compress and consolidate replayed episodes.
        
        Returns:
            Dictionary with consolidation results
        """
        if not replayed_episodes:
            return {'consolidated': 0}
        
        consolidated_count = 0
        
        # Option 1: Consolidate each episode individually
        # (Simple approach for initial implementation)
        for episode in replayed_episodes:
            if verbose:
                logger.info(f"  Compressing: {episode.content[:60]}...")
            
            # Use LLM to compress
            compression_result = self.compressor.compress_single_episode(episode)
            
            # Create consolidated memory
            consolidated = self.consolidated_store.add_memory(
                summary=compression_result.summary,
                source_episode_ids=[episode.id],
                key_concepts=compression_result.key_concepts,
                importance=episode.importance,
                tags=episode.tags + compression_result.themes
            )
            
            # Mark episode as consolidated
            self.episodic_store.mark_consolidated(episode.id)
            
            consolidated_count += 1
            
            if verbose:
                logger.info(f"  ✓ Consolidated: {consolidated.summary[:60]}...")
                logger.info(f"    Concepts: {', '.join(compression_result.key_concepts[:5])}")
        
        self.total_consolidated += consolidated_count
        
        return {'consolidated': consolidated_count}
    
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
        
        schemas_formed = 0
        
        # Simple schema induction: group memories by shared concepts
        concept_groups = self._group_memories_by_concepts(all_memories)
        
        for concepts, memories in concept_groups.items():
            if len(memories) < self.schema_min_memories:
                continue
            
            # Check if schema already exists for these concepts
            existing = self.schema_store.find_by_concepts(list(concepts), min_overlap=len(concepts))
            
            if existing:
                # Update existing schema
                schema = existing[0]
                new_memory_ids = [m.id for m in memories if m.id not in schema.related_memory_ids]
                if new_memory_ids:
                    self.schema_store.update_schema(
                        schema.id,
                        new_memory_ids=new_memory_ids,
                        confidence_boost=0.1
                    )
                    if verbose:
                        logger.info(f"  ↑ Updated schema: {schema.name}")
            else:
                # Create new schema
                schema_name = f"Schema: {', '.join(list(concepts)[:3])}"
                schema_description = self._generate_schema_description(concepts, memories)
                
                schema = self.schema_store.add_schema(
                    name=schema_name,
                    description=schema_description,
                    core_concepts=list(concepts),
                    related_memory_ids=[m.id for m in memories],
                    examples=[m.summary for m in memories[:3]],
                    confidence=0.6
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
        # Decay episodes older than 30 days that haven't been consolidated
        forgotten_ids = self.episodic_store.decay_episodes(decay_threshold_days=30)
        
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
