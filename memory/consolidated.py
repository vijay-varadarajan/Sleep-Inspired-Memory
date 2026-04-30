"""
Consolidated Memory Store

Stores compressed, integrated memories after sleep-based consolidation.
Inspired by cortical long-term memory in biological systems.

Design Principles:
- Consolidated memories are compressed summaries of episodic experiences
- Multiple episodes can be merged into a single consolidated memory
- Consolidated memories are more stable and less prone to decay
- They serve as the basis for semantic knowledge and schemas
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from uuid import uuid4
import json

# Simplified imports and removed unused code
@dataclass
class ConsolidatedMemory:
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    summary: str = ""
    source_episode_ids: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_access: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    core_fact: str = ""
    supporting_context: str = ""
    confidence: float = 0.5
    time_span: str = ""
    persona_link: str = ""
    evidence_link: str = ""
    evidence_strength: float = 0.0
    contradiction_flags: List[str] = field(default_factory=list)
    schema_label: str = ""
    memory_type: str = "mixed"
    stability_score: float = 0.5
    memory_consistency_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'summary': self.summary,
            'source_episode_ids': self.source_episode_ids,
            'key_concepts': self.key_concepts,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_access': self.last_access.isoformat() if self.last_access else None,
            'tags': self.tags,
            'core_fact': self.core_fact,
            'supporting_context': self.supporting_context,
            'confidence': self.confidence,
            'time_span': self.time_span,
            'persona_link': self.persona_link,
            'evidence_link': self.evidence_link,
            'evidence_strength': self.evidence_strength,
            'contradiction_flags': self.contradiction_flags,
            'schema_label': self.schema_label,
            'memory_type': self.memory_type,
            'stability_score': self.stability_score,
            'memory_consistency_score': self.memory_consistency_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsolidatedMemory':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_access'):
            data['last_access'] = datetime.fromisoformat(data['last_access'])
        return cls(**data)


class ConsolidatedMemoryStore:
    """
    Storage system for consolidated, compressed memories.
    
    Inspired by neocortical long-term memory, which stores
    integrated, abstracted knowledge rather than raw experiences.
    
    Key Operations:
    - add_memory: Store a new consolidated memory
    - get_memory: Retrieve by ID
    - search_by_concepts: Find memories containing specific concepts
    - get_relevant: Retrieve memories relevant to a query
    """
    
    def __init__(self):
        self.memories: Dict[str, ConsolidatedMemory] = {}
    
    def add_memory(
        self,
        summary: str,
        source_episode_ids: List[str],
        key_concepts: List[str],
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        core_fact: str = "",
        supporting_context: str = "",
        confidence: float = 0.5,
        time_span: str = "",
        persona_link: str = "",
        evidence_link: str = "",
        evidence_strength: float = 0.0,
        contradiction_flags: Optional[List[str]] = None,
        schema_label: str = "",
        memory_type: str = "mixed",
        stability_score: float = 0.5,
        memory_consistency_score: float = 0.5,
    ) -> ConsolidatedMemory:
        """
        Add a new consolidated memory.
        
        Args:
            summary: Compressed summary of the source episodes
            source_episode_ids: IDs of episodes that were consolidated
            key_concepts: Key concepts/entities extracted
            importance: Importance score (0-1)
            tags: Optional categorization tags
            
        Returns:
            The created ConsolidatedMemory object
        """
        memory = ConsolidatedMemory(
            summary=summary,
            source_episode_ids=source_episode_ids,
            key_concepts=key_concepts,
            importance=importance,
            tags=tags or [],
            core_fact=core_fact,
            supporting_context=supporting_context,
            confidence=confidence,
            time_span=time_span,
            persona_link=persona_link,
            evidence_link=evidence_link,
            evidence_strength=evidence_strength,
            contradiction_flags=contradiction_flags or [],
            schema_label=schema_label,
            memory_type=memory_type,
            stability_score=stability_score,
            memory_consistency_score=memory_consistency_score,
        )

        # Simple conflict detection over core facts for stability tracking.
        new_fact = (memory.core_fact or memory.summary).strip().lower()
        if new_fact:
            for existing in self.memories.values():
                old_fact = (existing.core_fact or existing.summary).strip().lower()
                if not old_fact or old_fact == new_fact:
                    continue
                overlap = len(set(new_fact.split()) & set(old_fact.split()))
                denom = max(1, min(len(new_fact.split()), len(old_fact.split())))
                if overlap / denom > 0.6 and new_fact != old_fact:
                    flag = f"possible_conflict_with:{existing.id}"
                    memory.contradiction_flags.append(flag)
                    existing.contradiction_flags.append(f"possible_conflict_with:{memory.id}")
                    existing.memory_consistency_score = max(0.0, existing.memory_consistency_score - 0.1)

        if memory.contradiction_flags:
            memory.memory_consistency_score = max(0.0, memory.memory_consistency_score - 0.15)
            memory.stability_score = max(0.0, memory.stability_score - 0.1)

        self.memories[memory.id] = memory
        return memory

    def _lexical_overlap(self, query: str, text: str) -> float:
        q = [t for t in (query or "").lower().split() if t]
        if not q:
            return 0.0
        t = set((text or "").lower().split())
        hits = sum(1 for token in q if token in t)
        return min(1.0, hits / len(q))

    def _entity_overlap(self, query: str, text: str) -> float:
        # Lightweight entity proxy: title-cased tokens in query.
        entities = [tok for tok in (query or "").split() if tok[:1].isupper() and len(tok) > 2]
        if not entities:
            return 0.0
        tl = (text or "")
        hits = sum(1 for e in entities if e in tl)
        return min(1.0, hits / len(entities))

    def search_hybrid(
        self,
        query: str,
        persona: str = "",
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        evidence_required: bool = False,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval with interpretable scoring bundles."""
        if not self.memories:
            return []

        w = {
            "semantic": 0.25,
            "lexical": 0.20,
            "recency": 0.10,
            "salience": 0.15,
            "persona": 0.15,
            "evidence": 0.10,
            "contradiction_penalty": 0.25,
        }
        if weights:
            w.update(weights)

        query_l = (query or "").lower()
        persona_l = (persona or "").lower()
        ranked: List[Tuple[ConsolidatedMemory, float, Dict[str, float]]] = []

        for mem in self.memories.values():
            text = f"{mem.summary} {mem.core_fact} {mem.supporting_context}".strip()
            lexical = self._lexical_overlap(query_l, text)
            entity = self._entity_overlap(query, text)
            semantic = min(1.0, 0.6 * lexical + 0.4 * entity)
            persona_match = self._lexical_overlap(persona_l, f"{mem.persona_link} {text}") if persona_l else 0.0

            # Recency proxy from access history and timestamp order.
            recency = 0.2 + 0.8 * (1.0 / (1.0 + mem.access_count))
            salience = 0.5 * mem.importance + 0.5 * mem.confidence
            evidence = mem.evidence_strength if mem.evidence_link else 0.0
            contradiction_penalty = 1.0 if mem.contradiction_flags else 0.0

            final_score = (
                w["semantic"] * semantic
                + w["lexical"] * lexical
                + w["recency"] * recency
                + w["salience"] * salience
                + w["persona"] * persona_match
                + w["evidence"] * evidence
                - w["contradiction_penalty"] * contradiction_penalty
            )

            if evidence_required and not mem.evidence_link:
                final_score *= 0.6

            ranked.append(
                (
                    mem,
                    final_score,
                    {
                        "semantic": semantic,
                        "lexical": lexical,
                        "recency": recency,
                        "salience": salience,
                        "persona": persona_match,
                        "evidence": evidence,
                        "contradiction_penalty": contradiction_penalty,
                    },
                )
            )

        ranked.sort(key=lambda x: x[1], reverse=True)

        bundles: List[Dict[str, Any]] = []
        for mem, score, parts in ranked[:top_k]:
            rationale = ", ".join(f"{k}={v:.2f}" for k, v in parts.items())
            bundles.append(
                {
                    "memory_id": mem.id,
                    "text": mem.summary,
                    "core_fact": mem.core_fact,
                    "source_type": "consolidated",
                    "confidence": mem.confidence,
                    "score": score,
                    "why_retrieved": rationale,
                    "is_evidence_grounded": bool(mem.evidence_link),
                    "is_contradictory": bool(mem.contradiction_flags),
                    "supporting": not bool(mem.contradiction_flags),
                    "parts": parts,
                }
            )

        return bundles
    
    def get_memory(self, memory_id: str) -> Optional[ConsolidatedMemory]:
        """Retrieve a specific memory by ID."""
        return self.memories.get(memory_id)
    
    def get_all_memories(self) -> List[ConsolidatedMemory]:
        """Get all memories sorted by timestamp (newest first)."""
        return sorted(
            self.memories.values(),
            key=lambda m: m.timestamp,
            reverse=True
        )
    
    def search_by_concepts(self, concepts: List[str]) -> List[ConsolidatedMemory]:
        """
        Find memories that contain any of the given concepts.
        
        Args:
            concepts: List of concepts to search for
            
        Returns:
            List of memories containing those concepts, sorted by relevance
        """
        matching_memories = []
        
        for memory in self.memories.values():
            # Count how many query concepts appear in this memory
            overlap = len(set(concepts) & set(memory.key_concepts))
            if overlap > 0:
                matching_memories.append((memory, overlap))
        
        # Sort by overlap count (most relevant first)
        matching_memories.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in matching_memories]
    
    def search_by_text(self, query: str) -> List[ConsolidatedMemory]:
        """
        Simple text-based search in summaries.
        
        Args:
            query: Search query string
            
        Returns:
            Memories whose summaries contain the query text
        """
        query_lower = query.lower()
        return [
            m for m in self.memories.values()
            if query_lower in m.summary.lower()
        ]
    
    def get_by_importance(self, threshold: float = 0.5, limit: int = 10) -> List[ConsolidatedMemory]:
        """
        Get most important consolidated memories.
        
        Args:
            threshold: Minimum importance score
            limit: Maximum number of memories to return
            
        Returns:
            List of important memories, sorted by importance
        """
        important = [
            m for m in self.memories.values()
            if m.importance >= threshold
        ]
        important.sort(key=lambda m: m.importance, reverse=True)
        return important[:limit]
    
    def mark_accessed(self, memory_id: str) -> None:
        """
        Mark a memory as accessed.
        Updates access count and timestamp.
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access_count += 1
            memory.last_access = datetime.now()
    
    def save_to_file(self, filepath: str) -> None:
        """Serialize consolidated memory to JSON file."""
        data = {
            'memories': [m.to_dict() for m in self.memories.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load consolidated memory from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.memories = {}
        for memory_data in data.get('memories', []):
            memory = ConsolidatedMemory.from_dict(memory_data)
            self.memories[memory.id] = memory
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        self.memories = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the consolidated memory store."""
        memories_list = list(self.memories.values())
        return {
            'total_memories': len(memories_list),
            'avg_importance': sum(m.importance for m in memories_list) / len(memories_list) if memories_list else 0,
            'avg_confidence': sum(m.confidence for m in memories_list) / len(memories_list) if memories_list else 0,
            'avg_stability': sum(m.stability_score for m in memories_list) / len(memories_list) if memories_list else 0,
            'conflicted_memories': sum(1 for m in memories_list if m.contradiction_flags),
            'total_accesses': sum(m.access_count for m in memories_list),
            'unique_concepts': len(set(
                concept
                for m in memories_list
                for concept in m.key_concepts
            ))
        }
