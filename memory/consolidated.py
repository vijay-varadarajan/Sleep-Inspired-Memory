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
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from uuid import uuid4
import json


@dataclass
class ConsolidatedMemory:
    """
    A consolidated memory representing compressed/summarized knowledge.
    
    Created during sleep consolidation from one or more episodic memories.
    
    Attributes:
        id: Unique identifier
        timestamp: When this memory was consolidated
        summary: Compressed, LLM-generated summary of source episodes
        source_episode_ids: IDs of episodes that contributed to this memory
        key_concepts: Extracted concepts/entities from the episodes
        importance: Aggregate importance score
        access_count: Number of times accessed during retrieval
        last_access: Timestamp of last access
        tags: Categorization tags
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    summary: str = ""
    source_episode_ids: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_access: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    
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
            'tags': self.tags
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
        tags: Optional[List[str]] = None
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
            tags=tags or []
        )
        self.memories[memory.id] = memory
        return memory
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the consolidated memory store."""
        memories_list = list(self.memories.values())
        return {
            'total_memories': len(memories_list),
            'avg_importance': sum(m.importance for m in memories_list) / len(memories_list) if memories_list else 0,
            'total_accesses': sum(m.access_count for m in memories_list),
            'unique_concepts': len(set(
                concept
                for m in memories_list
                for concept in m.key_concepts
            ))
        }
