"""
Episodic Memory Store

Maintains raw, uncompressed interaction episodes with metadata.
Inspired by hippocampal episodic memory in biological systems.

Design Principles:
- Each episode is stored as-is (no compression at encoding)
- Metadata tracks importance, novelty, recency for replay prioritization
- Access patterns influence consolidation priority
- Episodes can decay over time if not consolidated
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from uuid import uuid4
import json


@dataclass
class Episode:
    """
    A single episodic memory representing a raw interaction or experience.
    
    Attributes:
        id: Unique identifier
        timestamp: When the episode was created
        content: Raw text content of the episode
        context: Optional contextual information (user query, action taken, etc.)
        importance: Subjective importance score (0-1), can be manually set or computed
        novelty: Estimated novelty score (0-1), based on semantic similarity to existing episodes
        access_count: Number of times this episode has been accessed/replayed
        last_access: Timestamp of last access
        tags: Optional tags for categorization
        consolidated: Whether this episode has been consolidated into long-term memory
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # Default medium importance
    novelty: float = 0.5  # Default medium novelty
    access_count: int = 0
    last_access: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    consolidated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'context': self.context,
            'importance': self.importance,
            'novelty': self.novelty,
            'access_count': self.access_count,
            'last_access': self.last_access.isoformat() if self.last_access else None,
            'tags': self.tags,
            'consolidated': self.consolidated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Create episode from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_access'):
            data['last_access'] = datetime.fromisoformat(data['last_access'])
        return cls(**data)


class EpisodicMemoryStore:
    """
    Storage and retrieval system for episodic memories.
    
    Inspired by the hippocampus, which rapidly encodes new experiences
    without immediately integrating them into semantic memory.
    
    Key Operations:
    - add_episode: Store a new raw episode
    - get_episode: Retrieve episode by ID
    - get_recent: Get most recent episodes
    - get_unconsolidated: Get episodes not yet consolidated
    - mark_accessed: Update access metadata (used during replay)
    """
    
    def __init__(self):
        self.episodes: Dict[str, Episode] = {}
    
    def add_episode(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        novelty: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> Episode:
        """
        Add a new episodic memory.
        
        Args:
            content: Raw text content of the episode
            context: Optional contextual information
            importance: Importance score (0-1)
            novelty: Novelty score (0-1)
            tags: Optional categorization tags
            
        Returns:
            The created Episode object
        """
        episode = Episode(
            content=content,
            context=context or {},
            importance=importance,
            novelty=novelty,
            tags=tags or []
        )
        self.episodes[episode.id] = episode
        return episode
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve a specific episode by ID."""
        return self.episodes.get(episode_id)
    
    def get_all_episodes(self) -> List[Episode]:
        """Get all episodes sorted by timestamp (newest first)."""
        return sorted(
            self.episodes.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )

    def get_count(self) -> int:
        """Get the total number of stored episodes."""
        return len(self.episodes)
    
    def get_recent(self, n: int = 10) -> List[Episode]:
        """Get the n most recent episodes."""
        return self.get_all_episodes()[:n]
    
    def get_unconsolidated(self) -> List[Episode]:
        """
        Get all episodes that haven't been consolidated yet.
        These are candidates for the next sleep cycle.
        """
        return [e for e in self.episodes.values() if not e.consolidated]
    
    def mark_accessed(self, episode_id: str) -> None:
        """
        Mark an episode as accessed (used during replay).
        Updates access count and timestamp.
        """
        if episode_id in self.episodes:
            episode = self.episodes[episode_id]
            episode.access_count += 1
            episode.last_access = datetime.now()
    
    def mark_consolidated(self, episode_id: str) -> None:
        """Mark an episode as consolidated into long-term memory."""
        if episode_id in self.episodes:
            self.episodes[episode_id].consolidated = True
    
    def decay_episodes(self, decay_threshold_days: int = 30) -> List[str]:
        """
        Remove old, unconsolidated episodes that haven't been accessed.
        Simulates forgetting of unimportant memories.
        
        Args:
            decay_threshold_days: Number of days after which unconsolidated,
                                  unaccessed episodes are forgotten
        
        Returns:
            List of forgotten episode IDs
        """
        now = datetime.now()
        forgotten_ids = []
        
        for episode_id, episode in list(self.episodes.items()):
            # Skip consolidated episodes (they're protected)
            if episode.consolidated:
                continue
            
            # Calculate age in days
            age_days = (now - episode.timestamp).days
            
            # Forget if old enough and low importance/access
            if age_days > decay_threshold_days and episode.access_count == 0:
                # Low importance episodes are more likely to be forgotten
                if episode.importance < 0.3:
                    del self.episodes[episode_id]
                    forgotten_ids.append(episode_id)
        
        return forgotten_ids
    
    def save_to_file(self, filepath: str) -> None:
        """Serialize episodic memory to JSON file."""
        data = {
            'episodes': [e.to_dict() for e in self.episodes.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load episodic memory from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.episodes = {}
        for episode_data in data.get('episodes', []):
            episode = Episode.from_dict(episode_data)
            self.episodes[episode.id] = episode
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the episodic memory store."""
        episodes_list = list(self.episodes.values())
        return {
            'total_episodes': len(episodes_list),
            'consolidated': sum(1 for e in episodes_list if e.consolidated),
            'unconsolidated': sum(1 for e in episodes_list if not e.consolidated),
            'avg_importance': sum(e.importance for e in episodes_list) / len(episodes_list) if episodes_list else 0,
            'avg_novelty': sum(e.novelty for e in episodes_list) / len(episodes_list) if episodes_list else 0,
            'total_accesses': sum(e.access_count for e in episodes_list)
        }
