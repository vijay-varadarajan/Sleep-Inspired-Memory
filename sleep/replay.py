"""
Prioritized Replay

Selects episodes for consolidation based on biological principles:
- Importance: Emotionally salient or high-stakes experiences
- Novelty: New, unexpected information
- Recency: Recent experiences (with temporal decay)
- Access patterns: Frequently accessed memories

Inspired by:
- Replay of place cells in hippocampus during sleep
- Preferential consolidation of important/novel memories
- Synaptic homeostasis hypothesis
"""

from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import math

from memory.episodic import Episode
from memory.config import DatasetMemoryConfig

# Simplified imports and removed unused code


def calculate_replay_priority(
    episode: Episode,
    current_time: datetime,
    recency_weight: float = 0.3,
    importance_weight: float = 0.4,
    novelty_weight: float = 0.3,
    access_bonus: float = 0.1
) -> float:
    """
    Calculate priority score for replaying/consolidating an episode.
    
    The priority function combines multiple factors that are known to
    influence memory consolidation in biological systems:
    
    1. Recency: Recently encoded memories are preferentially replayed
       (decays exponentially with time)
    2. Importance: Emotionally salient or goal-relevant memories
    3. Novelty: New, unexpected information
    4. Access patterns: Memories that have been retrieved are strengthened
    
    Args:
        episode: The episode to score
        current_time: Current timestamp for recency calculation
        recency_weight: Weight for recency component (0-1)
        importance_weight: Weight for importance component (0-1)
        novelty_weight: Weight for novelty component (0-1)
        access_bonus: Bonus multiplier for accessed episodes
        
    Returns:
        Priority score (higher = more likely to be replayed)
    """
    # Recency score: exponential decay (half-life of 7 days)
    time_delta = current_time - episode.timestamp
    hours_elapsed = time_delta.total_seconds() / 3600
    half_life_hours = 7 * 24  # 7 days
    recency_score = math.exp(-hours_elapsed * math.log(2) / half_life_hours)
    
    # Importance score: directly from metadata
    importance_score = episode.importance
    
    # Novelty score: directly from metadata
    novelty_score = episode.novelty
    
    # Access bonus: frequently accessed memories get a boost
    # (but not too much to avoid rich-get-richer dynamics)
    access_multiplier = 1.0 + (access_bonus * min(episode.access_count, 5))
    
    # Weighted combination
    base_priority = (
        recency_weight * recency_score +
        importance_weight * importance_score +
        novelty_weight * novelty_score
    )
    
    # Apply access bonus
    final_priority = base_priority * access_multiplier
    
    return final_priority


def calculate_weighted_replay_priority(
    episode: Episode,
    current_time: datetime,
    config: DatasetMemoryConfig,
) -> float:
    """Dataset-aware replay priority using biologically-inspired encoding signals."""
    w = dict(config.salience_weights or {})

    # Defaults for backwards compatibility.
    salience_w = w.get("salience", 0.2)
    novelty_w = w.get("novelty", 0.2)
    uncertainty_w = w.get("uncertainty", 0.1)
    repetition_w = w.get("repetition", 0.2)
    recency_w = w.get("recency", 0.2)
    persona_w = w.get("persona_relevance", 0.1)
    evidence_w = w.get("evidence_strength", 0.0)

    # Fresh recency estimate with decay.
    time_delta = current_time - episode.timestamp
    hours_elapsed = max(0.0, time_delta.total_seconds() / 3600.0)
    half_life_hours = 72.0 if config.dataset_name == "locomo" else 96.0
    recency = math.exp(-hours_elapsed * math.log(2) / half_life_hours)

    repetition = min(1.0, episode.repetition_count / 5.0)
    uncertainty = min(1.0, max(0.0, episode.uncertainty))

    score = (
        salience_w * episode.salience_score
        + novelty_w * episode.novelty_score
        + uncertainty_w * uncertainty
        + repetition_w * repetition
        + recency_w * recency
        + persona_w * episode.persona_relevance
        + evidence_w * episode.evidence_strength
    )

    # Penalize risky low-confidence memories except when evidence-grounded.
    risk_penalty = episode.factuality_risk * (0.25 if episode.evidence_strength > 0.5 else 0.45)
    score -= risk_penalty

    return max(0.0, score)


def adaptive_replay_top_k(episodes: List[Episode], config: DatasetMemoryConfig) -> int:
    """Adaptive replay size based on pool size and uncertainty."""
    if not episodes:
        return config.replay_top_k_min

    avg_uncertainty = sum(ep.uncertainty for ep in episodes) / len(episodes)
    avg_novelty = sum(ep.novelty_score for ep in episodes) / len(episodes)
    uncertainty_factor = 1.0 + 0.5 * avg_uncertainty + 0.3 * avg_novelty

    base = int(round(config.replay_top_k * uncertainty_factor))
    return max(config.replay_top_k_min, min(config.replay_top_k_max, min(base, len(episodes))))


def select_replay_mode(episode: Episode, config: DatasetMemoryConfig) -> str:
    """Select replay mode: verbatim / lossy / contrastive / schema / evidence."""
    if config.dataset_name == "locomo" and episode.evidence_strength >= 0.45:
        return "evidence"
    if episode.contradiction_flags:
        return "contrastive"
    if episode.episode_type in {"fact", "instruction"} and episode.confidence >= config.factuality_threshold:
        return "verbatim"
    if episode.episode_type in {"preference", "event"} and episode.salience_score >= 0.6:
        return "schema"
    return "lossy"


def select_episodes_for_replay_weighted(
    episodes: List[Episode],
    config: DatasetMemoryConfig,
    current_time: datetime | None = None,
    exclude_consolidated: bool = True,
) -> List[Tuple[Episode, float, str]]:
    """Dataset-aware replay selection returning (episode, priority, replay_mode)."""
    if current_time is None:
        current_time = datetime.now()

    pool = [e for e in episodes if not e.consolidated] if exclude_consolidated else list(episodes)
    if not pool:
        return []

    k = adaptive_replay_top_k(pool, config)
    scored = []
    for ep in pool:
        if config.ablations.disable_replay_selection:
            priority = 1.0
        else:
            priority = calculate_weighted_replay_priority(ep, current_time, config)
        scored.append((ep, priority, select_replay_mode(ep, config)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def select_episodes_for_replay(
    episodes: List[Episode],
    n_replay: int = 10,
    current_time: datetime = None,
    recency_weight: float = 0.3,
    importance_weight: float = 0.4,
    novelty_weight: float = 0.3,
    access_bonus: float = 0.1,
    exclude_consolidated: bool = True
) -> List[Tuple[Episode, float]]:
    """
    Select top-N episodes for replay during consolidation.
    
    Uses a priority-based selection mechanism inspired by biological
    memory consolidation during sleep.
    
    Args:
        episodes: List of candidate episodes
        n_replay: Number of episodes to select
        current_time: Current time (defaults to now)
        recency_weight: Weight for recency in priority calculation
        importance_weight: Weight for importance
        novelty_weight: Weight for novelty
        access_bonus: Bonus for accessed memories
        exclude_consolidated: Whether to exclude already consolidated episodes
        
    Returns:
        List of (episode, priority_score) tuples, sorted by priority (descending)
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Filter out consolidated episodes if requested
    if exclude_consolidated:
        episodes = [e for e in episodes if not e.consolidated]
    
    # Calculate priority for each episode
    priorities = []
    for episode in episodes:
        priority = calculate_replay_priority(
            episode=episode,
            current_time=current_time,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            novelty_weight=novelty_weight,
            access_bonus=access_bonus
        )
        priorities.append((episode, priority))
    
    # Sort by priority (highest first)
    priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    return priorities[:n_replay]


def calculate_batch_diversity(episodes: List[Episode]) -> float:
    """
    Calculate diversity of a batch of episodes based on tags and concepts.
    
    Higher diversity is generally desirable for consolidation as it
    promotes integration across different types of experiences.
    
    Args:
        episodes: Batch of episodes
        
    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    if not episodes:
        return 0.0
    
    # Collect all unique tags across episodes
    all_tags = set()
    for episode in episodes:
        all_tags.update(episode.tags)
    
    # Diversity is roughly the average number of unique tags per episode
    # normalized by the total unique tags
    if len(all_tags) == 0:
        return 0.5  # Neutral diversity if no tags
    
    avg_tags_per_episode = sum(len(e.tags) for e in episodes) / len(episodes)
    diversity = min(avg_tags_per_episode / len(all_tags), 1.0)
    
    return diversity


def select_diverse_batch(
    episodes: List[Episode],
    batch_size: int = 5,
    current_time: datetime = None
) -> List[Episode]:
    """
    Select a diverse batch of episodes for consolidation.
    
    Uses a greedy algorithm to maximize both priority and diversity:
    1. Select the highest priority episode
    2. Iteratively add episodes that are high priority AND increase diversity
    
    Args:
        episodes: Candidate episodes
        batch_size: Number of episodes to select
        current_time: Current time for priority calculation
        
    Returns:
        List of selected episodes
    """
    if not episodes:
        return []
    
    if current_time is None:
        current_time = datetime.now()
    
    # Calculate priorities for all episodes
    episode_priorities = [
        (ep, calculate_replay_priority(ep, current_time))
        for ep in episodes
    ]
    episode_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Start with the highest priority episode
    selected = [episode_priorities[0][0]]
    remaining = [ep for ep, _ in episode_priorities[1:]]
    
    # Greedily add episodes that maximize priority + diversity
    while len(selected) < batch_size and remaining:
        best_score = -1
        best_episode = None
        
        for episode in remaining:
            # Calculate priority
            priority = calculate_replay_priority(episode, current_time)
            
            # Calculate diversity if we add this episode
            test_batch = selected + [episode]
            diversity = calculate_batch_diversity(test_batch)
            
            # Combined score (equal weighting)
            score = 0.5 * priority + 0.5 * diversity
            
            if score > best_score:
                best_score = score
                best_episode = episode
        
        if best_episode:
            selected.append(best_episode)
            remaining.remove(best_episode)
        else:
            break
    
    return selected
