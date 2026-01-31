"""
LLM-Based Memory Compression

Uses Gemini to perform generative compression of episodic memories:
1. Summarize multiple episodes into concise descriptions
2. Extract key concepts and entities
3. Identify relationships and patterns
4. Generate abstract representations

Inspired by:
- Systems consolidation in neocortex
- Semantic abstraction during sleep
- Generative compression in AI (e.g., Deepmind's work)
"""

from typing import List, Dict, Any, Optional
import os
import json
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from memory.episodic import Episode


@dataclass
class CompressionResult:
    """
    Result of compressing one or more episodes.
    
    Attributes:
        summary: Compressed textual summary
        key_concepts: Extracted key concepts/entities
        themes: High-level themes identified
        relationships: Identified relationships between concepts
        confidence: Confidence in the compression (0-1)
    """
    summary: str
    key_concepts: List[str]
    themes: List[str]
    relationships: List[str]
    confidence: float = 0.7


class MemoryCompressor:
    """
    LLM-based memory compression using Gemini.
    
    Performs generative compression to transform raw episodic memories
    into consolidated, semantic representations.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest",
        temperature: float = 0.3
    ):
        """
        Initialize the memory compressor.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature
        )

    @staticmethod
    def _ensure_text(value: Any) -> str:
        """Ensure a value is converted to a safe string for prompting/parsing."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    
    def compress_single_episode(self, episode: Episode) -> CompressionResult:
        """
        Compress a single episode into a consolidated memory.
        
        Args:
            episode: Episode to compress
            
        Returns:
            CompressionResult with summary and extracted information
        """
        # Create compression prompt
        prompt = f"""You are a memory consolidation system. Your task is to compress and abstract the following memory episode.

Episode Content:
{episode.content}

Context:
{episode.context}

Tags: {', '.join(episode.tags)}

Please provide:
1. A concise summary (2-3 sentences) capturing the essential information
2. Key concepts/entities (comma-separated list)
3. High-level themes (comma-separated list)
4. Important relationships between concepts (if any)

Format your response as:
SUMMARY: [your summary]
CONCEPTS: [concept1, concept2, ...]
THEMES: [theme1, theme2, ...]
RELATIONSHIPS: [relationship1; relationship2; ...]
"""
        
        try:
            # Call Gemini
            messages = [
                SystemMessage(content="You are an expert at memory consolidation and information compression."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            
            # Parse response
            result = self._parse_compression_response(self._ensure_text(response.content))
            return result
        
        except Exception as e:
            # Fallback: basic extraction if LLM fails
            print(f"Warning: LLM compression failed: {e}")
            return CompressionResult(
                summary=episode.content[:200] + "..." if len(episode.content) > 200 else episode.content,
                key_concepts=episode.tags,
                themes=[],
                relationships=[],
                confidence=0.3
            )
    
    def compress_episode_batch(
        self,
        episodes: List[Episode],
        find_commonalities: bool = True
    ) -> CompressionResult:
        """
        Compress multiple related episodes into a single consolidated memory.
        
        This is particularly useful when episodes share themes or concepts,
        enabling integration and abstraction.
        
        Args:
            episodes: List of episodes to compress together
            find_commonalities: Whether to explicitly look for common patterns
            
        Returns:
            CompressionResult representing the integrated memory
        """
        if not episodes:
            return CompressionResult(
                summary="",
                key_concepts=[],
                themes=[],
                relationships=[],
                confidence=0.0
            )
        
        if len(episodes) == 1:
            return self.compress_single_episode(episodes[0])
        
        # Construct batch prompt
        episodes_text = ""
        for i, ep in enumerate(episodes, 1):
            episodes_text += f"\n--- Episode {i} ---\n"
            episodes_text += f"Content: {ep.content}\n"
            episodes_text += f"Tags: {', '.join(ep.tags)}\n"
        
        commonality_instruction = ""
        if find_commonalities:
            commonality_instruction = """
Pay special attention to:
- Common themes across episodes
- Recurring concepts or entities
- Patterns or relationships that span multiple episodes
"""
        
        prompt = f"""You are a memory consolidation system. Your task is to compress and integrate the following related memory episodes into a single consolidated memory.

{episodes_text}

{commonality_instruction}

Please provide:
1. An integrated summary (3-5 sentences) capturing the essential information from all episodes
2. Key concepts/entities that appear across episodes (comma-separated list)
3. High-level themes connecting the episodes (comma-separated list)
4. Important relationships or patterns (if any)

Format your response as:
SUMMARY: [your integrated summary]
CONCEPTS: [concept1, concept2, ...]
THEMES: [theme1, theme2, ...]
RELATIONSHIPS: [relationship1; relationship2; ...]
"""
        
        try:
            messages = [
                SystemMessage(content="You are an expert at memory consolidation, integration, and pattern recognition."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            
            result = self._parse_compression_response(self._ensure_text(response.content))
            # Higher confidence for batch compression (more evidence)
            result.confidence = min(0.8, result.confidence + 0.2)
            return result
        
        except Exception as e:
            print(f"Warning: LLM batch compression failed: {e}")
            # Fallback: concatenate summaries
            combined_content = " ".join(ep.content[:100] for ep in episodes)
            all_tags = list(set(tag for ep in episodes for tag in ep.tags))
            return CompressionResult(
                summary=combined_content[:300] + "...",
                key_concepts=all_tags,
                themes=[],
                relationships=[],
                confidence=0.3
            )
    
    def extract_concepts_from_text(self, text: str) -> List[str]:
        """
        Extract key concepts from raw text.
        
        Useful for estimating novelty or finding related memories.
        
        Args:
            text: Raw text to analyze
            
        Returns:
            List of key concepts
        """
        text_value = self._ensure_text(text)
        prompt = f"""Extract the key concepts, entities, and important terms from the following text.
Return only a comma-separated list.

Text:
    {text_value}

Concepts:"""
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            # Parse comma-separated list
            response_text = self._ensure_text(response.content)
            concepts = [c.strip() for c in response_text.split(',')]
            return [c for c in concepts if c]  # Remove empty strings
        
        except Exception as e:
            print(f"Warning: Concept extraction failed: {e}")
            return []
    
    def _parse_compression_response(self, response_text: str) -> CompressionResult:
        """
        Parse the LLM's formatted response into a CompressionResult.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed CompressionResult
        """
        response_value = self._ensure_text(response_text)
        lines = response_value.strip().split('\n')
        
        summary = ""
        concepts = []
        themes = []
        relationships = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("SUMMARY:"):
                summary = line[len("SUMMARY:"):].strip()
            elif line.startswith("CONCEPTS:"):
                concepts_str = line[len("CONCEPTS:"):].strip()
                concepts = [c.strip() for c in concepts_str.split(',') if c.strip()]
            elif line.startswith("THEMES:"):
                themes_str = line[len("THEMES:"):].strip()
                themes = [t.strip() for t in themes_str.split(',') if t.strip()]
            elif line.startswith("RELATIONSHIPS:"):
                rel_str = line[len("RELATIONSHIPS:"):].strip()
                relationships = [r.strip() for r in rel_str.split(';') if r.strip()]
        
        return CompressionResult(
            summary=summary,
            key_concepts=concepts,
            themes=themes,
            relationships=relationships,
            confidence=0.7
        )


def estimate_novelty(
    episode_concepts: List[str],
    existing_concepts: List[str],
    threshold: float = 0.5
) -> float:
    """
    Estimate novelty of an episode based on concept overlap with existing memories.
    
    Args:
        episode_concepts: Concepts in the new episode
        existing_concepts: Concepts in existing memories
        threshold: Baseline novelty score
        
    Returns:
        Novelty score (0-1), where 1 = completely novel
    """
    if not episode_concepts:
        return 0.5  # Neutral
    
    if not existing_concepts:
        return 1.0  # Everything is novel if no prior knowledge
    
    # Calculate concept overlap
    episode_set = set(episode_concepts)
    existing_set = set(existing_concepts)
    
    intersection = len(episode_set & existing_set)
    union = len(episode_set | existing_set)
    
    if union == 0:
        return 0.5
    
    # Jaccard similarity
    similarity = intersection / union
    
    # Novelty is inverse of similarity
    novelty = 1.0 - similarity
    
    return novelty
