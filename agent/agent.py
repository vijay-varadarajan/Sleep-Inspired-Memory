"""
Sleep-Inspired Memory Agent

An LLM agent with biologically-inspired memory consolidation.
Integrates episodic, consolidated, and schema memory systems.

Key Features:
- Stores interactions as episodic memories
- Performs periodic sleep-based consolidation
- Recalls from both episodic and consolidated memory
- Uses schemas for generalization
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from memory.episodic import EpisodicMemoryStore, Episode
from memory.consolidated import ConsolidatedMemoryStore
from memory.schema import SchemaStore
from sleep.consolidation import SleepCycle
from sleep.compression import MemoryCompressor, estimate_novelty


class MemoryAgent:
    """
    LLM agent with sleep-inspired memory consolidation.
    
    The agent:
    1. Interacts with users and stores experiences as episodic memories
    2. Periodically runs sleep cycles to consolidate memories
    3. Retrieves relevant memories to inform responses
    4. Uses schemas for generalization and transfer
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest",
        temperature: float = 0.7,
        auto_sleep_threshold: int = 10
    ):
        """
        Initialize the memory agent.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            temperature: Sampling temperature for responses
            auto_sleep_threshold: Number of new episodes before auto-sleep
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=temperature
        )
        
        # Initialize memory systems
        self.episodic_store = EpisodicMemoryStore()
        self.consolidated_store = ConsolidatedMemoryStore()
        self.schema_store = SchemaStore()
        
        # Initialize compressor and sleep system
        self.compressor = MemoryCompressor(api_key=self.api_key)
        self.sleep_cycle = SleepCycle(
            episodic_store=self.episodic_store,
            consolidated_store=self.consolidated_store,
            schema_store=self.schema_store,
            compressor=self.compressor
        )
        
        # Conversation tracking
        self.conversation_history: List[Dict[str, str]] = []
        self.interaction_count = 0
        self.auto_sleep_threshold = auto_sleep_threshold
        self.episodes_since_sleep = 0
    
    def interact(
        self,
        user_input: str,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        use_memory: bool = True,
        persona: Optional[str] = None
    ) -> str:
        """
        Interact with the agent.
        
        Args:
            user_input: User's message
            importance: Optional importance score (auto-estimated if None)
            tags: Optional tags for categorization
            use_memory: Whether to retrieve and use memory context
            
        Returns:
            Agent's response
        """
        self.interaction_count += 1
        
        # Retrieve relevant memories if requested
        memory_context = ""
        if use_memory:
            memory_context = self._retrieve_relevant_memories(user_input)
        
        # Build prompt with memory context and optional persona
        system_prompt = self._build_system_prompt(memory_context, persona)
        
        # Get response from LLM
        messages = [
            SystemMessage(content=system_prompt),
        ]
        
        # Add conversation history (last 5 turns)
        for turn in self.conversation_history[-5:]:
            if turn['role'] == 'user':
                messages.append(HumanMessage(content=turn['content']))
            else:
                messages.append(AIMessage(content=turn['content']))
        
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        # Generate response
        response = self.llm.invoke(messages)
        agent_response = response.content
        
        # Update conversation history
        self.conversation_history.append({'role': 'user', 'content': user_input})
        self.conversation_history.append({'role': 'agent', 'content': agent_response})
        
        # Store interaction as episodic memory
        self._store_interaction(
            user_input=user_input,
            agent_response=agent_response,
            importance=importance,
            tags=tags,
            persona=persona
        )
        
        # Check if auto-sleep should trigger
        if self.episodes_since_sleep >= self.auto_sleep_threshold:
            print("\n[Auto-sleep triggered]")
            self.sleep()
        
        return agent_response
    
    def _build_system_prompt(self, memory_context: str, persona: Optional[str] = None) -> str:
        """
        Build system prompt with memory context.
        
        Args:
            memory_context: Retrieved memory context
            
        Returns:
            System prompt string
        """
        base_prompt = """You are a helpful AI assistant with a sophisticated memory system.
You can remember past interactions and learn from them over time.

Be conversational, helpful, and make use of your memories when relevant."""
        
        if persona:
            base_prompt += f"\n\nUser Profile:\n{persona}"

        if memory_context:
            base_prompt += f"\n\n--- RELEVANT MEMORIES ---\n{memory_context}\n--- END MEMORIES ---"
        
        return base_prompt
    
    def _retrieve_relevant_memories(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant memories for a query.
        
        Uses a simple approach:
        1. Extract concepts from query
        2. Search consolidated memories by concepts
        3. Search schemas by concepts
        
        Args:
            query: User query
            top_k: Number of memories to retrieve
            
        Returns:
            Formatted memory context string
        """
        # Extract concepts from query
        query_concepts = self.compressor.extract_concepts_from_text(query)
        
        if not query_concepts:
            return ""
        
        memory_lines = []
        
        # Retrieve consolidated memories
        consolidated_matches = self.consolidated_store.search_by_concepts(query_concepts)
        for memory in consolidated_matches[:top_k]:
            memory_lines.append(f"• {memory.summary}")
            self.consolidated_store.mark_accessed(memory.id)
        
        # Retrieve schemas
        schema_matches = self.schema_store.find_by_concepts(query_concepts, min_overlap=1)
        for schema in schema_matches[:2]:
            memory_lines.append(f"• [Schema] {schema.name}: {schema.description}")
            self.schema_store.mark_accessed(schema.id)
        
        if memory_lines:
            return "\n".join(memory_lines)
        else:
            return ""
    
    def _store_interaction(
        self,
        user_input: str,
        agent_response: str,
        importance: Optional[float],
        tags: Optional[List[str]],
        persona: Optional[str] = None
    ) -> Episode:
        """
        Store an interaction as an episodic memory.
        
        Args:
            user_input: User's input
            agent_response: Agent's response
            importance: Importance score
            tags: Tags for categorization
            
        Returns:
            Created Episode
        """
        # Format episode content
        content = f"User: {user_input}\nAgent: {agent_response}"
        
        # Auto-estimate importance if not provided
        if importance is None:
            # Simple heuristic: length and question marks
            importance = 0.5
            if len(user_input) > 100:
                importance += 0.1
            if '?' in user_input:
                importance += 0.1
            importance = min(1.0, importance)
        
        # Estimate novelty based on existing concepts
        all_existing_concepts = []
        for memory in self.consolidated_store.get_all_memories():
            all_existing_concepts.extend(memory.key_concepts)
        
        episode_concepts = self.compressor.extract_concepts_from_text(content)
        novelty = estimate_novelty(episode_concepts, all_existing_concepts)
        
        # Create episode
        episode = self.episodic_store.add_episode(
            content=content,
            context={
                'user_input': user_input,
                'agent_response': agent_response,
                'interaction_number': self.interaction_count,
                'persona': persona
            },
            importance=importance,
            novelty=novelty,
            tags=tags or []
        )
        
        self.episodes_since_sleep += 1
        
        return episode
    
    def sleep(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a sleep cycle to consolidate memories.
        
        Args:
            verbose: Whether to log progress
            
        Returns:
            Sleep cycle statistics
        """
        stats = self.sleep_cycle.run_sleep_cycle(verbose=verbose)
        self.episodes_since_sleep = 0
        return stats
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the agent's memory state.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            'interactions': self.interaction_count,
            'episodes_since_sleep': self.episodes_since_sleep,
            'episodic': self.episodic_store.get_stats(),
            'consolidated': self.consolidated_store.get_stats(),
            'schemas': self.schema_store.get_stats(),
            'sleep_cycles': self.sleep_cycle.cycle_count
        }
    
    def recall_episode(self, episode_id: str) -> Optional[Episode]:
        """
        Explicitly recall a specific episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            Episode if found, None otherwise
        """
        episode = self.episodic_store.get_episode(episode_id)
        if episode:
            self.episodic_store.mark_accessed(episode_id)
        return episode
    
    def get_recent_interactions(self, n: int = 5) -> List[str]:
        """
        Get recent interaction summaries.
        
        Args:
            n: Number of recent interactions
            
        Returns:
            List of formatted interaction strings
        """
        recent_episodes = self.episodic_store.get_recent(n)
        summaries = []
        
        for episode in recent_episodes:
            summaries.append(
                f"[{episode.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"{episode.content[:100]}..."
            )
        
        return summaries
    
    def save_memories(self, directory: str = "saved_memories") -> None:
        """
        Save all memory stores to files.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        self.episodic_store.save_to_file(f"{directory}/episodic.json")
        self.consolidated_store.save_to_file(f"{directory}/consolidated.json")
        self.schema_store.save_to_file(f"{directory}/schemas.json")
        
        print(f"Memories saved to {directory}/")
    
    def load_memories(self, directory: str = "saved_memories") -> None:
        """
        Load memory stores from files.
        
        Args:
            directory: Directory to load from
        """
        self.episodic_store.load_from_file(f"{directory}/episodic.json")
        self.consolidated_store.load_from_file(f"{directory}/consolidated.json")
        self.schema_store.load_from_file(f"{directory}/schemas.json")
        
        print(f"Memories loaded from {directory}/")
