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
import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Simplified imports and removed unused code

from memory.episodic import EpisodicMemoryStore, Episode
from memory.consolidated import ConsolidatedMemoryStore
from memory.schema import SchemaStore
from sleep.consolidation import SleepCycle
from sleep.compression import MemoryCompressor, estimate_novelty
from memory.config import DatasetMemoryConfig, resolve_dataset_config
from utils.api_counter import increment_llm_call


class MemoryAgent:
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest",
        temperature: float = 0.7,
        auto_sleep_threshold: int = 4,
        dataset_name: str = "personamem",
        memory_config: Optional[DatasetMemoryConfig] = None,
    ):
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
        self.dataset_name = dataset_name.lower()
        self.memory_config = memory_config or resolve_dataset_config(self.dataset_name)
        
        # Initialize compressor and sleep system
        self.compressor = MemoryCompressor(api_key=self.api_key)
        self.sleep_cycle = SleepCycle(
            episodic_store=self.episodic_store,
            consolidated_store=self.consolidated_store,
            schema_store=self.schema_store,
            compressor=self.compressor,
            dataset_name=self.dataset_name,
            policy=self.memory_config,
        )
        
        # Conversation tracking
        self.conversation_history: List[Dict[str, str]] = []
        self.interaction_count = 0
        self.auto_sleep_threshold = auto_sleep_threshold
        self.episodes_since_sleep = 0
        self.last_retrieval_bundles: List[Dict[str, Any]] = []
        self.last_latency_ms: float = 0.0
    
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
        retrieval_bundles: List[Dict[str, Any]] = []
        if use_memory:
            retrieval_bundles = self._retrieve_memory_bundles(user_input, persona or "")
            self.last_retrieval_bundles = retrieval_bundles
            memory_context = self._render_memory_context(retrieval_bundles)
        
        # Build prompt with memory context and optional persona
        system_prompt = self._build_system_prompt(
            user_input=user_input,
            memory_context=memory_context,
            retrieval_bundles=retrieval_bundles,
            persona=persona,
        )
        
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
        t0 = time.perf_counter()
        increment_llm_call("agent_generation")
        response = self.llm.invoke(messages)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        # Standardize response format: if content is list of dicts, extract text
        if isinstance(response.content, list):
            # Response is list of content blocks, extract text from each
            agent_response = "".join(
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in response.content
            )
        else:
            # Response is plain string
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
        # if self.episodes_since_sleep >= self.auto_sleep_threshold:
        #     print("\n[Auto-sleep triggered]")
        #     self.sleep()
        
        return agent_response
    
    def _build_system_prompt(
        self,
        user_input: str,
        memory_context: str,
        retrieval_bundles: List[Dict[str, Any]],
        persona: Optional[str] = None,
    ) -> str:
        """
        Build system prompt with memory context.
        
        Args:
            memory_context: Retrieved memory context
            
        Returns:
            System prompt string
        """
        base_prompt = """[Task Instruction]
You are a helpful AI assistant with a controlled memory system.
Use retrieved memories only when relevant to the current query.
If memories conflict, acknowledge uncertainty instead of inventing details.

[Conversation Goal]
Answer the user naturally while preserving factual fidelity."""

        if self.dataset_name == "locomo":
            base_prompt += (
                "\n\n[Grounding Constraint]"
                "\nPrioritize evidence-grounded details."
                "\nDo not over-generalize from schema when exact evidence exists."
            )
        elif self.dataset_name == "personamem":
            base_prompt += (
                "\n\n[Preference Alignment Constraint]"
                "\nPrioritize stable user preferences and long-term identity facts."
                "\nIf preference memories conflict, mention uncertainty or recency explicitly."
            )
        elif self.dataset_name == "personachat":
            base_prompt += (
                "\n\n[Dialogue Continuity Constraint]"
                "\nPrioritize turn-to-turn continuity, persona consistency, and natural conversation style."
            )
        
        if persona:
            base_prompt += f"\n\n[User Profile]\n{persona}"

        if memory_context:
            base_prompt += f"\n\n[Retrieved Memory Bundles]\n{memory_context}"

        if retrieval_bundles:
            contradictory = [b for b in retrieval_bundles if b.get("is_contradictory", False)]
            if contradictory:
                base_prompt += (
                    "\n\n[Conflict Handling]"
                    "\nSome retrieved memories are contradictory; prefer high-confidence grounded memories."
                    "\nIf unresolved, state uncertainty briefly and avoid fabricated resolution."
                )

        base_prompt += f"\n\n[Current User Query]\n{user_input}"
        
        return base_prompt
    
    def _retrieve_memory_bundles(self, query: str, persona: str, top_k: int = 5) -> List[Dict[str, Any]]:
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

        consolidated = self.consolidated_store.search_hybrid(
            query=query,
            persona=persona,
            top_k=top_k,
            weights=self.memory_config.retrieval_weights,
            evidence_required=self.memory_config.evidence_required and not self.memory_config.ablations.disable_evidence_priority,
        )
        for b in consolidated:
            self.consolidated_store.mark_accessed(b["memory_id"])

        episodic = []
        if not self.memory_config.ablations.episodic_only or not consolidated:
            for ep in self.episodic_store.get_candidates(query=query, persona=persona, top_k=top_k):
                self.episodic_store.mark_accessed(ep.id)
                episodic.append(
                    {
                        "memory_id": ep.id,
                        "text": ep.content[:260],
                        "core_fact": ep.content.split("\n")[0][:140],
                        "source_type": "episodic",
                        "confidence": ep.confidence,
                        "score": 0.5 * ep.salience_score + 0.5 * ep.temporal_recency,
                        "why_retrieved": f"salience={ep.salience_score:.2f}, recency={ep.temporal_recency:.2f}",
                        "is_evidence_grounded": ep.evidence_strength > 0.4,
                        "is_contradictory": bool(ep.contradiction_flags),
                        "supporting": not bool(ep.contradiction_flags),
                    }
                )

        schema_bundles = []
        if not self.memory_config.ablations.disable_schema:
            schema_matches = self.schema_store.find_by_concepts(query_concepts, min_overlap=1)
            for schema in schema_matches[:2]:
                self.schema_store.mark_accessed(schema.id)
                schema_bundles.append(
                    {
                        "memory_id": schema.id,
                        "text": f"{schema.name}: {schema.description}",
                        "core_fact": schema.name,
                        "source_type": "schema",
                        "confidence": schema.confidence,
                        "score": 0.45 + 0.1 * schema.confidence,
                        "why_retrieved": f"schema_status={schema.status}, version={schema.version}",
                        "is_evidence_grounded": False,
                        "is_contradictory": schema.status == "conflicted",
                        "supporting": schema.status != "conflicted",
                    }
                )

        bundles = consolidated + episodic + schema_bundles
        bundles.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)

        # Conflict-aware suppression unless ablation disables it.
        if not self.memory_config.ablations.disable_conflict_handling:
            filtered = []
            for b in bundles:
                if b.get("is_contradictory") and float(b.get("confidence", 0.0)) < self.memory_config.factuality_threshold:
                    continue
                filtered.append(b)
            bundles = filtered

        return bundles[:top_k]

    def _render_memory_context(self, bundles: List[Dict[str, Any]]) -> str:
        """Render retrieval bundles as structured prompt sections."""
        if not bundles:
            return ""

        lines = [
            "[Retrieved Episodic Memories]",
        ]
        for b in bundles:
            source = b.get("source_type", "unknown")
            block = (
                f"- ({source}) conf={float(b.get('confidence', 0.0)):.2f} "
                f"evidence={b.get('is_evidence_grounded', False)} "
                f"supporting={b.get('supporting', True)}\n"
                f"  text: {str(b.get('text', ''))[:260]}\n"
                f"  why: {b.get('why_retrieved', '')}"
            )
            lines.append(block)
        return "\n".join(lines)

    def _retrieve_relevant_memories(self, query: str, top_k: int = 3) -> str:
        """Backward-compatible helper returning text context only."""
        bundles = self._retrieve_memory_bundles(query=query, persona="", top_k=top_k)
        return self._render_memory_context(bundles)
    
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
                'persona': persona,
                'dataset_name': self.dataset_name,
                'episode_type': self._infer_episode_type(user_input),
                'time_span': datetime.now().isoformat(),
                'evidence_link': self._infer_evidence_link(user_input),
            },
            importance=importance,
            novelty=novelty,
            tags=tags or [],
            salience_score=self._estimate_salience(user_input, importance),
            novelty_score=novelty,
            persona_relevance=self._estimate_persona_relevance(user_input, persona),
            factuality_risk=self._estimate_factuality_risk(user_input),
            temporal_recency=1.0,
            evidence_strength=self._estimate_evidence_strength(user_input),
            episode_type=self._infer_episode_type(user_input),
            confidence=max(0.3, 1.0 - self._estimate_factuality_risk(user_input)),
            uncertainty=min(1.0, self._estimate_factuality_risk(user_input) + 0.2),
            contradiction_flags=[],
        )
        
        self.episodes_since_sleep += 1
        
        return episode

    def _estimate_salience(self, text: str, importance: float) -> float:
        urgency = 0.0
        if any(tok in text.lower() for tok in ["important", "remember", "never", "always"]):
            urgency += 0.2
        if "?" in text:
            urgency += 0.1
        return max(0.0, min(1.0, 0.7 * importance + urgency))

    def _estimate_persona_relevance(self, text: str, persona: Optional[str]) -> float:
        if not persona:
            return 0.4
        t = set(re.findall(r"[a-zA-Z]+", text.lower()))
        p = set(re.findall(r"[a-zA-Z]+", persona.lower()))
        if not p:
            return 0.4
        return max(0.0, min(1.0, len(t & p) / max(1, len(p) * 0.2)))

    def _estimate_factuality_risk(self, text: str) -> float:
        l = text.lower()
        has_precise_claim = any(k in l for k in ["when", "where", "date", "time", "exact", "evidence"])
        hedge = any(k in l for k in ["maybe", "probably", "guess", "might"])
        risk = 0.35 + (0.25 if has_precise_claim else 0.0) + (0.2 if hedge else 0.0)
        return max(0.0, min(1.0, risk))

    def _estimate_evidence_strength(self, text: str) -> float:
        l = text.lower()
        if re.search(r"\bd\d+:\d+\b", text):
            return 0.9
        if any(k in l for k in ["evidence", "source", "according to", "session"]):
            return 0.6
        return 0.2

    def _infer_evidence_link(self, text: str) -> str:
        match = re.search(r"\b(D\d+:\d+)\b", text)
        return match.group(1) if match else ""

    def _infer_episode_type(self, text: str) -> str:
        l = text.lower()
        if any(w in l for w in ["prefer", "preference", "favorite", "like", "dislike"]):
            return "preference"
        if any(w in l for w in ["when", "where", "date", "time", "fact"]):
            return "fact"
        if any(w in l for w in ["event", "happened", "met", "went"]):
            return "event"
        if any(w in l for w in ["please", "instruction", "do this", "follow"]):
            return "instruction"
        if len(l.split()) <= 20:
            return "dialogue"
        return "mixed"
    
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
            'sleep_cycles': self.sleep_cycle.cycle_count,
            'dataset_name': self.dataset_name,
            'last_latency_ms': self.last_latency_ms,
            'retrieval_bundle_count': len(self.last_retrieval_bundles),
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
