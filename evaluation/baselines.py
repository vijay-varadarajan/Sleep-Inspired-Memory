"""
Baseline Methods for Comparison

Implements various baseline approaches for Table 1 comparisons:
1. Vanilla LLM: No memory, just LLM responses
2. RAG (Vector DB): Simple vector-based retrieval
3. Episodic Only: Only episodic memory, no consolidation
4. Episodic + Summarization: Basic summarization without sleep
5. Ours (Sleep-Consolidated): Full sleep-inspired consolidation
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from memory.episodic import EpisodicMemoryStore, Episode
from memory.consolidated import ConsolidatedMemoryStore
from memory.schema import SchemaStore
from sleep.consolidation import SleepCycle
from sleep.compression import MemoryCompressor
from agent.agent import MemoryAgent


class VanillaLLM:
    """
    Baseline 1: Vanilla LLM with no memory system.
    
    Simply uses the LLM to answer queries without any memory of past interactions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        """Initialize Vanilla LLM."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.7
        )
        self.interaction_count = 0
    
    def interact(self, user_input: str, persona: str = "", **kwargs) -> str:
        """
        Respond to user input with no memory.
        
        Args:
            user_input: User query
            persona: Optional persona information
            **kwargs: Ignored additional arguments
            
        Returns:
            Response string
        """
        self.interaction_count += 1
        
        # Build prompt with persona if provided
        system_msg = "You are a helpful AI assistant."
        if persona:
            system_msg += f"\n\nUser Profile:\n{persona}"
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_input)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary (empty for vanilla)."""
        return {
            'interactions': self.interaction_count,
            'episodic': {'total_episodes': 0},
            'consolidated': {'total_memories': 0},
            'schemas': {'total_schemas': 0}
        }


class RAGBaseline:
    """
    Baseline 2: Simple RAG with vector database.
    
    Stores interactions in a vector database and retrieves relevant ones
    for context, without any consolidation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest", top_k: int = 3):
        """Initialize RAG baseline."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.7
        )
        
        # Initialize embeddings and vector store
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        self.vectorstore = None
        self.documents = []
        self.top_k = top_k
        self.interaction_count = 0
    
    def interact(
        self,
        user_input: str,
        persona: str = "",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Respond using RAG retrieval.
        
        Args:
            user_input: User query
            persona: Optional persona information
            importance: Ignored (for compatibility)
            tags: Ignored (for compatibility)
            **kwargs: Additional arguments
            
        Returns:
            Response string
        """
        self.interaction_count += 1
        
        # Retrieve relevant past interactions if available
        context = ""
        if self.vectorstore is not None:
            try:
                relevant_docs = self.vectorstore.similarity_search(user_input, k=self.top_k)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
            except:
                context = ""
        
        # Build prompt
        system_msg = "You are a helpful AI assistant."
        if persona:
            system_msg += f"\n\nUser Profile:\n{persona}"
        if context:
            system_msg += f"\n\nRelevant Past Interactions:\n{context}"
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_input)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Store interaction in vector store
        interaction_text = f"User: {user_input}\nAssistant: {response_text}"
        self.documents.append(interaction_text)
        
        # Rebuild vector store
        if len(self.documents) > 0:
            self.vectorstore = FAISS.from_texts(
                self.documents,
                self.embeddings
            )
        
        return response_text
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        return {
            'interactions': self.interaction_count,
            'episodic': {'total_episodes': len(self.documents)},
            'consolidated': {'total_memories': 0},
            'schemas': {'total_schemas': 0}
        }


class EpisodicOnlyAgent:
    """
    Baseline 3: Episodic memory only, no consolidation.
    
    Stores all interactions as episodic memories but never consolidates them.
    Retrieves episodic memories based on recency and importance.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-latest"):
        """Initialize episodic-only agent."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.7
        )
        
        self.episodic_store = EpisodicMemoryStore()
        self.interaction_count = 0
    
    def interact(
        self,
        user_input: str,
        persona: str = "",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        use_memory: bool = True,
        **kwargs
    ) -> str:
        """
        Respond using episodic memory only.
        
        Args:
            user_input: User query
            persona: Optional persona information
            importance: Importance score for this interaction
            tags: Tags for categorization
            use_memory: Whether to retrieve episodic memories
            **kwargs: Additional arguments
            
        Returns:
            Response string
        """
        self.interaction_count += 1
        
        # Retrieve recent episodic memories
        context = ""
        if use_memory and self.episodic_store.get_count() > 0:
            recent_episodes = self.episodic_store.get_recent(n=5)
            context = "\n\n".join([
                f"Previous interaction: {ep.content[:200]}..."
                for ep in recent_episodes
            ])
        
        # Build prompt
        system_msg = "You are a helpful AI assistant."
        if persona:
            system_msg += f"\n\nUser Profile:\n{persona}"
        if context:
            system_msg += f"\n\nRecent Interactions:\n{context}"
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_input)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Store as episodic memory
        interaction_text = f"User: {user_input}\nAssistant: {response_text}"
        self.episodic_store.add_episode(
            content=interaction_text,
            importance=importance,
            tags=tags or []
        )
        
        return response_text
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        return {
            'interactions': self.interaction_count,
            'episodic': {
                'total_episodes': self.episodic_store.get_count(),
                'unconsolidated': self.episodic_store.get_count()
            },
            'consolidated': {'total_memories': 0},
            'schemas': {'total_schemas': 0}
        }


class EpisodicSummarizationAgent:
    """
    Baseline 4: Episodic + Basic Summarization (no sleep).
    
    Periodically summarizes episodic memories into simple summaries
    without the full sleep-inspired consolidation process.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest",
        summarize_threshold: int = 5
    ):
        """Initialize episodic + summarization agent."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.7
        )
        
        self.episodic_store = EpisodicMemoryStore()
        self.summaries = []
        self.interaction_count = 0
        self.summarize_threshold = summarize_threshold
    
    def _summarize_recent_episodes(self, num_episodes: int = 5):
        """Create a simple summary of recent episodes."""
        recent = self.episodic_store.get_recent(n=num_episodes)
        if not recent:
            return
        
        # Combine episodes
        combined_text = "\n\n".join([ep.content for ep in recent])
        
        # Create summary using LLM
        prompt = f"""Summarize the following interactions concisely:

{combined_text}

Provide a brief summary capturing the key points and topics discussed."""
        
        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            summary = result.content
            self.summaries.append(summary)
            
            # Mark episodes as summarized (we'll use consolidated flag for this)
            for ep in recent:
                ep.consolidated = True
        except Exception as e:
            print(f"Error creating summary: {e}")
    
    def interact(
        self,
        user_input: str,
        persona: str = "",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        use_memory: bool = True,
        **kwargs
    ) -> str:
        """
        Respond using episodic memory + summaries.
        
        Args:
            user_input: User query
            persona: Optional persona information
            importance: Importance score
            tags: Tags for categorization
            use_memory: Whether to use memory
            **kwargs: Additional arguments
            
        Returns:
            Response string
        """
        self.interaction_count += 1
        
        # Build context from summaries and recent episodes
        context = ""
        if use_memory:
            # Add summaries
            if self.summaries:
                context += "Previous Conversation Summaries:\n"
                context += "\n".join(self.summaries[-3:])  # Last 3 summaries
                context += "\n\n"
            
            # Add recent unconsolidated episodes
            unconsolidated = [ep for ep in self.episodic_store.get_recent(n=5) if not ep.consolidated]
            if unconsolidated:
                context += "Recent Interactions:\n"
                context += "\n".join([ep.content[:200] for ep in unconsolidated])
        
        # Build prompt
        system_msg = "You are a helpful AI assistant."
        if persona:
            system_msg += f"\n\nUser Profile:\n{persona}"
        if context:
            system_msg += f"\n\nMemory:\n{context}"
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_input)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Store as episodic memory
        interaction_text = f"User: {user_input}\nAssistant: {response_text}"
        self.episodic_store.add_episode(
            content=interaction_text,
            importance=importance,
            tags=tags or []
        )
        
        # Periodically summarize
        if self.interaction_count % self.summarize_threshold == 0:
            self._summarize_recent_episodes()
        
        return response_text
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        consolidated_count = sum(1 for ep in self.episodic_store.episodes.values() if ep.consolidated)
        return {
            'interactions': self.interaction_count,
            'episodic': {
                'total_episodes': self.episodic_store.get_count(),
                'consolidated': consolidated_count,
                'unconsolidated': self.episodic_store.get_count() - consolidated_count
            },
            'consolidated': {'total_memories': len(self.summaries)},
            'schemas': {'total_schemas': 0}
        }


class SleepConsolidatedAgent(MemoryAgent):
    """
    Our Method: Full sleep-inspired consolidation.
    
    This is a wrapper around the existing MemoryAgent with sleep consolidation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest",
        auto_sleep_threshold: int = 10
    ):
        """Initialize sleep-consolidated agent."""
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            auto_sleep_threshold=auto_sleep_threshold
        )
    
    def sleep(self, verbose: bool = False) -> Dict[str, Any]:
        """Run sleep consolidation cycle."""
        return super().sleep(verbose=verbose)


def create_agent(method: str, api_key: Optional[str] = None) -> Any:
    """
    Factory function to create agents of different types.
    
    Args:
        method: Method name ('vanilla', 'rag', 'episodic', 'summarization', 'sleep')
        api_key: Google API key
        
    Returns:
        Agent instance
    """
    if method == 'vanilla':
        return VanillaLLM(api_key=api_key)
    elif method == 'rag':
        return RAGBaseline(api_key=api_key)
    elif method == 'episodic':
        return EpisodicOnlyAgent(api_key=api_key)
    elif method == 'summarization':
        return EpisodicSummarizationAgent(api_key=api_key)
    elif method == 'sleep':
        return SleepConsolidatedAgent(api_key=api_key)
    else:
        raise ValueError(f"Unknown method: {method}")
