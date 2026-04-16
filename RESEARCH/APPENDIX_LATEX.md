\begin{center}
	{\Large \textbf{Appendix A}}-\vspace*{0.5 cm}
	{\Large \textbf{Sample Code}}
	
\end{center}
\begin{lstlisting}
/* agent/agent.py **/

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
        auto_sleep_threshold: int = 4,
        dataset_name: str = "personamem",
        memory_config: Optional[DatasetMemoryConfig] = None,
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

    def interact(
        self,
        user_input: str,
        use_memory: bool = True,
        persona: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Main interaction loop.
        
        Args:
            user_input: User's message
            use_memory: Whether to use memory retrieval
            persona: Optional persona context
            importance: Importance of the interaction
            tags: Optional tags for the episode
            
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
        
        return agent_response

/* sleep/consolidation.py **/

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
            logger.info(f"\\n{'='*60}")
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
            logger.info("\\n[Phase 1] Prioritized Replay")
        
        replay_results = self._phase_1_replay(current_time, verbose)
        stats['episodes_replayed'] = replay_results['replayed']
        stats['replay_modes'] = replay_results.get('replay_modes', {})
        
        # Phase 2: Generative Compression & Consolidation
        if verbose:
            logger.info("\\n[Phase 2] Generative Compression & Consolidation")
        
        consolidation_results = self._phase_2_consolidation(
            replay_results['selected_episodes'],
            replay_results.get('replay_annotations', []),
            verbose
        )
        stats['memories_consolidated'] = consolidation_results['consolidated']
        stats['conflicts_detected'] = consolidation_results.get('conflicts_detected', 0)
        
        # Phase 3: Schema Formation
        if verbose:
            logger.info("\\n[Phase 3] Schema Formation")
        
        schema_results = self._phase_3_schema_formation(verbose)
        stats['schemas_formed'] = schema_results['schemas_formed']
        
        # Phase 4: Forgetting & Decay
        if verbose:
            logger.info("\\n[Phase 4] Forgetting & Decay")
        
        decay_results = self._phase_4_decay(verbose)
        stats['episodes_forgotten'] = decay_results['forgotten']
        
        if verbose:
            logger.info(f"\\n{'='*60}")
            logger.info(f"Sleep Cycle #{self.cycle_count} Complete")
            logger.info(f"  Replayed: {stats['episodes_replayed']} episodes")
            logger.info(f"  Consolidated: {stats['memories_consolidated']} memories")
            logger.info(f"  Schemas: {stats['schemas_formed']} new schemas")
            logger.info(f"  Forgotten: {stats['episodes_forgotten']} episodes")
            logger.info(f"{'='*60}\\n")

/* memory/consolidated.py **/

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

/* benchmark_runner.py **/

class BenchmarkRunner:
    """Main benchmark runner for PersonaMem experiments."""
    
    def __init__(
        self,
        split: str = "benchmark",
        num_samples: int = 100,
        methods: List[str] = None,
        output_dir: str = "results",
        ablation_flags: Optional[Dict[str, bool]] = None,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            split: Dataset split to use ('benchmark', 'val', 'train')
            num_samples: Number of samples to evaluate
            methods: List of methods to evaluate
            output_dir: Directory to save results
        """
        load_dotenv()
        
        self.split = split
        self.num_samples = num_samples
        self.methods = methods or ['vanilla', 'rag', 'episodic', 'summarization', 'sleep']
        self.ablation_flags = ablation_flags or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load preprocessed data
        self.data_dir = Path("PERSONAMEM/preprocessed")
        self.samples = self._load_data()
        self.persona_groups = self._load_persona_groups()
        
        # Initialize evaluator
        self.evaluator = PersonaMemEvaluator()
        
        print(f"\\n{'='*70}")
        print(f"Benchmark Runner Initialized")
        print(f"{'='*70}")
        print(f"Split: {split}")
        print(f"Samples: {len(self.samples)} (will use {num_samples})")
        print(f"Methods: {', '.join(self.methods)}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\\n")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load preprocessed data."""
        data_file = self.data_dir / f"{self.split}_processed.json"
        with open(data_file, 'r') as f:
            return json.load(f)
    
    def _load_persona_groups(self) -> Dict[int, List[Dict[str, Any]]]:
        """Load persona-grouped data."""
        persona_file = self.data_dir / f"{self.split}_persona_sessions.json"
        with open(persona_file, 'r') as f:
            data = json.load(f)
            # Convert string keys to int
            return {int(k): v for k, v in data.items()}
    
    def run_table1_evaluation(self, method: str) -> Dict[str, Any]:
        """
        Run Table 1 evaluation: Task-Based Memory Performance.
        
        - qa_results: Results from QA tasks
        - continuity_results: Results from continuity tasks
        - hallucination_results: Results from hallucination tasks
        - utility_results: Results from utility tasks
        - retrieval_results: Results from retrieval tasks
        - fidelity_results: Results from fidelity tasks
        - unit_hallu_results: Results from unit hallucination tasks
        - latency_ms: Latency in milliseconds
        """
        
        # Results storage
        qa_results = []
        continuity_results = []
        hallucination_results = []
        utility_results = []
        retrieval_results = []
        fidelity_results = []
        unit_hallu_results = []
        latency_ms = []
        
        # Process each persona (multi-session) - limit to match num_samples
        max_personas = min(self.num_samples, 10)  # Cap at 10 personas max
        for persona_id, persona_sample_list in tqdm(list(persona_samples.items())[:max_personas], desc=f"Evaluating {method}"):
            # Sort by any available timestamp or use order
            persona_sample_list = persona_sample_list[:2]  # Limit to 2 interactions per persona for speed
            
            persona_info = persona_sample_list[0]['persona']
            
            # Simulate multi-session interactions
            for i, sample in enumerate(persona_sample_list):
                query = sample['query']
                correct_answer = sample['correct_answer']
                incorrect_answers = sample['incorrect_answers']
                related_snippet = sample['related_conversation_snippet']
                
                # Agent interaction
                use_memory = (i > 0)  # Use memory after first interaction
\end{lstlisting}
