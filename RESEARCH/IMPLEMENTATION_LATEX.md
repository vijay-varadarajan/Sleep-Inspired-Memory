\chapter{\MakeUppercase{Project implementation}}

\section{Overview}
\paragraph{}
The project is implemented in Python 3.10+ and structured into several key modules, each with a distinct responsibility. The source code is organized as follows:
\begin{itemize}
    \item \verb|agent/|: Contains the core \verb|MemoryAgent| class that orchestrates the interaction and memory management logic.
    \item \verb|memory/|: Implements the three-tier memory system: \verb|EpisodicMemoryStore|, \verb|ConsolidatedMemoryStore|, and \verb|SchemaStore|. It also includes the dataset-specific configuration file, \verb|config.py|.
    \item \verb|sleep/|: Contains the logic for the offline sleep cycle, including replay, compression, and consolidation.
    \item \verb|evaluation/|: Includes scripts for running benchmarks, defining baseline models, and calculating performance metrics.
    \item \verb|*\_preprocessing.py| scripts: A set of scripts at the root level for preparing each of the four datasets.
    \item \verb|public/|: A Flask-based web application for interactively demonstrating the agent's capabilities.
    \item \verb|RESEARCH/|: Contains all research drafts, result summaries, and generated graphs.
\end{itemize}

\section{Step A: Data Preparation}
\paragraph{}
The first step in the pipeline is data preprocessing. Each of the four datasets has a dedicated script that transforms its raw format into a standardized JSON structure that the agent can consume. These scripts are:
\begin{itemize}
    \item \verb|personamem_preprocessing.py|
    \item \verb|personachat_preprocessing.py|
    \item \verb|locomo_preprocessing.py|
    \item \verb|okvqa_preprocessing.py|
\end{itemize}

\paragraph{}
A key function in these scripts is grouping conversational turns into sessions and associating them with the correct persona or context. For example, in \verb|personamem_preprocessing.py|, sessions are grouped by \verb|persona_id| to ensure all related interactions are processed together.

\begin{verbatim}
# Example from personamem_preprocessing.py
def group_by_persona(data):
    personas = {}
    for item in data:
        persona_id = item['persona_id']
        if persona_id not in personas:
            personas[persona_id] = {
                "persona_id": persona_id,
                "persona": item['persona'],
                "sessions": []
            }
        personas[persona_id]["sessions"].append(item)
    return list(personas.values())
\end{verbatim}

\paragraph{}
Across all preprocessing pipelines, a shared normalization step is applied to numeric fields like importance and salience to prevent scale imbalances during retrieval scoring. This is followed by concept extraction using a TF-IDF-style weighting to identify key terms.
\begin{equation}
\hat{x} = \frac{x - \mu_{\text{field}}}{\sigma_{\text{field}} + \epsilon}
\end{equation}
\begin{equation}
w_{t,d} = \text{tf}(t,d) \cdot \log\left(\frac{N}{1 + \text{df}(t)}\right)
\end{equation}

\section{Step B: Agent and Memory Implementation}
\paragraph{}
The core of the system is the \verb|MemoryAgent| class in \verb|agent/agent.py|. It initializes the three memory stores and manages the main interaction loop.

\begin{verbatim}
# Snippet from agent/agent.py
class MemoryAgent:
    def __init__(self, agent_name, model, config, memory_stores=None):
        self.name = agent_name
        self.model = model
        self.config = config
        if memory_stores:
            self.episodic_memory = memory_stores['episodic']
            self.consolidated_memory = memory_stores['consolidated']
            self.schema_memory = memory_stores['schema']
        else:
            self.episodic_memory = EpisodicMemoryStore()
            self.consolidated_memory = ConsolidatedMemoryStore()
            self.schema_memory = SchemaStore()
        # ...
\end{verbatim}

\paragraph{}
The agent's main entry point is the \verb|interact| method, which orchestrates retrieval, prompt construction, and response generation.

\section{Step C: Dataset-Aware Policy}
\paragraph{}
To adapt the agent's behavior to different tasks, a dataset-aware policy is implemented in \verb|memory/config.py|. This file contains a dictionary that maps each dataset to a specific set of hyperparameters, such as retrieval weights and replay size.

\begin{verbatim}
# Snippet from memory/config.py
DATASET_MEMORY_CONFIG = {
    "personamem": DatasetMemoryConfig(
        dataset_name="personamem",
        replay_top_k=10,
        retrieval_weights={"semantic": 0.6, "lexical": 0.2, "recency": 0.2},
        # ...
    ),
    "personachat": DatasetMemoryConfig(
        dataset_name="personachat",
        replay_top_k=6,
        retrieval_weights={"semantic": 0.5, "lexical": 0.1, "recency": 0.4},
        # ...
    ),
    # ... configs for locomo and okvqa
}
\end{verbatim}
\paragraph{}
This allows the agent to use different memory strategies for different datasets without changing the core code.

\section{Step D: Runtime Interaction and Retrieval}
\paragraph{}
During an interaction, the agent first retrieves relevant memories using the \verb|_retrieve_memory_bundles| method. This method queries all three memory stores and combines the results. The retrieval score is a weighted sum of semantic similarity, lexical match, recency, and evidence signals.

\begin{equation}
r(q, m) = w_s \cdot \text{sem}(q,m) + w_l \cdot \text{lex}(q,m) + w_r \cdot \text{rec}(m) + w_e \cdot \text{evid}(m)
\end{equation}

\paragraph{}
The retrieved memories are then used to build a structured system prompt, which is passed to the LLM to generate a response.

\section{Step E: Sleep Consolidation}
\paragraph{}
When the number of new episodes reaches a threshold, the \verb|sleep| method is triggered. This initiates the five-phase consolidation process managed by the \verb|SleepCycle| class in \verb|sleep/consolidation.py|.

\paragraph{}
A critical phase is replay selection, where episodes are prioritized for consolidation. The priority is calculated based on recency, importance, novelty, and access count.
\begin{equation}
P(e) = w_1 \cdot e^{-\ln 2 \cdot \Delta t / 7} + w_2 \cdot \text{importance}(e) + w_3 \cdot \text{novelty}(e) + w_4 \cdot \log(1 + \text{access\_count}(e))
\end{equation}

\paragraph{}
Selected episodes are then compressed and merged into the \verb|ConsolidatedMemoryStore|. During this process, a decay mechanism reduces the importance of older, less relevant episodes.
\begin{equation}
\text{importance}_{t+1}(e) = \text{importance}_t(e) \cdot (1 - \delta)
\end{equation}

\section{Step F: Evaluation and Output}
\paragraph{}
The framework's performance is evaluated using the scripts in the \verb|evaluation/| directory. The main entry point is \verb|benchmark_runner.py|, which iterates through datasets and methods, calling the appropriate evaluator.

\paragraph{}
The \verb|evaluation/baselines.py| script defines the different memory models for comparison, including \verb|vanilla|, \verb|rag|, and the full \verb|sleep|-enabled agent.

\begin{verbatim}
# Snippet from benchmark_runner.py
def run_benchmark(method, dataset_name, num_samples):
    # ...
    evaluator = get_evaluator(dataset_name)
    agent = load_agent(method, dataset_name)
    
    for sample in benchmark_data:
        response = agent.interact(sample['input'])
        metrics = evaluator.evaluate(response, sample['ground_truth'])
        results.append(metrics)
    # ...
\end{verbatim}

\paragraph{}
Results are saved in timestamped JSON and CSV files in the \verb|results/| directory, ensuring that all experimental outcomes are reproducible and available for analysis. These results are then used by scripts in the \verb|RESEARCH/| folder to generate the tables and figures for the final paper.

\section{Project Plan}
\paragraph{}
The project was executed over several distinct phases, beginning with a 10-day sprint to develop the foundational architecture, including the core agent and the three-tiered memory system. Initial implementation revealed challenges in balancing retrieval across the different memory stores, leading to suboptimal response generation. To address this, the next phase focused on creating a dataset-aware policy, which took approximately 7 days to perfect. This allowed for dynamic adjustment of retrieval weights, significantly improving performance.

\paragraph{}
With the core logic in place, the sleep-consolidation cycle was implemented to enable long-term memory formation. Subsequently, a comprehensive evaluation framework was established over 5 days, defining baselines and performance metrics. The experimentation phase involved running extensive benchmarks across all datasets, a process that spanned 10 days of computation and data gathering. The final phase was dedicated to documentation and reporting. Over a period of two weeks, all experimental findings were analyzed, and the complete research paper, including detailed methodology and results, was drafted in LaTeX. The project concluded with a robust, well-documented system and a comprehensive research summary.
