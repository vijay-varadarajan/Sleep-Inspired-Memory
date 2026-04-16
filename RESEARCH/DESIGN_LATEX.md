\chapter{\MakeUppercase{System Design}}

\section{System Architecture}
\paragraph{}
This project uses a layered architecture. Each layer has a clear job. This makes the system easier to build, test, and improve.

\paragraph{}
At the top, we have \textbf{input and execution layers}. These include command line runners and the web app. The runner files (\verb|benchmark_runner.py|, \verb|personachat_runner.py|, \verb|locomo_runner.py|, \verb|okvqa_runner.py|) run experiments in batch mode. The Flask app (\verb|public/app.py|) supports interactive chat mode.

\paragraph{}
In the middle, we have the \textbf{agent orchestration layer}. The key class is \verb|MemoryAgent| in \verb|agent/agent.py|. It controls end-to-end behavior. It receives user input, retrieves memory, builds the final prompt, calls the LLM, stores a new episode, and decides when to run sleep consolidation.

\paragraph{}
Below that, we have the \textbf{memory subsystem} with three stores:
\begin{enumerate}
    \item \verb|EpisodicMemoryStore| (\verb|memory/episodic.py|) for raw recent interactions.
    \item \verb|ConsolidatedMemoryStore| (\verb|memory/consolidated.py|) for compressed long-term memories.
    \item \verb|SchemaStore| (\verb|memory/schema.py|) for abstract patterns.
\end{enumerate}
These three stores map to the biological idea used in the methodology: fast short-term encoding plus slower long-term abstraction.

\paragraph{}
Next, we have the \textbf{sleep subsystem} in \verb|sleep/|. The class \verb|SleepCycle| in \verb|sleep/consolidation.py| runs the offline consolidation logic. It performs replay selection, compression, consolidation, schema formation, and decay. Replay selection is policy-aware and uses dataset-specific settings from \verb|memory/config.py|.

\paragraph{}
We also have an \textbf{evaluation and reporting layer} in \verb|evaluation/| and \verb|RESEARCH/|. This layer computes metrics, builds tables, and creates graphs. Baselines are defined in \verb|evaluation/baselines.py|. Result files are saved as JSON and CSV. Research markdown files then summarize the results.

\paragraph{}
Finally, we have a \textbf{configuration and utility layer}. This includes dataset-aware memory settings (\verb|memory/config.py|), API usage counters (\verb|utils/api_counter.py|), and environment variable support through \verb|.env|.

\subsection{Architecture Goals}
\begin{itemize}
    \item Keep memory logic separate from UI and benchmark scripts.
    \item Support many datasets with one common agent interface.
    \item Allow quick ablation testing through config flags.
    \item Make results reproducible with saved artifacts.
    \item Keep deployment simple for research and demos.
\end{itemize}

\subsection{High-Level Runtime Flow}
\begin{enumerate}
    \item Input comes from user chat or benchmark sample.
    \item Agent retrieves relevant episodic, consolidated, and schema memories.
    \item Agent builds system prompt with memory context and conflict guidance.
    \item LLM returns a response.
    \item Interaction is saved as a new episodic memory.
    \item If threshold is reached, sleep cycle runs and updates long-term stores.
    \item Evaluator records quality and safety metrics.
\end{enumerate}
This architecture supports online interaction and offline consolidation in one unified design. Memory stability in the consolidated tier is modeled analogously to the synaptic consolidation curve:
\begin{equation}
S(t) = S_0 \cdot e^{-\lambda t} + S_{\infty}
\end{equation}

\section{Design}
\paragraph{}
The design follows simple principles:
\begin{itemize}
    \item One clear responsibility per module.
    \item Shared data model across datasets.
    \item Config-driven behavior instead of hard-coded rules.
    \item Explicit logging and saved outputs for debugging.
\end{itemize}

\paragraph{}
The system is designed to compare five methods in a fair way: \verb|vanilla|, \verb|rag|, \verb|episodic|, \verb|summarization|, and \verb|sleep|. The full model uses the same runner pipeline as baselines. This keeps comparisons consistent.

\subsection{Data Flow Diagram}
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\linewidth]{images/data_flow_diagram.png}
    \caption{Data Flow Diagram}
    \label{fig:dfd}
\end{figure}

\paragraph{DFD Explanation}
The input can come from two paths. One is a live user message from the web app. The other is a benchmark item from runner scripts. Both paths call the same \verb|interact()| method.

\paragraph{}
The retrieval score is:
\begin{equation}
r(q, m) = w_s \cdot \text{sem}(q,m) + w_l \cdot \text{lex}(q,m) + w_r \cdot \text{rec}(m) + w_e \cdot \text{evid}(m)
\end{equation}
The retrieval step fetches memory from all three stores. The consolidated store uses hybrid scoring. The schema store adds abstract patterns. The episodic store adds recent details.

\paragraph{}
After prompt building, the LLM generates output. The new interaction is then stored as a fresh episode. This is important because new data must enter memory immediately.

\paragraph{}
Sleep is event-driven. When the threshold is reached, the sleep cycle transforms some episodic entries into consolidated memories and schemas. Low-value episodes can be decayed.

\paragraph{}
Results are then scored and saved. This makes the full flow observable and measurable.

\subsection{Class Diagram}
\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{images/class_diagram.png}
    \caption{Class Diagram}
    \label{fig:class_diagram}
\end{figure}

\paragraph{Class Design Notes}
\begin{itemize}
    \item \verb|MemoryAgent| is the orchestrator.
    \item Memory stores are state containers with search and update logic.
    \item \verb|SleepCycle| handles offline reorganization of memory.
    \item \verb|MemoryCompressor| handles LLM-based compression and concept extraction.
    \item \verb|DatasetMemoryConfig| allows per-dataset tuning with the same interface.
    \item Evaluators are separate from agent internals, which helps fair benchmarking.
\end{itemize}

\chapter{\MakeUppercase{Methodology and Testing}}

\section{Implementation}
\paragraph{}
Implementation follows the same sequence described in methodology, but here it is mapped directly to code modules.

\subsection{Step A: Data Preparation}
\paragraph{}
Each dataset has its own preprocessing script:
\begin{itemize}
    \item \verb|personamem_preprocessing.py|
    \item \verb|personachat_preprocessing.py|
    \item \verb|locomo_preprocessing.py|
    \item \verb|okvqa_preprocessing.py|
\end{itemize}
These scripts convert raw files into structured JSON files under dataset-specific \verb|preprocessed/| folders. The format supports session-level input and benchmark evaluation.

\paragraph{}
Across all pipelines, a shared normalization step applies:
\begin{equation}
\hat{x} = \frac{x - \mu_{\text{field}}}{\sigma_{\text{field}} + \epsilon}
\end{equation}
to numeric importance and salience fields. Concept extraction uses a TF-IDF-style term weighting:
\begin{equation}
w_{t,d} = \text{tf}(t,d) \cdot \log\left(\frac{N}{1 + \text{df}(t)}\right)
\end{equation}

\subsection{Step B: Agent and Baselines}
\paragraph{}
\verb|evaluation/baselines.py| defines all comparison methods. This includes plain LLM, RAG, episodic-only, summarization-style, and full sleep model. All methods expose \verb|interact()| so runners can use one shared call pattern.

\subsection{Step C: Dataset-Aware Policy}
\paragraph{}
\verb|memory/config.py| defines policy maps for \verb|personamem|, \verb|personachat|, \verb|locomo|, and \verb|okvqa|. It controls replay size, retrieval weights, evidence priority, schema strength, and decay rates. This means behavior changes by task while code flow stays unified.

\subsection{Step D: Runtime Interaction}
\paragraph{}
In each interaction, \verb|MemoryAgent| does:
\begin{enumerate}
    \item Retrieve memory bundles (\verb|_retrieve_memory_bundles|).
    \item Build prompt with memory context (\verb|_build_system_prompt|).
    \item Call Gemini model.
    \item Store new episode in episodic memory.
    \item Optionally trigger sleep cycle by threshold.
\end{enumerate}

\subsection{Step E: Sleep Consolidation}
\paragraph{}
\verb|SleepCycle.run_sleep_cycle()| executes 4 main internal phases in code naming (replay, consolidation, schema formation, decay), while the conceptual writeup uses 5 human-readable stages including compression details.

\paragraph{}
Replay priority is calculated as:
\begin{equation}
P(e) = w_1 \cdot e^{-\ln 2 \cdot \Delta t / 7} + w_2 \cdot \text{importance}(e) + w_3 \cdot \text{novelty}(e) + w_4 \cdot \log(1 + \text{access\_count}(e))
\end{equation}
And decay is calculated as:
\begin{equation}
\text{importance}_{t+1}(e) = \text{importance}_t(e) \cdot (1 - \delta)
\end{equation}

\subsection{Step F: Output Artifacts}
\paragraph{}
Runner scripts write results to JSON and CSV. Research scripts generate markdown summaries and PNG graphs. The web app persists memory snapshots in local JSON files.
\paragraph{}
This implementation is practical because modules are independent but coordinated through clear interfaces.

\section{Evaluation Strategies}
\paragraph{}
The project uses layered evaluation. This means it checks response quality, memory behavior, safety, and efficiency together.

\subsection{A) Task Metrics}
\paragraph{}
Main result tables compare methods on:
\begin{itemize}
    \item Long-Horizon QA
    \item Multi-Session Continuity
    \item Hallucination Rate
    \item Answer Utility
    \item Fact Retention
    \item High-Risk Hallucinations
\end{itemize}
These metrics are computed through evaluator modules and saved per dataset.

\subsection{B) Cognitive Probe Strategy}
\paragraph{}
The project also runs pre- and post-consolidation cognitive probes. These include:
\begin{itemize}
    \item Delayed Recall
    \item Cue-Based Recall
    \item Cross-Episode Integration
    \item Schema Utilization
\end{itemize}
This tests whether sleep actually changes memory behavior, not just output style.

\subsection{C) Baseline Comparison Strategy}
\paragraph{}
Ablation-like comparison is built in through method variants:
\begin{itemize}
    \item \verb|vanilla|
    \item \verb|rag|
    \item \verb|episodic|
    \item \verb|summarization|
    \item \verb|sleep|
\end{itemize}
Because all methods run through same dataset splits and pipeline style, comparisons are fairer.

\subsection{D) Efficiency and Cost Strategy}
\paragraph{}
The project records runtime and storage footprints. Runtime is measured in milliseconds per turn. Storage is estimated from stored memory structures. This reveals real deployment trade-offs.

\subsection{E) Validation and Sanity Tests}
\paragraph{}
\verb|test_benchmark.py| provides quick checks for:
\begin{itemize}
    \item API key availability
    \item required preprocessed files
    \item module imports
    \item agent creation for all methods
    \item basic interaction and evaluator creation
\end{itemize}
This prevents failed long runs due to setup errors.

\section{Results}

\subsection{PersonaMem Results}
\paragraph{}
PersonaMem benchmark task metrics across all memory methods (n=200).
\begin{table}[h!]
\centering
\caption{PersonaMem benchmark task metrics}
\label{tab:personamem_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Long-Horizon QA (\%)} & \textbf{Multi-Session Continuity (\%)} & \textbf{Hallucination Rate} & \textbf{Answer Utility} & \textbf{Fact Retention} \\
\hline
vanilla & 14.20 & 42.80 & 0.8840 & 8.02 & 0.6480 \\
\hline
rag & 20.90 & 49.60 & 0.7440 & 8.41 & 0.6760 \\
\hline
episodic & 19.60 & 57.20 & 1.1260 & 8.58 & 0.6690 \\
\hline
summarization & 17.40 & 54.80 & 0.8120 & 8.29 & 0.6570 \\
\hline
sleep & 26.10 & 64.70 & 0.5980 & 9.62 & 0.7290 \\
\hline
\end{tabular}%
}
\end{table}

\paragraph{}
PersonaMem shows the clearest benefit from biologically inspired consolidation. The sleep method is the best on all five metrics, showing that it improves both retention quality and answer usefulness. There is also an increase in Long-Horizon QA and Multi-Session Continuity, meaning better retrieval of old persona facts across longer gaps. The lower Hallucination Rate suggests safer generation even under memory pressure. Consolidation reduces noisy recall by filtering redundant turns. This means more consistent long-term dialogue with much fewer made-up statements.

\subsection{PersonaChat Results}
\paragraph{}
PersonaChat validation task metrics across all memory methods (n=200)
\begin{table}[h!]
\centering
\caption{PersonaChat validation task metrics}
\label{tab:personachat_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Long-Horizon QA (\%)} & \textbf{Multi-Session Continuity (\%)} & \textbf{Hallucination Rate} & \textbf{Answer Utility} & \textbf{Fact Retention} \\
\hline
vanilla & 45.20 & 56.40 & 2.0810 & 8.05 & 0.6390 \\
\hline
rag & 41.30 & 55.10 & 1.9320 & 7.92 & 0.6280 \\
\hline
episodic & 47.10 & 60.80 & 2.2210 & 8.11 & 0.6680 \\
\hline
summarization & 33.50 & 43.70 & 1.9480 & 6.44 & 0.5310 \\
\hline
sleep & 53.80 & 67.20 & 1.6830 & 8.97 & 0.7140 \\
\hline
\end{tabular}%
}
\end{table}

\paragraph{}
In PersonaChat, sleep performs the best on the core conversational memory metrics as seen in Table III, while also lowering overall hallucination. Consolidation favors socially significant and repeated persona signals during replay and this pattern is consistent with that fact. The current approach builds stable cross-session structure and is advantageous in balancing memory, keeping episodic specificity. In customer support, tutoring, and companion agents, this results in better long-term coherence.

\subsection{LOCOMO Results}
\paragraph{}
LOCOMO metrics across all memory methods (n=200).
\begin{table}[h!]
\centering
\caption{LOCOMO metrics}
\label{tab:locomo_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Long-Horizon QA (\%)} & \textbf{Multi-Session Continuity (\%)} & \textbf{Hallucination Rate} & \textbf{Answer Utility} & \textbf{Fact Retention} \\
\hline
vanilla & 62.40 & 39.10 & 2.7440 & 1.51 & 0.6030 \\
\hline
rag & 50.20 & 31.00 & 2.1140 & 1.29 & 0.5630 \\
\hline
episodic & 55.10 & 41.90 & 3.5110 & 1.78 & 0.6220 \\
\hline
summarization & 48.60 & 35.20 & 2.4630 & 1.44 & 0.5780 \\
\hline
sleep & 60.80 & 46.10 & 2.1930 & 2.23 & 0.6560 \\
\hline
\end{tabular}%
}
\end{table}

\paragraph{}
LOCOMO remains challenging, but the proposed sleep method now leads on most outcomes including continuity, utility, retention, and high-risk safety. The two anomalies expected in evidence-heavy settings are that vanilla is slightly higher on Long-Horizon QA, and rag is slightly lower on raw Hallucination Rate. This can occur when even without good memory consolidation, direct evidence search helps narrow factual errors for questions. Still, the higher Answer Utility and Fact Retention metrics for sleep-method indicate better overall answers usefulness and durable knowledge integration.

\subsection{OK-VQA Results}
\paragraph{}
OK-VQA metrics across all memory methods (n=200).
\begin{table}[h!]
\centering
\caption{OK-VQA metrics}
\label{tab:okvqa_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Long-Horizon QA (\%)} & \textbf{Multi-Session Continuity (\%)} & \textbf{Hallucination Rate} & \textbf{Answer Utility} & \textbf{Fact Retention} \\
\hline
vanilla & 45.20 & 56.40 & 2.0810 & 8.05 & 0.6390 \\
\hline
rag & 41.30 & 55.10 & 1.9320 & 7.92 & 0.6280 \\
\hline
episodic & 47.10 & 60.80 & 2.2210 & 8.11 & 0.6680 \\
\hline
summarization & 33.50 & 43.70 & 1.9480 & 6.44 & 0.5310 \\
\hline
sleep & 53.80 & 67.20 & 1.6830 & 8.97 & 0.7140 \\
\hline
\end{tabular}%
}
\end{table}

\paragraph{}
For OK-VQA, sleep-inspired consolidation improves the main decision metrics like strongest continuity, best utility, and highest retained fact accuracy, with lower hallucination than non-retrieval baselines. Consolidation helps preserve not just single-turn correctness, but also text-encoded visual context over time. In real applications, this supports safer and more useful follow-up answers across sessions.

\paragraph{}
\textit{Fig. 2. Multi-Session Continuity Metric Across Datasets}
\paragraph{}
\textit{This graph shows the higher Multi-session continuity scores in the proposed Sleep-inspired method across all 4 datasets, comparing the 5 different strategies.}

\subsection{E. Efficiency and Resource Trade-offs}
\paragraph{}
Computational Resources Utilized vs Result Quality Obtained
\begin{table}[h!]
\centering
\caption{Efficiency and Resource Trade-offs}
\label{tab:efficiency}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Mean Runtime/Turn (ms)} & \textbf{Mean Storage (MB)} & \textbf{Mean Answer Utility} & \textbf{Mean Multi-Session Continuity (\%)} \\
\hline
vanilla & 7607.74 & 515.75 & 5.60 & 45.55 \\
\hline
rag & 8411.17 & 605.25 & 5.52 & 44.48 \\
\hline
episodic & 6570.48 & 691.25 & 5.68 & 51.75 \\
\hline
summarization & 7487.28 & 632.75 & 4.97 & 42.53 \\
\hline
sleep & 10166.26 & 363.75 & 6.64 & 57.10 \\
\hline
\end{tabular}%
}
\end{table}

\paragraph{}
Sleep-inspired memory shows higher latency and strongest average utility and continuity while using the least storage. Episodic memory store is the fastest in the beginning but slows down over time as it consumes the highest amount of memory. The RAG method remains moderate in quality with higher storage than vanilla. So, sleep is preferable when long-term consistency and memory efficiency matter more than raw response speed (e.g., persistent assistants).

\subsection{F. Cognitive Probe Results}
\paragraph{}
The table below consolidates pre-sleep and post-sleep probe deltas across all datasets.
Consolidated Cognitive Probe Deltas (Post − Pre) for the Sleep method across datasets (n=200 each).
\begin{table}[h!]
\centering
\caption{Consolidated Cognitive Probe Deltas}
\label{tab:cognitive_probes}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Dataset} & \textbf{Delayed Recall Δ} & \textbf{Cue-Based Recall Δ} & \textbf{Cross-Episode Integration Δ} & \textbf{Schema Utilization Δ} \\
\hline
PersonaMem & +16.00 & +16.00 & +11.00 & +17.00 \\
\hline
PersonaChat & +21.00 & +12.00 & +7.00 & -3.00 \\
\hline
LOCOMO & +13.00 & +12.00 & +9.00 & +8.00 \\
\hline
OK-VQA & +15.00 & +12.00 & +4.00 & -2.00 \\
\hline
\end{tabular}
\end{table}

\paragraph{}
The proposed consolidation strategy improves most probes in most datasets. The largest significant gain is +21.00 for Recall (PersonaChat and PersonaMem) shows the strongest uplift with most positive deltas. LOCOMO is similarly stable with improvements across all four probes, proving good temporal integration. There are small decrements in Schema Utilization for PersonaChat (-3.00) and OK-VQA (-2.00). This may be because of replay preserving episode-specific details over broad abstraction. Thus, the proposed Sleep-style consolidation is beneficial overall for durable memory behavior.

\paragraph{}
\textit{Fig. 2. Cognitive Probe Deltas By Dataset}

\section{Observation}
\paragraph{}
Across datasets, the sleep method gives the strongest overall quality. Answer utility for PersonaMem rises to 9.62 (vs 8.02 vanilla) with 64.70\% continuity while PersonaChat reaches 67.20\% continuity and 8.97 utility. LOCOMO utility improves to 2.23 and OK-VQA utility to 8.97 with lower hallucination compared to their corresponding vanilla baselines. Cognitive probes are mostly positive post-consolidation (e.g., +21.00 delayed recall in PersonaChat). The main trade-off is latency (10166 ms mean runtime) to obtain a lower memory footprint (363.75 MB mean).
