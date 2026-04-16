## Abstract

Large language model (LLM) agents excel at in-context reasoning but lack persistent memory across sessions. This limits their ability to build on prior interactions, track evolving user context, or abstract general knowledge from experience. Biological memory research offers a compelling analogy: during sleep, the hippocampus replays recent episodes and transfers structured representations to the neocortex through slow oscillations and spindle-mediated consolidation.

We draw directly on this mechanism to design a sleep-inspired memory consolidation framework for LLM agents. The system implements a three-tier memory architecture — episodic, consolidated, and schema — and a five-phase sleep cycle covering replay selection, compression, consolidation, schema formation, and decay.

We evaluate the framework against four baselines (vanilla LLM, RAG, episodic-only, summarization) across four datasets: PersonaChat, PersonaMem, LOCOMO, and OK-VQA. Under the updated evaluation setting ($n=200$ per dataset), the sleep agent achieves an answer utility of 9.62 on PersonaMem (versus 8.02 for the vanilla baseline), 67.20% multi-session continuity on PersonaChat, and a +21.00 delayed-recall gain in post-consolidation cognitive probes. These results suggest that structured offline consolidation — modeled after the sleeping brain — can substantially improve long-horizon agent memory.

---

## Keywords

memory consolidation, LLM agents, sleep-inspired AI, episodic memory, hippocampal replay, schema formation, multimodal memory, retrieval-augmented generation, long-term memory, personalized conversational agents

---

## 1. Introduction

LLM agents deployed in open-ended, multi-session environments cannot reliably remember what happened in prior conversations. Each new session begins from a blank context window. This is not merely a storage problem — it is a structural one: there is no mechanism to transform raw episodic traces into durable, generalized knowledge over time. The biological brain solves a closely related problem during sleep. Hippocampal replay events, orchestrated by slow oscillations and sleep spindles during slow-wave sleep, gradually transfer memories from fast-binding hippocampal circuits to stable neocortical representations. The result is not just retention — it is abstraction, with sequential structure and gist preserved over perceptual detail. This paper proposes a framework that maps this process onto an LLM agent architecture. We implement a three-tier memory store and a five-phase consolidation cycle, then evaluate the full system on four benchmark datasets spanning dialogue, question answering, and multimodal reasoning. The goal is to determine whether sleep-cycle-structured offline consolidation improves agent memory performance across diverse task types.

---

## 2. Motivation and Scope

Existing approaches to LLM agent memory each address part of the problem but leave important gaps. Retrieval-augmented generation (RAG) enables access to large memory stores but treats all retrieved chunks equally, without accounting for recency, relevance weighting, or accumulated redundancy. Summarization methods compress context but discard episodic structure and cannot recover fine-grained details on demand. Episodic-only systems store raw interaction logs but lack any mechanism for offline reorganization — memories accumulate without being refined, abstracted, or pruned. None of these approaches model the transition from episodic trace to schematic knowledge that biological consolidation achieves.

A sleep-inspired approach is motivated by the robustness of hippocampal-neocortical transfer as a memory design principle. Replay-based consolidation prioritizes emotionally or contextually salient events, resolves contradictions between overlapping memories, and builds schema-level representations that generalize across episodes. These are precisely the operations that agent memory systems lack.

This paper scopes its evaluation to four datasets — PersonaChat, PersonaMem, LOCOMO, and OK-VQA — covering conversational personalization, long-horizon recall, and multimodal question answering, compared across five methods.

---

## 3. Literature Review

Research on biological memory consolidation and AI agent memory systems collectively motivates the sleep-inspired design explored here. Brodt et al. [1] established that slow-wave sleep drives active memory consolidation through hippocampal replay, sleep spindles, and slow oscillations, though questions remain about whether sleep also supports abstraction of gist-level memory. Squire et al. [2] described systems consolidation as a process in which the hippocampus guides neocortical reorganization over time, with transfer rates modulated by how well new information relates to prior knowledge structures. Bendor and Wilson [3] demonstrated that hippocampal replay can be biased by external cues during sleep to selectively reinforce specific memories. Diamond et al. [4] showed that sleep actively transforms memory rather than merely preserving it, boosting sequential and organizational structure while allowing perceptual detail to fade — with effects persisting over days to years. Spens and Burgess [12] modeled the hippocampus as a fast teacher that encodes episodes for replay to a generative neocortical student, producing schematic representations through iterative consolidation.

On the AI side, Pink et al. [5] argued that LLM agents fundamentally need episodic memory with mechanisms for segmentation, retrieval, and periodic consolidation into base parameters. Du et al. [6] categorized agent memory operations — including consolidation, forgetting, and condensation — and identified unified evaluation and multi-source consistency as open challenges. Hu et al. [8] surveyed agent memory across forms and functions, projecting offline consolidation analogous to biological sleep as the next major architectural development. Kagaya et al. [7] demonstrated that retrieving past multimodal experiences can improve agent planning in embodied settings. Jiang et al. [9] proposed a spreading activation memory graph achieving strong performance on LOCOMO, but without explicit consolidation or schema abstraction. Sarin et al. [10] combined session summarization with knowledge graphs for personalization, but without offline consolidation or multimodal support. Hassell et al. [11] showed that episodic memory with critique-driven reflection improves classification accuracy by up to 24.8% over RAG-style baselines.

### Summary of Related Work

| Author(s) & Year | Approach | Key Contribution | Limitation / Gap |
|---|---|---|---|
| Brodt et al. (2023) [1] | Neuroscience review | Sleep drives memory consolidation via hippocampal replay and slow oscillations | Gist/abstraction mechanisms not fully characterized |
| Squire et al. (2015) [2] | Systems consolidation theory | Hippocampus reorganizes neocortical memory; rate depends on prior knowledge | Does not address AI applications |
| Bendor & Wilson (2012) [3] | Targeted memory reactivation | Hippocampal replay is steerable via external cues during sleep | Limited to rodent models |
| Diamond et al. (2025) [4] | Longitudinal human memory study | Sleep boosts sequential structure over detail; effects persist years | No computational model proposed |
| Spens & Burgess (2024) [12] | Generative consolidation model | Hippocampus-as-teacher with generative neocortical student via replay | Theoretical; not validated on LLM agents |
| Pink et al. (2025) [5] | LLM agent memory survey | Episodic memory with consolidation identified as critical missing piece | No implementation or benchmark results |
| Du et al. (2025) [6] | Memory operations taxonomy | Categorizes consolidation, forgetting, retrieval in LLM agents | Evaluation remains fragmented across tasks |
| Hu et al. (2026) [8] | Agent memory survey | Projects offline sleep-like consolidation as the next major stage | Prospective; no system proposed |
| Kagaya et al. (2024) [7] | RAP: retrieval-augmented planning | Past multimodal experiences improve embodied agent planning | No offline consolidation or schema formation |
| Jiang et al. (2026) [9] | SYNAPSE: spreading activation | Associative memory graph achieves strong LOCOMO results | No consolidation cycle or schema abstraction |
| Sarin et al. (2025) [10] | Memoria framework | Session summaries + knowledge graphs for personalization | No offline consolidation; no multimodal support |
| Hassell et al. (2025) [11] | Episodic memory with critique | Up to 24.8% accuracy gain over RAG baselines via reflection | No sleep cycle; limited to classification tasks |

### Summary of Related Work - Latex

\begin{table}[h!]
\centering
\caption{Comparison of Existing Approaches}
\label{tab:related_work}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|p{5cm}|p{5cm}|}
\hline
\textbf{Author(s)} & \textbf{Key Contribution} & \textbf{Limitation / Gap} \\
\hline
Brodt et al. (2023) [1] & Sleep drives memory consolidation via hippocampal replay & Abstraction mechanisms not characterized \\
\hline
Squire et al. (2015) [2] & Memory reorganization rate depends on prior knowledge & Does not address AI applications \\
\hline
Bendor \& Wilson (2012) [3] & Hippocampal replay is steerable via external cues during sleep & Limited to rodent models \\
\hline
Diamond et al. (2025) [4] & Sleep boosts sequential structure over detail; effects persist years & No computational model proposed \\
\hline
Pink et al. (2025) [5] & Episodic memory with consolidation identified & No implementation or benchmark results \\
\hline
Du et al. (2025) [6] & Categorizes consolidation, forgetting and retrieval & Evaluation remains fragmented \\
\hline
Hu et al. (2026) [8] & Projects sleep consolidation as next major stage & Prospective; no system proposed \\
\hline
Kagaya et al. (2024) [7] & Past multimodal experiences improve agent planning & No consolidation cycle \\
\hline
Jiang et al. (2026) [9] & Associative memory graph achieves strong results & No schema abstraction \\
\hline
Sarin et al. (2025) [10] & Session summaries + knowledge graphs for personalization & no multimodal support \\
\hline
Hassell et al. (2025) [11] & Up to 24.8\% accuracy gain over RAG baselines & No sleep cycle; limited to classification tasks \\
\hline
\end{tabular}%
}
\end{table}

### References

[1] Brodt, S. et al. (2023). "Sleep—A brain-state serving systems memory consolidation." *Neuron*. PMID: 37023710.

[2] Squire, L.R. et al. (2015). "Memory Consolidation." *PMC*. PMID: 26238360.

[3] Bendor, D. & Wilson, M.A. (2012). "Biasing the content of hippocampal replay during sleep." *PMC*. PMID: 22941111.

[4] Diamond, N.B. et al. (2025). "Sleep selectively and durably enhances memory for the sequence of real-world experiences." *Nature Human Behaviour*. PMID: 40069368.

[5] Pink, M. et al. (2025). "Episodic Memory is the Missing Piece for Long-Term LLM Agents." arXiv:2502.06975.

[6] Du, Y. et al. (2025). "Rethinking Memory in LLM-based Agents." arXiv:2505.00675.

[7] Kagaya, T. et al. (2024). "RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents." arXiv:2402.03610.

[8] Hu, Y. et al. (2026). "Memory in the Age of AI Agents: A Survey." arXiv:2512.13564.

[9] Jiang, H. et al. (2026). "SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation." arXiv:2601.02744.

[10] Sarin, S. et al. (2025). "Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI." *IEEE*.

[11] Hassell, J. et al. (2025). "Learning from Supervision with Semantic and Episodic Memory." arXiv:2510.19897.

[12] Spens, E. & Burgess, N. (2024). "A generative model of memory construction and consolidation." *Nature Human Behaviour*.

---

## 4. Gaps Identified

Prior work on LLM agent memory lacks offline consolidation mechanisms — no system runs a structured sleep-like cycle between sessions to reorganize stored information. Existing AI memory architectures do not model the episodic-to-schema transition that biological consolidation achieves: raw experiences accumulate without being abstracted into reusable knowledge. No prior system applies structured sleep cycles to multimodal memory, leaving visual and textual information siloed. Memory policies are rarely adapted to dataset characteristics — a single retrieval or compression strategy is applied regardless of task type. Finally, conflict detection and contradiction-aware memory updating are absent from current agent memory systems, producing inconsistent long-term representations that degrade over repeated sessions.
