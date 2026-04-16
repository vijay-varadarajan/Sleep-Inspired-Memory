\chapter{\MakeUppercase{Technical Specification}}

\section{Requirements}
\subsection{Functional Requirements}
\paragraph{}
This project must support a full memory lifecycle for an LLM agent. From the codebase, the main functional flow is clear: input comes in, the agent retrieves memory context, generates a response, stores a new episode, and then runs sleep consolidation when a threshold is reached. The \verb|MemoryAgent| in \verb|agent/agent.py| is the center of this behavior. It must accept user messages through \verb|interact()|, add optional persona context, and combine memory from three stores: episodic (\verb|memory/episodic.py|), consolidated (\verb|memory/consolidated.py|), and schema (\verb|memory/schema.py|). It must also keep recent chat history and expose retrieval bundles for evaluation.

\paragraph{}
The system must run different benchmark methods: \verb|vanilla|, \verb|rag|, \verb|episodic|, \verb|summarization|, and \verb|sleep|, as shown in \verb|benchmark_runner.py| and dataset-specific runner files. It must support dataset-aware memory policy using \verb|memory/config.py|, where settings change per dataset (\verb|personamem|, \verb|personachat|, \verb|locomo|, \verb|okvqa|). It must run evaluation metrics and save outputs as JSON and CSV in results folders. It must support preprocessed dataset input files and use split-based runs (\verb|train|, \verb|validation|, \verb|benchmark|).

\paragraph{}
The web demo in \verb|public/app.py| must provide chat APIs, memory APIs, and real-time process logs. It should persist memory snapshots to JSON files in \verb|public/data|. The system must provide manual and automatic sleep cycle behavior, contradiction-aware handling during retrieval, and memory summary outputs so users can inspect state. Finally, graph and report scripts in \verb|RESEARCH| must generate figures and tables from stored result values.

\subsection{Non-Functional Requirements}
\paragraph{}
The project has important non-functional needs because it is both a research system and a runnable demo. First, reliability: the agent should fail safely when API keys are missing or external calls fail. This is already visible in \verb|public/app.py|, where the app checks \verb|GOOGLE_API_KEY| and returns clear error messages. The same reliability principle should apply to benchmark runners, preprocessing scripts, and evaluation scripts.

\paragraph{}
Second, maintainability: the code should remain modular. The current structure supports this with separate folders (\verb|agent|, \verb|memory|, \verb|sleep|, \verb|evaluation|, \verb|public|, \verb|RESEARCH|). Changes to memory policy are isolated in \verb|memory/config.py|, and this should stay true for future updates. Third, reproducibility: results must be saved with timestamps and consistent file formats (\verb|.json|, \verb|.csv|, \verb|.md|, \verb|.png|). That is already used in the results folders and should remain required.

\paragraph{}
Fourth, performance and scalability: runtime is not minimal because sleep consolidation is heavier than basic methods, so the system should support batch-size limits, sample-size options, and threshold control. Runners already expose arguments like \verb|--num_samples| and split selection, and that should remain required for controlled experiments. Fifth, observability: logs and counters are needed for debugging and cost tracking. The project includes process logs and API counters (\verb|utils/api_counter.py|), and these should be enabled in all long runs.

\paragraph{}
Security and privacy are also required. No hard-coded API keys, no accidental dump of sensitive user text in public logs, and safe local persistence. The system should continue to use environment variables and avoid storing secrets in source files.

\subsection{Domain Requirements}
\paragraph{}
This project sits in the domain of long-term conversational memory for AI assistants. So the domain rules are not only software rules; they are memory-behavior rules. The system must preserve user-relevant facts across turns and across sessions, not only answer one prompt well. For this reason, the domain requires three memory levels: short episodic traces, compressed long-term memories, and schema-level abstractions. This is implemented directly in the code with separate stores and should not be merged into one flat memory list.

\paragraph{}
The domain also requires conflict-aware memory use. In real dialogue, facts can change and memories can disagree. The retrieval path in \verb|agent/agent.py| marks contradictory bundles, and prompt building includes conflict handling instructions. This is a core domain requirement because silent contradictions reduce trust.

\paragraph{}
The project must support different task types: social dialogue (PersonaChat), persona QA (PersonaMem), evidence-heavy long-context QA (LOCOMO), and multimodal knowledge QA (OK-VQA via text-encoded image context). Because of this diversity, dataset-aware policy tuning is required, not optional. In \verb|memory/config.py|, replay size, retrieval weights, evidence priority, decay rate, and abstraction strength are different by dataset. This is a direct domain need.

\paragraph{}
Another domain requirement is cognitive behavior measurement. The system must report not only accuracy-style metrics but also consolidation-related probes and safety indicators (hallucination patterns, high-risk claims, retention patterns). Finally, domain output should be human-readable for research communication: tables, narrative explanation, and clear graphs. The \verb|RESEARCH| folder and plotting scripts satisfy this requirement and should be maintained as first-class artifacts.

\section{Feasibility Study}
\subsection{Technical Feasibility}
\paragraph{}
The project is technically feasible and already largely implemented in modular form. The dependency stack in \verb|requirements.txt| is practical for current Python workflows: \verb|langchain|, \verb|langchain-google-genai|, \verb|google-generativeai|, \verb|numpy|, \verb|pydantic|, \verb|datasets|, \verb|faiss-cpu|, \verb|matplotlib|, and \verb|Pillow|. These libraries are widely used and stable enough for research and prototype deployments. The architecture is also feasible because responsibilities are separated: the agent handles orchestration, memory modules handle storage and search, sleep modules handle replay and consolidation, evaluators handle scoring, and runners handle experiment execution.

\paragraph{}
Data flow feasibility is good. Preprocessing scripts convert each dataset into structured files under \verb|*/preprocessed|, then runner scripts consume these files, then evaluators produce normalized result objects, and finally reporting scripts build tables and plots. This avoids tight coupling between raw datasets and inference code. The web app in \verb|public/app.py| further confirms practical integration feasibility by exposing REST endpoints and memory inspection.

\paragraph{}
Risks exist but are manageable. The biggest technical risk is external LLM dependence (network/API limits, pricing, quota). Another risk is evaluation variability due to stochastic generation. The code already uses controlled runner settings and repeatable artifacts, which helps. With proper environment management, logging, and test scripts (\verb|test_benchmark.py|, \verb|test_speed.py|), technical execution remains realistic on a single developer workstation.

\subsection{Economic Feasibility}
\paragraph{}
The project is economically feasible for research and prototype stages, with costs that can be controlled through sample size and method selection. Most software components are open-source and free to use locally. Hardware needs are moderate for preprocessing and evaluation logic because the heavy generation step is offloaded to hosted LLM APIs. This shifts spending from infrastructure purchase to pay-per-use model calls.

\paragraph{}
The codebase already supports practical cost controls. Runner scripts allow \verb|--num_samples|, and this can limit token spend during development. The system also supports baseline modes, so cheaper runs can be done first (\verb|vanilla| or small ablations), while full \verb|sleep| runs can be reserved for final reporting. API usage can be monitored with helper tools in \verb|utils/api_counter.py|, enabling per-run cost estimation and budget checks.

\paragraph{}
Storage costs are low. Results are mainly JSON/CSV/PNG and memory snapshots in local files. Even with many runs, this remains small compared to enterprise data systems. Labor cost is manageable because the project has clear modules and scripts, reducing debugging overhead.

\paragraph{}
Main economic risks are variable API pricing and longer runtime for sleep consolidation. Still, because runs are script-driven and configurable, teams can tune frequency, batch size, and sample count to keep monthly costs within budget. For an academic or startup environment, this is financially practical.

\subsection{Social Feasibility}
\paragraph{}
Social feasibility is strong because the project addresses a common user pain point: assistants that forget context across sessions. A system that remembers stable preferences, maintains conversation continuity, and reduces contradictory answers has clear user value in education, support, personal productivity, and companion-style applications. The sleep-inspired design also provides an understandable mental model for non-technical stakeholders: learn during interaction, organize memory offline, then respond better later.

\paragraph{}
The project can improve trust when implemented carefully. It includes conflict-aware prompting and hallucination-related evaluation, which are important for safer user-facing behavior. It also makes memory state more transparent in the demo app via logs and memory views. Transparency helps acceptance because users can see that memory is an explicit system feature, not hidden behavior.

\paragraph{}
There are social risks. Persistent memory can raise privacy concerns if users do not know what is stored. To be socially acceptable, deployments should include consent, clear retention policy, deletion controls, and visibility over stored data. The current file-based storage in the demo is simple, but production should add access control and better data governance.

\paragraph{}
Overall, the project is socially feasible if it keeps user agency central: explain memory use, allow opt-out, and avoid overclaiming certainty when memories conflict. These practices align with responsible AI expectations in real communities.

\section{System Specifications}
\subsection{Hardware Specs}
\paragraph{}
The project can run on a standard modern development machine because the core model inference is API-based. A practical minimum setup is: 4 CPU cores, 16 GB RAM, 20 GB free SSD storage, and stable internet access. This is enough for preprocessing, runner scripts, evaluation, and graph generation. For smoother experience, recommended specs are 8 CPU cores, 32 GB RAM, and 50+ GB SSD free space, especially when running multiple datasets and methods in sequence.

\paragraph{}
GPU is not required for current operation, since generation relies on hosted Gemini models. However, if future extensions add local embedding models or multimodal encoders, a GPU with 8–12 GB VRAM would help. Network quality matters more than GPU at present, because API latency affects runtime directly.

\paragraph{}
For the web demo (\verb|public/app.py|), hardware demand is low for single-user local testing. For small team demos, a server with 8 vCPU and 16 GB RAM is enough if requests are moderate. Disk IO needs are light because persistence is JSON-based. Backup storage should still be planned for result artifacts and experiment history.

\paragraph{}
In summary: this system is hardware-accessible for student and research labs. Better CPU and RAM reduce preprocessing and evaluation wait time, but no special accelerator is mandatory for the current codebase.

\subsection{Software Specs}
\paragraph{}
The software environment is Python-first and should be managed with a virtual environment (\verb|.venv|). Python 3.10+ is recommended; the current workspace already runs a newer Python release successfully. Required packages come from \verb|requirements.txt|: core LLM and orchestration (\verb|langchain|, \verb|langchain-google-genai|, \verb|google-generativeai|, \verb|langchain-community|), data utilities (\verb|numpy|, \verb|pydantic|, \verb|tqdm|, \verb|datasets|, \verb|pyarrow|), vector support (\verb|faiss-cpu|), plotting (\verb|matplotlib|), image utility (\verb|Pillow|), and optional env loading (\verb|python-dotenv|).

\paragraph{}
Runtime configuration needs environment variables, mainly \verb|GOOGLE_API_KEY|. Without it, the agent and web endpoints fail by design. The project should run on Linux, macOS, and Windows with minor path handling care. For web interface, Flask is used in \verb|public/app.py|, and templates/static assets are served from \verb|public/templates| and \verb|public/static|.

\paragraph{}
Execution entry points include dataset runners (\verb|personachat_runner.py|, \verb|locomo_runner.py|, \verb|okvqa_runner.py|, \verb|benchmark_runner.py|), preprocessing scripts, and research scripts (\verb|RESEARCH/GRAPHS_DRAFT.py|). Testing scripts (\verb|test_benchmark.py|, \verb|test_speed.py|) support quick checks. Recommended developer tools are VS Code, Git, and terminal shell with task automation.

\paragraph{}
For reproducible results, keep pinned dependency versions in a lock file, save timestamped outputs, and document run commands per dataset. This software stack is mature, portable, and aligned with the current project architecture.
