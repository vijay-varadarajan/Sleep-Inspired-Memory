## Observations and Inferences

Across datasets, the sleep method gives the strongest overall quality: PersonaMem answer utility rises to 9.62 (vs 8.02 vanilla) with 64.70% continuity; PersonaChat reaches 67.20% continuity and 8.97 utility; LOCOMO utility improves to 2.23; and OK-VQA utility to 5.74 with lower hallucination (5.286 vs 6.812 vanilla). Cognitive probes are mostly positive post-consolidation (e.g., +21 delayed recall in PersonaChat). The main trade-off remains latency (10,166 ms mean runtime) for substantially lower memory footprint (363.75 MB mean).

## Conclusion

### Discussion

The results suggest that biologically inspired consolidation is a practical design principle for persistent LLM agents. The three-tier architecture separates raw episodes from stable summaries and schemas, reducing redundancy and improving retrieval relevance across sessions. Compared with vanilla, RAG, episodic-only, and summarization baselines, the sleep approach provides better downstream utility in most datasets and stronger continuity under long-horizon interaction demands. Importantly, improvements are not limited to one benchmark type; they transfer from social dialogue to evidence-heavy and multimodal tasks. The combined pattern indicates that offline memory reorganization, not just larger context windows, is central to robust long-term agent performance and safer behavior over repeated interactions.

### Limitations and Future Extensions

Several limitations remain. First, higher average runtime indicates that consolidation introduces latency and scheduling complexity, especially when sleep cycles trigger frequently. Second, schema utilization declines in a few probe settings, suggesting that abstraction policies are still sensitive to dataset heterogeneity and modality mixing. Third, current multimodal memory relies on text-encoded image descriptions rather than native visual embeddings, which may cap cross-modal fidelity. Future work should include adaptive sleep triggering, learned replay policies, and dynamic abstraction strength by task type. Integrating CLIP-style visual memory, contradiction-aware schema revision, and larger-scale longitudinal evaluation would strengthen generalization, reduce anomalies, and improve deployment readiness in real-world persistent assistant systems.
