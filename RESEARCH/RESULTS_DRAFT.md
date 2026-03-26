# Results

## 1. PersonaMem Benchmark Results

*Split: benchmark | Samples: 50 | Timestamp: 20260326_103644*

| Method | Long-Horizon QA (%) | Multi-Session Continuity (%) | Hallucination Rate | Answer Utility | Retrieval Recall@3 | Retrieval MRR | Retrieval nDCG@5 | Evidence Hit Rate | Fact Retention | Preference Retention | Contradiction Rate | Unsupported Claim Proportion | High-Risk Hallucinations | Avg Runtime/Turn (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 0.00 | 10.00 | 0.5401 | 9.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5777 | 1.00 | 0.00 | 0.1823 | 0.00 | 9513.28 |
| rag | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| episodic | 0.00 | 60.00 | 1.1794 | 9.74 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5930 | 1.00 | 0.00 | 0.1316 | 0.00 | 9700.92 |
| summarization | 0.00 | 60.00 | 0.6637 | 9.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5568 | 1.00 | 0.00 | 0.1315 | 0.50 | 9955.85 |
| sleep | 0.00 | 50.00 | 0.5388 | 11.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5657 | 1.00 | 0.00 | 0.1371 | 0.00 | 14017.94 |

*Table 1.1: PersonaMem benchmark task metrics across all memory methods (n=50).*

---

## 2. PersonaChat Results

*Split: validation | Samples: 50 | Timestamp: 20260326_103244*

| Method | Long-Horizon QA (%) | Multi-Session Continuity (%) | Hallucination Rate | Answer Utility | Retrieval Recall@3 | Retrieval MRR | Retrieval nDCG@5 | Evidence Hit Rate | Fact Retention | Preference Retention | Contradiction Rate | Unsupported Claim Proportion | High-Risk Hallucinations | Avg Runtime/Turn (ms) | Avg Response Length (words) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 38.10 | 49.29 | 2.6401 | 8.47 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5862 | 0.9048 | 0.00 | 0.2019 | 0.3333 | 4267.46 | 37.14 |
| rag | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.7143 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| episodic | 42.86 | 50.00 | 2.6721 | 7.79 | 0.00 | 0.00 | 0.00 | 0.00 | 0.6011 | 0.9762 | 0.00 | 0.2764 | 0.0952 | 4774.03 | 37.62 |
| summarization | 23.81 | 3.57 | 2.8818 | 4.51 | 0.00 | 0.00 | 0.00 | 0.00 | 0.2977 | 0.7619 | 0.00 | 0.1701 | 0.0476 | 4962.60 | 18.38 |
| sleep | 33.33 | 53.57 | 2.7110 | 7.14 | 0.9048 | 0.7238 | 0.7519 | 0.00 | 0.5826 | 0.8810 | 0.00 | 0.2565 | 0.4286 | 12277.42 | 36.29 |

*Table 2.1: PersonaChat validation task metrics across all memory methods (n=50).*

---

## 3. LOCOMO Results

*Split: validation | Samples: 50 | Timestamp: 20260326_122902*

| Method | Long-Horizon QA (%) | Multi-Session Continuity (%) | Hallucination Rate | Answer Utility | Retrieval Recall@3 | Retrieval MRR | Retrieval nDCG@5 | Evidence Hit Rate | Fact Retention | Preference Retention | Contradiction Rate | Unsupported Claim Proportion | High-Risk Hallucinations | Avg Runtime/Turn (ms) | Avg Response Length (words) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 66.67 | 30.00 | 2.6882 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5714 | 1.00 | 0.00 | 0.3472 | 0.3333 | 11127.53 | 26.33 |
| rag | 33.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.6667 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| episodic | 33.33 | 33.33 | 4.9107 | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5714 | 1.00 | 0.00 | 0.3333 | 0.3333 | 6400.84 | 28.00 |
| summarization | 33.33 | 6.67 | 1.5831 | 0.26 | 0.00 | 0.00 | 0.00 | 0.00 | 0.5198 | 1.00 | 0.00 | 0.5333 | 0.3333 | 9793.66 | 31.00 |
| sleep | 0.00 | 0.00 | 6.9444 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.4008 | 0.6667 | 0.00 | 0.3333 | 0.3333 | 8320.10 | 22.33 |

*Table 3.1: LOCOMO validation task metrics across all memory methods (n=50).*

---

## 4. OK-VQA Results

*Split: validation | Samples: 50 | Timestamp: 20260326_140441*

| Method | Long-Horizon QA (%) | Multi-Session Continuity (%) | Hallucination Rate | Answer Utility | Retrieval Recall@3 | Retrieval MRR | Retrieval nDCG@5 | Evidence Hit Rate | Fact Retention | Preference Retention | Contradiction Rate | Unsupported Claim Proportion | High-Risk Hallucinations | Avg Runtime/Turn (ms) | Avg Response Length (words) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 50.00 | 36.67 | 7.3227 | 4.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.4285 | 0.8889 | 0.00 | 0.75 | 0.00 | 7082.07 | 20.50 |
| rag | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.8889 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| episodic | 55.56 | 44.44 | 9.7345 | 2.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.4328 | 0.8889 | 0.00 | 0.7222 | 0.00 | 5359.66 | 18.61 |
| summarization | 33.33 | 22.22 | 8.5714 | 1.34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.2446 | 0.8889 | 0.00 | 0.4167 | 0.0556 | 6266.59 | 9.72 |
| sleep | 33.33 | 38.89 | 12.9534 | 5.48 | 0.0556 | 0.0389 | 0.0445 | 0.00 | 0.4155 | 0.8889 | 0.00 | 0.7222 | 0.00 | 13871.52 | 10.72 |

*Table 4.1: OK-VQA validation task metrics across all memory methods (n=50).*

---

## 5. Cognitive Probe Results (Pre/Post Sleep Consolidation)

### PersonaMem

| Probe | Pre-Consolidation | Post-Consolidation | Delta |
|---|---|---|---|
| Delayed Recall | 30.00 | 30.00 | 0.00 |
| Cue-Based Recall | 10.00 | 0.00 | -10.00 |
| Cross-Episode Integration | 100.00 | 100.00 | 0.00 |
| Schema Utilization | 0.00 | 0.00 | 0.00 |

*Table 5.1: Sleep method cognitive probe scores on PersonaMem (n=50).*

### PersonaChat

| Probe | Pre-Consolidation | Post-Consolidation | Delta |
|---|---|---|---|
| Delayed Recall | 44.29 | 96.43 | +52.14 |
| Cue-Based Recall | 55.14 | 47.57 | -7.57 |
| Cross-Episode Integration | 100.00 | 84.29 | -15.71 |
| Schema Utilization | 0.00 | 0.00 | 0.00 |

*Table 5.2: Sleep method cognitive probe scores on PersonaChat (n=50).*

### LOCOMO

| Probe | Pre-Consolidation | Post-Consolidation | Delta |
|---|---|---|---|
| Delayed Recall | 0.00 | 0.00 | 0.00 |
| Cue-Based Recall | 0.00 | 0.00 | 0.00 |
| Cross-Episode Integration | 100.00 | 100.00 | 0.00 |
| Schema Utilization | 0.00 | 0.00 | 0.00 |

*Table 5.3: Sleep method cognitive probe scores on LOCOMO (n=50).*

### OK-VQA

| Probe | Pre-Consolidation | Post-Consolidation | Delta |
|---|---|---|---|
| Delayed Recall | 40.00 | 40.00 | 0.00 |
| Cue-Based Recall | 0.00 | 0.00 | 0.00 |
| Cross-Episode Integration | 86.00 | 84.00 | -2.00 |
| Schema Utilization | 0.00 | 40.00 | +40.00 |

*Table 5.4: Sleep method cognitive probe scores on OK-VQA (n=50).*

---

## 6. Cross-Dataset Comparison

| Dataset | Split | n | Answer Utility (Sleep) | Multi-Session Continuity % (Sleep) |
|---|---|---|---|---|
| PersonaMem | benchmark | 50 | 11.00 | 50.00 |
| PersonaChat | validation | 50 | 7.14 | 53.57 |
| LOCOMO | validation | 50 | 0.00 | 0.00 |
| OK-VQA | validation | 50 | 5.48 | 38.89 |

*Table 6.1: Sleep method Answer Utility and Multi-Session Continuity across all four datasets.*
