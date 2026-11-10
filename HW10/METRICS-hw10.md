# METRICS — Improving RAG with Reranking, Chunking, and Conversational Memory

**Course:** DATA 266 · **Author:** Ritika Mukesh Neema · **Notebook:** `DATA266_RAG_Reranking_Chunking_Memory_final.ipynb`
**Data:** `ccdv/arxiv-summarization`, `document` subset, train split, first 100 papers · **SEED:** 6638 · **Runtime:** Google Colab, CUDA GPU (T4)

All values are means over queries. K = 5 for P@K/R@K unless noted, plus R@10. Relevance is set by a manually defined rubric: the chunk comes from the target paper, contains a required term (detail queries), and contains enough hand-picked answer terms.

## 1. Setup

| Item | Value |
|---|---|
| Documents | 100 (mean 4,937 words; min 446; max 24,354) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-d, 256-token limit) |
| Vector DB | ChromaDB, persistent, cosine HNSW |
| Baseline reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-20 candidate pool |
| Improved reranker | `BAAI/bge-reranker-base`, top-40 candidate pool |
| Queries | 8 single-turn (Q1–Q4 topic, Q5–Q8 detail) + 3 two-turn conversations (C1, C2 dependent; C3 topic switch) |

### Chunking strategies

| Strategy | Chunks | Mean words | Min | Max | Ends on sentence | Mean tokens | Chunks > 256 tokens |
|---|---|---|---|---|---|---|---|
| S1 fixed window (120 words, 24 overlap) | 5,164 | 120.0 | 120 | 120 | 7% | 187 | 7.8% |
| S2 sentence-aware (≤120 words, 1-sentence overlap) | 5,723 | 106.3 | 8 | 120 | 98% | 165 | 5.7% |

### Relevant chunks per query (rubric)

| Query | Level | Target doc | Relevant S1 | Relevant S2 |
|---|---|---|---|---|
| Q1 learning rates, additive models | topic | 0 | 28 | 31 |
| Q2 phase diffuser, scintillations | topic | 3 | 23 | 24 |
| Q3 synaptic memory | topic | 7 | 15 | 18 |
| Q4 radial velocity, planets | topic | 9 | 31 | 34 |
| Q5 Gaussian in multivariate RBF RKHS | detail | 0 | 7 | 7 |
| Q6 slow phase variations, detector time | detail | 3 | 5 | 5 |
| Q7 short-term → long-term plasticity | detail | 7 | 3 | 5 |
| Q8 linear RV trends, outer companion | detail | 9 | 2 | 2 |
| C1-T2 "results … for it" | follow-up | 3 | 9 | 7 |
| C2-T2 "… were they obtained with" | follow-up | 9 | 12 | 11 |

Q8 has only 2 relevant chunks, so its P@5 is capped at 0.40 for every method.

## 2. Without vs with reranking (dense retrieval, MiniLM reranker)

| Configuration | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| S1, no rerank | 0.725 | 0.433 | 0.544 | 0.938 |
| S1, rerank | 0.775 | 0.483 | 0.619 | 1.000 |
| S2, no rerank | 0.725 | 0.385 | 0.588 | 1.000 |
| S2, rerank | **0.800** | 0.453 | **0.638** | 1.000 |

### By query level

| Level | Configuration | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| detail | S1, no rerank | 0.50 | 0.660 | 0.660 | 0.875 |
| detail | S1, rerank | 0.55 | 0.743 | 0.793 | 1.000 |
| detail | S2, no rerank | 0.45 | 0.571 | 0.793 | 1.000 |
| detail | S2, rerank | 0.60 | 0.707 | 0.879 | 1.000 |
| topic | S1, no rerank | 0.95 | 0.206 | 0.429 | 1.000 |
| topic | S1, rerank | 1.00 | 0.223 | 0.445 | 1.000 |
| topic | S2, no rerank | 1.00 | 0.199 | 0.383 | 1.000 |
| topic | S2, rerank | 1.00 | 0.199 | 0.397 | 1.000 |

## 3. Chunking strategy 1 vs 2 (all Q1–Q8)

| Reranking | Strategy | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| no | S1 fixed window | 0.725 | **0.433** | 0.544 | 0.938 |
| no | S2 sentence-aware | 0.725 | 0.385 | **0.588** | **1.000** |
| yes | S1 fixed window | 0.775 | **0.483** | 0.619 | 1.000 |
| yes | S2 sentence-aware | **0.800** | 0.453 | **0.638** | 1.000 |

## 4. Without vs with conversational memory (concatenation memory, dependent turns C1-T2 + C2-T2)

| Strategy | Reranking | Memory | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| S1 | no | off | 0.0 | 0.000 | 0.042 | 0.071 |
| S1 | no | on | 0.5 | 0.264 | 0.472 | 0.667 |
| S1 | yes | off | 0.0 | 0.000 | 0.000 | 0.000 |
| S1 | yes | on | **0.6** | 0.306 | **0.597** | **1.000** |
| S2 | no | off | 0.0 | 0.000 | 0.000 | 0.000 |
| S2 | no | on | 0.5 | 0.305 | 0.539 | 1.000 |
| S2 | yes | off | 0.0 | 0.000 | 0.045 | 0.083 |
| S2 | yes | on | **0.6** | **0.325** | 0.558 | 0.750 |

## 5. Improvements (Section 11)

### 5a. Retrieval pipelines — all queries (Q1–Q8)

| Strategy | Pipeline | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| S1 | Dense | 0.725 | 0.433 | 0.544 | 0.938 |
| S1 | Dense + MiniLM (baseline) | 0.775 | 0.483 | 0.619 | 1.000 |
| S1 | Hybrid | 0.725 | 0.423 | **0.644** | 0.875 |
| S1 | Hybrid + MiniLM | **0.800** | **0.508** | 0.638 | **1.000** |
| S1 | Hybrid + BGE (pool 40) | 0.650 | 0.416 | 0.628 | 0.875 |
| S2 | Dense | 0.725 | 0.385 | 0.588 | 1.000 |
| S2 | Dense + MiniLM (baseline) | **0.800** | 0.453 | **0.638** | 1.000 |
| S2 | Hybrid | 0.775 | 0.428 | **0.638** | 1.000 |
| S2 | Hybrid + MiniLM | **0.800** | 0.453 | 0.631 | 1.000 |
| S2 | Hybrid + BGE (pool 40) | **0.800** | **0.457** | 0.631 | 1.000 |

### 5b. Retrieval pipelines — detail queries (Q5–Q8)

| Strategy | Pipeline | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| S1 | Dense | 0.50 | 0.660 | 0.660 | 0.875 |
| S1 | Dense + MiniLM (baseline) | 0.55 | 0.743 | 0.793 | 1.000 |
| S1 | Hybrid | 0.45 | 0.624 | 0.843 | 0.750 |
| S1 | Hybrid + MiniLM | **0.60** | **0.793** | 0.864 | **1.000** |
| S1 | Hybrid + BGE (pool 40) | 0.50 | 0.660 | **0.879** | 0.875 |
| S2 | Dense | 0.45 | 0.571 | 0.793 | 1.000 |
| S2 | Dense + MiniLM (baseline) | 0.60 | 0.707 | 0.879 | 1.000 |
| S2 | Hybrid | 0.55 | 0.657 | **0.893** | 1.000 |
| S2 | Hybrid + MiniLM | 0.60 | 0.707 | 0.879 | 1.000 |
| S2 | Hybrid + BGE (pool 40) | **0.65** | **0.729** | 0.879 | 1.000 |

### 5c. Memory modes — dependent follow-ups (C1-T2, C2-T2)

| Strategy | Pipeline | Memory | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| S1 | Dense + MiniLM | none | 0.0 | 0.000 | 0.000 | 0.000 |
| S1 | Dense + MiniLM | concat | 0.6 | 0.306 | 0.597 | 1.000 |
| S1 | Dense + MiniLM | anchored | **0.7** | **0.347** | **0.639** | **1.000** |
| S1 | Hybrid + BGE | none | 0.1 | 0.042 | 0.042 | 0.100 |
| S1 | Hybrid + BGE | concat | 0.6 | 0.292 | 0.444 | 0.750 |
| S1 | Hybrid + BGE | anchored | **0.7** | **0.347** | 0.486 | **1.000** |
| S2 | Dense + MiniLM | none | 0.0 | 0.000 | 0.045 | 0.083 |
| S2 | Dense + MiniLM | concat | 0.6 | 0.325 | 0.558 | 0.750 |
| S2 | Dense + MiniLM | anchored | 0.6 | 0.325 | 0.558 | **1.000** |
| S2 | Hybrid + BGE | none | 0.0 | 0.000 | 0.045 | 0.050 |
| S2 | Hybrid + BGE | concat | 0.3 | 0.188 | 0.468 | 0.375 |
| S2 | Hybrid + BGE | anchored | 0.5 | 0.279 | 0.513 | 0.750 |

### 5d. Memory modes — topic switch (C3-T2: diffuser question, then an unrelated synaptic-memory question)

| Strategy | Pipeline | Memory | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| S1 | Dense + MiniLM | none | 1.0 | 0.333 | 0.667 | 1.000 |
| S1 | Dense + MiniLM | concat | 0.0 | 0.000 | 0.067 | 0.125 |
| S1 | Dense + MiniLM | anchored | 1.0 | 0.333 | 0.667 | 1.000 |
| S1 | Hybrid + BGE | none | 0.6 | 0.200 | 0.533 | 0.500 |
| S1 | Hybrid + BGE | concat | 0.0 | 0.000 | 0.000 | 0.000 |
| S1 | Hybrid + BGE | anchored | 0.6 | 0.200 | 0.533 | 0.500 |
| S2 | Dense + MiniLM | none | 1.0 | 0.278 | 0.556 | 1.000 |
| S2 | Dense + MiniLM | concat | 0.0 | 0.000 | 0.056 | 0.100 |
| S2 | Dense + MiniLM | anchored | 1.0 | 0.278 | 0.556 | 1.000 |
| S2 | Hybrid + BGE | none | 0.8 | 0.222 | 0.500 | 1.000 |
| S2 | Hybrid + BGE | concat | 0.0 | 0.000 | 0.056 | 0.167 |
| S2 | Hybrid + BGE | anchored | 0.8 | 0.222 | 0.500 | 1.000 |

## 6. Headline results

| Comparison | Result |
|---|---|
| Reranking (S2, all queries) | P@5 0.725 → 0.800, R@10 0.588 → 0.638 |
| Reranking (S2, detail queries) | P@5 0.45 → 0.60, R@10 0.793 → 0.879 |
| Chunking, with reranking (S1 → S2) | P@5 0.775 → 0.800, R@10 0.619 → 0.638, MRR 1.000 both |
| Memory (concat), reranked | P@5 0.0 → 0.6, MRR ≤ 0.083 → 0.75–1.00 |
| Anchored vs concat memory | Equal or better in all 4 configs; MRR 1.000 in 3 of 4 |
| Topic switch | Concat drops P@5 to 0.0 in all 4 configs; anchored equals no-memory in all 4 |
| Best detail-query P@5 | S2 + Hybrid + BGE: 0.65 (baseline 0.60) |
| Recommended overall | S2 + Dense + MiniLM rerank + anchored memory |
