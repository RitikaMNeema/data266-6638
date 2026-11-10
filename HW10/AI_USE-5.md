# AI Use Disclosure

**Assignment:** DATA 266 — Improving RAG with Reranking, Chunking, and Conversational Memory
**Student:** Ritika Mukesh Neema
**AI tool used:** Claude (Anthropic), through the Claude app

## What the AI tool did
- **Notebook code:** wrote the Colab notebook: preprocessing, both chunking strategies, ChromaDB indexing, dense retrieval, cross-encoder reranking, the conversational memory class, the Precision@K / Recall@K / MRR evaluation, comparison tables, and charts.
- **Queries and relevance rubric:** proposed the 8 single-turn queries, the 3 two-turn conversations, and the term-based relevance rubric. Each query was tied to a target paper among the first 100 documents, and the notebook asserts that the paper is found.
- **Testing:** smoke-tested the notebook with stand-in models and synthetic data before each version. It had no access to Hugging Face, so it never ran on the real dataset.
- **Diagnosis between runs:** read the executed notebooks I returned and found that:
  - Run 1's single-turn queries were too easy (every method scored about 1.0).
  - Chunks were exceeding the embedding model's 256-token limit.
  - Concatenation memory carried turn-1 wording into the follow-up query.
  
  It then changed the notebook to address each of these (detail queries, 120-word chunks, the Section 11 improvements).
- **Improvements section:** designed and coded Section 11: hybrid BM25 + dense retrieval with RRF, the `bge-reranker-base` reranker, anchored memory, and the topic-switch test.
- **Write-ups:** drafted the discussion cells, `METRICS.md`, this `RUN_LOG.txt` / `AI_USE.md`, and the analysis report, using the numbers printed in my executed notebook.

## What I did
- Provided the assignment specification and the course seed (`SEED = 6638`).
- Ran every version of the notebook end to end in Google Colab on a T4 GPU and returned the executed notebooks for review.
- Asked for the improvements and for the evaluation of the final run.
- Reviewed the outputs and deliverables before submission.

## Verification
- Every metric in the report, `METRICS.md`, and `RUN_LOG.txt` comes from the outputs of my final Colab run. None were estimated or typed in by hand.
- The notebook contains built-in checks: placeholder removal, chunk-length limits, ChromaDB counts, self-retrieval at rank 1, target-paper lookup, baseline reproduction, and follow-up detection. All passed in the final run.
- Relevance labels come from a rubric that the AI proposed. The notebook prints a pooled review table (Section 9) so the labels can be checked by hand. Any override goes through `MANUAL_OVERRIDES`; none were applied in the final run.

## Limitations of AI assistance
- The relevance rubric is term-based. It can mark a chunk relevant if the chunk contains the terms but does not answer the question, and it can miss paraphrased answers.
- The AI could not run the notebook on the real data. All results come from my Colab runs.
