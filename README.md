# ArXiv RAG: Embedding Comparison and Level-Aware Scientific Q&A

A three-phase retrieval-augmented generation pipeline over **500,000 ArXiv papers**, built to answer a practical question: when students search academic literature, LLMs tend to answer at a single register — too dense for a newcomer, too shallow for a researcher. This system retrieves from ArXiv and adapts its explanation to the reader's level.

> **Course project — Natural Language Processing & Data Analysis (MSc)**
> Team project. See [Contributions](#contributions).

---

## TL;DR

- Streamed and cleaned a **4.98 GB / 2.1M-record** ArXiv metadata dump, then drew a stratified 500K sample across **153 subject categories**.
- Benchmarked three sentence-embedding models on the full 500K corpus with FAISS IVF indexing. **MiniLM was ~5.6× faster than MPNet and BGE at identical retrieval accuracy** — the practical result being that the lightweight model is the right default at this scale.
- Served answers with a quantised Llama-3.1-8B at **~103 tokens/sec**, prompted across **four reader levels** (beginner → expert).

---

## Phase 1 — Data preparation

The raw ArXiv metadata snapshot is a 4.98 GB newline-delimited JSON file — too large to load into memory, so it is read as a stream.

| Step | Result |
|---|---|
| Streamed records | 2,100,000 |
| Removed (empty fields, duplicate IDs, abstracts under 100 chars) | 3,769 |
| Clean corpus | 2,096,231 |
| Stratified sample | **500,000** across 153 categories |

Stratifying by primary category matters: ArXiv is heavily skewed toward physics, and uniform sampling would have starved smaller CS and maths categories. The sample preserves the real distribution — `cs.CV` 4.62%, `quant-ph` 4.18%, `cs.LG` 4.09%, `cs.CL` 2.37% — while keeping all 153 categories represented. Papers span 2016–2025.

Each record's embedding text is the title and abstract concatenated. Outputs are written as CSV, pickle and a JSON metadata summary, plus a distribution figure.

## Phase 2 — Embedding model comparison

Three models were embedded over the full 500K corpus and indexed with **FAISS IVF**, which reduces search from linear to roughly √n scans.

| Model | Dim | Embed time | Throughput | Self-retrieval acc. |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | 406 s | **1,232 texts/s** | 1.000 |
| all-mpnet-base-v2 | 768 | 2,292 s | 218 texts/s | 1.000 |
| BGE-base-en-v1.5 | 768 | 2,258 s | 221 texts/s | 1.000 |

Query-time search speed on the BGE index: **1,256 queries/sec**.

**Reading the result honestly:** self-retrieval accuracy (can a document be found by its own embedding?) saturates at 1.000 for all three, so it does not discriminate between them — it confirms the indexes are sound but cannot rank quality. What the benchmark *does* establish is cost: MiniLM produces embeddings 5.6× faster and at half the storage (768 MB vs 1,536 MB per index). Ranking retrieval *quality* would need a labelled relevance set, which this phase did not have.

## Phase 3 — Level-aware RAG

Retrieval embeds the user's question, pulls the nearest papers from FAISS, and formats them as numbered context blocks. Generation runs **Llama-3.1-8B-Instruct** (Q4_K_M GGUF via `llama-cpp-python`) under one of four system prompts:

| Level | Prompting strategy |
|---|---|
| `beginner` | Everyday language, no technical jargon |
| `intermediate` | Some technical terms, general science background assumed |
| `advanced` | Full terminology, conceptual depth expected |
| `expert` | Academic register, methodological detail |

Evaluated on 12 test queries (3 per level):

| Metric | Value |
|---|---|
| Successful queries | 12 / 12 |
| Avg. generation time | 4.38 s |
| Avg. throughput | 103.13 tokens/s |

## Hardware

Google Colab Pro, NVIDIA A100-SXM4-40GB.

---

## Repository structure

```
.
├── ArXiv_RAG_Experiment.ipynb      # orchestration notebook
├── phase1_data_preparation.py      # streaming load, cleaning, stratified sampling
├── phase2_embedding_comparison.py  # embedding + FAISS indexing + benchmark
├── phase3_llm_comparison.py        # RAG retrieval + level-aware generation
├── docs/
│   └── project_documentation.md
└── requirements.txt
```

The notebook orchestrates only — all computation lives in the phase modules, so each phase can be re-run independently and checkpoints to disk.

## Reproducing

1. Download the [ArXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (~5 GB).
2. `pip install -r requirements.txt`
3. Download a Llama-3.1-8B-Instruct GGUF (Q4_K_M) for Phase 3.
4. Set the data path in the notebook and run phases in order.

Each phase writes to disk before the next begins, so an interrupted session can resume rather than restart.

## Limitations and next steps

- **Retrieval quality is unmeasured.** Self-retrieval accuracy saturates; a labelled query–document relevance set is needed to rank the embedding models on quality rather than speed.
- **Answer quality is unmeasured.** Phase 3 reports latency and throughput only. Adding ROUGE-L and BERTScore against reference answers was planned but not completed.
- **Level-appropriateness is unvalidated.** Whether the four prompts genuinely produce level-appropriate output needs human evaluation.
- A LoRA fine-tuning branch on `bert-base-uncased` was attempted as an additional embedding baseline but is not included here — it did not reach a working state.
- Single generator model; the original design compared four LLMs but was scaled back to fit the available compute budget.


## Tech stack

`sentence-transformers` · `FAISS` · `llama-cpp-python` · `Llama-3.1-8B-Instruct` · `pandas` · `NumPy` · `matplotlib` · `Google Colab (A100)`
