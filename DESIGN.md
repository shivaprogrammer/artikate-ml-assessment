# DESIGN.md — Section 02: Production-Grade RAG Pipeline for Legal Documents

---

## Problem Context

The corpus is 500+ PDF contracts and policy documents averaging 40 pages each. Queries are highly precise:
- "What is the notice period in the NDA signed with Vendor X?"
- "Which contracts contain a limitation of liability clause above ₹1 crore?"

These are not semantic similarity queries — they require **exact factual retrieval** from specific named entities within specific documents. Hallucination is unacceptable, and every answer must cite the source document and page number.

---

## Chunking Strategy

**Choice: Hierarchical semantic chunking with 512-token windows and 128-token overlap, anchored to document structure.**

**Why not fixed-size chunking?**

Fixed-size chunking (e.g., 256 tokens with no overlap) cuts across clause boundaries in legal text. A notice period clause like:

> "Section 12.3 — Termination Notice. Either party may terminate this Agreement by providing thirty (30) days written notice to the other party at the address listed in Schedule A."

...can be split such that "thirty (30) days" lands in one chunk and "written notice" lands in the next. A retriever searching for "notice period" may retrieve the wrong half.

**Chosen approach:**

1. **Structure-aware pre-processing**: Use `pdfplumber` to extract text with page/section metadata. Detect section headers using regex patterns common in legal docs (`^\d+\.\s`, `^ARTICLE`, `^SCHEDULE`). Mark section boundaries.

2. **Semantic chunking within sections**: Chunk text to ~512 tokens, but only break at sentence boundaries. Use 128-token overlap (25%) to preserve cross-sentence context — critical for clauses that reference earlier definitions.

3. **Metadata injection**: Each chunk stores `{document_name, page_number, section_heading, chunk_index}`. The page number is extracted from `pdfplumber`'s page-level iteration, not estimated.

4. **Parent-child chunk structure**: Store both 512-token "child chunks" for retrieval and 1024-token "parent chunks" for answer generation. Retrieve via child chunks (more precise), generate from parent chunks (more context). This is the Small-to-Big retrieval pattern and is critical for legal text where a clause may be brief but its interpretation depends on surrounding context.

---

## Embedding Model Choice

**Choice: `BAAI/bge-large-en-v1.5` (335M params, 1024-dim embeddings)**

**Why not OpenAI `text-embedding-3-large`?**

The question specifies a legal document repository where answers must cite exact sources. In this context:
- Data privacy: contract text is confidential. Sending it to an external embedding API creates a data egress risk. BAAI/bge runs locally.
- Latency: local embedding at ingestion time is done once. No per-query API cost or latency.
- Performance: BAAI/bge-large ranks #1–3 on the BEIR legal retrieval benchmark (specifically on TREC-Legal), outperforming OpenAI embeddings on long-form legal text.

**Why not `sentence-transformers/all-MiniLM-L6-v2`?**

MiniLM (384-dim) is fast but loses nuance on rare legal terminology (e.g., "indemnification", "laches", "force majeure"). BGE-large handles domain-specific vocabulary significantly better, at the cost of 4× slower embedding — acceptable since embedding is done at ingestion, not query time.

---

## Vector Store Choice

**Choice: ChromaDB (local, persistent) for development; Qdrant for production scale.**

**FAISS**: Excellent raw ANN performance but requires manual persistence serialization, lacks metadata filtering, and has no built-in server mode. For legal retrieval where we frequently filter by `document_name` or `date_range`, FAISS requires post-retrieval filtering which degrades effective recall.

**ChromaDB**: Supports metadata filtering natively, has a simple Python API, persists to disk automatically, and handles our 500-document corpus (≈200K chunks) without issue. Perfect for the scope of this assessment.

**Pinecone**: Managed, scalable, but sends data to an external server — same privacy concern as OpenAI embeddings.

**Qdrant (production recommendation)**: Open-source, self-hostable, supports payload filtering, sparse+dense hybrid search, and scales to 50M+ vectors efficiently. When the corpus grows to 50,000 documents (see scaling section in ANSWERS.md), Qdrant with a dedicated server becomes the correct choice.

---

## Retrieval Strategy

**Choice: Hybrid retrieval (dense + sparse BM25) with cross-encoder re-ranking.**

**Why not naive top-k dense retrieval?**

Dense embeddings are excellent at semantic similarity but miss **exact keyword matches** — critical for legal queries like "limitation of liability above ₹1 crore" where "₹1 crore" is a precise filter, not a semantic concept. BM25 (sparse, keyword-based) handles exact term matching extremely well.

**Pipeline:**

1. **Sparse retrieval (BM25)**: Run BM25 over the full chunk corpus. Return top 20 candidates. Use `rank_bm25` library with legal stopword removal.

2. **Dense retrieval (BGE-large)**: Embed the query, retrieve top 20 from ChromaDB by cosine similarity.

3. **Reciprocal Rank Fusion (RRF)**: Merge the two lists of 20 into a unified ranking using RRF scores: `score(d) = Σ 1/(k + rank(d))` where k=60. This avoids the score normalisation problem of simply averaging dense and sparse scores.

4. **Cross-encoder re-ranking**: Pass the top 10 RRF-ranked chunks through `cross-encoder/ms-marco-MiniLM-L-6-v2`. Cross-encoders jointly encode (query, chunk) pairs and are dramatically more accurate than bi-encoders for re-ranking. Return top 3 for generation.

**Why this matters for legal text**: A query about "Vendor X" must find chunks mentioning "Vendor X" by exact name. Pure dense retrieval may surface chunks about "Vendor Y" if they have similar surrounding context. BM25 ensures the exact name anchors retrieval.

---

## Hallucination Mitigation Strategy

**Choice: Retrieved Context Grounding Check with Confidence Scoring**

After the LLM generates an answer, before returning it to the user:

1. **Entailment check**: Pass `(retrieved_chunks, generated_answer)` through a Natural Language Inference model (`cross-encoder/nli-deberta-v3-small`). If the generated answer cannot be entailed from the retrieved chunks (entailment score < 0.6), the answer is flagged as potentially hallucinated.

2. **Answer refusal on low retrieval score**: If the top retrieved chunk's similarity score is below a threshold (cosine < 0.65), the system responds: "I could not find sufficient information in the available documents to answer this question reliably. Please consult the source documents directly."

3. **Citation verification**: For each claim in the answer, verify that at least one cited chunk contains the key entity/number mentioned (e.g., if the answer says "30 days", verify that the cited chunk contains "30" or "thirty"). Uses simple string matching post-generation.

**Why not RAG-Fusion or Chain-of-Thought grounding?**

RAG-Fusion (generating multiple query variants) improves recall but increases latency by 3–5× — unacceptable for a query-time operation over 500+ documents. The NLI-based entailment check is lightweight (MiniLM-sized) and adds only ~50ms per response.

---

## Scaling to 50,000 Documents

See ANSWERS.md Section 02 scaling discussion.

Summary of bottlenecks:
1. **Embedding at ingestion**: Move to async batch embedding with GPU acceleration. 500 docs → 50,000 docs at 512-token chunks means ~20M embeddings. Use `sentence-transformers` with `encode(batch_size=256)` on GPU → ~40 hours one-time cost, parallelisable.
2. **Vector store**: ChromaDB does not scale beyond ~1M vectors efficiently. Migrate to Qdrant with HNSW indexing and payload-based pre-filtering.
3. **BM25 index**: `rank_bm25` loads the full corpus into memory. Replace with Elasticsearch or OpenSearch for distributed BM25 with sub-100ms latency at 50K documents.
4. **Re-ranker**: Cross-encoder at 50K scale can become a bottleneck if retrieval recall is high. Implement a two-stage filter: BM25+dense fusion reduces to 50 candidates, cross-encoder re-ranks only those 50. Latency remains bounded regardless of corpus size.
