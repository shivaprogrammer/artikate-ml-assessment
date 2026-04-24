# Artikate Studio — AI / ML / LLM Engineer Assessment

This repository contains my submission for the Artikate Studio AI/ML/LLM Engineer technical assessment. The project demonstrates system design, implementation, and evaluation of real-world AI systems including LLM pipelines, Retrieval-Augmented Generation (RAG), and classification models.

---

## 📌 Project Structure

```bash
artikate-ml-assessment/
│
├── README.md
├── requirements.txt
├── ANSWERS.md
├── DESIGN.md
│
├── rag_pipeline.py
├── train.py
├── evaluate.py
├── test_latency.py
│
├── sample_docs/
│   ├── nda_sample.pdf
│   ├── contract_sample.pdf
│   └── policy_sample.pdf
```



## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/artikate-ml-assessment.git
cd artikate-ml-assessment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Configuration (Optional)

This project supports OpenAI-based answer generation.

### Linux / Mac

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Windows (PowerShell)

```bash
setx OPENAI_API_KEY "your_api_key_here"
```

If no API key is provided, the system runs in **stub mode** for offline execution.

---

## 🚀 Section 2 — RAG Pipeline

### Run Pipeline

```bash
python section2_rag/rag_pipeline.py
```

### Ask Custom Question

```bash
python section2_rag/rag_pipeline.py "What is the notice period in the NDA?"
```

### Run Evaluation (Precision@3)

```bash
python section2_rag/rag_pipeline.py eval
```

---

## 🧠 RAG Pipeline Overview

The system implements a production-grade Retrieval-Augmented Generation pipeline:

* PDF ingestion using `pdfplumber`
* Structure-aware chunking (512 tokens, 128 overlap)
* Embeddings using SentenceTransformers (BGE model)
* Vector database using ChromaDB
* Hybrid retrieval:

  * Dense retrieval
  * BM25 sparse retrieval
  * Reciprocal Rank Fusion (RRF)
* Cross-encoder re-ranking
* Hallucination mitigation via NLI-based grounding check
* Answer generation with source citation

---

## 📊 Section 2 Evaluation

Metric used:

* **Precision@3**

Example output:

```
Precision@3: 0.83 (5/6 correct)
```

---

## 🤖 Section 3 — Classification Model

### Train Model

```bash
python section3_classifier/train.py
```

### Evaluate Model

```bash
python section3_classifier/evaluate.py
```

### Run Latency Test

```bash
python section3_classifier/test_latency.py
```

---

## 📈 Classification Details

* Model: DistilBERT
* Classes:

  * billing
  * technical_issue
  * feature_request
  * complaint
  * other

### Metrics Reported:

* Accuracy
* F1 Score
* Confusion Matrix

### Constraint:

* Inference time < **500ms on CPU** (validated via test script)

---

## 🧾 Section 1 & Section 4

All written answers are included in:

```
ANSWERS.md
```

Includes:

* LLM failure diagnosis
* Prompt engineering fixes
* Latency analysis
* Security and system design answers

---

## 📐 Section 2 Design

Detailed architecture and reasoning documented in:

```
DESIGN.md
```

Includes:

* Chunking strategy
* Embedding model choice
* Retrieval design
* Hallucination mitigation
* Scaling strategy

---

## ⚠️ Notes

* No API keys or credentials are included
* All code runs locally in under 5 minutes
* Sample PDFs included for reproducibility
* Stub mode allows execution without external APIs

---

## 🟢 Summary

This project demonstrates:

* End-to-end RAG system design and implementation
* Real-world LLM failure diagnosis
* Efficient ML model deployment under constraints
* Strong evaluation and reasoning

---

## 📬 Submission

Repository Link: https://github.com/<your-username>/artikate-ml-assessment

---
