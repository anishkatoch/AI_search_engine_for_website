# AI_search_engine_for_website
An AI-powered product search backend that uses vector embeddings for semantic retrieval and hybrid ranking (similarity + review count). The engine reads from CSV data, supports Hugging Face E5 embeddings, and provides a REST-style search interface for downstream applications.


# 🔍 AI Semantic Product Search Engine  
**Streamlit · Sentence Transformers · all-MiniLM-L6-v2**

---

## 1. Introduction

This project implements a **semantic product search engine** for e-commerce–style catalogs using **vector embeddings** and **cosine similarity**.  
Unlike traditional keyword-based search systems, this engine understands **semantic meaning, intent, and category context**, enabling natural-language queries such as:

- `TV & Display`
- `wearable fitness devices`
- `home audio equipment`
- `printers and scanners`

The system is designed as a **Streamlit application** with a **live autocomplete dropdown**, suitable for demos, MVPs, and experimentation with modern information retrieval techniques.

---

## 2. Problem Statement

### 2.1 Issues with Keyword-Based Search

Traditional keyword search systems suffer from the following limitations:

- Exact keyword dependency  
- No understanding of synonyms or paraphrasing  
- Weak category and taxonomy awareness  
- Poor handling of exploratory queries  
- Low recall for natural language inputs  

### 2.2 Example Failures

| Query | Keyword Search Behavior |
|-----|-------------------------|
| `TV & Display` | Misses TVs, monitors, displays |
| `fitness wearables` | Misses smartwatches, bands |
| `home audio` | Misses speakers, soundbars |

---

## 3. Solution Overview

This project addresses the above limitations by introducing a **semantic retrieval pipeline**:

- Products are encoded into **dense vector embeddings**
- Queries are encoded into the **same vector space**
- Relevance is measured using **cosine similarity**
- Results are ranked using a **hybrid strategy**:
  - Semantic similarity
  - Popularity (`total_reviews`)
- Results are presented via a **real-time autocomplete UI**

---

### High-Level Architecture Flow

```text
Product CSV
    |
    v
Text Preprocessing
(title + category)
    |
    v
Sentence Transformer
(all-MiniLM-L6-v2)
    |
    v
Product Embeddings
(cached as .npy)
    |
    v
Query Embedding
    |
    v
Cosine Similarity
    |
    v
Popularity Boost
(total_reviews)
    |
    v
Streamlit Autocomplete UI```


