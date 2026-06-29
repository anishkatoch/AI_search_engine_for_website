# AI_search_engine_for_website
An AI-powered product search backend that uses vector embeddings for semantic retrieval and hybrid ranking (similarity + review count). The engine reads from CSV data, supports Hugging Face E5 embeddings, and provides a REST-style search interface for downstream applications.


# 🔍 AI Semantic Product Search Engine  
**React (Vite) · FastAPI · Sentence Transformers · all-MiniLM-L6-v2**

> A single FastAPI server serves both the **React UI** (at `/`) and the
> **search API** (at `/api/*`) — one command runs everything.

---

## 1. Introduction

This project implements a **semantic product search engine** for e-commerce–style catalogs using **vector embeddings** and **cosine similarity**.  
Unlike traditional keyword-based search systems, this engine understands **semantic meaning, intent, and category context**, enabling natural-language queries such as:

- `TV & Display`
- `wearable fitness devices`
- `home audio equipment`
- `printers and scanners`

The system is built as a **React single-page app** talking to a **FastAPI**
backend, suitable for demos, MVPs, and experimentation with modern information
retrieval techniques.

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
FastAPI  /api/search
    |
    v
React Search UI  (served at /)
```

---

## 4. Project layout

```text
app/                 FastAPI backend (serves the React build + /api/*)
  main.py            app entrypoint -> app.main:app
  search.py          embeddings + semantic ranking
  config.py          env-driven configuration
  schemas.py         API response models
frontend/            React (Vite) single-page app
  src/App.jsx        the search UI
  dist/              built static files served by FastAPI (generated)
amazon_products.csv  product catalog
pyproject.toml       Python deps  (locked in uv.lock)
Dockerfile           multi-stage: builds the UI, then serves everything
```

---

## 5. Quickstart — one command runs everything

Dependencies install automatically (declared in `pyproject.toml`, locked in
`uv.lock`); no manual `pip install` needed.

### Option A — bootstrap script (recommended)

`run.sh` installs [`uv`](https://docs.astral.sh/uv/) if missing, syncs the
locked Python deps, builds the React frontend, then starts the single server:

```bash
./run.sh
# -> open http://localhost:8005   (UI)
#         http://localhost:8005/docs   (API docs)
```

### Option B — manual (matches the raw command)

```bash
# 1) build the frontend once (creates frontend/dist)
cd frontend && npm install && npm run build && cd ..

# 2) install deps + run the one server that serves BOTH UI and API
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload
```

> The raw `uvicorn app.main:app` command serves the React UI **only after**
> `frontend/dist` has been built (step 1). `./run.sh` does both for you.

### Frontend dev mode (hot reload, optional)

```bash
# terminal 1: API
uv run uvicorn app.main:app --port 8005 --reload
# terminal 2: Vite dev server (proxies /api -> :8005)
cd frontend && npm run dev   # http://localhost:5173
```

---

## 6. Run with Docker

No Python, Node, or dependencies needed on the host — only Docker. The image is
built in two stages (build React → serve with FastAPI):

```bash
docker compose up --build
# -> http://localhost:8005
```

The generated `product_embeddings.npy` is cached in a named volume, so it is
only computed once across restarts.

---

## 7. Deploy on AWS (EC2)

On a fresh Ubuntu / Amazon Linux EC2 instance:

**Docker (recommended):**

```bash
curl -fsSL https://get.docker.com | sh       # install Docker once
git clone <this-repo> && cd AI_search_engine_for_website
docker compose up --build -d                 # http://<ec2-ip>:8005
```

**Or without Docker** — `./run.sh` installs `uv` + Node deps and builds itself:

```bash
git clone <this-repo> && cd AI_search_engine_for_website
./run.sh
```

Open port **8005** in the instance's Security Group.

---

## 8. How ranking works (hybrid search)

Each result is scored by combining two signals:

```
score = semantic_weight · cosine_similarity  +  keyword_weight · keyword_match
        (default 0.7)                            (default 0.3)
```

- **Semantic similarity** — cosine between the query and product embeddings.
- **Keyword match** — fraction of query words found in the product text.

A product is shown only if its **cosine similarity > 0.4** *or* it has a keyword
hit. Results are ranked by the hybrid score (ties broken by review count),
capped at `top_k`, and returned **10 per page** (`/api/search?...&page=2`).

---

## 9. Recommendations

The same cached embeddings power three recommenders (on-the-fly kNN, no extra
model or data needed):

- **Similar products** ("You may also like") — nearest neighbors of a product in
  embedding space, nudged toward the same category. Shown on the product detail
  view when a result box is clicked.
- **More from this category** — same-category items ranked by similarity.
- **Trending** — most-reviewed products, shown on the empty-state homepage.

> True personalized / collaborative recommendations ("customers who bought X…")
> would need per-user interaction history, which this dataset doesn't include.

---

## 10. Use your own data

The app searches the bundled Amazon catalog by default, but you can point it at
**your own data** — by file or by SQL query — without touching any code.

### A) Upload a file (CSV / XLS / XLSX)

1. Click **⬆ Upload file** and choose a `.csv`, `.xls`, or `.xlsx`.
2. The app shows its **columns** — tick the one(s) to search on.
3. Multiple columns are **concatenated** (separated by `, `) into one text field,
   which is then **embedded**.
4. **Reset to default** restores the Amazon catalog.

### B) Connect a SQL database

Click **🗄 Connect SQL**, choose the engine (**PostgreSQL, MySQL, SQL Server,
Snowflake**), enter the connection fields (host/port/user/password/database — plus
schema/warehouse/role for Snowflake) and a **read-only query** (single table or
joins). The query result becomes the dataset — pick the search columns exactly as
with a file.

**🔒 Credential security model:**
- The password is sent as a masked `SecretStr`, used **once** to open the
  connection, then the engine is **disposed**. It is **never written to disk,
  never logged, and never returned** in any response.
- Only the **query result rows** are kept in memory — **not** your credentials.
- Queries are restricted to **read-only** (`SELECT`/`WITH`/`SHOW`/`EXPLAIN`); a
  single statement only — so a query can't modify your database.
- DB error messages are **sanitized** to strip the password.
- ⚠️ Over plain HTTP the password still crosses the network in the request body —
  **serve over HTTPS in production** and avoid exposing the app publicly with real
  credentials.

When picking columns you can also:
- **Choose how many rows** to index (optional — default is all rows).
- **👁 Preview** the data, and **⬇ Download** it as CSV (so a SQL result can be
  saved and re-used without reconnecting).

Notes (both paths):
- A `category`-like and a `reviews`/`rating`-like column are auto-detected (by
  name) to power category recommendations + trending — optional.
- The active dataset lives **in memory** for the running server (single dataset).
  There's a hard cap of `upload.max_rows` rows (default **10,000**); larger inputs
  are **truncated** with a warning. Uploaded/queried embeddings are computed fresh.

---

## 11. API

| Method | Path                          | Description                                  |
|--------|-------------------------------|----------------------------------------------|
| `GET`  | `/`                           | React search UI                              |
| `GET`  | `/api/health`                 | Readiness + backend + item count             |
| `GET`  | `/api/search?query=…&page=1`  | Paginated hybrid search results (JSON)       |
| `GET`  | `/api/trending?limit=N`       | Most-reviewed (or first N) items             |
| `GET`  | `/api/product/{sku}`          | Item + "similar" + "category" recommendations |
| `GET`  | `/api/dataset`                | Active dataset info (source, columns, …)     |
| `POST` | `/api/upload`                 | Upload a CSV/XLS/XLSX → returns its columns  |
| `POST` | `/api/sql/query`              | Run a read-only SQL query → returns its columns |
| `POST` | `/api/upload/{id}/build`      | Build the index from chosen search columns   |
| `POST` | `/api/dataset/reset`          | Restore the default Amazon dataset           |
| `GET`  | `/docs`                       | Interactive OpenAPI docs                     |

---

## 12. Configuration

Configuration is split in two:

- **`.env`** — **secrets only** (git-ignored). Copy from `.env.example`:
  ```bash
  cp .env.example .env      # then edit OPEN_API_KEY if you want OpenAI embeddings
  ```
- **`config.yaml`** — all non-secret settings (models, thresholds, paging,
  hybrid weights). Any value can be overridden by an env var of the same
  UPPER_CASE name.

| Source        | Key                                  | Default                  | Description                                          |
|---------------|--------------------------------------|--------------------------|------------------------------------------------------|
| `.env`        | `OPEN_API_KEY`                       | *(empty)*                | Set to use OpenAI embeddings; empty → local HF model |
| `config.yaml` | `app.port`                           | `8005`                   | Server port                                          |
| `config.yaml` | `models.huggingface`                 | `all-MiniLM-L6-v2`       | Local embedding model                                |
| `config.yaml` | `models.openai_embedding`            | `text-embedding-3-large` | OpenAI embedding model                               |
| `config.yaml` | `data.product_csv`                   | `amazon_products.csv`    | Catalog source (local path or URL)                   |
| `config.yaml` | `data.embeddings_file`               | `product_embeddings.npy` | Cached vectors file                                  |
| `config.yaml` | `search.top_k`                       | `80`                     | Max ranked results kept (all pages)                  |
| `config.yaml` | `search.similarity_threshold`        | `0.4`                    | Minimum cosine similarity to show a product          |
| `config.yaml` | `search.results_per_page`            | `10`                     | Pagination page size                                 |
| `config.yaml` | `search.hybrid.semantic_weight`      | `0.7`                    | Weight of semantic similarity in the hybrid score    |
| `config.yaml` | `search.hybrid.keyword_weight`       | `0.3`                    | Weight of keyword match in the hybrid score          |
| `config.yaml` | `recommend.similar_count`            | `5`                      | "You may also like" item count                       |
| `config.yaml` | `recommend.category_count`           | `5`                      | "More from this category" item count                 |
| `config.yaml` | `recommend.trending_count`           | `12`                     | Trending row size                                    |
| `config.yaml` | `recommend.category_boost`           | `0.05`                   | Same-category nudge in "similar" ranking             |
| `config.yaml` | `upload.max_rows`                    | `20000`                  | Max rows kept from an uploaded file                  |

> ⚠️ If you change the catalog or the embedding model, delete
> `product_embeddings.npy` so the vectors are rebuilt.

