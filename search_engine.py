import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# ================== ENV ==================
USE_OPENAI = bool(os.getenv("OPEN_API_KEY"))

if USE_OPENAI:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
else:
    from sentence_transformers import SentenceTransformer
    hf_model = SentenceTransformer("all-MiniLM-L6-v2")

# ================== CONFIG ==================
PRODUCT_CSV = r"https://raw.githubusercontent.com/anishkatoch/AI_search_engine_for_webiste/main/amazon_products.csv"
TOP_K = 80
EMBEDDINGS_FILE = "product_embeddings.npy"
OPENAI_EMBED_MODEL = "text-embedding-3-large"

# ================== MODELS ==================
class ProductResult(BaseModel):
    title: Optional[str] = None
    sku: Optional[str] = None
    total_reviews: int = 0

class SearchResponse(BaseModel):
    products: List[ProductResult]
    total_results: int
    query: str

# ================== GLOBALS ==================
products_df = None
product_skus = None
embeddings_unit = None
review_counts = {}
is_ready = False

# ================== UTILS ==================
def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(n, 1e-12, None)

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

# ================== CSV LOADER ==================
def load_products_from_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"sku", "product_title"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["sku"] = df["sku"].astype(str).str.upper().str.strip()

    if "total_reviews" not in df.columns:
        df["total_reviews"] = 0

    df["total_reviews"] = (
        pd.to_numeric(df["total_reviews"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    def safe_combine(row):
        return " ".join([
            str(row.get("sku", "")),
            str(row.get("product_title", "")),
            str(row.get("product_category", "")),
        ]).lower()

    df["combined"] = df.apply(safe_combine, axis=1)

    df.drop_duplicates(subset=["sku"], inplace=True)
    return df.astype(object).where(pd.notna(df), None)

# ================== EMBEDDINGS ==================
def embed_texts(texts: List[str]) -> np.ndarray:
    if USE_OPENAI:
        embeddings = []
        for i in range(0, len(texts), 512):
            resp = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=texts[i:i + 512]
            )
            embeddings.extend(
                np.array(d.embedding, dtype=np.float32)
                for d in resp.data
            )
        return np.vstack(embeddings)
    else:
        return hf_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

def embed_query(text: str) -> np.ndarray:
    if USE_OPENAI:
        resp = client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=[text]
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)
    else:
        return hf_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype(np.float32)

def load_embeddings(df: pd.DataFrame):
    texts = df["combined"].tolist()

    if os.path.exists(EMBEDDINGS_FILE):
        embs = np.load(EMBEDDINGS_FILE)
    else:
        embs = embed_texts(texts)
        np.save(EMBEDDINGS_FILE, embs)

    embs = normalize_matrix(embs)
    return df["sku"].tolist(), embs

# ================== FASTAPI LIFESPAN ==================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global products_df, product_skus, embeddings_unit, review_counts, is_ready

    try:
        products_df = load_products_from_csv(PRODUCT_CSV)
        products_df.set_index("sku", inplace=True)

        product_skus, embeddings_unit = load_embeddings(
            products_df.reset_index()
        )

        review_counts = dict(
            zip(products_df.index, products_df["total_reviews"])
        )

        is_ready = True
        print("App ready | Products:", len(product_skus))
    except Exception as e:
        is_ready = False
        print("Startup error:", e)
        raise

    yield

# ================== FASTAPI APP ==================
app = FastAPI(
    title="CSV AI Product Search",
    description="Semantic product search with OpenAI / HuggingFace fallback",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== ENDPOINTS ==================
@app.get("/health")
def health():
    return {
        "ready": is_ready,
        "embedding_backend": "openai" if USE_OPENAI else "huggingface",
        "total_products": len(product_skus) if product_skus else 0
    }

@app.get("/search", response_model=SearchResponse)
def search(query: str):
    if not is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    if not query.strip():
        return SearchResponse(products=[], total_results=0, query=query)

    q_vec = normalize_vector(embed_query(query.strip()))
    sims = embeddings_unit @ q_vec
    reviews = np.array([review_counts.get(s, 0) for s in product_skus])

    order = np.lexsort((-reviews, -sims))[:TOP_K]

    results = []
    for i in order:
        sku = product_skus[i]
        row = products_df.loc[sku]

        results.append(ProductResult(
            title=row.get("product_title"),
            sku=sku,
            total_reviews=int(reviews[i])
        ))

    return SearchResponse(
        products=results,
        total_results=len(results),
        query=query
    )

# ================== RUN ==================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8024,
        reload=True
    )

