import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox
from sentence_transformers import SentenceTransformer

# ================== CONFIG ==================
PRODUCT_CSV = r"https://raw.githubusercontent.com/anishkatoch/AI_search_engine_for_webiste/main/amazon_products.csv"
TOP_K = 20
EMBEDDINGS_FILE = "product_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# ================== UTILS ==================
def normalize_matrix(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(n, 1e-12, None)

def normalize_vector(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

# ================== LOAD DATA ==================
@st.cache_resource
def load_data():
    df = pd.read_csv(PRODUCT_CSV)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"sku", "product_title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["sku"] = df["sku"].astype(str).str.upper().str.strip()

    df["total_reviews"] = (
        pd.to_numeric(df.get("total_reviews", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Text used for embeddings
    df["combined"] = (
        df["product_title"].astype(str)
        + " "
        + df.get("product_category", "").astype(str)
    )

    df.drop_duplicates(subset=["sku"], inplace=True)
    return df.reset_index(drop=True)

df = load_data()

# ================== LOAD / BUILD EMBEDDINGS ==================
@st.cache_resource
def load_embeddings(_df):
    texts = df["combined"].tolist()

    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype(np.float32)
        np.save(EMBEDDINGS_FILE, embeddings)

    return normalize_matrix(embeddings)

embeddings = load_embeddings(df)

# ================== SEARCH LOGIC ==================
def search_products(query: str):
    if not query.strip():
        return []

    q_vec = model.encode(
        [query],
        convert_to_numpy=True
    )[0].astype(np.float32)

    q_vec = normalize_vector(q_vec)

    sims = embeddings @ q_vec
    reviews = df["total_reviews"].values

    # similarity ↓ then reviews ↓
    order = np.lexsort((-reviews, -sims))[:TOP_K]
    return df.loc[order, "product_title"].tolist()

# ================== STREAMLIT UI ==================
st.set_page_config(
    page_title="AI Product Search",
    page_icon="🔍",
    layout="centered"
)

st.markdown(
    """
    <h2 style="text-align:center;">🔍 AI Product Search</h2>
    <p style="text-align:center;">Type to search products using semantic similarity</p>
    """,
    unsafe_allow_html=True
)

def search_callback(query):
    return search_products(query)

selected = st_searchbox(
    search_callback,
    key="product_search",
    placeholder="Try: TV & Display, Speakers, Headphones, Printers & Scanners, Wearables..."
)

if selected:
    st.success(f"Selected: {selected}")



