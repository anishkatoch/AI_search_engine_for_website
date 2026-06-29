"""Dataset-agnostic semantic + keyword (hybrid) search engine.

Works on the default Amazon catalog out of the box, but `build_index()` can
point it at ANY dataframe (e.g. an uploaded CSV/Excel) by concatenating the
chosen "search columns" into one text column, embedding it, and rebuilding the
in-memory index. The same embeddings power search + recommendations.
"""
import math
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from app import config
from app.schemas import ProductResult, SearchResponse

_TOKEN_RE = re.compile(r"\w+")
_MAX_FIELD_LEN = 400  # truncate long cell values returned to the UI

# Heuristics for auto-detecting special columns in an uploaded file.
CATEGORY_CANDIDATES = [
    "category", "product_category", "categories", "type", "group", "brand",
    "genre", "department", "section",
]
POPULARITY_CANDIDATES = [
    "total_reviews", "reviews", "review_count", "num_reviews", "rating",
    "ratings", "popularity", "sales", "stars", "votes",
]
ID_CANDIDATES = ["sku", "id", "product_id", "asin", "code", "item_id", "uid"]


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(n, 1e-12, None)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _read_csv(path: str) -> pd.DataFrame:
    """Read the default catalog from a local path or an http(s) URL."""
    if path.startswith(("http://", "https://")):
        return pd.read_csv(path)
    if not os.path.exists(path):
        raise RuntimeError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _detect(columns, candidates) -> Optional[str]:
    """Return the original column whose lowercased name matches a candidate."""
    low = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


class SearchEngine:
    """Holds the active dataset + embeddings and answers hybrid queries."""

    def __init__(self) -> None:
        self.ready = False
        self.df: pd.DataFrame | None = None
        self.ids: List[str] = []
        self.id_to_index: dict = {}
        self.combined_texts: List[str] = []     # lowercased, for embed + keyword
        self.combined_display: List[str] = []    # original case, fallback title
        self.embeddings_unit: np.ndarray | None = None
        self.reviews_arr: np.ndarray | None = None
        self.categories_arr: np.ndarray | None = None
        self.field_columns: List[str] = []
        self.search_columns: List[str] = []
        self.display_column: Optional[str] = None
        self.category_column: Optional[str] = None
        self.popularity_column: Optional[str] = None
        self.source: str = "default"
        self.name: str = "Amazon products"
        self._hf_model = None
        self._openai_client = None

    # ---------- backend setup ----------
    def _ensure_backend(self) -> None:
        if config.USE_OPENAI:
            if self._openai_client is None:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=config.OPEN_API_KEY)
        elif self._hf_model is None:
            from sentence_transformers import SentenceTransformer

            self._hf_model = SentenceTransformer(config.HF_MODEL_NAME)

    # ---------- embeddings ----------
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if config.USE_OPENAI:
            embeddings = []
            for i in range(0, len(texts), 512):
                resp = self._openai_client.embeddings.create(
                    model=config.OPENAI_EMBED_MODEL,
                    input=texts[i:i + 512],
                )
                embeddings.extend(
                    np.array(d.embedding, dtype=np.float32) for d in resp.data
                )
            return np.vstack(embeddings)
        return self._hf_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        if config.USE_OPENAI:
            resp = self._openai_client.embeddings.create(
                model=config.OPENAI_EMBED_MODEL, input=[text]
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        return self._hf_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)

    def _compute_embeddings(self, texts: List[str], cache_path: Optional[str] = None) -> np.ndarray:
        if cache_path and os.path.exists(cache_path):
            embs = np.load(cache_path)
            if embs.shape[0] != len(texts):  # stale cache -> rebuild
                embs = self._embed_texts(texts)
                np.save(cache_path, embs)
        else:
            embs = self._embed_texts(texts)
            if cache_path:
                np.save(cache_path, embs)
        return _normalize_matrix(embs)

    # ---------- index building (any dataframe) ----------
    def build_index(
        self,
        df: pd.DataFrame,
        search_columns: List[str],
        *,
        id_column: Optional[str] = None,
        display_column: Optional[str] = None,
        category_column: Optional[str] = None,
        popularity_column: Optional[str] = None,
        detect: bool = True,
        embeddings_path: Optional[str] = None,
        source: str = "uploaded",
        name: str = "uploaded data",
    ) -> None:
        """Concatenate `search_columns`, embed, and (re)build the index."""
        self._ensure_backend()
        df = df.reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]

        search_columns = [c for c in search_columns if c in df.columns]
        if not search_columns:
            raise ValueError("No valid search columns selected")

        if detect:
            id_column = id_column or _detect(df.columns, ID_CANDIDATES)
            category_column = category_column or _detect(df.columns, CATEGORY_CANDIDATES)
            popularity_column = popularity_column or _detect(df.columns, POPULARITY_CANDIDATES)
        display_column = display_column or search_columns[0]

        # Combined text = chosen columns joined by ", ".
        def join_row(row):
            return ", ".join(
                "" if pd.isna(row[c]) else str(row[c]) for c in search_columns
            )

        combined = df.apply(join_row, axis=1)
        self.combined_display = combined.tolist()
        self.combined_texts = combined.str.lower().tolist()

        self.embeddings_unit = self._compute_embeddings(self.combined_texts, embeddings_path)

        # Stable unique id per row (use id column if it's actually unique).
        if id_column and id_column in df.columns:
            ids = df[id_column].astype(str).str.strip().tolist()
            if len(set(ids)) != len(ids):
                ids = [str(i) for i in range(len(df))]
        else:
            ids = [str(i) for i in range(len(df))]
        self.ids = ids
        self.id_to_index = {s: i for i, s in enumerate(ids)}

        self.categories_arr = (
            df[category_column].astype(str).to_numpy()
            if category_column else np.array([""] * len(df))
        )
        if popularity_column and popularity_column in df.columns:
            self.reviews_arr = (
                pd.to_numeric(df[popularity_column], errors="coerce")
                .fillna(0).astype(int).to_numpy()
            )
        else:
            self.reviews_arr = np.zeros(len(df), dtype=int)

        self.df = df
        self.field_columns = list(df.columns)
        self.search_columns = search_columns
        self.display_column = display_column
        self.category_column = category_column
        self.popularity_column = popularity_column
        self.source = source
        self.name = name
        self.ready = True
        print(f"Index built | source={source} | rows={len(df)} | search={search_columns}")

    # ---------- default catalog ----------
    def _load_default_df(self) -> pd.DataFrame:
        df = _read_csv(config.PRODUCT_CSV)
        df.columns = [str(c).lower().strip() for c in df.columns]

        required = {"sku", "product_title"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")

        df["sku"] = df["sku"].astype(str).str.upper().str.strip()
        if "total_reviews" not in df.columns:
            df["total_reviews"] = 0
        df["total_reviews"] = (
            pd.to_numeric(df["total_reviews"], errors="coerce").fillna(0).astype(int)
        )
        if "product_category" not in df.columns:
            df["product_category"] = ""
        df.drop_duplicates(subset=["sku"], inplace=True)
        return df.reset_index(drop=True)

    def load(self) -> None:
        """Load the default Amazon catalog (cached embeddings). Called at startup."""
        df = self._load_default_df()
        self.build_index(
            df,
            search_columns=["product_title", "product_category"],
            id_column="sku",
            display_column="product_title",
            category_column="product_category",
            popularity_column="total_reviews",
            detect=False,
            embeddings_path=config.EMBEDDINGS_FILE,
            source="default",
            name="Amazon products",
        )

    def dataset_info(self) -> dict:
        return {
            "source": self.source,
            "name": self.name,
            "total": int(len(self.df)) if self.df is not None else 0,
            "columns": self.field_columns,
            "search_columns": self.search_columns,
            "category_column": self.category_column,
            "popularity_column": self.popularity_column,
            "has_trending": self.popularity_column is not None,
        }

    # ---------- result helpers ----------
    def _index_of(self, key: str):
        key = str(key).strip()
        if key in self.id_to_index:
            return self.id_to_index[key]
        return self.id_to_index.get(key.upper())

    def _row_result(self, i: int, rank: int = 0,
                    similarity: float = 0.0, score: float = 0.0) -> ProductResult:
        row = self.df.iloc[i]
        title = row[self.display_column]
        title = self.combined_display[i] if pd.isna(title) or str(title) == "" else str(title)
        category = None
        if self.category_column:
            cv = row[self.category_column]
            category = None if pd.isna(cv) else str(cv)
        fields = {
            c: ("" if pd.isna(row[c]) else str(row[c]))[:_MAX_FIELD_LEN]
            for c in self.field_columns
        }
        return ProductResult(
            rank=rank,
            sku=self.ids[i],
            title=title,
            category=category,
            total_reviews=int(self.reviews_arr[i]),
            fields=fields,
            similarity=round(float(similarity), 4),
            keyword=0.0,
            score=round(float(score), 4),
        )

    # ---------- search ----------
    def _keyword_scores(self, tokens: List[str]) -> np.ndarray:
        n = len(self.combined_texts)
        if not tokens:
            return np.zeros(n, dtype=np.float32)
        scores = np.empty(n, dtype=np.float32)
        ntok = len(tokens)
        for i, text in enumerate(self.combined_texts):
            hits = sum(1 for t in tokens if t in text)
            scores[i] = hits / ntok
        return scores

    def _empty(self, query: str, page: int) -> SearchResponse:
        return SearchResponse(
            products=[], total_results=0, page=page,
            per_page=config.RESULTS_PER_PAGE, total_pages=0, query=query,
        )

    def search(self, query: str, page: int = 1) -> SearchResponse:
        page = max(1, page)
        if not self.ready or not query.strip():
            return self._empty(query, page)

        q = query.strip()
        q_vec = _normalize_vector(self._embed_query(q))
        sims = self.embeddings_unit @ q_vec

        tokens = _TOKEN_RE.findall(q.lower())
        kw = self._keyword_scores(tokens)

        mask = (sims > config.SIMILARITY_THRESHOLD) | (kw > 0)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return self._empty(q, page)

        sem = np.clip(sims, 0.0, 1.0)
        hybrid = config.SEMANTIC_WEIGHT * sem + config.KEYWORD_WEIGHT * kw
        order = idxs[np.lexsort((-self.reviews_arr[idxs], -hybrid[idxs]))]
        order = order[:config.TOP_K]

        total = len(order)
        per = config.RESULTS_PER_PAGE
        total_pages = max(1, math.ceil(total / per))
        page = min(page, total_pages)
        start = (page - 1) * per
        page_idx = order[start:start + per]

        products = [
            self._row_result(i, rank=start + off + 1,
                             similarity=sims[i], score=hybrid[i])
            for off, i in enumerate(page_idx)
        ]
        return SearchResponse(
            products=products, total_results=total, page=page,
            per_page=per, total_pages=total_pages, query=q,
        )

    # ---------- recommendations ----------
    def similar(self, sku: str, top_n: int | None = None) -> List[ProductResult]:
        if not self.ready:
            return []
        top_n = top_n or config.REC_SIMILAR_COUNT
        i = self._index_of(sku)
        if i is None:
            return []
        sims = self.embeddings_unit @ self.embeddings_unit[i]
        same_cat = (self.categories_arr == self.categories_arr[i]).astype(np.float32)
        rank_score = sims + config.REC_CATEGORY_BOOST * same_cat
        rank_score[i] = -np.inf
        order = np.argsort(-rank_score)[:top_n]
        return [
            self._row_result(j, rank=r + 1, similarity=sims[j],
                             score=float(np.clip(sims[j], 0.0, 1.0)))
            for r, j in enumerate(order)
        ]

    def more_from_category(self, sku: str, top_n: int | None = None) -> List[ProductResult]:
        if not self.ready or not self.category_column:
            return []
        top_n = top_n or config.REC_CATEGORY_COUNT
        i = self._index_of(sku)
        if i is None:
            return []
        sims = self.embeddings_unit @ self.embeddings_unit[i]
        mask = self.categories_arr == self.categories_arr[i]
        mask[i] = False
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return []
        order = idxs[np.lexsort((-self.reviews_arr[idxs], -sims[idxs]))][:top_n]
        return [
            self._row_result(j, rank=r + 1, similarity=sims[j],
                             score=float(np.clip(sims[j], 0.0, 1.0)))
            for r, j in enumerate(order)
        ]

    def trending(self, top_n: int | None = None) -> List[ProductResult]:
        if not self.ready:
            return []
        top_n = top_n or config.REC_TRENDING_COUNT
        order = np.argsort(-self.reviews_arr)[:top_n]
        return [self._row_result(j, rank=r + 1) for r, j in enumerate(order)]

    def product_detail(self, sku: str):
        if not self.ready:
            return None
        i = self._index_of(sku)
        if i is None:
            return None
        return {
            "product": self._row_result(i, rank=0, similarity=1.0, score=1.0),
            "similar": self.similar(sku),
            "category": self.more_from_category(sku),
        }
