"""Pydantic request/response models."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr


class ProductResult(BaseModel):
    rank: int = 0               # 1-based position across the full ranked list
    sku: Optional[str] = None   # unique row id (sku or row index)
    title: Optional[str] = None
    category: Optional[str] = None
    total_reviews: int = 0
    fields: Dict[str, str] = {}  # all original columns of the row (stringified)
    similarity: float = 0.0     # cosine similarity (semantic)
    keyword: float = 0.0        # keyword-match fraction (0..1)
    score: float = 0.0          # hybrid score used for ranking


class SearchResponse(BaseModel):
    products: List[ProductResult]
    total_results: int
    page: int
    per_page: int
    total_pages: int
    query: str


class TrendingResponse(BaseModel):
    products: List[ProductResult]


class ProductDetailResponse(BaseModel):
    product: ProductResult
    similar: List[ProductResult]
    category: List[ProductResult]


class HealthResponse(BaseModel):
    ready: bool
    embedding_backend: str
    total_products: int


# ---------- dataset upload ----------
class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    columns: List[str]
    n_rows: int                      # rows currently available (after any cap)
    total_rows: int = 0              # rows in the original file/query result
    truncated: bool = False          # True if total_rows exceeded the cap
    max_rows: int = 0                # the hard row cap
    preview: List[Dict[str, str]]    # first few rows, stringified
    dropped_columns: List[str] = []  # columns dropped because every value was null


class BuildIndexRequest(BaseModel):
    search_columns: List[str]
    max_rows: Optional[int] = None   # optional row limit to index (default: all)


class DatasetInfo(BaseModel):
    source: str                     # "default" | "uploaded"
    name: str
    total: int
    columns: List[str]
    search_columns: List[str]
    category_column: Optional[str] = None
    popularity_column: Optional[str] = None
    has_trending: bool = True


class SQLQueryRequest(BaseModel):
    """Connect to a SQL database and run a read-only query.

    The password is a SecretStr so it is masked in logs/reprs, used once to
    connect, and never persisted or returned.
    """
    dialect: str                              # postgresql | mysql | mssql | snowflake
    host: str
    port: Optional[int] = None
    user: str = ""
    password: SecretStr = SecretStr("")
    database: str = ""
    query: str
    db_schema: Optional[str] = Field(default=None, alias="schema")   # snowflake
    warehouse: Optional[str] = None                                  # snowflake
    role: Optional[str] = None                                       # snowflake

    model_config = {"populate_by_name": True}
