"""FastAPI app: serves the search API under /api and the built React UI at /.

Run everything with one command:

    uv run uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload
"""
import io
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import config, sql_source
from app.schemas import (
    BuildIndexRequest,
    DatasetInfo,
    HealthResponse,
    ProductDetailResponse,
    SearchResponse,
    SQLQueryRequest,
    TrendingResponse,
    UploadResponse,
)
from app.search import SearchEngine

engine = SearchEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build/load embeddings once when the server starts.
    engine.load()
    yield


app = FastAPI(
    title="AI Product Search",
    description="Semantic product search with OpenAI / HuggingFace fallback",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Recently uploaded files, kept in memory until the user picks columns to build.
_uploads: "OrderedDict[str, pd.DataFrame]" = OrderedDict()


def _parse_table(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV / XLS / XLSX into a dataframe."""
    fn = (filename or "").lower()
    if fn.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if fn.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    if fn.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw), engine="xlrd")
    # Unknown extension: try CSV, then Excel.
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.read_excel(io.BytesIO(raw))


def _csv_response(df: pd.DataFrame, filename: str) -> Response:
    return Response(
        content=df.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _register_dataframe(df: pd.DataFrame, filename: str) -> UploadResponse:
    """Clean a dataframe, stash it for the build step, return its columns."""
    df.columns = [str(c) for c in df.columns]

    # Drop columns that are entirely null, and remember which ones (to tell user).
    before = list(df.columns)
    df = df.dropna(axis=1, how="all")
    dropped = [c for c in before if c not in df.columns]

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="No usable rows/columns")

    # Enforce the hard row cap, remembering the original size (to warn the user).
    total_rows = len(df)
    truncated = total_rows > config.UPLOAD_MAX_ROWS
    if truncated:
        df = df.head(config.UPLOAD_MAX_ROWS)

    uid = uuid.uuid4().hex[:12]
    _uploads[uid] = df
    while len(_uploads) > 5:
        _uploads.popitem(last=False)

    head = df.head(5)
    preview = head.astype(object).where(pd.notna(head), "").astype(str).to_dict(orient="records")
    return UploadResponse(
        upload_id=uid, filename=filename, columns=list(df.columns),
        n_rows=len(df), total_rows=total_rows, truncated=truncated,
        max_rows=config.UPLOAD_MAX_ROWS, preview=preview, dropped_columns=dropped,
    )


# ================== API (declared before the static mount so /api wins) ======
@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        ready=engine.ready,
        embedding_backend="openai" if config.USE_OPENAI else "huggingface",
        total_products=len(engine.ids),
    )


# ---------- dataset (default vs uploaded) ----------
@app.get("/api/dataset", response_model=DatasetInfo)
def dataset():
    return engine.dataset_info()


@app.post("/api/dataset/reset", response_model=DatasetInfo)
def dataset_reset():
    engine.load()
    return engine.dataset_info()


@app.get("/api/dataset/download")
def dataset_download():
    """Download the currently active dataset as CSV (e.g. a SQL query result)."""
    if engine.df is None:
        raise HTTPException(status_code=404, detail="No active dataset")
    return _csv_response(engine.df, "dataset.csv")


@app.get("/api/upload/{upload_id}/download")
def upload_download(upload_id: str):
    """Download a just-staged upload/query result as CSV (before building)."""
    df = _uploads.get(upload_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Upload expired — please re-run")
    return _csv_response(df, f"data_{upload_id}.csv")


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = _parse_table(raw, file.filename or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")
    return _register_dataframe(df, file.filename or "data")


@app.post("/api/sql/query", response_model=UploadResponse)
def sql_query(req: SQLQueryRequest):
    """Connect to a SQL DB, run a read-only query, and stage the result rows.

    The password is used only here to open the connection (then disposed) and is
    never stored, logged, or returned. Only the resulting rows are kept.
    """
    if req.dialect not in sql_source.DIALECTS:
        raise HTTPException(status_code=400, detail=f"Unsupported dialect: {req.dialect}")

    password = req.password.get_secret_value() if req.password else ""
    try:
        df = sql_source.run_query(
            req.dialect, req.host, req.port, req.user, password, req.database,
            req.query, max_rows=config.UPLOAD_MAX_ROWS,
            schema=req.db_schema, warehouse=req.warehouse, role=req.role,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModuleNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Database driver not installed: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection/query failed: {e}")
    finally:
        password = None  # drop the plaintext password reference promptly

    return _register_dataframe(df, f"{req.dialect}:{req.database or req.host}")


@app.post("/api/upload/{upload_id}/build", response_model=DatasetInfo)
def upload_build(upload_id: str, req: BuildIndexRequest):
    df = _uploads.get(upload_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Upload expired — please upload again")
    cols = [c for c in req.search_columns if c in df.columns]
    if not cols:
        raise HTTPException(status_code=400, detail="Select at least one valid column")

    # Optional user-chosen row limit (already bounded by the hard cap at upload).
    if req.max_rows and req.max_rows > 0:
        df = df.head(req.max_rows)

    try:
        engine.build_index(df, search_columns=cols, source="uploaded",
                           name=f"uploaded ({len(df)} rows)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not build index: {e}")
    _uploads.pop(upload_id, None)
    return engine.dataset_info()


@app.get("/api/search", response_model=SearchResponse)
def search(query: str, page: int = 1):
    return engine.search(query, page=page)


@app.get("/api/trending", response_model=TrendingResponse)
def trending(limit: int | None = None):
    return TrendingResponse(products=engine.trending(limit))


@app.get("/api/product/{sku}", response_model=ProductDetailResponse)
def product(sku: str):
    detail = engine.product_detail(sku)
    if detail is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return detail


# ================== Frontend (React build) ==================
# Mounted at "/" LAST so the explicit /api routes above take precedence.
if config.FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(config.FRONTEND_DIST), html=True),
        name="frontend",
    )
else:
    @app.get("/")
    def frontend_not_built():
        return JSONResponse(
            status_code=200,
            content={
                "message": (
                    "Frontend not built yet. Run `cd frontend && npm install && "
                    "npm run build`, or use ./run.sh / docker compose up."
                ),
                "api_docs": "/docs",
            },
        )
