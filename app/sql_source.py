"""Connect to a SQL database, run a read-only query, return the rows.

Security model (see README):
  - Credentials are used ONCE to open a connection, run the query, fetch rows,
    then the engine is disposed. They are never written to disk, never logged,
    and never returned to the client.
  - Only read-only statements (SELECT / WITH / SHOW / EXPLAIN / DESCRIBE) are
    allowed, so a query cannot modify the source database.
  - Errors are sanitized to strip the password before they leave this module.
"""
import re
from typing import Optional

import pandas as pd

# Supported dialects -> (SQLAlchemy driver, default port).
DIALECTS = {
    "postgresql": ("postgresql+psycopg2", 5432),
    "mysql": ("mysql+pymysql", 3306),
    "mssql": ("mssql+pymssql", 1433),
    "snowflake": ("snowflake", None),
}

_READ_FIRST_WORDS = {"select", "with", "show", "explain", "describe", "desc"}
_WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|"
    r"call|exec|execute|replace|attach|copy|into)\b",
    re.IGNORECASE,
)


def is_read_only(query: str) -> bool:
    """Allow a single read-only statement only."""
    if not query or not query.strip():
        return False
    # Strip block + line comments.
    q = re.sub(r"/\*.*?\*/", " ", query, flags=re.S)
    q = re.sub(r"--[^\n]*", " ", q)
    q = q.strip().rstrip(";").strip()
    if not q:
        return False
    # Reject multiple statements.
    if ";" in q:
        return False
    first = q.split(None, 1)[0].lower()
    if first not in _READ_FIRST_WORDS:
        return False
    if _WRITE_WORDS.search(q):
        return False
    return True


def _sanitize(text: str, secret: Optional[str]) -> str:
    """Remove the password (if present) from any text we surface."""
    if secret:
        text = text.replace(secret, "***")
    return text


def build_url(dialect: str, host: str, port, user: str, password: str,
              database: str, *, schema=None, warehouse=None, role=None):
    """Build a SQLAlchemy URL/object. Password is URL-escaped safely."""
    if dialect not in DIALECTS:
        raise ValueError(f"Unsupported dialect: {dialect}")

    if dialect == "snowflake":
        # snowflake-sqlalchemy provides its own URL builder; host == account.
        from snowflake.sqlalchemy import URL as SnowflakeURL  # lazy import

        params = dict(account=host, user=user, password=password, database=database)
        if schema:
            params["schema"] = schema
        if warehouse:
            params["warehouse"] = warehouse
        if role:
            params["role"] = role
        return SnowflakeURL(**params)

    from sqlalchemy.engine import URL  # lazy import

    driver, default_port = DIALECTS[dialect]
    return URL.create(
        drivername=driver,
        username=user,
        password=password,        # URL.create escapes special characters
        host=host,
        port=int(port) if port else default_port,
        database=database,
    )


def run_query(dialect: str, host: str, port, user: str, password: str,
              database: str, query: str, *, max_rows: int = 20000,
              schema=None, warehouse=None, role=None) -> pd.DataFrame:
    """Open a connection, run the query, return up to `max_rows` rows, dispose."""
    from sqlalchemy import create_engine, text  # lazy import
    from sqlalchemy.pool import NullPool

    if not is_read_only(query):
        raise ValueError("Only a single read-only query (SELECT/WITH/SHOW) is allowed")

    url = build_url(dialect, host, port, user, password, database,
                    schema=schema, warehouse=warehouse, role=role)

    # connect timeout per driver (best effort).
    connect_args = {}
    if dialect == "postgresql":
        connect_args = {"connect_timeout": 10}
    elif dialect == "mysql":
        connect_args = {"connect_timeout": 10}
    elif dialect == "mssql":
        connect_args = {"timeout": 15, "login_timeout": 10}
    elif dialect == "snowflake":
        connect_args = {"login_timeout": 15}

    engine = create_engine(
        url,
        poolclass=NullPool,        # no pooled/lingering connections
        hide_parameters=True,      # keep query params out of error messages
        connect_args=connect_args,
    )
    try:
        with engine.connect() as conn:
            chunks = pd.read_sql(text(query), conn, chunksize=max_rows)
            try:
                df = next(chunks)          # first chunk = up to max_rows
            except StopIteration:
                df = pd.DataFrame()
        return df
    except Exception as e:  # sanitize before re-raising
        raise RuntimeError(_sanitize(str(e), password)) from None
    finally:
        engine.dispose()                   # drop the connection immediately
