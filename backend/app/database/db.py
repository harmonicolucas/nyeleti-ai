"""Read-only access to the analysis database."""
from __future__ import annotations

import json
import threading
from functools import lru_cache

import duckdb
from fastapi import HTTPException

from ml import config

_lock = threading.Lock()
_connection: duckdb.DuckDBPyConnection | None = None


def connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        with _lock:
            if _connection is None:
                if not config.DB_PATH.exists():
                    raise HTTPException(
                        status_code=503,
                        detail="Database not built yet. Run the data pipeline first (see README).",
                    )
                _connection = duckdb.connect(str(config.DB_PATH), read_only=True)
    return _connection


def cursor() -> duckdb.DuckDBPyConnection:
    """A cursor of its own for the calling thread."""
    return connection().cursor()


@lru_cache(maxsize=1)
def metadata() -> dict:
    rows = cursor().execute("SELECT key, value FROM metadata").fetchall()
    return {key: json.loads(value) for key, value in rows}


def has_table(name: str) -> bool:
    row = cursor().execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row[0])
