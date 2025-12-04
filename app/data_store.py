from __future__ import annotations

import sqlite3
import threading
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4

from .settings import CATALOG_DB
from app.logger import setup_logger

logger = setup_logger("data_store")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT UNIQUE NOT NULL,
    owner_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    source_url TEXT,
    mime_type TEXT,
    file_size INTEGER,
    n_rows INTEGER,
    n_cols INTEGER,
    created_at TEXT NOT NULL
);
"""

_CREATE_CACHED_SHEETS_SQL = """
CREATE TABLE IF NOT EXISTS cached_sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_id TEXT UNIQUE NOT NULL,
    dataset_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    sheet_name TEXT,
    display_name TEXT NOT NULL,
    n_rows INTEGER,
    n_cols INTEGER,
    cached_at TEXT NOT NULL,
    description TEXT,
    column_descriptions TEXT,
    transform_explanation TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);
"""

_LIST_SQL = """
SELECT dataset_id, owner_id, display_name, original_name, stored_path, source_url, mime_type,
       file_size, n_rows, n_cols, created_at
FROM datasets
WHERE owner_id = ?
ORDER BY datetime(created_at) DESC;
"""

_GET_SQL = """
SELECT dataset_id, owner_id, display_name, original_name, stored_path, source_url, mime_type,
       file_size, n_rows, n_cols, created_at
FROM datasets
WHERE dataset_id = ? AND (owner_id = ? OR ? IS NULL)
LIMIT 1;
"""

_LIST_CACHED_SHEETS_SQL = """
SELECT cs.cache_id, cs.dataset_id, cs.owner_id, cs.sheet_name, cs.display_name,
       cs.n_rows, cs.n_cols, cs.cached_at, cs.description, cs.column_descriptions, cs.transform_explanation, d.stored_path, d.source_url
FROM cached_sheets cs
JOIN datasets d ON cs.dataset_id = d.dataset_id
WHERE cs.owner_id = ?
ORDER BY datetime(cs.cached_at) DESC;
"""


@dataclass
class DatasetRecord:
    dataset_id: str
    owner_id: str
    display_name: str
    original_name: str
    stored_path: str
    source_url: Optional[str]
    mime_type: Optional[str]
    file_size: Optional[int]
    n_rows: Optional[int]
    n_cols: Optional[int]
    created_at: str


@dataclass
class CachedSheetRecord:
    cache_id: str
    dataset_id: str
    owner_id: str
    sheet_name: Optional[str]
    display_name: str
    n_rows: Optional[int]
    n_cols: Optional[int]
    cached_at: str
    stored_path: str
    source_url: Optional[str]
    description: Optional[str] = None
    column_descriptions: Optional[str] = None  # JSON string
    transform_explanation: Optional[str] = None


class DatasetCatalog:
    """Simple SQLite-backed catalog for uploaded datasets."""

    def __init__(self, db_path: Path = CATALOG_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_CACHED_SHEETS_SQL)
            
            # Migration: Add new columns if they don't exist
            try:
                conn.execute("ALTER TABLE cached_sheets ADD COLUMN description TEXT")
            except sqlite3.OperationalError:
                pass  # Column likely exists
                
            try:
                conn.execute("ALTER TABLE cached_sheets ADD COLUMN column_descriptions TEXT")
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute("ALTER TABLE cached_sheets ADD COLUMN transform_explanation TEXT")
            except sqlite3.OperationalError:
                pass
                
            # Migration: Add source_url to datasets
            try:
                conn.execute("ALTER TABLE datasets ADD COLUMN source_url TEXT")
            except sqlite3.OperationalError:
                pass

    def add_dataset(
        self,
        owner_id: str,
        display_name: str,
        original_name: str,
        stored_path: Path,
        source_url: Optional[str] = None,
        mime_type: Optional[str] = None,
        file_size: Optional[int] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> str:
        dataset_id = str(uuid4())
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, owner_id, display_name, original_name, stored_path, source_url,
                    mime_type, file_size, n_rows, n_cols, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    dataset_id,
                    owner_id,
                    display_name,
                    original_name,
                    str(stored_path),
                    source_url,
                    mime_type,
                    file_size,
                    n_rows,
                    n_cols,
                    created_at,
                ),
            )
        return dataset_id

    def list_datasets(self, owner_id: str) -> List[DatasetRecord]:
        with self._connect() as conn:
            rows = conn.execute(_LIST_SQL, (owner_id,)).fetchall()
        return [DatasetRecord(**dict(row)) for row in rows]

    def get_dataset(self, dataset_id: str, owner_id: Optional[str] = None) -> Optional[DatasetRecord]:
        with self._connect() as conn:
            row = conn.execute(_GET_SQL, (dataset_id, owner_id, owner_id)).fetchone()
        return DatasetRecord(**dict(row)) if row else None

    def delete_dataset(self, dataset_id: str, owner_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM datasets WHERE dataset_id = ? AND owner_id = ?",
                (dataset_id, owner_id),
            )
            return cur.rowcount > 0

    def purge_orphans(self, existing_paths: Iterable[Path]) -> int:
        existing = {str(p) for p in existing_paths}
        with self._connect() as conn:
            rows = conn.execute("SELECT dataset_id, stored_path FROM datasets").fetchall()
            missing = [row["dataset_id"] for row in rows if row["stored_path"] not in existing]
            for dataset_id in missing:
                conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
        return len(missing)

    # ---- Cached Sheets Management ----

    def add_cached_sheet(
        self,
        dataset_id: str,
        owner_id: str,
        sheet_name: Optional[str],
        display_name: str,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        description: Optional[str] = None,
        column_descriptions: Optional[dict] = None,
        transform_explanation: Optional[str] = None,
    ) -> str:
        """Register a cached sheet. Returns cache_id."""
        cache_id = str(uuid4())
        cached_at = datetime.now(UTC).isoformat()
        
        col_desc_json = json.dumps(column_descriptions) if column_descriptions else None
        
        with self._lock, self._connect() as conn:
            # Check if already cached
            existing = conn.execute(
                "SELECT cache_id FROM cached_sheets WHERE dataset_id = ? AND (sheet_name = ? OR (sheet_name IS NULL AND ? IS NULL))",
                (dataset_id, sheet_name, sheet_name),
            ).fetchone()
            if existing:
                return existing["cache_id"]
            
            conn.execute(
                """
                INSERT INTO cached_sheets (
                    cache_id, dataset_id, owner_id, sheet_name, display_name,
                    n_rows, n_cols, cached_at, description, column_descriptions, transform_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (cache_id, dataset_id, owner_id, sheet_name, display_name, n_rows, n_cols, cached_at, description, col_desc_json, transform_explanation),
            )
        logger.info(f"Created cache for dataset {dataset_id}, sheet {sheet_name} (ID: {cache_id})")
        return cache_id

    def list_cached_sheets(self, owner_id: str) -> List[CachedSheetRecord]:
        """List all cached sheets for a user."""
        with self._connect() as conn:
            rows = conn.execute(_LIST_CACHED_SHEETS_SQL, (owner_id,)).fetchall()
        return [CachedSheetRecord(**dict(row)) for row in rows]

    def get_cached_sheet(self, cache_id: str) -> Optional[CachedSheetRecord]:
        """Get a cached sheet by cache_id."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cs.cache_id, cs.dataset_id, cs.owner_id, cs.sheet_name, cs.display_name,
                       cs.n_rows, cs.n_cols, cs.cached_at, cs.description, cs.column_descriptions, cs.transform_explanation, d.stored_path
                FROM cached_sheets cs
                JOIN datasets d ON cs.dataset_id = d.dataset_id
                WHERE cs.cache_id = ?
                LIMIT 1;
                """,
                (cache_id,),
            ).fetchone()
        return CachedSheetRecord(**dict(row)) if row else None

    def delete_cached_sheet(self, cache_id: str, owner_id: str) -> bool:
        """Delete a cached sheet record."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cached_sheets WHERE cache_id = ? AND owner_id = ?",
                (cache_id, owner_id),
            )
            return cur.rowcount > 0

    def update_cached_sheet_metadata(
        self,
        cache_id: str,
        description: Optional[str] = None,
        column_descriptions: Optional[dict] = None,
        transform_explanation: Optional[str] = None,
    ) -> bool:
        """Update metadata for a cached sheet."""
        col_desc_json = json.dumps(column_descriptions) if column_descriptions else None
        
        with self._lock, self._connect() as conn:
            # Build query dynamically based on provided fields
            updates = []
            params = []
            
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            
            if column_descriptions is not None:
                updates.append("column_descriptions = ?")
                params.append(col_desc_json)
                
            if transform_explanation is not None:
                updates.append("transform_explanation = ?")
                params.append(transform_explanation)
            
            if not updates:
                return False
                
            params.append(cache_id)
            
            cur = conn.execute(
                f"""
                UPDATE cached_sheets 
                SET {', '.join(updates)}
                WHERE cache_id = ?
                """,
                tuple(params)
            )
            return cur.rowcount > 0

    def update_cached_sheet_stats(
        self,
        cache_id: str,
        n_rows: int,
        n_cols: int
    ) -> bool:
        """Update row/col counts for a cached sheet."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE cached_sheets 
                SET n_rows = ?, n_cols = ?
                WHERE cache_id = ?
                """,
                (n_rows, n_cols, cache_id),
            )
            return cur.rowcount > 0
