from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd
from pandas import DataFrame

from .data_store import DatasetCatalog, DatasetRecord
from .settings import AppSettings

settings = AppSettings()

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xls", ".xlsx"}
PARQUET_CACHE_DIR = settings.upload_dir / "_parquet_cache"
PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Metadata file to track cache info
CACHE_METADATA_FILE = PARQUET_CACHE_DIR / "_metadata.json"

# Chunked upload settings
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks for writing to disk


@dataclass
class CachedDataInfo:
    """Info about a cached parquet file."""
    cache_path: Path
    display_name: str
    original_file: str
    sheet_name: Optional[str]
    n_rows: int
    n_cols: int
    cached_at: str
    file_size_mb: float
    transform_code: Optional[str] = None
    source_metadata: Optional[dict] = None
    description: Optional[str] = None
    column_descriptions: Optional[dict] = None
    stored_path: Optional[str] = None  # Original source file path
    source_url: Optional[str] = None  # Original source URL (e.g. OneDrive)
    transform_explanation: Optional[str] = None


def _load_cache_metadata() -> dict:
    """Load cache metadata from JSON file."""
    if CACHE_METADATA_FILE.exists():
        try:
            return json.loads(CACHE_METADATA_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache_metadata(metadata: dict) -> None:
    """Save cache metadata to JSON file."""
    CACHE_METADATA_FILE.write_text(json.dumps(metadata, indent=2))


def list_all_cached_data() -> List[CachedDataInfo]:
    """List all parquet files in the cache folder with their metadata."""
    result = []
    metadata = _load_cache_metadata()
    
    for parquet_file in PARQUET_CACHE_DIR.glob("*.parquet"):
        cache_key = parquet_file.stem  # filename without extension
        
        # Get metadata if available
        info = metadata.get(cache_key, {})
        
        # Read parquet to get row/col count if not in metadata
        try:
            df = pd.read_parquet(parquet_file)
            n_rows, n_cols = df.shape
        except Exception:
            continue  # Skip corrupted files
        
        file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
        cached_at = datetime.fromtimestamp(parquet_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        
        result.append(CachedDataInfo(
            cache_path=parquet_file,
            display_name=info.get("display_name", parquet_file.stem),
            original_file=info.get("original_file", "Unknown"),
            sheet_name=info.get("sheet_name"),
            n_rows=n_rows,
            n_cols=n_cols,
            cached_at=cached_at,
            file_size_mb=round(file_size_mb, 2),
            transform_code=info.get("transform_code"),
            source_metadata=info.get("source_metadata"),
            transform_explanation=info.get("transform_explanation"),
        ))
    
    # Sort by cached date, newest first
    result.sort(key=lambda x: x.cached_at, reverse=True)
    return result


def delete_cached_data(cache_path: Path) -> bool:
    """Delete a cached parquet file and its metadata."""
    try:
        cache_key = cache_path.stem
        if cache_path.exists():
            cache_path.unlink()
        
        # Remove from metadata
        metadata = _load_cache_metadata()
        if cache_key in metadata:
            del metadata[cache_key]
            _save_cache_metadata(metadata)
        return True
    except Exception:
        return False


def _detect_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _parquet_cache_path(original_path: Path, sheet_name: str | int | None = None) -> Path:
    """Generate a unique parquet cache filename based on source path + sheet."""
    key = f"{original_path}:{sheet_name or 0}"
    h = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]
    return PARQUET_CACHE_DIR / f"{h}.parquet"


def has_parquet_cache(path: Path, sheet_name: str | int | None = None) -> bool:
    """Check if a valid parquet cache exists for the given file/sheet."""
    cache_path = _parquet_cache_path(path, sheet_name)
    if not cache_path.exists():
        return False
    # If source file doesn't exist anymore, cache is still valid
    if not path.exists():
        return True
    return cache_path.stat().st_mtime >= path.stat().st_mtime


def build_parquet_cache(
    path: Path, 
    sheet_name: str | int | None = None,
    display_name: str | None = None,
    source_metadata: dict | None = None,
    transform_explanation: Optional[str] = None
) -> Tuple[Path, int, int]:
    """Build parquet cache for a file/sheet if it doesn't exist.
    
    Args:
        path: Path to source file
        sheet_name: Sheet name for Excel files
        display_name: Human-readable name for the cache
    
    Returns: (cache_path, n_rows, n_cols)
    """
    cache_path = _parquet_cache_path(path, sheet_name)
    cache_key = cache_path.stem
    
    if not has_parquet_cache(path, sheet_name):
        df = _read_dataframe_raw(path, sheet_name)
        df = _downcast_dtypes(df)
        df = _sanitize_for_parquet(df)
        n_rows, n_cols = df.shape
        df.to_parquet(cache_path, index=False)
    else:
        # Cache exists - read to get shape
        df = pd.read_parquet(cache_path)
        n_rows, n_cols = df.shape
    
    # Save metadata
    metadata = _load_cache_metadata()
    if display_name is None:
        display_name = f"{path.stem}"
        if sheet_name:
            display_name += f" - {sheet_name}"
    
    metadata[cache_key] = {
        "display_name": display_name,
        "original_file": str(path.name),
        "sheet_name": sheet_name if isinstance(sheet_name, str) else None,
        "n_rows": n_rows,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "source_metadata": source_metadata,
        "transform_explanation": transform_explanation,
    }
    _save_cache_metadata(metadata)
    
    return cache_path, n_rows, n_cols


def build_parquet_cache_from_df(
    df: DataFrame,
    display_name: str,
    original_file: str = "transformed",
    sheet_name: str | None = None,
    transform_code: str | None = None,
    source_metadata: dict | None = None,
    transform_explanation: Optional[str] = None,
) -> Tuple[Path, int, int]:
    """Build parquet cache directly from a DataFrame (for transformed data).
    
    Args:
        df: DataFrame to cache
        display_name: Human-readable name for the cache
        original_file: Original filename for reference
        sheet_name: Original sheet name for reference
    
    Returns: (cache_path, n_rows, n_cols)
    """
    import hashlib
    from datetime import datetime
    
    # Generate unique cache key based on display name + timestamp
    key = f"{display_name}:{datetime.now().isoformat()}"
    h = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:12]
    cache_path = PARQUET_CACHE_DIR / f"{h}.parquet"
    cache_key = cache_path.stem
    
    # Process and save
    df = _downcast_dtypes(df.copy())
    df = _sanitize_for_parquet(df)
    n_rows, n_cols = df.shape
    df.to_parquet(cache_path, index=False)
    
    # Save metadata
    metadata = _load_cache_metadata()
    metadata[cache_key] = {
        "display_name": display_name,
        "original_file": original_file,
        "sheet_name": sheet_name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "transformed": True,
        "transform_code": transform_code,
        "source_metadata": source_metadata,
        "transform_explanation": transform_explanation,
    }
    _save_cache_metadata(metadata)
    
    return cache_path, n_rows, n_cols


def update_existing_parquet_cache(
    cache_path: Path,
    df: DataFrame,
    transform_code: str | None = None,
    transform_explanation: Optional[str] = None,
) -> Tuple[int, int]:
    """Update an existing parquet cache file with new data (overwrite)."""
    
    # Process and save
    df = _downcast_dtypes(df.copy())
    df = _sanitize_for_parquet(df)
    n_rows, n_cols = df.shape
    df.to_parquet(cache_path, index=False)
    
    # Update metadata
    cache_key = cache_path.stem
    metadata = _load_cache_metadata()
    
    if cache_key in metadata:
        metadata[cache_key]["n_rows"] = n_rows
        metadata[cache_key]["n_cols"] = n_cols
        if transform_code:
            metadata[cache_key]["transform_code"] = transform_code
            metadata[cache_key]["transformed"] = True
            
        if transform_explanation:
            metadata[cache_key]["transform_explanation"] = transform_explanation
            
        _save_cache_metadata(metadata)
    
    return n_rows, n_cols


def _sanitize_for_parquet(df: DataFrame) -> DataFrame:
    """Convert ALL object columns to strings for parquet compatibility.
    
    This is the simplest and most robust approach - any column with dtype 'object'
    (which includes mixed types, bytes, ints disguised as objects, etc.) gets
    converted to string. This avoids all Arrow type errors.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)
    return df


def _downcast_dtypes(df: DataFrame) -> DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def _read_dataframe_raw(path: Path, sheet_name: str | int | None = None, nrows: int | None = None) -> DataFrame:
    """Read from original file format (CSV/Excel). Use nrows to limit rows read.
    
    IMPORTANT: Reads ALL columns - no column limit applied.
    """
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=nrows)
        return df.fillna("")
    if suffix in {".xls", ".xlsx"}:
        # Read ALL columns (no usecols limit)
        # Use header=None initially to get ALL data including potential merged headers
        df = pd.read_excel(
            path, 
            sheet_name=sheet_name or 0, 
            nrows=nrows,
            # Don't skip any columns - read everything
        )
        return df.fillna("")
    raise ValueError(f"Unsupported file type: {suffix}")


def _read_dataframe(path: Path, sheet_name: str | int | None = None) -> DataFrame:
    """Read DataFrame, using parquet cache if available for faster loads."""
    cache_path = _parquet_cache_path(path, sheet_name)
    
    # Use cache if it exists and is newer than source
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache_path)
    
    # Read from original, downcast, and cache as parquet
    df = _read_dataframe_raw(path, sheet_name)
    df = _downcast_dtypes(df)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass  # Caching is best-effort; don't fail if it doesn't work
    return df


def get_excel_sheet_names(path: Path) -> List[str]:
    """Return list of sheet names for an Excel file, or empty list for non-Excel."""
    suffix = path.suffix.lower()
    if suffix not in {".xls", ".xlsx"}:
        return []
    xls = pd.ExcelFile(path)
    return xls.sheet_names


def _profile_df(df: DataFrame) -> Tuple[int, int]:
    return df.shape if isinstance(df, DataFrame) else (None, None)


def ensure_supported(upload_name: str) -> None:
    suffix = Path(upload_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )


def persist_upload(upload, owner_id: str, catalog: DatasetCatalog, progress_callback=None) -> Tuple[str, DataFrame]:  # pragma: no cover - streamlit runtime
    """Save the uploaded file to disk in chunks, register metadata, and return (dataset_id, DataFrame).
    
    Args:
        upload: Streamlit UploadedFile object
        owner_id: User/session identifier
        catalog: DatasetCatalog instance
        progress_callback: Optional callable(bytes_written, total_bytes) for progress updates
    """

    ensure_supported(upload.name)
    upload_dir = settings.upload_dir / owner_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    size_bytes = upload.size
    max_bytes = settings.upload_max_mb * 1024 * 1024
    if size_bytes and size_bytes > max_bytes:
        raise ValueError(
            f"File too large ({size_bytes/1_048_576:.1f} MB). Max allowed is {settings.upload_max_mb} MB."
        )

    suffix = Path(upload.name).suffix.lower()
    stored_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{owner_id}{suffix}"
    stored_path = upload_dir / stored_name

    # Write file in chunks to avoid memory spikes for large files
    bytes_written = 0
    with open(stored_path, "wb") as fh:
        while True:
            chunk = upload.read(CHUNK_SIZE)
            if not chunk:
                break
            fh.write(chunk)
            bytes_written += len(chunk)
            if progress_callback and size_bytes:
                progress_callback(bytes_written, size_bytes)
    
    # Reset upload file pointer in case it's needed again
    upload.seek(0)

    df = _read_dataframe(stored_path)
    n_rows, n_cols = _profile_df(df)
    dataset_id = catalog.add_dataset(
        owner_id=owner_id,
        display_name=upload.name,
        original_name=upload.name,
        stored_path=stored_path,
        mime_type=_detect_mime(upload.name),
        file_size=size_bytes,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    return dataset_id, df


def load_dataset(record: DatasetRecord, sheet_name: str | int | None = None, nrows: int | None = None) -> DataFrame:
    """Load dataset from cache/file. Use nrows to limit rows for preview."""
    path = Path(record.stored_path)
    if not path.exists():
        raise FileNotFoundError(f"Stored dataset not found: {record.stored_path}")
    
    df = _read_dataframe(path, sheet_name=sheet_name)
    if nrows is not None:
        df = df.head(nrows)
    return df


def load_dataset_preview(record: DatasetRecord, sheet_name: str | int | None = None, nrows: int = 20) -> DataFrame:
    """Load only top N rows for lightweight preview.
    
    - If parquet cache exists: read from cache (fast)
    - If no cache: read only N rows from source (don't build full cache yet)
    """
    path = Path(record.stored_path)
    if not path.exists():
        raise FileNotFoundError(f"Stored dataset not found: {record.stored_path}")
    
    # If cache exists, read from parquet (fast)
    if has_parquet_cache(path, sheet_name):
        cache_path = _parquet_cache_path(path, sheet_name)
        return pd.read_parquet(cache_path).head(nrows)
    
    # No cache yet - read only N rows directly from source (quick preview without full load)
    df = _read_dataframe_raw(path, sheet_name=sheet_name, nrows=nrows)
    return _downcast_dtypes(df)
