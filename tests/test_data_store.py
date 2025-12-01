from __future__ import annotations

from pathlib import Path

from app.data_store import DatasetCatalog


def test_add_and_get_dataset(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    catalog = DatasetCatalog(db_path=db_path)

    dataset_id = catalog.add_dataset(
        owner_id="user-1",
        display_name="sales.csv",
        original_name="sales.csv",
        stored_path=tmp_path / "sales.csv",
        mime_type="text/csv",
        file_size=123,
        n_rows=10,
        n_cols=3,
    )

    record = catalog.get_dataset(dataset_id, owner_id="user-1")
    assert record is not None
    assert record.display_name == "sales.csv"
    assert record.n_rows == 10


def test_list_scoped_by_owner(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    catalog = DatasetCatalog(db_path=db_path)

    catalog.add_dataset(
        owner_id="alice",
        display_name="a.csv",
        original_name="a.csv",
        stored_path=tmp_path / "a.csv",
    )
    catalog.add_dataset(
        owner_id="bob",
        display_name="b.csv",
        original_name="b.csv",
        stored_path=tmp_path / "b.csv",
    )

    alice_records = catalog.list_datasets("alice")
    assert len(alice_records) == 1
    assert alice_records[0].owner_id == "alice"


def test_add_and_list_cached_sheets(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    catalog = DatasetCatalog(db_path=db_path)

    # First create a dataset
    dataset_id = catalog.add_dataset(
        owner_id="user-1",
        display_name="data.xlsx",
        original_name="data.xlsx",
        stored_path=tmp_path / "data.xlsx",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        file_size=5000,
        n_rows=100,
        n_cols=5,
    )

    # Add a cached sheet
    cache_id = catalog.add_cached_sheet(
        dataset_id=dataset_id,
        owner_id="user-1",
        sheet_name="Sheet1",
        display_name="data.xlsx - Sheet1",
        n_rows=100,
        n_cols=5,
    )
    assert cache_id is not None

    # List cached sheets
    cached = catalog.list_cached_sheets("user-1")
    assert len(cached) == 1
    assert cached[0].display_name == "data.xlsx - Sheet1"
    assert cached[0].sheet_name == "Sheet1"

    # Get specific cached sheet
    record = catalog.get_cached_sheet(cache_id)
    assert record is not None
    assert record.n_rows == 100

    # Adding same sheet again should return existing cache_id
    same_cache_id = catalog.add_cached_sheet(
        dataset_id=dataset_id,
        owner_id="user-1",
        sheet_name="Sheet1",
        display_name="data.xlsx - Sheet1",
        n_rows=100,
        n_cols=5,
    )
    assert same_cache_id == cache_id

    # Delete cached sheet
    deleted = catalog.delete_cached_sheet(cache_id, "user-1")
    assert deleted is True
    assert catalog.get_cached_sheet(cache_id) is None
