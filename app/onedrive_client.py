"""
OneDrive client for PandasAI App.
Simplified from AppV3 - only list files and download.
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import List, Optional
from urllib.parse import quote

import pandas as pd
import requests

from . import onedrive_config as config


def get_access_token() -> str:
    """Get Azure AD access token via client credentials."""
    if not (config.MS_TENANT_ID and config.MS_CLIENT_ID and config.MS_CLIENT_SECRET):
        raise RuntimeError("OneDrive credentials not configured")

    token_url = f"https://login.microsoftonline.com/{config.MS_TENANT_ID}/oauth2/v2.0/token"
    payload = {
        "client_id": config.MS_CLIENT_ID,
        "client_secret": config.MS_CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": config.GRAPH_SCOPE,
    }

    resp = requests.post(token_url, data=payload, timeout=10)
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("No access_token in response")
    return token


def _graph_get(url: str, token: str) -> dict:
    """GET request with retry on 429."""
    headers = {"Authorization": f"Bearer {token}"}
    for _ in range(5):
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "2"))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def list_files(token: str) -> List[dict]:
    """List all Excel/CSV files in configured OneDrive folder."""
    results: List[dict] = []
    root_path = config.ONEDRIVE_ROOT_PATH.strip("/")
    encoded_path = quote(root_path, safe="/")
    drive_id = config.ONEDRIVE_DRIVE_ID

    # Get root folder
    root_url = f"{config.GRAPH_BASE_URL}/drives/{drive_id}/root:/{encoded_path}"
    try:
        root_item = _graph_get(root_url, token)
    except Exception:
        return results

    if "id" not in root_item:
        return results

    # Traverse folders
    stack = [(root_item["id"], root_path)]
    while stack:
        parent_id, parent_path = stack.pop()
        next_url = f"{config.GRAPH_BASE_URL}/drives/{drive_id}/items/{parent_id}/children"

        while next_url:
            data = _graph_get(next_url, token)
            for item in data.get("value", []):
                name = item.get("name", "")
                item_id = item.get("id")
                if not item_id:
                    continue

                child_path = f"{parent_path}/{name}"

                if item.get("folder"):
                    stack.append((item_id, child_path))
                    continue

                if not any(name.lower().endswith(ext) for ext in config.SUPPORTED_EXTENSIONS):
                    continue

                results.append({
                    "id": item_id,
                    "name": name,
                    "path": child_path,
                    "size": item.get("size", 0),
                    "downloadUrl": item.get("@microsoft.graph.downloadUrl"),
                    "webUrl": item.get("webUrl"),
                    "lastModified": item.get("lastModifiedDateTime"),
                })

            next_url = data.get("@odata.nextLink")

    return results


def get_file_details(token: str, file_id: str) -> dict:
    """Get file details by ID (useful for refreshing download URL)."""
    drive_id = config.ONEDRIVE_DRIVE_ID
    url = f"{config.GRAPH_BASE_URL}/drives/{drive_id}/items/{file_id}"
    
    try:
        return _graph_get(url, token)
    except Exception:
        return {}


def download_file(download_url: str) -> bytes:
    """Download file bytes."""
    resp = requests.get(download_url, timeout=300)
    resp.raise_for_status()
    return resp.content


def get_excel_sheets(file_bytes: bytes) -> List[str]:
    """Get sheet names from Excel file."""
    try:
        xls = pd.ExcelFile(BytesIO(file_bytes))
        return xls.sheet_names
    except Exception:
        return []


def read_file_to_df(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read file bytes to DataFrame.
    
    Args:
        file_bytes: File content bytes
        filename: Filename (for format detection)
        sheet_name: Sheet name for Excel files
        nrows: Optional - limit number of rows to read (for quick preview)
    """
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes), nrows=nrows)
        return df.fillna("")
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name or 0, nrows=nrows)
        return df.fillna("")
    else:
        raise ValueError(f"Unsupported file: {filename}")


def upload_file(file_path: Path, destination_filename: str) -> dict:
    """Upload a file to the configured OneDrive root path.
    
    Args:
        file_path: Local path to the file to upload
        destination_filename: Name of the file in OneDrive
        
    Returns:
        dict: API response with file metadata
    """
    token = get_access_token()
    
    # Prepare upload URL
    root_path = config.ONEDRIVE_ROOT_PATH.strip("/")
    encoded_path = quote(f"{root_path}/{destination_filename}", safe="")
    drive_id = config.ONEDRIVE_DRIVE_ID
    
    upload_url = f"{config.GRAPH_BASE_URL}/drives/{drive_id}/root:/{encoded_path}:/content"
    
    # Read file content
    with open(file_path, "rb") as f:
        file_content = f.read()
        
    # Upload (PUT)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    }
    
    resp = requests.put(upload_url, headers=headers, data=file_content, timeout=60)
    resp.raise_for_status()
    
    return resp.json()
