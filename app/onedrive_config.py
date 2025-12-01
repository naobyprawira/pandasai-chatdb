"""
OneDrive configuration for PandasAI App.
Uses same credentials as AppV3.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Microsoft Graph configuration
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "").strip()
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID", "").strip()
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET", "").strip()
ONEDRIVE_DRIVE_ID = os.getenv("ONEDRIVE_DRIVE_ID", "").strip()
ONEDRIVE_ROOT_PATH = os.getenv("ONEDRIVE_ROOT_PATH", "Purchasing/Data Files").strip()

GRAPH_SCOPE = "https://graph.microsoft.com/.default"
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

# Supported file types
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


def is_configured() -> tuple[bool, str]:
    """Check if OneDrive is properly configured. Returns (ok, error_msg)."""
    if not MS_TENANT_ID:
        return False, "MS_TENANT_ID not set"
    if not MS_CLIENT_ID:
        return False, "MS_CLIENT_ID not set"
    if not MS_CLIENT_SECRET:
        return False, "MS_CLIENT_SECRET not set"
    if not ONEDRIVE_DRIVE_ID:
        return False, "ONEDRIVE_DRIVE_ID not set"
    return True, ""
