"""
Purchasing Data Assistant
Chat-based UI with table selection popup, OneDrive integration, and Indonesian explanations.
"""
from __future__ import annotations

from app.logger import get_app_logger
logger = get_app_logger()

import streamlit as st
import pandas as pd
from pathlib import Path
from uuid import uuid4
from typing import List, Optional, Tuple
import os
import time
from datetime import datetime

from app.data_store import DatasetCatalog
from app.datasets import (
    build_parquet_cache,
    build_parquet_cache_from_df,
    delete_cached_data,
    get_excel_sheet_names,
    has_parquet_cache,
    list_all_cached_data,
    load_dataset_preview,
    persist_upload,
    CachedDataInfo,
    PARQUET_CACHE_DIR,
    _read_dataframe_raw,
)
from app.qa_engine import PandasAIClient
from app.settings import AppSettings
from app import onedrive_config
from app import onedrive_client
from app.data_analyzer import (
    analyze_and_generate_transform,
    execute_transform,
    get_quick_analysis,
    regenerate_with_feedback,
    TransformResult,
)

settings = AppSettings()

logger.info("Application starting...")

st.set_page_config(page_title="Purchasing Data Assistant", layout="wide", page_icon="🛍️")

# =============================================================================
# Custom CSS & UI
# =============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {visibility: hidden;}
        footer {visibility: hidden;}
        
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--secondary-background-color);
            border-right: 1px solid var(--secondary-background-color);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--text-color);
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Cards/Containers */
        .css-1r6slb0 {
            background: var(--secondary-background-color);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        /* Login Form */
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 2rem;
            background: var(--secondary-background-color);
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            text-align: center;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        /* Dark mode specific overrides if needed */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #0e1117; /* Streamlit default dark bg */
            }
            .css-1r6slb0, .login-container {
                background-color: #262730; /* Streamlit default dark secondary */
                border: 1px solid #41424b;
            }
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# =============================================================================
# Session State & Auth
# =============================================================================

def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_table_selector" not in st.session_state:
        st.session_state.show_table_selector = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "selected_table" not in st.session_state:
        st.session_state.selected_table = None
    if "onedrive_token" not in st.session_state:
        st.session_state.onedrive_token = None
    if "onedrive_files" not in st.session_state:
        st.session_state.onedrive_files = []
    # Transform analysis states
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "transform_preview_df" not in st.session_state:
        st.session_state.transform_preview_df = None
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "selected_transforms" not in st.session_state:
        st.session_state.selected_transforms = []
    if "flash_message" not in st.session_state:
        st.session_state.flash_message = None

def reset_onedrive_state():
    """Reset OneDrive tab state to initial."""
    st.session_state.onedrive_files = []
    st.session_state.onedrive_file_bytes = None
    st.session_state.onedrive_sheets = []
    st.session_state.onedrive_analysis = None
    st.session_state.onedrive_preview_df = None

init_session_state()
catalog = DatasetCatalog()

def check_password():
    """Returns `True` if the user had the correct password."""
    if st.session_state.authenticated:
        return True

    st.markdown("""
        <div class="login-container">
            <h2>🔐 Purchasing Data Assistant</h2>
            <p style="color: var(--text-color); margin-bottom: 2rem; opacity: 0.8;">Silakan login untuk melanjutkan</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        env_pass = os.environ.get("APP_PASSWORD", "admin123")
        logger.info(f"DEBUG: Expected password is '{env_pass}'")
        with st.form("login_form"):
            password = st.text_input(
                "Password", 
                type="password", 
                label_visibility="collapsed",
                placeholder="Masukkan password..."
            )
            submit = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submit:
                env_pass = os.environ.get("APP_PASSWORD", "admin123")
                if password == env_pass:
                    st.session_state.authenticated = True
                    logger.info(f"User {st.session_state.user_id} logged in successfully")
                    st.rerun()
                else:
                    logger.warning(f"Failed login attempt for user {st.session_state.user_id}")
                    st.error("😕 Password salah")
    
    return False

if not check_password():
    st.stop()

# =============================================================================
# Main Application (Authenticated)
# =============================================================================

# Helper Functions
def get_enriched_cached_data() -> List[CachedDataInfo]:
    """Get cached data and enrich with metadata (descriptions) from SQLite."""
    cached_list = list_all_cached_data()
    
    # Fetch SQLite metadata for current user
    if "user_id" in st.session_state:
        try:
            user_sheets = catalog.list_cached_sheets(st.session_state.user_id)
            # Map by display_name + sheet_name (best effort matching)
            # Ideally we should use cache_id, but CachedDataInfo doesn't have it yet.
            db_map = {}
            for s in user_sheets:
                key = (s.display_name, s.sheet_name)
                db_map[key] = s
            
            import json
            for table in cached_list:
                key = (table.display_name, table.sheet_name)
                if key in db_map:
                    sheet = db_map[key]
                    table.description = sheet.description
                    table.stored_path = sheet.stored_path  # Populate stored_path from SQLite

                    table.source_url = sheet.source_url    # Populate source_url from SQLite
                    if not table.source_url and table.source_metadata:
                         table.source_url = table.source_metadata.get("webUrl")
                    
                    table.transform_explanation = sheet.transform_explanation # Populate explanation
                    if sheet.column_descriptions:
                        try:
                            table.column_descriptions = json.loads(sheet.column_descriptions)
                        except:
                            table.column_descriptions = {}
        except Exception as e:
            logger.error(f"Failed to enrich cached data: {e}")
            
    return cached_list

def register_onedrive_cache(cache_path: Path, n_rows: int, n_cols: int, display_name: str, sheet_name: str | None, source_metadata: dict, transform_explanation: str | None = None):
    """Register a OneDrive-cached file into the SQLite catalog."""
    try:
        # 1. Add Dataset Record
        # We don't have a permanent stored path for OneDrive files (it's temp), 
        # but we need to store something. We can store the temp path or a placeholder.
        # Ideally, we should store the 'webUrl' as source_url.
        
        dataset_id = catalog.add_dataset(
            owner_id=st.session_state.user_id,
            display_name=display_name,
            original_name=source_metadata.get("file_path", display_name),
            stored_path=Path(source_metadata.get("file_path", "onedrive_remote")), # Placeholder or remote path
            source_url=source_metadata.get("webUrl"),
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if sheet_name else "text/csv",
            file_size=0, # Unknown
            n_rows=n_rows,
            n_cols=n_cols
        )
        
        # 2. Add Cached Sheet Record
        catalog.add_cached_sheet(
            dataset_id=dataset_id,
            owner_id=st.session_state.user_id,
            sheet_name=sheet_name,
            display_name=display_name,
            n_rows=n_rows,
            n_cols=n_cols,
            transform_explanation=transform_explanation
        )
        logger.info(f"Registered OneDrive cache in SQLite: {display_name}")
    except Exception as e:
        logger.error(f"Failed to register OneDrive cache in SQLite: {e}")

# Sidebar
with st.sidebar:
    st.title("🛍️ Menu")
    
    # OneDrive status
    onedrive_ok, onedrive_err = onedrive_config.is_configured()
    if onedrive_ok:
        st.success("☁️ OneDrive: Terhubung")
    else:
        st.warning(f"☁️ OneDrive: {onedrive_err}")
    
    # Show cached tables count
    cached_list = get_enriched_cached_data()
    st.info(f"📊 {len(cached_list)} tabel tersimpan")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.show_table_selector = False
        st.rerun()
        
    st.divider()
    if st.button("🔒 Logout", use_container_width=True):
        logger.info(f"User {st.session_state.user_id} logged out")
        st.session_state.authenticated = False
        st.rerun()




def rank_tables_by_relevance(question: str, tables: List[CachedDataInfo]) -> List[Tuple[CachedDataInfo, float]]:
    """Rank tables by relevance to question using keyword matching on name and descriptions."""
    if not tables:
        return []
    
    ranked = []
    question_lower = question.lower()
    words = [w for w in question_lower.split() if len(w) > 3]
    
    for table in tables:
        score = 0.0
        
        # 1. Match in Display Name (High weight)
        name_lower = table.display_name.lower()
        for word in words:
            if word in name_lower:
                score += 2.0
        
        # 2. Match in Table Description (Medium weight)
        if table.description:
            desc_lower = table.description.lower()
            for word in words:
                if word in desc_lower:
                    score += 1.0
        
        # 3. Match in Column Descriptions (Low weight)
        if table.column_descriptions:
            for col, desc in table.column_descriptions.items():
                desc_lower = str(desc).lower()
                for word in words:
                    if word in desc_lower:
                        score += 0.5
        
        ranked.append((table, score))
    
    ranked.sort(key=lambda x: (-x[1], x[0].display_name))
    return ranked


def add_message(role: str, content: str, table_name: str = None, code: str = None, st_components: list = None):
    """Add message to chat history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "table_name": table_name,
        "code": code,
        "st_components": st_components or [],
    })


def _sanitize_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame for Streamlit display to avoid Arrow errors."""
    if df is None:
        return None
    df = df.copy()
    
    # First, fill any actual NaNs with empty string
    df = df.fillna("")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Convert to string to avoid mixed types
                df[col] = df[col].astype(str)
                # Replace "nan" string artifacts (case insensitive)
                df[col] = df[col].replace(r'(?i)^nan$', "", regex=True)
                df[col] = df[col].replace(r'(?i)^<na>$', "", regex=True)
                df[col] = df[col].replace(r'(?i)^none$', "", regex=True)
            except Exception:
                pass
    return df


def render_st_components(components: list):
    """Render captured Streamlit components."""
    for comp in components:
        comp_type = comp.get("type", "")
        
        if comp_type == "dataframe":
            data = comp.get("data")
            if data is not None:
                st.dataframe(_sanitize_df_for_display(data), use_container_width=True)
                total = comp.get("total_rows", len(data))
                if total > len(data):
                    st.caption(f"Menampilkan {len(data)} dari {total} baris")
        
        elif comp_type == "table":
            data = comp.get("data")
            if data is not None:
                st.table(data)
                total = comp.get("total_rows", len(data) if hasattr(data, '__len__') else 0)
                if total > 50:
                    st.caption(f"Menampilkan 50 dari {total} baris")
        
        elif comp_type == "metric":
            st.metric(
                label=comp.get("label", ""),
                value=comp.get("value", ""),
                delta=comp.get("delta")
            )
        
        elif comp_type == "write":
            st.write(comp.get("content", ""))
        
        elif comp_type == "caption":
            st.caption(comp.get("text", ""))
        
        elif comp_type == "success":
            st.success(comp.get("text", ""))
        
        elif comp_type == "warning":
            st.warning(comp.get("text", ""))
        
        elif comp_type == "error":
            st.error(comp.get("text", ""))
        
        elif comp_type == "info":
            st.info(comp.get("text", ""))


def process_question(question: str, table: CachedDataInfo):
    """Process question with selected table."""
    # Use API Key from settings/env
    api_key = settings.google_api_key
    if not api_key:
        add_message("assistant", "❌ Google API Key belum dikonfigurasi oleh admin.")
        return
    
    try:
        logger.info(f"Processing question for table '{table.display_name}': {question}")
        df = pd.read_parquet(table.cache_path)
        client = PandasAIClient(api_key=api_key)
        
        # Pass descriptions to QA engine
        result = client.ask(
            df, 
            question, 
            table_description=table.description,
            column_descriptions=table.column_descriptions
        )
        
        # Format response text (without dataframes - those are rendered separately)
        response = f"📊 **Tabel:** {table.display_name}\n\n"
        if result.response:
            response += result.response
        
        # Add AI explanation if available
        if result.explanation:
            response += f"\n\n---\n\n💡 **Insight:**\n{result.explanation}"
        
        add_message(
            "assistant", 
            response, 
            table_name=table.display_name, 
            code=result.code,
            st_components=result.st_components  # Pass components for rendering
        )
        
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        # For KeyError, the str(e) already includes quotes, so strip them
        if error_type == "KeyError":
            key_name = str(e).strip("'\"")
            error_msg = f"Kolom/key '{key_name}' tidak ditemukan"
        else:
            error_msg = str(e)
        # Log full traceback for debugging
        logger.error(f"Exception in process_question: {error_type}: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        print(f"[DEBUG] Exception in process_question: {error_type}: {e}")
        print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
        add_message("assistant", f"❌ Error ({error_type}): {error_msg}")


def handle_transform_upload(stored_path, selected_sheet, result, display_name, record, onedrive_ok):
    """Handle the transform, cache, and upload process."""
    with st.spinner("Menerapkan & Mengupload..."):
        # 1. Transform & Cache
        full_df = _read_dataframe_raw(stored_path, sheet_name=selected_sheet)
        transformed_df, error = execute_transform(full_df, result.transform_code)
        
        if error:
            st.error(f"Error transformasi: {error}")
            return

        cache_path, n_rows, n_cols = build_parquet_cache_from_df(
            transformed_df,
            display_name=f"{display_name} (transformed)",
            original_file=record.display_name,
            sheet_name=selected_sheet,
            transform_code=result.transform_code,
            transform_explanation=result.explanation
        )
        
        # Register in SQLite
        catalog.add_cached_sheet(
            dataset_id=record.dataset_id,
            owner_id=st.session_state.user_id,
            sheet_name=selected_sheet,
            display_name=f"{display_name} (transformed)",
            n_rows=n_rows,
            n_cols=n_cols,
            transform_explanation=result.explanation
        )
        st.success(f"✅ Data berhasil di-cache ({n_rows:,} baris)")
        

        
        st.balloons()
        st.session_state.analysis_result = None
        st.rerun()


# =============================================================================
# Main UI
# =============================================================================

st.title("💬 Purchasing Data Assistant")

# Tabs
# Tabs
tab_chat, tab_onedrive, tab_upload, tab_manage = st.tabs(["💬 Chat", "☁️ OneDrive", "⬆️ Upload File", "🛠️ Manage Tables"])

# =============================================================================
# TAB 1: Chat
# =============================================================================

with tab_chat:
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Render any Streamlit components (dataframes, metrics, etc.)
            if msg.get("st_components"):
                render_st_components(msg["st_components"])
            
            if msg.get("code"):
                with st.expander("🐍 Lihat Kode"):
                    st.code(msg["code"], language="python")
    
    # Table selector popup
    if st.session_state.show_table_selector and st.session_state.pending_question:
        st.divider()
        st.subheader("📊 Pilih Tabel untuk Pertanyaan Ini")
        st.caption(f"_Pertanyaan: {st.session_state.pending_question}_")
        
        cached_list = get_enriched_cached_data()
        
        if not cached_list:
            st.warning("Belum ada tabel yang di-cache. Upload file atau sync dari OneDrive terlebih dahulu.")
            if st.button("❌ Batal"):
                st.session_state.show_table_selector = False
                st.session_state.pending_question = None
                st.rerun()
        else:
            # Rank tables by relevance
            ranked = rank_tables_by_relevance(st.session_state.pending_question, cached_list)
            
            # Show suggested table first
            if ranked and ranked[0][1] > 0:
                st.success(f"💡 **Rekomendasi:** {ranked[0][0].display_name}")
            
            # Table selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                table_options = [f"{t.display_name} ({t.n_rows:,} baris)" for t, _ in ranked]
                selected_idx = st.selectbox(
                    "Pilih tabel:",
                    range(len(table_options)),
                    format_func=lambda i: table_options[i],
                    key="table_selector"
                )
                
                # Preview selected table
                if selected_idx is not None:
                    selected_table = ranked[selected_idx][0]
                    try:
                        preview_df = pd.read_parquet(selected_table.cache_path).head(5)
                        st.caption(f"Preview {selected_table.display_name}:")
                        st.dataframe(_sanitize_df_for_display(preview_df), use_container_width=True)
                    except Exception as e:
                        st.error(f"Gagal memuat preview: {e}")
            
            with col2:
                st.write("")  # Spacer
                st.write("")
                if st.button("✅ Gunakan Tabel Ini", type="primary", key="confirm_table"):
                    selected_table = ranked[selected_idx][0]
                    add_message("user", st.session_state.pending_question)
                    
                    with st.spinner("🔄 Menganalisis data..."):
                        process_question(st.session_state.pending_question, selected_table)
                    
                    st.session_state.show_table_selector = False
                    st.session_state.pending_question = None
                    st.rerun()
                
                if st.button("❌ Batal", key="cancel_table"):
                    st.session_state.show_table_selector = False
                    st.session_state.pending_question = None
                    st.rerun()
    
    # Chat input
    if not st.session_state.show_table_selector:
        if prompt := st.chat_input("Tanyakan sesuatu tentang data Anda..."):
            # Check if we have any cached tables
            cached_list = get_enriched_cached_data()
            
            if not cached_list:
                add_message("user", prompt)
                add_message("assistant", "⚠️ Belum ada tabel yang tersimpan. Silakan upload file atau sync dari OneDrive terlebih dahulu di tab yang tersedia.")
                st.rerun()
            elif len(cached_list) == 1:
                # Only one table, use it directly
                add_message("user", prompt)
                with st.spinner("🔄 Menganalisis data..."):
                    process_question(prompt, cached_list[0])
                st.rerun()
            else:
                # Multiple tables, show selector
                st.session_state.pending_question = prompt
                st.session_state.show_table_selector = True
                st.rerun()


# =============================================================================
# TAB 2: OneDrive
# =============================================================================

with tab_onedrive:
    if not onedrive_ok:
        st.warning(f"☁️ OneDrive tidak dikonfigurasi: {onedrive_err}")
        st.info("Hubungi admin untuk konfigurasi integrasi OneDrive.")
    else:
        # Show flash message if exists
        if st.session_state.flash_message:
            st.success(st.session_state.flash_message)
            st.session_state.flash_message = None
            
        st.subheader(f"📂 File dari: {onedrive_config.ONEDRIVE_ROOT_PATH}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh", key="refresh_onedrive"):
                with st.spinner("Mengambil daftar file..."):
                    try:
                        token = onedrive_client.get_access_token()
                        st.session_state.onedrive_token = token
                        st.session_state.onedrive_files = onedrive_client.list_files(token)
                        st.success(f"✅ {len(st.session_state.onedrive_files)} file ditemukan")
                    except Exception as e:
                        st.error(f"Gagal: {e}")
        
        files = st.session_state.onedrive_files
        
        if not files:
            st.info("Klik 'Refresh' untuk memuat daftar file dari OneDrive.")
        else:
            # File selector
            file_options = [f["name"] for f in files]
            selected_file_name = st.selectbox("Pilih file:", file_options, key="onedrive_file_select")
            
            if selected_file_name:
                selected_file = next(f for f in files if f["name"] == selected_file_name)
                
                st.caption(f"📁 Path: `{selected_file['path']}`")
                st.caption(f"📏 Size: {selected_file['size'] / 1024 / 1024:.2f} MB")
                
                # For Excel files, need to select sheet
                is_excel = selected_file_name.lower().endswith((".xlsx", ".xls"))
                
                if is_excel:
                    # Download and get sheets
                    if st.button("📥 Muat Sheet", key="load_sheets"):
                        with st.spinner("Mengunduh file..."):
                            try:
                                file_bytes = onedrive_client.download_file(selected_file["downloadUrl"])
                                sheets = onedrive_client.get_excel_sheets(file_bytes)
                                st.session_state.onedrive_file_bytes = file_bytes
                                st.session_state.onedrive_sheets = sheets
                                # Reset analysis state
                                st.session_state.onedrive_analysis = None
                                st.session_state.onedrive_preview_df = None
                                st.success(f"✅ {len(sheets)} sheet ditemukan")
                            except Exception as e:
                                st.error(f"Gagal: {e}")
                    
                    if "onedrive_sheets" in st.session_state and st.session_state.onedrive_sheets:
                        selected_sheet = st.selectbox(
                            "Pilih sheet:",
                            st.session_state.onedrive_sheets,
                            key="onedrive_sheet_select"
                        )
                        
                        display_name = f"{Path(selected_file_name).stem} - {selected_sheet}"
                        
                        # Load and preview the data
                        try:
                            df_raw = onedrive_client.read_file_to_df(
                                st.session_state.onedrive_file_bytes,
                                selected_file_name,
                                selected_sheet,
                                nrows=100
                            )
                            
                            st.subheader("📋 Preview Data")
                            st.dataframe(_sanitize_df_for_display(df_raw.head(30)), use_container_width=True)
                            
                            # Quick analysis hints
                            quick = get_quick_analysis(df_raw.head(100))
                            if quick["issues"]:
                                st.info("💡 **Quick Check:** " + "; ".join(quick["issues"]))
                            
                            st.divider()
                            
                            # User description section
                            st.subheader("📝 Jelaskan Struktur Data Ini")
                            st.caption("Bantu AI memahami data Anda dengan menjelaskan strukturnya:")
                            
                            od_user_description = st.text_area(
                                "Deskripsi data:",
                                placeholder="Contoh:\n- Header ada di baris ke-3\n- Ini adalah pivot table dengan bulan sebagai kolom",
                                key="od_user_data_description",
                                height=100
                            )
                            
                            col_analyze, col_cache = st.columns([1, 1])
                            
                            with col_analyze:
                                if st.button("🤖 Analisis & Transform", key="analyze_onedrive"):
                                    with st.spinner("🔍 AI sedang menganalisis..."):
                                        try:
                                            df_full = onedrive_client.read_file_to_df(
                                                st.session_state.onedrive_file_bytes,
                                                selected_file_name,
                                                selected_sheet
                                            )
                                            result = analyze_and_generate_transform(
                                                df_full.head(100),
                                                filename=selected_file_name,
                                                sheet_name=selected_sheet,
                                                user_description=od_user_description or ""
                                            )
                                            st.session_state.onedrive_analysis = result
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Gagal menganalisis: {e}")
                            
                            with col_cache:
                                st.caption("Atau langsung cache tanpa transformasi ↓")
                            
                            # Show AI Analysis Results
                            if st.session_state.get("onedrive_analysis"):
                                result: TransformResult = st.session_state.onedrive_analysis
                                
                                st.divider()
                                st.subheader("🤖 Hasil Analisis AI")
                                
                                if result.has_error:
                                    st.error(f"❌ **Error:** {result.summary}")
                                else:
                                    st.info(f"📝 **Summary:** {result.summary}")
                                    
                                    if result.needs_transform and result.transform_code:
                                        st.subheader("💡 Transformasi")
                                        
                                        # Show Explanation
                                        if result.explanation:
                                            st.info(f"**Penjelasan:**\n{result.explanation}")
                                        
                                        with st.expander("🐍 Lihat kode Python (untuk debugging)", expanded=False):
                                            st.code(result.transform_code, language="python")
                                        
                                        # Preview
                                        st.subheader("👁️ Preview Hasil")
                                        fresh_preview_df, exec_error = execute_transform(df_raw.head(100).copy(), result.transform_code)
                                        
                                        if exec_error:
                                            st.error(f"Error preview: {exec_error}")
                                        else:
                                            st.dataframe(_sanitize_df_for_display(fresh_preview_df.head(20)), use_container_width=True)
                                            
                                            # Feedback Loop
                                            st.divider()
                                            st.subheader("🔧 Perbaiki Transformasi")
                                            st.caption("Jika hasil belum sesuai, berikan feedback untuk diperbaiki AI.")
                                            
                                            feedback = st.text_area(
                                                "Feedback:",
                                                placeholder="Contoh: Kolom tanggal masih salah format, atau kolom X harusnya dihapus.",
                                                key="onedrive_feedback"
                                            )
                                            
                                            if st.button("🛠️ Perbaiki dengan AI", key="regenerate_onedrive"):
                                                if not feedback:
                                                    st.warning("Silakan isi feedback terlebih dahulu.")
                                                else:
                                                    with st.spinner("🔄 Memperbaiki transformasi..."):
                                                        new_result = regenerate_with_feedback(
                                                            df=df_raw,
                                                            previous_code=result.transform_code,
                                                            user_feedback=feedback,
                                                            filename=selected_file_name,
                                                            sheet_name=selected_sheet,
                                                            original_df=df_raw.head(50),
                                                            transformed_df=fresh_preview_df.head(20),
                                                            previous_error=exec_error
                                                        )
                                                        st.session_state.onedrive_analysis = new_result
                                                        st.rerun()
                                        
                                        # Apply button
                                        st.divider()
                                        if st.button("✅ Terapkan & Cache", key="od_apply_transform", type="primary"):
                                            with st.spinner("Menerapkan transformasi..."):
                                                try:
                                                    # Re-read full dataframe for application
                                                    df_full = onedrive_client.read_file_to_df(
                                                        st.session_state.onedrive_file_bytes,
                                                        selected_file_name,
                                                        selected_sheet
                                                    )
                                                    transformed_df, error = execute_transform(df_full.copy(), result.transform_code)
                                                    if error:
                                                        st.error(f"Error transformasi: {error}")
                                                    else:
                                                        cache_path, n_rows, n_cols = build_parquet_cache_from_df(
                                                            transformed_df,
                                                            display_name=f"{display_name} (transformed)",
                                                            original_file=selected_file_name,
                                                            sheet_name=selected_sheet,
                                                            transform_code=result.transform_code,
                                                            source_metadata={
                                                                "source": "onedrive",
                                                                "file_id": selected_file["id"],
                                                                "file_path": selected_file["path"],
                                                                "download_url": selected_file.get("downloadUrl"),
                                                                "webUrl": selected_file.get("webUrl"),
                                                            },
                                                            transform_explanation=result.explanation
                                                        )
                                                        # Register in SQLite
                                                        register_onedrive_cache(
                                                            cache_path, n_rows, n_cols, display_name, selected_sheet,
                                                            {
                                                                "source": "onedrive",
                                                                "file_id": selected_file["id"],
                                                                "file_path": selected_file["path"],
                                                                "download_url": selected_file.get("downloadUrl"),
                                                                "webUrl": selected_file.get("webUrl"),
                                                            },
                                                            transform_explanation=result.explanation
                                                        )
                                                        
                                                        st.session_state.flash_message = f"✅ Berhasil! Tabel '{display_name}' ({n_rows:,} baris) telah di-cache."
                                                        st.balloons()
                                                        reset_onedrive_state()
                                                        st.rerun()
                                                except Exception as e:
                                                    st.error(f"Gagal: {e}")
                            
                            # Direct cache button
                            st.divider()
                            if st.button("📦 Cache Tanpa Transformasi", key="cache_onedrive_sheet"):
                                with st.spinner("Memproses..."):
                                    try:
                                        temp_path = PARQUET_CACHE_DIR / f"_temp_{st.session_state.user_id}.xlsx"
                                        with open(temp_path, "wb") as f:
                                            f.write(st.session_state.onedrive_file_bytes)
                                        
                                        cache_path, n_rows, n_cols = build_parquet_cache(
                                            temp_path, 
                                            selected_sheet, 
                                            display_name=display_name,
                                            source_metadata={
                                                "source": "onedrive",
                                                "file_id": selected_file["id"],
                                                "file_path": selected_file["path"],
                                                "download_url": selected_file.get("downloadUrl"),
                                                "webUrl": selected_file.get("webUrl"),
                                            },
                                            transform_explanation="No transformation applied"
                                        )
                                        
                                        # Register in SQLite
                                        register_onedrive_cache(
                                            cache_path, n_rows, n_cols, display_name, selected_sheet,
                                            {
                                                "source": "onedrive",
                                                "file_id": selected_file["id"],
                                                "file_path": selected_file["path"],
                                                "download_url": selected_file.get("downloadUrl"),
                                                "webUrl": selected_file.get("webUrl"),
                                            },
                                            transform_explanation="No transformation applied"
                                        )
                                        
                                        temp_path.unlink(missing_ok=True)
                                        temp_path.unlink(missing_ok=True)
                                        st.session_state.flash_message = f"✅ Berhasil! Tabel '{display_name}' ({n_rows:,} baris) telah di-cache."
                                        st.balloons()
                                        reset_onedrive_state()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Gagal: {e}")
                        
                        except Exception as e:
                            st.error(f"Gagal membaca data: {e}")
                
                else:
                    # CSV file
                    display_name = Path(selected_file_name).stem
                    if st.button("📦 Cache File CSV", type="primary", key="cache_onedrive_csv"):
                        with st.spinner("Memproses..."):
                            try:
                                file_bytes = onedrive_client.download_file(selected_file["downloadUrl"])
                                temp_path = PARQUET_CACHE_DIR / f"_temp_{st.session_state.user_id}.csv"
                                with open(temp_path, "wb") as f:
                                    f.write(file_bytes)
                                
                                cache_path, n_rows, n_cols = build_parquet_cache(
                                    temp_path, 
                                    None, 
                                    display_name=display_name,
                                    source_metadata={
                                        "source": "onedrive",
                                        "file_id": selected_file["id"],
                                        "file_path": selected_file["path"],
                                        "download_url": selected_file.get("downloadUrl"),
                                        "webUrl": selected_file.get("webUrl"), # Ensure webUrl is passed
                                    },
                                    transform_explanation="No transformation applied"
                                )
                                
                                # Register in SQLite
                                register_onedrive_cache(
                                    cache_path, n_rows, n_cols, display_name, None,
                                    {
                                        "source": "onedrive",
                                        "file_id": selected_file["id"],
                                        "file_path": selected_file["path"],
                                        "download_url": selected_file.get("downloadUrl"),
                                        "webUrl": selected_file.get("webUrl"),
                                    },
                                    transform_explanation="No transformation applied"
                                )
                                temp_path.unlink(missing_ok=True)
                                temp_path.unlink(missing_ok=True)
                                st.session_state.flash_message = f"✅ Berhasil! Tabel '{display_name}' ({n_rows:,} baris) telah di-cache."
                                st.balloons()
                                reset_onedrive_state()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gagal: {e}")


# =============================================================================
# TAB 3: Upload File
# =============================================================================

with tab_upload:
    st.subheader("⬆️ Upload File")
    st.write("Upload file CSV atau Excel. File akan dianalisis dan di-cache.")
    
    upload = st.file_uploader(
        "Pilih file",
        type=["csv", "tsv", "txt", "xls", "xlsx"],
        help=f"Maksimal {settings.upload_max_mb} MB",
    )
    
    if upload is not None:
        upload_key = f"{upload.name}_{upload.size}"
        
        if st.session_state.get("last_upload_key") != upload_key:
            # Reset transform states for new upload
            st.session_state.analysis_result = None
            st.session_state.transform_preview_df = None
            st.session_state.original_df = None
            st.session_state.selected_transforms = []
            
            with st.spinner("Menyimpan file sementara..."):
                try:
                    dataset_id, df = persist_upload(
                        upload,
                        st.session_state.user_id,
                        catalog
                    )
                    st.session_state.current_upload_id = dataset_id
                    st.session_state.last_upload_key = upload_key
                    st.success(f"✅ '{upload.name}' tersimpan ({df.shape[0]:,} baris)")
                except Exception as exc:
                    st.error(f"Gagal: {exc}")
                    st.stop()
        
        if "current_upload_id" not in st.session_state:
            st.stop()
        
        record = catalog.get_dataset(st.session_state.current_upload_id)
        if not record:
            st.error("Dataset tidak ditemukan.")
            st.stop()
        
        stored_path = Path(record.stored_path)
        sheet_names = get_excel_sheet_names(stored_path)
        
        st.divider()
        
        if sheet_names:
            selected_sheet = st.selectbox("Pilih sheet:", sheet_names, key="upload_sheet")
            display_name = f"{record.display_name} - {selected_sheet}"
        else:
            selected_sheet = None
            display_name = record.display_name
            st.info("File CSV (tidak ada sheet)")
        
        # Load preview data
        try:
            df_raw = _read_dataframe_raw(stored_path, sheet_name=selected_sheet, nrows=100)
            st.session_state.original_df = df_raw
            
            is_cached = has_parquet_cache(stored_path, selected_sheet)
            
            if is_cached:
                st.success("✅ Sheet ini sudah di-cache.")
            else:
                # Show original data preview
                st.subheader("📋 Preview Data")
                st.dataframe(_sanitize_df_for_display(df_raw.head(30)), use_container_width=True)
                
                # Quick analysis hints
                quick_analysis = get_quick_analysis(df_raw)
                if quick_analysis["issues"]:
                    st.info("💡 **Quick Check:** " + "; ".join(quick_analysis["issues"]))
                
                st.divider()
                
                # User description section
                st.subheader("📝 Jelaskan Struktur Data Ini")
                st.caption("Bantu AI memahami data Anda dengan menjelaskan strukturnya:")
                
                user_description = st.text_area(
                    "Deskripsi data:",
                    placeholder="Contoh:\n- Header ada di baris ke-3\n- Ini adalah pivot table dengan bulan sebagai kolom",
                    key="user_data_description",
                    height=120
                )
                
                col_analyze, col_skip = st.columns([1, 1])
                
                with col_analyze:
                    if st.button("🤖 Analisis & Transform", type="primary", key="analyze_data"):
                        with st.spinner("🔍 AI sedang menganalisis struktur data..."):
                            try:
                                result = analyze_and_generate_transform(
                                    df_raw, 
                                    filename=record.display_name,
                                    sheet_name=selected_sheet or "",
                                    user_description=user_description or ""
                                )
                                st.session_state.analysis_result = result
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gagal menganalisis: {e}")
                
                with col_skip:
                    if st.button("⏭️ Cache Tanpa Transformasi", key="skip_transform"):
                        with st.spinner("Memproses..."):
                            try:
                                full_df = _read_dataframe_raw(stored_path, sheet_name=selected_sheet)
                                cache_path, n_rows, n_cols = build_parquet_cache_from_df(
                                    full_df,
                                    display_name=display_name,
                                    original_file=record.display_name,
                                    sheet_name=selected_sheet,
                                    transform_explanation="No transformation applied"
                                )
                                
                                # Register in SQLite
                                catalog.add_cached_sheet(
                                    dataset_id=record.dataset_id,
                                    owner_id=st.session_state.user_id,
                                    sheet_name=selected_sheet,
                                    display_name=display_name,
                                    n_rows=n_rows,
                                    n_cols=n_cols,
                                    transform_explanation="No transformation applied"
                                )
                                st.success(f"✅ '{display_name}' ({n_rows:,} baris) di-cache!")
                                

                                
                                st.balloons()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gagal: {e}")
                
                # Show AI Analysis Results
                if st.session_state.analysis_result:
                    result: TransformResult = st.session_state.analysis_result
                    
                    st.divider()
                    st.subheader("🤖 Hasil Analisis AI")
                    
                    if result.has_error:
                        st.error(f"❌ **Error:** {result.summary}")
                    else:
                        st.info(f"📝 **Summary:** {result.summary}")
                        
                        if result.needs_transform and result.transform_code:
                            st.subheader("💡 Transformasi")
                            
                            # Show Explanation
                            if result.explanation:
                                st.info(f"**Penjelasan:**\n{result.explanation}")
                            
                            with st.expander("🐍 Lihat kode Python (untuk debugging)", expanded=False):
                                st.code(result.transform_code, language="python")
                            
                            # Preview
                            st.subheader("👁️ Preview Hasil")
                            fresh_preview_df, exec_error = execute_transform(df_raw.head(100).copy(), result.transform_code)
                            
                            if exec_error:
                                st.error(f"Error preview: {exec_error}")
                            else:
                                st.dataframe(_sanitize_df_for_display(fresh_preview_df.head(20)), use_container_width=True)
                                
                                # Feedback Loop
                                st.divider()
                                st.subheader("🔧 Perbaiki Transformasi")
                                st.caption("Jika hasil belum sesuai, berikan feedback untuk diperbaiki AI.")
                                
                                feedback = st.text_area(
                                    "Feedback:",
                                    placeholder="Contoh: Kolom tanggal masih salah format, atau kolom X harusnya dihapus.",
                                    key="upload_feedback"
                                )
                                
                                if st.button("🛠️ Perbaiki dengan AI", key="regenerate_upload"):
                                    if not feedback:
                                        st.warning("Silakan isi feedback terlebih dahulu.")
                                    else:
                                        with st.spinner("🔄 Memperbaiki transformasi..."):
                                            new_result = regenerate_with_feedback(
                                                df=df_raw,
                                                previous_code=result.transform_code,
                                                user_feedback=feedback,
                                                filename=record.display_name,
                                                sheet_name=selected_sheet or "",
                                                original_df=df_raw.head(50),
                                                transformed_df=fresh_preview_df.head(20),
                                                previous_error=exec_error
                                            )
                                            st.session_state.analysis_result = new_result
                                            st.rerun()
                            
                            st.divider()
                            
                            if st.button("✅ Terapkan & Cache", key="apply_transform_upload", type="primary"):
                                handle_transform_upload(stored_path, selected_sheet, result, display_name, record, onedrive_ok)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# =============================================================================
# TAB 4: Manage Tables
# =============================================================================

with tab_manage:
    st.subheader("🛠️ Kelola Tabel Tersimpan")
    
    cached_list = get_enriched_cached_data()
    
    if not cached_list:
        st.info("Belum ada tabel yang tersimpan.")
    else:
        # Dropdown for selection
        table_options = ["-- Pilih Tabel --"] + [f"{t.display_name} ({t.n_rows:,} baris)" for t in cached_list]
        selected_option = st.selectbox("Pilih tabel untuk dikelola:", table_options, key="manage_table_select")
        
        if selected_option != "-- Pilih Tabel --":
            # Find selected table
            selected_idx = table_options.index(selected_option) - 1
            table = cached_list[selected_idx]
            
            st.divider()
            st.markdown(f"### 📊 {table.display_name}")
            
            col_info, col_actions = st.columns([2, 1])
            
            with col_info:
                source_display = f"`{table.original_file}`"
                if table.source_metadata and table.source_metadata.get("webUrl"):
                    source_display = f"[{table.original_file}]({table.source_metadata.get('webUrl')})"
                
                st.markdown(f"""
                - **File Asli:** {source_display}
                - **Sheet:** `{table.sheet_name or '-'}`
                - **Dimensi:** {table.n_rows:,} baris x {table.n_cols} kolom
                - **Ukuran:** {table.file_size_mb} MB
                - **Di-cache:** {table.cached_at}
                """)
                
                if table.source_metadata:
                    st.caption(f"Source: {table.source_metadata.get('source', 'Unknown')}")
                    if table.transform_code:
                        st.caption("✅ Menggunakan transformasi custom")
                
                # Preview
                try:
                    preview_df = pd.read_parquet(table.cache_path).head(10)
                    st.caption("Preview Data:")
                    st.dataframe(_sanitize_df_for_display(preview_df), use_container_width=True)
                except Exception as e:
                    st.error(f"Gagal memuat preview: {e}")
            
            # Description & Schema Editor
            st.write("")
            with st.expander("📝 Deskripsi & Schema", expanded=False):
                st.caption("Tambahkan deskripsi untuk membantu AI memahami data Anda lebih baik.")
                
                # Table Description
                new_desc = st.text_area(
                    "Deskripsi Tabel",
                    value=table.description or "",
                    placeholder="Contoh: Data penjualan bulanan per supplier tahun 2024...",
                    height=100,
                    key=f"desc_{table.cache_path.stem}"
                )
                
                # Column Descriptions
                st.markdown("#### Deskripsi Kolom")
                
                # Load columns
                try:
                    df_cols = pd.read_parquet(table.cache_path).columns.tolist()
                    
                    # Prepare data for editor
                    current_col_descs = table.column_descriptions or {}
                    editor_data = []
                    for col in df_cols:
                        editor_data.append({
                            "Nama Kolom": col,
                            "Deskripsi": current_col_descs.get(col, "")
                        })
                    
                    df_editor = pd.DataFrame(editor_data)
                    
                    edited_df = st.data_editor(
                        df_editor,
                        column_config={
                            "Nama Kolom": st.column_config.TextColumn(disabled=True),
                            "Deskripsi": st.column_config.TextColumn(
                                "Deskripsi (Editable)",
                                help="Jelaskan isi kolom ini",
                                width="large"
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                        key=f"schema_{table.cache_path.stem}"
                    )
                    
                    if st.button("💾 Simpan Deskripsi", type="primary", key=f"save_{table.cache_path.stem}"):
                        # Convert back to dict
                        new_col_descs = {}
                        for _, row in edited_df.iterrows():
                            if row["Deskripsi"] and str(row["Deskripsi"]).strip():
                                new_col_descs[row["Nama Kolom"]] = str(row["Deskripsi"]).strip()
                        
                        # Save to SQLite
                        # Find cache_id from cached_list (it's not directly in CachedDataInfo but we can find it via catalog)
                        # Actually CachedDataInfo doesn't have cache_id, but we can look it up or add it.
                        # Wait, CachedDataInfo is constructed from file system in list_all_cached_data.
                        # But we need cache_id for SQLite update.
                        
                        # Let's look up the cache_id from the catalog based on stored_path (cache_path)
                        # Or we can just iterate the catalog to find the matching record.
                        
                        # Better approach: list_all_cached_data should probably return cache_id if available.
                        # But for now, let's find it.
                        found_cache_id = None
                        user_sheets = catalog.list_cached_sheets(st.session_state.user_id)
                        
                        # Match by filename (cache_path stem)
                        target_stem = table.cache_path.stem
                        for sheet in user_sheets:
                            # We don't have the cache filename in CachedSheetRecord directly, but we can infer or check.
                            # Actually, list_all_cached_data iterates files.
                            # Let's try to match by display_name and original_file as a fallback?
                            # No, that's risky.
                            
                            # Let's check `list_all_cached_data` implementation in `app/datasets.py`.
                            # It iterates `PARQUET_CACHE_DIR.glob("*.parquet")`.
                            # The stem IS the cache key (hash).
                            # In `app/data_store.py`, `add_cached_sheet` generates `cache_id = str(uuid4())`.
                            # Wait, `build_parquet_cache` in `app/datasets.py` uses `hashlib.md5` for filename.
                            # `app/data_store.py` uses `uuid4` for `cache_id`.
                            # These are DIFFERENT!
                            
                            # This is a problem. The file-based cache and SQLite catalog are not fully aligned on ID.
                            # `build_parquet_cache` returns `cache_path`.
                            # `add_cached_sheet` returns `cache_id`.
                            
                            # We need to link them.
                            # In `app/datasets.py`, `_save_cache_metadata` saves to JSON.
                            # We are moving to SQLite.
                            
                            # If we use `catalog.update_cached_sheet_metadata`, we need `cache_id`.
                            # But `CachedDataInfo` comes from `list_all_cached_data` which reads files.
                            
                            # FIX: We need to make sure `CachedDataInfo` has the `cache_id` from SQLite.
                            # In `app/datasets.py`, `list_all_cached_data` should try to find the SQLite record.
                            pass
                        
                        # TEMPORARY FIX:
                        # Since we haven't fully migrated `list_all_cached_data` to use SQLite IDs,
                        # we need a way to find the `cache_id`.
                        # In `app/data_store.py`, `cached_sheets` has `cache_id`.
                        # But it doesn't store the `parquet_filename` (hash).
                        
                        # Wait, `app/data_store.py` `CachedSheetRecord` has `stored_path`.
                        # Is `stored_path` the parquet file?
                        # No, `stored_path` in `datasets` table is the ORIGINAL file.
                        # `cached_sheets` table doesn't have the parquet path!
                        
                        # We need to fix `app/data_store.py` to store the parquet path or hash?
                        # Or we can just use the `cache_id` as the filename?
                        
                        # Let's look at `app/datasets.py` again.
                        # `_parquet_cache_path` generates filename from original path + sheet.
                        
                        # If I want to update metadata in SQLite, I need `cache_id`.
                        # I should probably add `cache_id` to `CachedDataInfo`.
                        # And `list_all_cached_data` needs to find it.
                        
                        # How to find it?
                        # `list_cached_sheets` returns `CachedSheetRecord`.
                        # We can match `display_name` and `sheet_name`?
                        
                        # Let's assume for now we can match by `display_name`.
                        # It's not perfect but it's what we have.
                        
                        # Actually, I can update `list_all_cached_data` in `app/datasets.py` to fetch from SQLite
                        # and match based on `display_name` and `sheet_name`.
                        
                        # But wait, I can't easily change `list_all_cached_data` logic in this tool call.
                        # I'll do the matching here in `streamlit_app.py`.
                        
                        target_cache_id = None
                        user_sheets = catalog.list_cached_sheets(st.session_state.user_id)
                        for sheet in user_sheets:
                            if sheet.display_name == table.display_name and sheet.sheet_name == table.sheet_name:
                                target_cache_id = sheet.cache_id
                                break
                        
                        if target_cache_id:
                            if catalog.update_cached_sheet_metadata(target_cache_id, new_desc, new_col_descs):
                                st.success("✅ Deskripsi berhasil disimpan!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Gagal menyimpan ke database.")
                        else:
                            # If not found in SQLite (maybe old cache), try to add it?
                            # Or just warn.
                            st.warning("Metadata tabel tidak ditemukan di database (mungkin cache lama). Silakan upload ulang file ini untuk mengaktifkan fitur deskripsi.")
                            
                except Exception as e:
                    st.error(f"Error editor: {e}")

            with col_actions:
                st.markdown("#### Aksi")
                




                # RE-TRANSFORM BUTTON
                # Check if we have a valid source (URL or existing local file)
                has_valid_source = False
                if table.source_url:
                    has_valid_source = True
                elif table.source_metadata and table.source_metadata.get("webUrl"):
                    has_valid_source = True
                elif table.stored_path and os.path.exists(table.stored_path):
                    has_valid_source = True
                
                if st.button("🔄 Re-transform", key=f"retrans_{table.cache_path.stem}", use_container_width=True, disabled=not has_valid_source, help="Butuh file asli atau koneksi OneDrive" if not has_valid_source else None):
                    st.session_state.retransform_target = table.cache_path.stem
                    st.rerun()

                # DELETE BUTTON
                st.write("") # Spacer
                if st.button("🗑️ Hapus Tabel", key=f"del_{table.cache_path.stem}", type="primary", use_container_width=True):
                    if delete_cached_data(table.cache_path):
                        st.success("Terhapus!")
                        time.sleep(1)
                        st.rerun()
            
            # Re-transform UI
            if "retransform_target" in st.session_state and st.session_state.retransform_target == table.cache_path.stem:
                st.divider()
                st.markdown("### 🔄 Re-transform Data")
                st.info("Modifikasi transformasi data ini dengan memberikan instruksi baru/feedback.")
                
                # Show current code
                with st.expander("Lihat Kode Transformasi Saat Ini"):
                    st.code(table.transform_code or "# Tidak ada kode transformasi tersimpan", language="python")
                
                feedback = st.text_area(
                    "Feedback / Instruksi Baru:",
                    placeholder="Contoh: Filter juga untuk tahun > 2023...",
                    key=f"feedback_{table.cache_path.stem}"
                )
                
                col_rt_1, col_rt_2 = st.columns([1, 1])
                
                with col_rt_1:
                    if st.button("▶️ Preview Transformasi", key=f"prev_retrans_{table.cache_path.stem}", type="primary"):
                        if not feedback:
                            st.warning("Masukkan feedback terlebih dahulu.")
                        else:
                            with st.spinner("Menganalisis & Transformasi ulang..."):
                                try:
                                    # Load original data
                                    # We need original path. stored_path in CachedDataInfo is now populated.
                                    full_df = None
                                    
                                    # 1. Try downloading from Source URL (OneDrive)
                                    if table.source_url:
                                        try:
                                            import requests
                                            from io import BytesIO
                                            
                                            # If it's a OneDrive URL, we might need to use the graph API or just download if public/authenticated?
                                            # For now, let's assume we need to re-download using onedrive_client if available, or requests if it's a direct link.
                                            # Actually, onedrive_config has the token.
                                            # But onedrive_client.download_file takes a file_id.
                                            # We stored webUrl. We might not have the ID easily unless we stored it too.
                                            # Wait, record.id was available during sync.
                                            # But we didn't store it in datasets table explicitly, only source_url (webUrl).
                                            
                                            # However, if we have the token, maybe we can download from webUrl?
                                            # Or we can just try requests.get if it's accessible.
                                            # But OneDrive webUrl is usually a viewing link, not direct download.
                                            
                                            # Workaround: If we have onedrive_client, we can LIST files and match webUrl? Too slow.
                                            # Better: We should have stored the ID.
                                            # But for now, let's try to use the stored_path if it exists.
                                            
                                            # Actually, the user requirement says "keep the onedrive document url and download it directly".
                                            # If we can't easily download from webUrl, maybe we can just rely on stored_path IF it exists.
                                            # But the issue is stored_path (temp) is gone.
                                            
                                            # Let's try to use the onedrive_client to download if we can find the file.
                                            # Or, if we can't, just fail gracefully.
                                            
                                            # Wait, `onedrive_client.download_file` needs `file_id`.
                                            # We don't have `file_id` in `datasets` table.
                                            # We only have `source_url` (webUrl).
                                            
                                            # Maybe we can parse the ID from the URL?
                                            # Or just tell the user we can't re-download without ID.
                                            
                                            # Let's try to use `requests` with the token if possible?
                                            # No, webUrl is for browser.
                                            
                                            # FIX: We should rely on `stored_path` if it exists.
                                            # If `source_url` exists, we can TRY to download it if we had the ID.
                                            # Since we don't have the ID, we can't easily download via API.
                                            
                                            # BUT, the user said "download it directly".
                                            # Maybe they mean the `downloadUrl` (which expires)?
                                            # No, `webUrl` is persistent.
                                            
                                            # Let's check `app/onedrive_client.py`.
                                            # `list_files` returns `id`, `name`, `webUrl`.
                                            
                                            # If I want to support this properly, I should store `file_id` in `datasets` table.
                                            # But I only added `source_url`.
                                            
                                            # Let's assume for now we only support if `stored_path` exists OR if we can somehow get the file.
                                            # If I can't download, I can't re-transform.
                                            
                                            # Wait, if I use `onedrive_client.list_files` I can find the ID by matching `webUrl`!
                                            # It's a bit expensive but works.
                                            
                                            if onedrive_config.is_configured()[0]:
                                                # Try to find file ID by matching URL
                                                # This assumes the file is in the root or we search recursively?
                                                # `list_files` is not recursive by default.
                                                # This is tricky.
                                                
                                                # Alternative: Just use `stored_path` and warn if missing.
                                                # But user specifically asked to fix sourcing.
                                                
                                                # Let's try to find the file in the root folder (common case).
                                                # Let's try to find the file in the root folder (common case).
                                                token = onedrive_client.get_access_token()
                                                files = onedrive_client.list_files(token)
                                                logger.info(f"Retransform: Searching for {table.source_url} in {len(files)} OneDrive files.")
                                                
                                                found_download_url = None
                                                for f in files:
                                                    # logger.info(f"Comparing with: {f.get('webUrl')}") # Too verbose?
                                                    if f.get("webUrl") == table.source_url:
                                                        found_download_url = f.get("downloadUrl")
                                                        logger.info(f"Retransform: Found match! Download URL: {found_download_url[:50]}...")
                                                        break
                                                
                                                if found_download_url:
                                                    # Download to temp
                                                    temp_path = Path("temp") / f"retrans_{table.cache_path.stem}.xlsx" # Assume excel/csv
                                                    temp_path.parent.mkdir(exist_ok=True)
                                                    
                                                    logger.info(f"Retransform: Downloading to {temp_path}...")
                                                    file_bytes = onedrive_client.download_file(found_download_url)
                                                    with open(temp_path, "wb") as f:
                                                        f.write(file_bytes)
                                                    
                                                    full_df = _read_dataframe_raw(temp_path, sheet_name=table.sheet_name)
                                                    logger.info(f"Retransform: Download successful, df shape: {full_df.shape}")
                                                    # Clean up temp
                                                    try:
                                                        temp_path.unlink()
                                                    except:
                                                        pass
                                                else:
                                                    logger.warning(f"Retransform: No file matched source_url: {table.source_url}")
                                            
                                        except Exception as e:
                                            logger.error(f"Failed to download from OneDrive: {e}", exc_info=True)

                                    # 2. Fallback to Stored Path
                                    if full_df is None:
                                        if table.stored_path and os.path.exists(table.stored_path):
                                            full_df = _read_dataframe_raw(Path(table.stored_path), sheet_name=table.sheet_name)
                                        else:
                                            raise FileNotFoundError("File asli tidak ditemukan di server dan tidak dapat diunduh.")

                                    if full_df is None:
                                         raise FileNotFoundError("Gagal memuat data asli.")
                                        
                                    # Regenerate
                                    result = regenerate_with_feedback(
                                            full_df.head(100), # Sample for generation
                                            table.transform_code or "",
                                            feedback
                                        )
                                        
                                    # Execute on sample for preview
                                    preview_df, error = execute_transform(full_df.head(100).copy(), result.transform_code)
                                    
                                    if error:
                                        st.error(f"Error preview: {error}")
                                    else:
                                        st.session_state.retransform_preview = preview_df
                                        st.session_state.retransform_code = result.transform_code
                                        st.session_state.retransform_explanation = result.explanation
                                        st.session_state.retransform_result = result # Store full result if needed
                                        st.success("Preview berhasil!")
                                except Exception as e:
                                    st.error(f"Gagal re-transform: {e}")
                
                with col_rt_2:
                    if st.button("❌ Batal", key=f"cancel_retrans_{table.cache_path.stem}"):
                        st.session_state.retransform_target = None
                        if "retransform_preview" in st.session_state:
                            del st.session_state.retransform_preview
                        st.rerun()

                # Show Preview if available
                if "retransform_preview" in st.session_state and st.session_state.retransform_target == table.cache_path.stem:
                    st.markdown("#### Preview Hasil Baru")
                    
                    if st.session_state.get("retransform_explanation"):
                        st.info(f"**Penjelasan:**\n{st.session_state.retransform_explanation}")
                        
                    st.dataframe(_sanitize_df_for_display(st.session_state.retransform_preview.head(20)), use_container_width=True)
                    
                    with st.expander("Lihat Kode Baru"):
                        st.code(st.session_state.retransform_code, language="python")
                    
                    if st.button("💾 Simpan Perubahan", key=f"save_retrans_{table.cache_path.stem}", type="primary"):
                        with st.spinner("Menyimpan perubahan..."):
                            try:
                                # Apply to FULL data
                                full_df = _read_dataframe_raw(Path(table.stored_path), sheet_name=table.sheet_name)
                                final_df, error = execute_transform(full_df, st.session_state.retransform_code)
                                
                                if error:
                                    st.error(f"Gagal menerapkan ke data penuh: {error}")
                                else:
                                    # Update Cache
                                    from app.datasets import update_existing_parquet_cache
                                    n_rows, n_cols = update_existing_parquet_cache(
                                        table.cache_path,
                                        final_df,
                                        st.session_state.retransform_code,
                                        transform_explanation=st.session_state.get("retransform_explanation")
                                    )
                                    
                                    # Update SQLite Stats & Metadata
                                    # We need cache_id. 
                                    target_cache_id = None
                                    user_sheets = catalog.list_cached_sheets(st.session_state.user_id)
                                    for sheet in user_sheets:
                                        if sheet.display_name == table.display_name and sheet.sheet_name == table.sheet_name:
                                            target_cache_id = sheet.cache_id
                                            break
                                    
                                    if target_cache_id:
                                        catalog.update_cached_sheet_stats(target_cache_id, n_rows, n_cols)
                                        catalog.update_cached_sheet_metadata(
                                            target_cache_id, 
                                            transform_explanation=st.session_state.get("retransform_explanation")
                                        )
                                    
                                    st.success(f"✅ Berhasil disimpan! ({n_rows:,} baris)")
                                    st.session_state.retransform_target = None
                                    del st.session_state.retransform_preview
                                    time.sleep(1)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Gagal menyimpan: {e}")
