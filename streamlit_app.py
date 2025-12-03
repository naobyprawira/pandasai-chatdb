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
    cached_list = list_all_cached_data()
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

# Helper Functions
def rank_tables_by_relevance(question: str, tables: List[CachedDataInfo]) -> List[Tuple[CachedDataInfo, float]]:
    """Rank tables by relevance to question using simple keyword matching."""
    if not tables:
        return []
    
    ranked = []
    question_lower = question.lower()
    words = [w for w in question_lower.split() if len(w) > 3]
    
    for table in tables:
        score = 0.0
        name_lower = table.display_name.lower()
        
        for word in words:
            if word in name_lower:
                score += 1.0
        
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
        result = client.ask(df, question)
        
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
            sheet_name=selected_sheet
        )
        st.success(f"✅ Data berhasil di-cache ({n_rows:,} baris)")
        
        # 2. Upload ORIGINAL file to OneDrive (as requested)
        if onedrive_ok:
            with st.spinner("☁️ Mengupload file asli ke OneDrive..."):
                onedrive_client.upload_file(stored_path, record.original_name)
                st.success(f"✅ File asli '{record.original_name}' berhasil di-upload ke OneDrive!")
        
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
        
        cached_list = list_all_cached_data()
        
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
            cached_list = list_all_cached_data()
            
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
                                        with st.expander("Lihat kode Python", expanded=False):
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
                                                            }
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
                                            }
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
                                    }
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
    st.write("Upload file CSV atau Excel. File akan dianalisis, di-cache, dan **di-upload ke OneDrive**.")
    
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
                    if st.button("⏭️ Skip, Cache & Upload", key="skip_transform"):
                        with st.spinner("Memproses..."):
                            try:
                                full_df = _read_dataframe_raw(stored_path, sheet_name=selected_sheet)
                                cache_path, n_rows, n_cols = build_parquet_cache_from_df(
                                    full_df,
                                    display_name=display_name,
                                    original_file=record.display_name,
                                    sheet_name=selected_sheet
                                )
                                st.success(f"✅ '{display_name}' ({n_rows:,} baris) di-cache!")
                                
                                # Trigger OneDrive Upload
                                if onedrive_ok:
                                    with st.spinner("☁️ Mengupload ke OneDrive..."):
                                        onedrive_client.upload_file(stored_path, record.original_name)
                                        st.success(f"✅ File '{record.original_name}' berhasil di-upload ke OneDrive!")
                                
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
                            with st.expander("Lihat kode Python", expanded=False):
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
                            
                            if st.button("✅ Terapkan, Cache & Upload", key="apply_transform_upload", type="primary"):
                                handle_transform_upload(stored_path, selected_sheet, result, display_name, record, onedrive_ok)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# =============================================================================
# TAB 4: Manage Tables
# =============================================================================

with tab_manage:
    st.subheader("🛠️ Kelola Tabel Tersimpan")
    
    cached_list = list_all_cached_data()
    
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

            with col_actions:
                st.markdown("#### Aksi")
                
                # RESYNC BUTTON (OneDrive only)
                if table.source_metadata and table.source_metadata.get("source") == "onedrive":
                    if st.button("🔄 Resync Data", key=f"sync_{table.cache_path.stem}", use_container_width=True):
                        with st.spinner("Resyncing data..."):
                            try:
                                meta = table.source_metadata
                                token = onedrive_client.get_access_token()
                                
                                # Get fresh download URL
                                details = onedrive_client.get_file_details(token, meta["file_id"])
                                if not details or "id" not in details:
                                    st.error("File tidak ditemukan di OneDrive (mungkin sudah dihapus/dipindah).")
                                else:
                                    download_url = details.get("@microsoft.graph.downloadUrl")
                                    file_bytes = onedrive_client.download_file(download_url)
                                    
                                    # Re-process
                                    if table.transform_code:
                                        df_full = onedrive_client.read_file_to_df(
                                            file_bytes, 
                                            table.original_file, 
                                            meta.get("sheet_name")
                                        )
                                        transformed_df, error = execute_transform(df_full, table.transform_code)
                                        
                                        if error:
                                            st.error(f"Transform error: {error}")
                                        else:
                                            build_parquet_cache_from_df(
                                                transformed_df,
                                                display_name=table.display_name,
                                                original_file=table.original_file,
                                                sheet_name=meta.get("sheet_name"),
                                                transform_code=table.transform_code,
                                                source_metadata=meta
                                            )
                                            st.success("✅ Resync berhasil!")
                                            time.sleep(1)
                                            st.rerun()
                                    else:
                                        # Direct cache (no transform)
                                        suffix = Path(table.original_file).suffix
                                        temp_path = PARQUET_CACHE_DIR / f"_resync_{st.session_state.user_id}{suffix}"
                                        with open(temp_path, "wb") as f:
                                            f.write(file_bytes)
                                        
                                        build_parquet_cache(
                                            temp_path,
                                            meta.get("sheet_name"),
                                            display_name=table.display_name,
                                            source_metadata=meta
                                        )
                                        temp_path.unlink(missing_ok=True)
                                        st.success("✅ Resync berhasil!")
                                        time.sleep(1)
                                        st.rerun()
                                        
                            except Exception as e:
                                st.error(f"Resync gagal: {e}")

                # EDIT BUTTON (OneDrive + Transform only)
                if table.source_metadata and table.source_metadata.get("source") == "onedrive" and table.transform_code:
                    if st.button("✏️ Edit Transformasi", key=f"edit_{table.cache_path.stem}", use_container_width=True):
                        # Load data into session state for OneDrive tab
                        try:
                            meta = table.source_metadata
                            token = onedrive_client.get_access_token()
                            details = onedrive_client.get_file_details(token, meta["file_id"])
                            download_url = details.get("@microsoft.graph.downloadUrl")
                            file_bytes = onedrive_client.download_file(download_url)
                            
                            st.session_state.onedrive_file_bytes = file_bytes
                            # Reconstruct file entry with all necessary metadata
                            st.session_state.onedrive_files = [{
                                "name": table.original_file, 
                                "path": meta.get("file_path", table.original_file), 
                                "id": meta["file_id"], 
                                "downloadUrl": download_url, 
                                "webUrl": meta.get("webUrl"),
                                "size": 0
                            }] 
                            
                            # Reconstruct TransformResult
                            df_raw = onedrive_client.read_file_to_df(file_bytes, table.original_file, meta.get("sheet_name"), nrows=100)
                            
                            # Execute current transform for preview
                            transformed_df, _ = execute_transform(df_raw.copy(), table.transform_code)
                            
                            st.session_state.onedrive_analysis = TransformResult(
                                summary="Loaded from cache for editing",
                                issues_found=[],
                                transform_code=table.transform_code,
                                needs_transform=True,
                                preview_df=transformed_df.head(20),
                                original_df=df_raw.head(50),
                                validation_notes=["Loaded for editing"]
                            )
                            
                            st.session_state.flash_message = f"✏️ Mode Edit: {table.display_name}. Silakan buka tab OneDrive."
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Gagal memuat untuk edit: {e}")

                # DELETE BUTTON
                st.write("") # Spacer
                if st.button("🗑️ Hapus Tabel", key=f"del_{table.cache_path.stem}", type="primary", use_container_width=True):
                    if delete_cached_data(table.cache_path):
                        st.success("Terhapus!")
                        time.sleep(1)
                        st.rerun()
