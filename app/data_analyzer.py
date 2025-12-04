"""
Data Analyzer Module
Uses AI to detect data issues and generate Python transformation code.
With iterative validation to ensure no data is lost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame

from .settings import AppSettings
from app.logger import get_transform_logger

settings = AppSettings()
logger = get_transform_logger()

# Maximum rows for sample
MAX_SAMPLE_ROWS = 30
MAX_ITERATIONS = 3


@dataclass
class TransformResult:
    """Result of AI analysis with transformation code and preview."""
    summary: str  # Indonesian summary of what was transformed
    issues_found: List[str]  # List of detected issues
    transform_code: str  # Python code used for transformation
    needs_transform: bool  # Whether transformation was needed
    preview_df: Optional[DataFrame] = None  # Cached preview (already transformed)
    original_df: Optional[DataFrame] = None  # Original data before transformation (for feedback context)
    validation_notes: List[str] = field(default_factory=list)  # Validation feedback
    iterations_used: int = 1  # How many iterations to get valid result
    has_error: bool = False  # True if generation failed (dangerous code, all iterations failed, etc.)
    has_error: bool = False  # True if generation failed (dangerous code, all iterations failed, etc.)
    failed_code: str = ""  # Store the failed code for debugging
    explanation: str = ""  # Natural language explanation of the transformation


def _dataframe_to_sample_text(df: DataFrame, max_rows: int = MAX_SAMPLE_ROWS) -> str:
    """Convert DataFrame to text representation for AI analysis."""
    sample = df.head(max_rows)
    
    lines = []
    lines.append(f"=== COLUMNS ({len(df.columns)} total) ===")
    lines.append(", ".join([f'"{c}"' for c in df.columns]))
    lines.append("")
    lines.append(f"=== DATA SAMPLE ({len(sample)} of {len(df)} rows) ===")
    
    # Show with row numbers
    for idx, row in sample.iterrows():
        row_num = idx if isinstance(idx, int) else sample.index.get_loc(idx)
        values = [str(v) if pd.notna(v) else "" for v in row.values]
        # Truncate long values
        values = [v[:50] + "..." if len(v) > 50 else v for v in values]
        lines.append(f"[{row_num}] {' | '.join(values)}")
    
    return "\n".join(lines)


from openai import OpenAI

def _get_client():
    """Get OpenAI client."""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    return OpenAI(api_key=settings.openai_api_key)


def _compare_dataframes(original: DataFrame, transformed: DataFrame) -> List[str]:
    """
    Compare original and transformed DataFrames.
    Hanya flag jika DATA LOSS signifikan.
    """
    issues = []
    
    # Check if empty - CRITICAL
    if len(transformed) == 0:
        issues.append("❌ CRITICAL: Hasil transformasi kosong")
        return issues
    
    # Check if no columns - CRITICAL
    if len(transformed.columns) == 0:
        issues.append("❌ CRITICAL: Tidak ada kolom")
        return issues
    # Check column loss - CRITICAL if >50% columns removed (unless it's unpivot which reduces columns intentionally)
    orig_cols = len(original.columns)
    trans_cols = len(transformed.columns)
    if orig_cols > 5:  # Only check if original has enough columns
        col_loss = (orig_cols - trans_cols) / orig_cols
        # If columns reduced by >50% AND rows didn't increase significantly (not an unpivot)
        rows_increased = len(transformed) > len(original) * 1.5
        if col_loss > 0.5 and not rows_increased:
            issues.append(f"⚠️ WARNING: {int(col_loss*100)}% kolom hilang ({orig_cols}->{trans_cols} kolom). Pastikan ini disengaja.")
    
    # Check data loss - hanya jika >80% hilang
    if len(original) > 5:
        loss = (len(original) - len(transformed)) / len(original)
        if loss > 0.8:  # 80% hilang - CRITICAL
            issues.append(f"❌ CRITICAL: {int(loss*100)}% data hilang ({len(original)}->{len(transformed)} baris)")
    
    return issues


def _generate_transform_code(
    client,
    df: DataFrame,
    filename: str,
    sheet_name: str,
    previous_issues: List[str] = None,
    user_description: str = None,
    previous_code: str = None,
    error_history: List[Dict] = None,
    original_df: DataFrame = None  # NEW: Reference to untouched original data
) -> tuple[str, str, List[str], bool, str]:
    """
    Generate transformation code using AI.
    Generate transformation code using AI.
    Returns: (code, summary, issues, needs_transform, failed_code, explanation)
    """
    sample_text = _dataframe_to_sample_text(df, MAX_SAMPLE_ROWS)
    
    # Build prompt with previous issues if this is a retry
    previous_context = ""
    if previous_issues or error_history:
        code_context = ""
        if previous_code:
            code_context = f"""

KODE YANG ERROR SEBELUMNYA:
```python
{previous_code}
```
"""
        
        # Format issues clearly
        issues_text = "\n".join(f"  • {issue}" for issue in (previous_issues or []))
        
        # Add error history analysis
        history_text = ""
        if error_history and len(error_history) > 1:
            history_text = "\n\n📋 ERROR HISTORY - HINDARI ERROR INI LAGI:\n"
            for idx, err_entry in enumerate(error_history[-2:], 1):  # Show last 2 errors
                history_text += f"\n  Iterasi {err_entry['iteration']}: {err_entry['error']}\n"
                history_text += f"  Kode yang error:\n"
                # Show just first 3 lines of problematic code
                code_lines = err_entry['code'].split('\n')[:3]
                for line in code_lines:
                    history_text += f"    {line}\n"
        
        # Add ORIGINAL DATA REFERENCE for debugging
        original_ref = ""
        if original_df is not None:
            original_ref = f"""

📊 REFERENCE - DATA ORIGINAL (JANGAN DIUBAH, HANYA UNTUK REFERENSI):
Kolom asli: {list(original_df.columns)}
Jumlah kolom: {len(original_df.columns)}
Jumlah baris: {len(original_df)}
Sample baris pertama: {original_df.iloc[0].tolist() if len(original_df) > 0 else 'kosong'}
"""
        
        # Add debugging context based on error type
        debug_section = ""
        all_errors = [str(e) for e in (previous_issues or [])] + [e.get('error', '') for e in (error_history or [])]
        
        if any("Columns must be same length" in str(e) for e in all_errors):
            debug_section = f"""

🔴 RECURRING ERROR: "Columns must be same length as key"
Penyebab: Assign split result ke fixed columns
Solusi: Assign satu-satu - df['col1'] = split[0]; df['col2'] = split[1]
JANGAN: df[['col1', 'col2']] = split
"""
        elif any("Duplicate" in str(e) or "duplicate" in str(e) for e in all_errors):
            debug_section = f"""

🔴 RECURRING ERROR: Duplicate column names
Penyebab: Hasil transformasi masih punya kolom duplikat
CONTOH ERROR: ['Order Qty', 'Value before Disc', ...] muncul berulang kali

SOLUSI WAJIB - LANGKAH DEMI LANGKAH:
1. PERTAMA: Rename SEMUA kolom ke unique index:
   df.columns = [f"col_{{i}}" for i in range(len(df.columns))]
   
2. KEDUA: Baru buat DataFrame baru dengan kolom yang bersih:
   rows = []
   for idx, row in df.iterrows():
       # Process setiap grup bulan
       for month_idx, month_name in enumerate(['Jan', 'Feb', ...]):
           base = 3 + (month_idx * 4)  # Adjust sesuai struktur
           rows.append({{
               'supplier': row['col_0'],
               'month': month_name,
               'qty': row[f'col_{{base}}'],
               'value': row[f'col_{{base+1}}'],
           }})
   df = pd.DataFrame(rows)

3. PASTIKAN: Output df tidak punya kolom duplikat!
"""
        elif any("has no attribute" in str(e) for e in all_errors):
            debug_section = f"""

🔴 RECURRING ERROR: Attribute tidak ada
Penyebab: df.dtype ❌, df.size ❌, dll
Solusi: Gunakan yang benar - df.dtypes ✅, len(df) ✅
"""
        elif any("not in index" in str(e) or "not found in axis" in str(e) for e in all_errors):
            debug_section = f"""

🔴 RECURRING ERROR: Column tidak ada / not found in axis
Penyebab: Nama kolom yang dicari tidak ada atau sudah berubah
Solusi: 
1. Jangan hardcode nama kolom dari data mentah
2. Gunakan df.columns.tolist() untuk cek kolom yang tersedia
3. Gunakan index (df.iloc[:, 0]) bukan nama kolom
"""
        elif any("melt" in str(e).lower() or "pivot" in str(e).lower() for e in all_errors):
            debug_section = f"""

🔴 RECURRING ERROR: melt/pivot gagal
SOLUSI: JANGAN pakai melt/pivot! Gunakan approach tradisional:
```python
rows = []
for idx, row in df.iterrows():
    for col in month_columns:
        rows.append({{'id': row['id'], 'month': col, 'value': row[col]}})
df_new = pd.DataFrame(rows)
```
"""
        
        previous_context = f"""

⚠️ ITERASI SEBELUMNYA GAGAL - PERBAIKI! ⚠️
{code_context}
ERROR TERBARU:
{issues_text}{history_text}{debug_section}{original_ref}

INSTRUKSI FIX:
1. Pahami error yang terjadi
2. Lihat error history - jangan repeat error yang sama
3. Bandingkan dengan DATA ORIGINAL di atas untuk validasi
4. Tulis kode baru yang LEBIH SIMPLE dan TRADISIONAL
5. JANGAN pakai melt/pivot - gunakan for loop
6. JANGAN ambil nama kolom dari data - buat sendiri
"""
    
    # User description context
    user_context = ""
    if user_description and user_description.strip():
        user_context = f"""

DESKRIPSI USER TENTANG DATA INI:
"{user_description}"

Gunakan deskripsi user di atas sebagai panduan utama untuk transformasi!
"""
    
    prompt = f"""TUGAS: Normalisasi tabel Excel/CSV.
Ubah data mentah menjadi: [Header di baris 1] + [Data mulai baris 2]

File: {filename} | Sheet: {sheet_name or 'N/A'}
{user_context}

DATA SAAT INI:
{sample_text}

{previous_context}

STRATEGI WAJIB:
1. JANGAN ambil nama kolom langsung dari data - BUAT nama kolom bersih sendiri
   Contoh: Jika data punya "YTD 2025" duplikat, rename jadi "ytd_2025_1", "ytd_2025_2"
2. JANGAN pakai fungsi kompleks: melt(), pivot(), pivot_table(), stack(), unstack()
   Gunakan: for loop, df.iloc[], assignment manual, concat sederhana
3. Handle DUPLICATE COLUMNS dulu sebelum proses apapun:
   - df.columns = [f"col_{{i}}" for i in range(len(df.columns))]  # Rename semua dulu
   - Lalu set nama yang benar secara manual
4. Jika iterasi sebelumnya error - FIX dengan approach LEBIH SIMPLE
5. Debug: print(df.columns.tolist()), print(len(df.columns))
6. JANGAN HAPUS KOLOM kecuali kolom itu benar-benar kosong/tidak berguna

APPROACH TRADISIONAL (WAJIB):
- Untuk unpivot: Gunakan for loop + list append + pd.DataFrame(), BUKAN melt()
- Untuk rename: df.columns = ['col1', 'col2', ...], BUKAN rename()
- Untuk filter: df = df[df['col'] != value], BUKAN query()
- Untuk transform: df['new'] = df['old'].apply(lambda x: ...), BUKAN complex functions

RULES KETAT:
✅ BOLEH: pd, np, re, datetime, for loop, list comprehension
✅ BOLEH: df.iloc[], df.columns = [], df.astype(), df.apply()
✅ BOLEH: pd.DataFrame(), pd.concat(), df.reset_index()
❌ DILARANG: pd.read_excel/csv, open(), file I/O
❌ DILARANG: df.melt(), df.pivot(), df.pivot_table(), df.stack(), df.unstack()
❌ DILARANG: Menggunakan nama kolom dari data mentah yang mungkin duplikat
❌ DILARANG: df.dtype (gunakan df.dtypes), .str tanpa .astype(str)
❌ DILARANG: Menghapus kolom tanpa alasan jelas (df = df[['col1','col2']] atau df.drop())

⚠️ PENTING: PERTAHANKAN SEMUA KOLOM dari data original kecuali benar-benar tidak diperlukan
⚠️ WAJIB: Simpan hasil akhir ke variable `normalized_df`. Jangan ubah `df` original.

FORMAT JAWABAN (JSON):

{{
  "needs_transform": true/false,
  "issues": ["masalah 1", "masalah 2"],
  "summary": "Ringkasan singkat",
  "explanation": "Penjelasan detail langkah demi langkah untuk user non-teknis (contoh: 'Filter baris kosong, lalu ubah format tanggal')",
  "code": "normalized_df = df.copy()..."
}}
"""

    system_instruction = """Kamu adalah pandas expert. Normalisasi tabel dengan cara TRADISIONAL dan SIMPLE.

PRINSIP UTAMA:
1. BUAT nama kolom sendiri yang bersih - JANGAN copy dari data mentah
2. Handle duplicate columns PERTAMA dengan rename semua ke col_0, col_1, dst
3. Gunakan FOR LOOP dan LIST untuk unpivot, BUKAN melt/pivot
4. Semakin simple kode = semakin bagus

DILARANG KERAS:
- df.melt(), df.pivot(), df.pivot_table() 
- df.stack(), df.unstack()
- Mengambil nama kolom dari data yang mungkin duplikat
- df.dtype (gunakan df.dtypes)
- File I/O apapun
- Mengubah `df` original (gunakan `normalized_df` untuk hasil)

OUTPUT HARUS JSON VALID."""

    response = client.responses.create(
        model=settings.default_llm_model,
        instructions=system_instruction,
        input=prompt,

    )
    
    raw_response = response.output_text
    if not raw_response:
        return "df = df.copy()", "AI tidak memberikan response", [], False, "", "Tidak ada penjelasan"
    
    return _parse_ai_response(raw_response.strip())


def _parse_ai_response(response: str) -> tuple[str, str, List[str], bool, str, str]:
    """Parse AI response. Returns: (code, summary, issues, needs_transform, failed_code, explanation)"""
    import json
    
    # Try parsing as JSON first
    try:
        # Clean up potential markdown code blocks around JSON
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        data = json.loads(cleaned_response)
        
        needs_transform = data.get("needs_transform", False)
        issues = data.get("issues", [])
        summary = data.get("summary", "Tidak ada ringkasan")
        explanation = data.get("explanation", "Tidak ada penjelasan detail")
        code = data.get("code", "df = df.copy()")
        
        # Validate code
        if not code or len(code) < 3:
            code = "df = df.copy()"
            
        return code, summary, issues, needs_transform, "", explanation
        
    except json.JSONDecodeError:
        # Fallback to regex parsing (legacy format support)
        pass

    # Extract NEEDS_TRANSFORM
    needs_match = re.search(r'NEEDS_TRANSFORM:\s*(YES|NO)', response, re.IGNORECASE)
    needs_transform = needs_match.group(1).upper() == "YES" if needs_match else False
    
    # Extract ISSUES
    issues = []
    issues_match = re.search(r'ISSUES:\s*\n(.*?)(?=\nSUMMARY:)', response, re.DOTALL | re.IGNORECASE)
    if issues_match:
        for line in issues_match.group(1).split('\n'):
            line = line.strip()
            if line and line.startswith('-'):
                issues.append(line.lstrip('- ').strip())
    
    # Extract SUMMARY
    summary = "Tidak ada ringkasan"
    summary_match = re.search(r'SUMMARY:\s*\n?(.*?)(?=\nPYTHON_CODE:)', response, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()
    
    # Extract PYTHON_CODE
    code = "df = df.copy()"
    code_match = re.search(r'PYTHON_CODE:\s*\n(.*?)(?:#\s*END_CODE|$)', response, re.DOTALL | re.IGNORECASE)
    if code_match:
        code = code_match.group(1).strip()
        code = re.sub(r'^```python\s*\n?', '', code)
        code = re.sub(r'^```\s*\n?', '', code)
        code = re.sub(r'\n?```\s*$', '', code)
        code = code.strip()
    
    if 'def ' in code and 'return' in code:
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if func_match:
            func_name = func_match.group(1)
            code = code + f"\ndf = {func_name}(df)"
    
    # VALIDATE CODE
    dangerous_patterns = [
        (r'pd\.read_excel\s*\(', 'pd.read_excel()'),
        (r'pd\.read_csv\s*\(', 'pd.read_csv()'),
        (r'pd\.read_parquet\s*\(', 'pd.read_parquet()'),
        (r'open\s*\([\'"]', 'open()'),
    ]
    
    for pattern, error_msg in dangerous_patterns:
        if re.search(pattern, code):
            return (
                "df = df.copy()",
                f"❌ AI generate kode berbahaya: {error_msg}",
                [f"Kode mencoba {error_msg}"],
                True,
                code,
                "Gagal generate penjelasan karena kode berbahaya"
            )
    
    if not code or len(code) < 3:
        code = "df = df.copy()"
    
    return code, summary, issues, needs_transform, "", summary # Use summary as explanation fallback


def execute_transform(df: DataFrame, code: str) -> tuple[DataFrame, str]:
    """Execute transformation code on DataFrame."""
    import numpy as np
    import re as re_module
    import datetime
    
    try:
        local_ns = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "re": re_module,
            "datetime": datetime,
        }
        
        # Merge globals and locals to ensure imports are available
        global_ns = local_ns.copy()
        global_ns["__builtins__"] = __builtins__
        
        exec(code, global_ns)
        
        result_df = None
        for var_name in ["normalized_df", "df_result", "df_new", "df_transformed", "df_melted", "df_final", "result", "df"]:
            if var_name in global_ns and isinstance(global_ns[var_name], DataFrame):
                result_df = global_ns[var_name]
                break
        
        if result_df is None:
            for var_name, var_value in global_ns.items():
                if isinstance(var_value, DataFrame) and var_name != "_":
                # Check if it's a new dataframe or modified original
                    if var_name == "df" or (len(var_value) != len(df) or list(var_value.columns) != list(df.columns)):
                        result_df = var_value
                        break
        
        if result_df is None:
            result_df = global_ns.get("df", df)
        
        if not isinstance(result_df, DataFrame):
            return df, "Kode tidak menghasilkan DataFrame"
        
        return result_df, ""
        
    except Exception as e:
        return df, f"Error: {str(e)}"


def analyze_and_generate_transform(
    df: DataFrame,
    filename: str = "",
    sheet_name: str = "",
    user_description: str = ""
) -> TransformResult:
    """Analyze data structure, generate transformation code, execute it, validate, and iterate."""
    try:
        client = _get_client()
    except Exception as e:
        return TransformResult(
            summary=f"Error koneksi API: {str(e)}",
            issues_found=[],
            transform_code="df = df.copy()",
            needs_transform=False,
            preview_df=df.head(20).copy(),
            original_df=df.head(50).copy(),
            validation_notes=["Tidak bisa konek ke AI"],
            explanation="Gagal koneksi API"
        )
    
    original_sample = df.head(50).copy()
    validation_notes = []
    previous_issues = None
    previous_code = None
    error_history = []
    last_failed_code = ""
    
    logger.info(f"Starting analysis for file: {filename}, sheet: {sheet_name}")
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        try:
            code, summary, issues, needs_transform, failed_code, explanation = _generate_transform_code(
                client, df, filename, sheet_name, previous_issues, user_description, 
                previous_code, error_history, original_df=original_sample
            )
            
            if failed_code:
                logger.warning(f"Iteration {iteration}: Dangerous code detected")
                validation_notes.append(f"Iterasi {iteration}: ❌ Kode berbahaya")
                error_history.append({"iteration": iteration, "error": "Dangerous code", "code": failed_code})
                previous_issues = ["Kode berbahaya detected"]
                previous_code = failed_code
                last_failed_code = failed_code
                continue
            
            if not needs_transform:
                logger.info("Analysis complete: No transformation needed")
                return TransformResult(
                    summary=summary,
                    issues_found=issues,
                    transform_code="df = df.copy()",
                    needs_transform=False,
                    preview_df=df.head(20).copy(),
                    original_df=original_sample,
                    validation_notes=["Data sudah OK"],
                    iterations_used=iteration,
                    explanation=explanation
                )
            
            sample_df = df.head(100).copy()
            transformed_df, error = execute_transform(sample_df, code)
            
            if error:
                logger.warning(f"Iteration {iteration}: Execution error: {error}")
                validation_notes.append(f"Iterasi {iteration}: Error eksekusi - {error}")
                error_history.append({"iteration": iteration, "error": error, "code": code})
                previous_issues = [f"Error: {error}"]
                previous_code = code
                last_failed_code = code
                continue
            
            # Check for duplicate columns
            if transformed_df is not None and transformed_df.columns.duplicated().any():
                dup_cols = transformed_df.columns[transformed_df.columns.duplicated()].tolist()
                error = f"Duplicate columns: {dup_cols[:5]}"
                logger.warning(f"Iteration {iteration}: {error}")
                validation_notes.append(f"Iterasi {iteration}: {error}")
                error_history.append({"iteration": iteration, "error": error, "code": code})
                previous_issues = [error]
                previous_code = code
                last_failed_code = code
                continue

            comparison_issues = _compare_dataframes(sample_df, transformed_df)
            critical_issues = [i for i in comparison_issues if i.startswith("❌")]
            
            if critical_issues:
                logger.warning(f"Iteration {iteration}: Validation failed: {critical_issues}")
                validation_notes.append(f"Iterasi {iteration}: Validation failed - {critical_issues}")
                error_history.append({"iteration": iteration, "error": str(critical_issues), "code": code})
                previous_issues = critical_issues
                previous_code = code
                last_failed_code = code
                continue
            
            logger.info("Analysis complete: Transformation generated successfully")
            return TransformResult(
                summary=summary,
                issues_found=issues,
                transform_code=code,
                needs_transform=True,
                preview_df=transformed_df.head(20).copy(),
                original_df=original_sample,
                validation_notes=validation_notes + ["Success"],
                iterations_used=iteration,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Iteration {iteration}: Unexpected exception: {e}")
            validation_notes.append(f"Iterasi {iteration}: Exception - {str(e)}")
            previous_issues = [str(e)]
            
    logger.error("Analysis failed after max iterations")
    return TransformResult(
        summary="Gagal setelah max iterations",
        issues_found=previous_issues or [],
        transform_code="df = df.copy()",
        needs_transform=False,
        preview_df=df.head(20).copy(),
        original_df=original_sample,
        validation_notes=validation_notes,
        has_error=True,
        failed_code=last_failed_code,
        explanation="Gagal setelah max iterations"
    )


def regenerate_with_feedback(
    df: DataFrame,
    previous_code: str,
    user_feedback: str,
    filename: str = "",
    sheet_name: str = "",
    original_df: DataFrame = None,
    transformed_df: DataFrame = None,
    previous_error: str = None
) -> TransformResult:
    """
    Regenerate transformation based on user feedback.
    
    Args:
        df: Raw DataFrame (full data)
        previous_code: The code that was previously generated
        user_feedback: User's feedback about what's wrong
        filename: Original filename for context
        sheet_name: Sheet name for context
        original_df: Original sample before any transformation (for context)
        transformed_df: Current transformed preview (for context)
        previous_error: Error message from previous execution (if any)
        
    Returns:
        TransformResult with new transformation based on feedback
    """
    try:
        client = _get_client()
    except Exception as e:
        return TransformResult(
            summary=f"Error koneksi API: {str(e)}",
            issues_found=[],
            transform_code=previous_code,
            needs_transform=True,
            preview_df=df.head(20).copy(),
            original_df=original_df,
            validation_notes=["Tidak bisa konek ke AI"],
            explanation="Gagal koneksi API"
        )
    
    # Keep original for context
    if original_df is None:
        original_df = df.head(50).copy()
    
    original_sample_text = _dataframe_to_sample_text(original_df, MAX_SAMPLE_ROWS)
    
    # Show current transformed state if available
    transformed_text = ""
    if transformed_df is not None:
        transformed_text = f"""
HASIL TRANSFORMASI SAAT INI:
{_dataframe_to_sample_text(transformed_df, 20)}
"""

    # Add error context if available
    error_context = ""
    if previous_error:
        error_context = f"""
❌ ERROR PADA EKSEKUSI SEBELUMNYA:
{previous_error}
"""
    
    prompt = f"""Perbaiki kode transformasi berdasarkan feedback user.

DATA ASLI:
{original_sample_text}
{transformed_text}

KODE SEBELUMNYA:
```python
{previous_code}
```

{error_context}

FEEDBACK USER: "{user_feedback}"

RULES:
❌ JANGAN: pd.read_excel/csv, df.dtype, df[['a','b']] = split, melt(), pivot()
✅ BOLEH: df.iloc, df.columns, df.astype(str), for loop, pd.DataFrame()

⚠️ PENTING: Hasil akhir HARUS disimpan di variable `normalized_df` (bukan df, df_new, dll). Jangan ubah `df` original.

FORMAT (JSON):
{{
  "summary": "Penjelasan singkat",
  "explanation": "Penjelasan detail langkah demi langkah untuk user non-teknis",
  "code": "normalized_df = ..."
}}
"""

    try:
        system_instruction = """Fix kode pandas. Rules:
❌ df.dtype, df[['a','b']], file I/O, melt(), pivot()
✅ df.dtypes, split satu-satu, .astype(str), for loop
⚠️ WAJIB: Simpan hasil akhir ke variable `normalized_df`. Jangan ubah `df` original.

OUTPUT HARUS JSON VALID."""

        response = client.responses.create(
            model=settings.default_llm_model,
            instructions=system_instruction,
            input=prompt,
    
        )
        
        raw_response = response.output_text
        if not raw_response:
            return TransformResult(
                summary="AI tidak memberikan response",
                issues_found=[],
                transform_code=previous_code,
                needs_transform=True,
                preview_df=df.head(20).copy(),
                original_df=original_df,
                validation_notes=["Response kosong"],
            )
        
        # Parse response (JSON)
        import json
        try:
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            data = json.loads(cleaned_response)
            summary = data.get("summary", "Transformasi diperbaiki")
            explanation = data.get("explanation", "Tidak ada penjelasan detail")
            code = data.get("code", previous_code)
        except json.JSONDecodeError:
            # Fallback regex
            summary = "Transformasi diperbaiki berdasarkan feedback"
            summary_match = re.search(r'SUMMARY:\s*\n?(.*?)(?=\nPYTHON_CODE:)', raw_response, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary = summary_match.group(1).strip()
            
            explanation = summary # Fallback
            
            code = previous_code
            code_match = re.search(r'PYTHON_CODE:\s*\n(.*?)(?:#\s*END_CODE|$)', raw_response, re.DOTALL | re.IGNORECASE)
            if code_match:
                code = code_match.group(1).strip()
                code = re.sub(r'^```python\s*\n?', '', code)
                code = re.sub(r'^```\s*\n?', '', code)
                code = re.sub(r'\n?```\s*$', '', code)
                code = code.strip()
        
        # Validate code - block dangerous file read patterns
        dangerous_patterns = [
            (r'pd\.read_excel\s*\(', 'pd.read_excel() - data sudah ada di df!'),
            (r'pd\.read_csv\s*\(', 'pd.read_csv() - data sudah ada di df!'),
            (r'pd\.read_parquet\s*\(', 'pd.read_parquet() - data sudah ada di df!'),
            (r'open\s*\([\'"]', 'open() untuk file - tidak diizinkan!'),
        ]
        
        for pattern, error_msg in dangerous_patterns:
            if re.search(pattern, code):
                return TransformResult(
                    summary=f"Error: Kode mencoba baca file ({error_msg})",
                    issues_found=[f"Kode tidak valid: {error_msg}"],
                    transform_code=previous_code,  # Keep previous code
                    needs_transform=True,
                    preview_df=df.head(20).copy(),
                    original_df=original_df,
                    validation_notes=[f"Blocked: AI mencoba {error_msg}"],
                    explanation="Gagal: Kode berbahaya"
                )
        
        # Handle function definitions
        if 'def ' in code and 'return' in code:
            func_match = re.search(r'def\s+(\w+)\s*\(', code)
            if func_match:
                func_name = func_match.group(1)
                code = code + f"\ndf = {func_name}(df)"
        
        # Execute and get preview - use MORE rows for better testing
        sample_df = df.head(100).copy()
        new_transformed_df, error = execute_transform(sample_df, code)
        
        if error:
            return TransformResult(
                summary=f"Error: {error}",
                issues_found=[error],
                transform_code=code,
                needs_transform=True,
                preview_df=df.head(20).copy(),  # Show original if error
                original_df=original_df,
                validation_notes=[f"Error eksekusi: {error}"],
                explanation=f"Gagal eksekusi: {error}"
            )
        
        # Debug: Log the transformation result
        print(f"[DEBUG] regenerate_with_feedback:")
        print(f"  - Input df shape: {sample_df.shape}")
        print(f"  - Output df shape: {new_transformed_df.shape}")
        print(f"  - Output columns: {list(new_transformed_df.columns)[:5]}...")
        print(f"  - Preview will show: {min(20, len(new_transformed_df))} rows")
        
        return TransformResult(
            summary=summary,
            issues_found=[],
            transform_code=code,
            needs_transform=True,
            preview_df=new_transformed_df.head(20).copy(),
            original_df=original_df,
            validation_notes=[f"Diperbaiki berdasarkan feedback: {user_feedback[:50]}..."],
            iterations_used=1,
            explanation=explanation
        )
        
    except Exception as e:
        return TransformResult(
            summary=f"Error: {str(e)}",
            issues_found=[str(e)],
            transform_code=previous_code,
            needs_transform=True,
            preview_df=df.head(20).copy(),
            original_df=original_df,
            validation_notes=[f"Exception: {str(e)}"],
            explanation=f"Error: {str(e)}"
        )


def get_quick_analysis(df: DataFrame) -> Dict[str, Any]:
    """
    Quick heuristic analysis without AI call.
    Good for initial fast check.
    """
    issues = []
    
    # Check for unnamed columns
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        issues.append(f"Ada {len(unnamed_cols)} kolom tanpa nama")
    
    # Check if first row looks like a header
    if len(df) > 0:
        first_row = df.iloc[0]
        all_str = all(isinstance(v, str) or pd.isna(v) for v in first_row.values)
        if all_str and unnamed_cols:
            issues.append("Baris pertama mungkin adalah header yang benar")
    
    # Check for potential pivot format
    month_patterns = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_cols = [c for c in df.columns if any(m in str(c).lower() for m in month_patterns)]
    if len(month_cols) >= 3:
        issues.append("Kemungkinan format pivot (kolom bulan terdeteksi)")
    
    # Check for empty first rows
    if len(df) > 0:
        empty_rows = 0
        for i in range(min(5, len(df))):
            if df.iloc[i].isna().all() or (df.iloc[i].astype(str).str.strip() == '').all():
                empty_rows += 1
            else:
                break
        if empty_rows > 0:
            issues.append(f"Ada {empty_rows} baris kosong di awal")
    
    return {
        "issues": issues,
        "unnamed_columns": unnamed_cols,
        "has_potential_issues": len(issues) > 0,
    }
