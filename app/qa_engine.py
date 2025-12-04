from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, List


import pandas as pd
from pandas import DataFrame
from openai import OpenAI

from .settings import AppSettings
from app.logger import get_chat_logger

settings = AppSettings()
logger = get_chat_logger()


@dataclass
class QAResult:
    prompt: str
    response: Any
    code: str | None = None
    explanation: str | None = None  # AI explanation of the results
    st_components: List[dict] = field(default_factory=list)  # Streamlit components to render
    iterations_used: int = 1  # How many iterations to get valid result
    has_error: bool = False  # True if generation failed (all iterations failed)
    failed_code: str = ""  # Store the failed code for debugging
    validation_notes: List[str] = field(default_factory=list)  # Validation feedback


_SYSTEM_PROMPT = """\
Kamu adalah asisten analisis data Python yang berjalan di dalam Streamlit app.

## KRITIS - BACA INI DULU!
Variable `df` SUDAH BERISI DATA LENGKAP ({total_rows} baris) yang di-load dari parquet.
JANGAN PERNAH membuat DataFrame baru! Langsung gunakan `df`.

## Kolom yang Tersedia
{columns}

⚠️ PENTING: Gunakan EXACT nama kolom seperti di atas. Jika tidak yakin nama kolom, gunakan:
```python
available_cols = df.columns.tolist()
print(f"Kolom tersedia:", available_cols)
```

## Sample Data (5 baris pertama, untuk referensi struktur saja):
{sample}

{description_section}

## Aturan Kode
1. LANGSUNG gunakan variable `df` - sudah berisi semua data
2. **WAJIB FUZZY MATCH untuk search nama/teks** - lihat section di bawah!
3. Untuk tanggal, gunakan `pd.to_datetime(..., errors='coerce')`
4. **UNTUK MENAMPILKAN DATAFRAME:** Gunakan `st.dataframe(df.head(50))` - MAKSIMAL 50 ROWS!
5. Untuk hasil angka/teks: gunakan `st.write()` atau `st.metric()`
```

**Contoh penggunaan:**
```python
# Cari supplier
mask = fuzzy_match(df['Supplier Name'], 'SUNG DONG IL', threshold=80)

# Cari dengan filter tambahan (tahun)
df['PO Date'] = pd.to_datetime(df['PO Date'], errors='coerce')
mask_supplier = fuzzy_match(df['Supplier Name'], 'SUNG DONG IL', threshold=80)
mask_year = df['PO Date'].dt.year == 2025
result = df[mask_supplier & mask_year]
st.dataframe(result.head(50))
```

**Parameter threshold:**
- `threshold=70` untuk match lebih loose (typo-tolerant)
- `threshold=80` untuk balance (RECOMMENDED)
- `threshold=90` untuk match lebih strict

# → KEDUA: Tampilkan: Supplier (context) + Score (jawaban) + Value (supporting)
available_cols = df.columns.tolist()
# Tentukan kolom yang akan digunakan
context_col = 'Supplier' if 'Supplier' in available_cols else 'Name'
score_col = 'Score' if 'Score' in available_cols else 'Performance'
value_col = 'Value' if 'Value' in available_cols else 'Cost'

mask = df[score_col] > df[score_col].quantile(0.9)
result = df[mask][[context_col, score_col, value_col]].head(50)
st.dataframe(result, use_container_width=True)  ✅ BENAR
```

⚠️ PENTING: SELALU CHECK NAMA KOLOM YANG SEBENARNYA ADA DI `df.columns` SEBELUM MENGGUNAKAN!


### Cara Memilih Kolom yang Tepat:

**Analisis pertanyaan user dan pilih kolom:**

1. **Identifikasi konteks pertanyaan:**
   - Siapa/apa yang ditanya? (supplier, kategori, periode, dll)
   - Ini adalah kolom "context" yang harus di-include

2. **Identifikasi metrik yang ditanya:**
   - Apa yang ingin dijawab? (total, performa, perbandingan, dll)
   - Ini adalah kolom "jawaban" yang harus di-include

3. **Tambah supporting info:**
   - Data apa lagi yang penting untuk memahami? (periode, status, detail, dll)
   - Include jika relevant dengan pertanyaan

```python
# Contoh:
# Pertanyaan: "Berapa order dari supplier ABC bulan lalu?"
# Context: Supplier ABC
# Jawaban: Order Quantity
# Supporting: Date/Period, Status
result = df[df['Supplier'] == 'ABC'].tail(30)[['Supplier', 'Date', 'Order Qty', 'Value']]
st.dataframe(result, use_container_width=True)

# Pertanyaan: "Mana bulan dengan penjualan terbesar?"
# Context: Period
# Jawaban: Sales/Revenue
# Supporting: Trend info
monthly = df.groupby('Period').agg({'Sales': 'sum'}).reset_index()
monthly = monthly.sort_values('Sales', ascending=False)
st.dataframe(monthly.head(12), use_container_width=True)
```

### DataFrame Display Rules:
- LIMIT 50 ROWS dengan `df.head(50)`

## DILARANG (LIBRARY & FUNGSI):
- ⛔ JANGAN import library eksternal apapun (contoh: fuzzywuzzy, rapidfuzz, numpy, pandas, re, process, tabulate, dsb) — SEMUA SUDAH TERSEDIA di environment!
- ⛔ JANGAN definisikan ulang fungsi yang sudah ada di environment (misal: fuzzy_match, pd.to_datetime, dsb)

Selalu CEK fungsi yang sudah tersedia di environment sebelum membuat fungsi baru!

## ANTI-HALLUCINATION & ERROR PREVENTION:
1. **NameError Prevention**:
   - JANGAN: `if condition: x = 1` lalu print(x) ← Error jika condition False!
   - HARUS: `x = 0` (default) di awal, baru `if condition: x = 1`

2. **KeyError Prevention**:
   - JANGAN: `df['Disc %']` jika belum yakin kolom ada.
   - HARUS: `cols = [c for c in ['Disc %', 'Discount'] if c in df.columns]`

3. **Empty Data Prevention**:
   - JANGAN: `st.dataframe(df[mask])` jika mask kosong.
   - HARUS:
     ```python
     if result.empty:
         st.warning("Data tidak ditemukan")
     else:
         st.dataframe(result.head(50))
     ```

Balas HANYA dengan blok kode Python (```python ... ```).
"""

# Sampling config - keep small to avoid AI copying data
SAMPLE_SIZE = 5  # only show 5 rows for structure reference

# Prompt for explaining query results
_EXPLAIN_PROMPT = """\
Kamu adalah asisten analisis data yang membantu menjelaskan hasil query.

## Pertanyaan User:
{user_question}

## Hasil Query (Data Aktual):
{query_result}

## Tugas:
Analisis data di atas dan berikan INSIGHT yang menjawab pertanyaan user.

## Aturan WAJIB:
1. BACA DATA YANG ADA - jangan bilang "tidak ada informasi" jika data sudah tersedia!
2. MENTION identifier yang relevan: nama, kategori, periode, tipe, dll (berguna agar user tahu siapa/apa yang dimaksud)
3. Jika ada kolom numerik (value, qty, score, dll) - ANALISIS nilainya:
   - Rata-rata, min, max, trend, perbandingan
   - Apakah bagus atau perlu perhatian
   - Highlight outlier/anomali jika ada
4. Jawab LANGSUNG pertanyaan user dengan data yang ada
5. Gunakan angka spesifik dari tabel
6. Berikan insight/rekomendasi praktis jika relevan

## Contoh Insight yang Bagus:
- "TOP 3 Category: ABC (Rp 5M), DEF (Rp 3M), GHI (Rp 2M) - ABC mendominasi 50% total value"
- "Trend naik dari Jan (Rp 1M) ke Jul (Rp 2M) - growth 100% year-to-date"
- "5 supplier dengan value <Rp 500K - pertimbangkan konsolidasi untuk efisiensi"
- "April mencatat peak dengan 300 qty (2x rata-rata) - ada seasonal factor"

## Format:
Bullet points singkat dan informatif (2-5 poin).
WAJIB menyebut angka spesifik dan identifier dari data.
"""


def _build_system_prompt(df: DataFrame, table_description: str = None, column_descriptions: dict = None) -> str:
    cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
    n_rows = len(df)
    
    # Only show first 5 rows for structure reference
    sample_df = df.head(SAMPLE_SIZE)
    sample = sample_df.to_csv(index=False)
    
    # Build description section
    desc_parts = []
    if table_description:
        desc_parts.append(f"## Deskripsi Tabel:\n{table_description}")
    
    if column_descriptions:
        desc_parts.append("## Deskripsi Kolom:")
        for col, desc in column_descriptions.items():
            if col in df.columns:
                desc_parts.append(f"- **{col}**: {desc}")
    
    description_section = "\n\n".join(desc_parts)
    
    return _SYSTEM_PROMPT.format(
        columns=cols, 
        sample=sample, 
        total_rows=n_rows,
        description_section=description_section
    )


def _extract_code(text: str) -> str:
    """Extract the first ```python ... ``` block from the response."""
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: assume entire response is code
    return text.strip()


def _fuzzy_match(series: pd.Series, query: str, threshold: int = 85) -> pd.Series:
    """Return a boolean Series where True means the value fuzzy-matches the query.
    
    More strict matching - query words must be found in value, not just similar.
    
    Matching priority:
    1. Exact substring: "DONG JIN" in "DONG JIN TEXTILE CO" -> match
    2. All query words present: "JIN DONG" matches "DONG JIN TEXTILE" -> match  
    3. Fuzzy match: "DONGJIN" matches "DONG JIN" -> match
    """
    import re
    
    # Normalize query
    query = str(query).upper().strip()
    query_words = set(re.findall(r'\w+', query))
    
    if not query_words:
        return pd.Series([False] * len(series), index=series.index)
    
    def check_match(val):
        if pd.isna(val):
            return False
        val_str = str(val).upper()
        
        # 1. Exact substring check (fastest)
        if query in val_str:
            return True
            
        # 2. All words check (order independent)
        val_words = set(re.findall(r'\w+', val_str))
        if query_words.issubset(val_words):
            return True
            
        # 3. Fuzzy check (slower, only if needed)
        # Simple fuzzy: check if query without spaces is in value without spaces
        query_nospace = query.replace(" ", "")
        val_nospace = val_str.replace(" ", "")
        if query_nospace in val_nospace:
            return True
            
        return False

    return series.apply(check_match)


def _safe_exec(code: str, df: DataFrame) -> tuple[str, List[dict]]:
    """
    Execute generated code safely and capture output.
    Returns: (output_string, list_of_streamlit_components)
    """
    import io
    import pandas as pd
    import numpy as np
    import re as re_module
    import datetime
    from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz
    
    buf = io.StringIO()
    st_components = []
    
    # Mock streamlit to capture output
    class MockStreamlit:
        def dataframe(self, data, use_container_width=False, **kwargs):
            """Capture dataframe display."""
            if isinstance(data, DataFrame):
                # Sanitize for display (handle non-serializable types)
                display_df = _sanitize_df_for_display(data.head(50).copy())
                st_components.append({
                    "type": "dataframe",
                    "data": display_df,
                    "total_rows": len(data),
                    "kwargs": {"use_container_width": use_container_width, **kwargs}
                })
            else:
                st_components.append({"type": "dataframe", "data": data})
        
        def metric(self, label, value, delta=None, **kwargs):
            """Capture metric call."""
            st_components.append({
                "type": "metric",
                "label": label,
                "value": value,
                "delta": delta,
                "kwargs": kwargs
            })
        
        def write(self, *args, **kwargs):
            """Capture write call."""
            st_components.append({
                "type": "write",
                "content": " ".join(str(a) for a in args),
                "kwargs": kwargs
            })
        
        def caption(self, text, **kwargs):
            """Capture caption call."""
            st_components.append({
                "type": "caption",
                "text": text,
                "kwargs": kwargs
            })
        
        def success(self, text, **kwargs):
            st_components.append({"type": "success", "text": text})
        
        def warning(self, text, **kwargs):
            st_components.append({"type": "warning", "text": text})
        
        def error(self, text, **kwargs):
            st_components.append({"type": "error", "text": text})
        
        def info(self, text, **kwargs):
            st_components.append({"type": "info", "text": text})
        
        def table(self, data, **kwargs):
            """Capture table call."""
            if isinstance(data, DataFrame):
                display_df = _sanitize_df_for_display(data.head(50).copy())
                st_components.append({
                    "type": "table",
                    "data": display_df,
                    "total_rows": len(data)
                })
            else:
                st_components.append({"type": "table", "data": data})
    
    mock_st = MockStreamlit()
    
    local_ns: dict[str, Any] = {
        "__builtins__": __builtins__,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "re": re_module,
        "datetime": datetime,
        "tabulate": __import__("tabulate").tabulate,
        "fuzzy_match": _fuzzy_match,
        "fuzz": fuzzywuzzy_fuzz,  # For AI-generated code using fuzzywuzzy
        "fuzzywuzzy": type('fuzzywuzzy', (), {'fuzz': fuzzywuzzy_fuzz})(),  # Mock module
        "st": mock_st,  # Mock Streamlit
    }
    try:
        with redirect_stdout(buf):
            exec(code, local_ns)  # noqa: S102
    except KeyError as exc:
        # KeyError usually means column not found
        col_name = str(exc).strip("'\"")
        available_cols = list(df.columns)
        return f"❌ Error: Kolom '{col_name}' tidak ditemukan!\n\nKolom yang tersedia: {', '.join(available_cols[:10])}" + (f" ... dan {len(available_cols)-10} lainnya" if len(available_cols) > 10 else ""), []
    except Exception as exc:
        exc_type = type(exc).__name__
        return f"❌ Execution error ({exc_type}): {str(exc)}\n\nPastikan kolom yang digunakan ada di data.", []
    output = buf.getvalue()
    return output if output.strip() else "", st_components


def _sanitize_df_for_display(df: DataFrame) -> DataFrame:
    """Convert non-serializable columns to string for display."""
    for col in df.columns:
        # Check if column has complex types (lists, dicts, objects)
        if df[col].dtype == 'object':
            try:
                # Try to see if it's safe
                df[col].head(1).to_json()
            except:
                # If not serializable, convert to string
                df[col] = df[col].astype(str)
    return df


class PandasAIClient:
    """Wrapper that asks the LLM to generate pandas code, then executes it."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        if not api_key:
            api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model or settings.default_llm_model

    def _generate_explanation(self, user_question: str, query_result: str) -> str:
        """Generate AI explanation of the query results."""
        # Skip explanation for errors or empty results
        if not query_result or "❌ Error" in query_result or "Execution error" in query_result:
            return ""
        
        # Skip if result is too short (probably just a simple number already explained)
        if len(query_result.strip()) < 20 and "\n" not in query_result:
            return ""

        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=_EXPLAIN_PROMPT,
                input=f"PERTANYAAN USER: {user_question}\n\nHASIL QUERY:\n{query_result}",
            )
            return response.output_text
        except Exception as e:
            return f"Gagal generate penjelasan: {str(e)}"

    def ask(self, df: DataFrame, prompt: str, explain: bool = True, 
            table_description: str = None, column_descriptions: dict = None) -> QAResult:
        """
        Ask a question about the DataFrame with iterative retry (max 3 attempts).
        
        Args:
            df: The DataFrame to query
            prompt: User's question
            explain: If True, generate AI explanation of results (default: True)
            
        Returns:
            QAResult with response, code, and optional explanation
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        MAX_ITERATIONS = 3
        error_history = []
        validation_notes = []
        system_prompt = _build_system_prompt(df, table_description, column_descriptions)
        last_failed_code = ""
        
        logger.info(f"Starting ask() with prompt: {prompt[:50]}...")
        
        for iteration in range(1, MAX_ITERATIONS + 1):
            logger.info(f"=== Iteration {iteration}/{MAX_ITERATIONS} ===")
            try:
                # Build messages with error context if retry
                current_prompt = prompt
                
                if error_history:
                    # Add error context for retry (similar to Data Analyzer)
                    error_context = "\n\n⚠️ RETRY - ERROR SEBELUMNYA:\n"
                    for idx, err in enumerate(error_history[-2:], 1):  # Show last 2 errors
                        error_context += f"Attempt {err['iteration']}: {err['error']}\n"
                    error_context += "\nCara fix:\n"
                    error_context += "1. Gunakan nama kolom EXACT seperti di list 'Kolom yang Tersedia'\n"
                    error_context += "2. Jika tidak yakin, print: available_cols = df.columns.tolist()\n"
                    error_context += "3. Gunakan approach lebih simple jika kode complex\n\n"
                    error_context += f"USER QUESTION: {prompt}"
                    current_prompt = error_context
                
                # Generate code
                response = self.client.responses.create(
                    model=self.model_name,
                    instructions=system_prompt,
                    input=current_prompt,
                )
                
                raw_answer = response.output_text
                code = _extract_code(raw_answer)
                logger.info(f"Generated code:\n{code[:200]}..." if len(code) > 200 else f"Generated code:\n{code}")
                result, st_components = _safe_exec(code, df)
                
                # Check if execution successful (no error)
                # Handle None/empty result safely
                result_str = result if result else ""
                has_error = "❌ Error" in result_str or "Execution error" in result_str
                logger.info(f"Execution result: has_error={has_error}, result_preview={result_str[:100]}..." if len(result_str) > 100 else f"Execution result: has_error={has_error}, result={result_str}")
                
                if not has_error:
                    # Success! Build explanation and return
                    validation_notes.append(f"Iterasi {iteration}: Query berhasil")
                    try:
                        text_for_explain = result if result else ""
                        for comp in st_components:
                            if comp["type"] == "dataframe" and "data" in comp:
                                comp_df = comp["data"]
                                total_rows = comp.get("total_rows", len(comp_df))
                                text_for_explain += f"\n\n📊 Data ({len(comp_df)} baris ditampilkan"
                                if total_rows > len(comp_df):
                                    text_for_explain += f" dari {total_rows} total"
                                text_for_explain += "):\n"
                                try:
                                    text_for_explain += comp_df.head(20).to_markdown(index=False)
                                except:
                                    text_for_explain += comp_df.head(20).to_string(index=False)
                            elif comp["type"] == "metric":
                                text_for_explain += f"\n📈 {comp.get('label', 'Metric')}: {comp.get('value', 'N/A')}"
                            elif comp["type"] == "write":
                                text_for_explain += f"\n{comp.get('content', '')}"
                            elif comp["type"] == "caption":
                                text_for_explain += f"\n({comp.get('text', '')})"
                        
                        explanation = ""
                        if explain and (text_for_explain.strip() or st_components):
                            explanation = self._generate_explanation(prompt, text_for_explain)
                        
                        return QAResult(
                            prompt=prompt, 
                            response=result, 
                            code=code, 
                            explanation=explanation, 
                            st_components=st_components,
                            iterations_used=iteration,
                            validation_notes=validation_notes
                        )
                    except Exception as build_err:
                        # Error building explanation - still return result without explanation
                        logger.error(f"Failed to build explanation: {build_err}")
                        validation_notes.append(f"Iterasi {iteration}: Gagal build explanation - {build_err}")
                        return QAResult(
                            prompt=prompt,
                            response=result if result else "(Hasil tersedia di tabel)",
                            code=code,
                            explanation=f"(Gagal build explanation: {build_err})",
                            st_components=st_components,
                            iterations_used=iteration,
                            validation_notes=validation_notes
                        )
                else:
                    # Error occurred - save to history and retry (similar to Data Analyzer)
                    validation_notes.append(f"Iterasi {iteration}: Error eksekusi - {result[:100]}...")
                    error_entry = {
                        "iteration": iteration,
                        "error": result,
                        "code": code
                    }
                    error_history.append(error_entry)
                    
                    # If last iteration, return error result
                    if iteration == MAX_ITERATIONS:
                        logger.error(f"Max iterations reached, returning error")
                        return QAResult(
                            prompt=prompt,
                            response=result + f"\n\n(Sudah dicoba {MAX_ITERATIONS}x - masih error)",
                            code=code,
                            explanation="",
                            st_components=st_components,
                            iterations_used=MAX_ITERATIONS,
                            has_error=True,
                            failed_code=last_failed_code,
                            validation_notes=validation_notes
                        )
                    # Continue to next iteration
                    logger.info(f"Continuing to iteration {iteration + 1}")
                    last_failed_code = code  # Track failed code
                    continue
                    
            except Exception as e:
                # Unexpected exception (similar to Data Analyzer)
                logger.error(f"Unexpected exception in ask(): {e}")
                validation_notes.append(f"Iterasi {iteration}: Exception - {str(e)}")
                error_entry = {
                    "iteration": iteration,
                    "error": f"Exception: {str(e)}",
                    "code": ""
                }
                error_history.append(error_entry)
                
                if iteration == MAX_ITERATIONS:
                    return QAResult(
                        prompt=prompt,
                        response=f"❌ Error setelah {MAX_ITERATIONS} attempt: {str(e)}",
                        code="",
                        explanation="",
                        st_components=[],
                        iterations_used=MAX_ITERATIONS,
                        has_error=True,
                        failed_code=last_failed_code,
                        validation_notes=validation_notes
                    )
                last_failed_code = code if 'code' in locals() else ""
                continue
        
        # Fallback (shouldn't reach here)
        return QAResult(
            prompt=prompt,
            response="❌ Query gagal setelah semua retry",
            code="",
            explanation="",
            st_components=[],
            iterations_used=MAX_ITERATIONS,
            has_error=True,
            failed_code=last_failed_code,
            validation_notes=validation_notes
        )
