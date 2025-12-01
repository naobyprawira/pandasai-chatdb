# PandasAI File Q&A

Simple Streamlit app that lets users upload tabular data, catalog it, and ask questions powered by [PandasAI](https://github.com/sinaptik-ai/pandas-ai).

## Features

- Upload CSV/TSV/Excel files (per-user storage)
- Catalog of previously uploaded datasets with metadata persisted in SQLite
- Natural language Q&A using PandasAI + OpenAI
- Configurable via `.env`

## Getting started

1. **Install dependencies** (Windows `cmd.exe` assumed):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  2>nul && echo (optional)
```

2. **Configure environment**:
  - Copy `.env.example` to `.env` and set `OPENAI_API_KEY` plus optional overrides like `PANDASAI_LLM_MODEL` (defaults to `gpt-5.1-mini`).

3. **Run the Streamlit app**:

```cmd
.venv\Scripts\activate
streamlit run streamlit_app.py
```

4. Open the provided local URL, upload a dataset, enter your OpenAI key in the sidebar, and start asking questions.

## Testing

Run unit tests (covers the dataset catalog layer):

```cmd
.venv\Scripts\activate
pytest
```

## Project structure

```
app/
  data_store.py   # SQLite-backed dataset catalog
  datasets.py     # Upload persistence & dataframe helpers
  qa_engine.py    # PandasAI wrapper
  settings.py     # Centralized configuration
streamlit_app.py  # Streamlit UI entry point
requirements.txt  # Runtime deps
```

Uploads and the SQLite catalog live under `data/`. Clean them up anytime by deleting the directory.
