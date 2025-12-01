# PandasAI Purchasing Chat

An intelligent data analysis application tailored for purchasing data, powered by **Google Gemini 2.5 Flash** and **PandasAI**. This tool allows users to upload datasets, perform natural language queries, and automatically transform data for analysis.

## Features

-   **AI-Powered Q&A**: Ask questions about your data in plain English using Google's Gemini 2.5 Flash model.
-   **Intelligent Data Transformation**: Automatically cleans and normalizes messy Excel/CSV data. The AI iteratively generates, executes, and validates transformation code, self-correcting if errors occur.
-   **OneDrive Integration**: Directly browse, select, and upload files from your OneDrive for Business.
-   **Secure Access**: Simple password-based authentication to protect your data.
-   **Data Catalog**: Keeps track of uploaded files and their metadata using a local SQLite database.
-   **Interactive Dashboard**: Built with Streamlit for a responsive and user-friendly experience.

## Prerequisites

-   **Docker Desktop** (Recommended)
-   **Google API Key** (Get one from [Google AI Studio](https://aistudio.google.com/))
-   **OneDrive Credentials** (Client ID, Secret, Tenant ID) - *Optional, for OneDrive features*

## Configuration

1.  Clone the repository.
2.  Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```
3.  Edit `.env` and set the following variables:
    ```ini
    # Required
    GOOGLE_API_KEY=your_google_api_key_here
    APP_PASSWORD=admin123  # Set your desired login password

    # Optional (for OneDrive Integration)
    ONEDRIVE_CLIENT_ID=your_client_id
    ONEDRIVE_CLIENT_SECRET=your_client_secret
    ONEDRIVE_TENANT_ID=your_tenant_id
    ONEDRIVE_USER_ID=your_user_id
    ```

## Running the Application

### Option 1: Using Docker (Recommended)

The easiest way to run the app is using Docker Compose. This ensures all dependencies and the environment are correctly set up.

1.  Build and start the container:
    ```bash
    docker-compose up -d --build
    ```
2.  Open your browser and navigate to:
    ```
    http://localhost:8505
    ```

### Option 2: Running Locally

If you prefer to run it without Docker:

1.  Create a virtual environment:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py --server.port 8505
    ```

## Project Structure

```
.
├── app/
│   ├── data_analyzer.py  # AI-driven data transformation logic
│   ├── data_store.py     # SQLite database operations
│   ├── datasets.py       # File handling and dataframe management
│   ├── onedrive_client.py# OneDrive API integration
│   ├── qa_engine.py      # PandasAI and Gemini client wrapper
│   └── settings.py       # Configuration management
├── data/                 # Storage for uploaded files and SQLite DB
├── streamlit_app.py      # Main Streamlit application entry point
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker services configuration
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables (not committed)
```

## Tech Stack

-   **Frontend**: Streamlit
-   **AI Engine**: Google Gemini 2.5 Flash (via `google-genai` SDK)
-   **Data Processing**: Pandas, PandasAI
-   **Database**: SQLite
-   **Containerization**: Docker

## Troubleshooting

-   **Port Conflicts**: If port `8505` is in use, modify the `ports` section in `docker-compose.yml` and the `EXPOSE` instruction in `Dockerfile`.
-   **API Errors**: Ensure your `GOOGLE_API_KEY` is valid and has access to the `gemini-1.5-flash` model.
-   **OneDrive Issues**: Verify your Azure App Registration permissions if OneDrive sync fails.
