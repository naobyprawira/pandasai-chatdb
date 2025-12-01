FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8505

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8505/_stcore/health || exit 1

# Run the application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8505", "--server.address=0.0.0.0"]
