FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Environment defaults
ENV LLM_PROVIDER=groq
ENV DATABASE_URL=sqlite:///./beastlife_care.db
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose ports
EXPOSE 8000 8501

# Default command - start API server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
