# Use official lightweight Python base image
FROM python:3.11-slim

# Set environment variables for clean logging and memory stability
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Set working directory inside container
WORKDIR /app

# Install minimal OS dependencies required for OpenCV headless & Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements definition first (to leverage Docker build caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files (including model_loader.py)
COPY app.py .
COPY model_loader.py .
COPY static/ ./static/
COPY models/ ./models/

# Expose port 8000 for FastAPI / Uvicorn
EXPOSE 8000

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]