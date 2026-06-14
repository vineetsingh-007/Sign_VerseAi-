# Use official light Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create necessary directories
RUN mkdir -p datasets models logs reports database

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit dashboard by default
CMD ["streamlit", "run", "dashboard/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
