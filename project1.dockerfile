# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    default-mysql-client \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/uploads /app/logs

# Expose ports
# 8000 for FastAPI
# 8501 for Streamlit
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Smart Health Analyzer..."\n\
python main.py &\n\
streamlit run interface.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]