# Underconfidence Adversarial Training (UAT) - Docker Image
# Python 3.11 with PyTorch 2.6.0

FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command (can be overridden)
CMD ["/bin/bash"]
