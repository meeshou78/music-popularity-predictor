# Use a slim base image to reduce size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt .

# Install system dependencies required by XGBoost and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Now copy the full app source
COPY . .

# Port for Heroku (Flask default is overridden via PORT env var)
EXPOSE 5000

# Default command
CMD ["python", "main.py"]

