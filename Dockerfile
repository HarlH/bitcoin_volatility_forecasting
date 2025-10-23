# Use a secure, small base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies first (for Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Set the entry point to run the ingestion script
# Note: We use the API keys via the `docker run` command for production, 
# but for now, we'll set it up for manual testing.
CMD ["python", "src/ingest_data.py"]