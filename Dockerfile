# Docker v2 - improved version

# Step 1 — Base image (Python version)
FROM python:3.11-slim

# Step 2 — Set working directory inside container
WORKDIR /app

# Step 3 — Copy requirements first
COPY requirements.txt .

# Step 4 — Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 5 — Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY main.py .

# Step 6 — Default command
CMD ["python", "main.py"]
