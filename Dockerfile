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
COPY main.py .

# Step 6 — Default command
CMD ["python", "main.py"]


# docker build -t customer-segmentation .           Step 1 — BUILD (reads Dockerfile, creates image, saves it)
# docker run customer-segmentation                  Step 2 — RUN (takes saved image, starts a container from it)
# docker run -v local_path:container_path image_name

# docker run `
#   -v "${PWD}/data:/app/data" `
#   -v "${PWD}/models:/app/models" `
#   -v "${PWD}/outputs:/app/outputs" `
#   customer-segmentation