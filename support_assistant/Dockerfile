# Zepto Support Assistant — FastAPI + LangGraph + ChromaDB
# Required graded baseline: builds and runs locally. DO NOT push to a registry
# unless you want the optional, ungraded Hugging Face Spaces stretch.

FROM python:3.12-slim

WORKDIR /app

# hnswlib (used by ChromaDB) needs libgomp on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (avoids the multi-GB CUDA wheel)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache the embedding model inside the image so runtime stays offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Build / refresh the ChromaDB index at image-build time
RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]