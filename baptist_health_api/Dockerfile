# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.13-slim AS builder
WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.13-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (venv/ and model_files/ excluded via .dockerignore)
COPY . .

EXPOSE 8080

# Single worker — models are loaded into process memory at startup.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
