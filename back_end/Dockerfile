# Google Cloud Run: build context = back_end directory.
#   gcloud run deploy baptist-mortality-api --source ./back_end --region REGION
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
