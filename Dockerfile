FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py .
COPY models/ ./models/
ENV MODEL_PATH=./models/bilstm_sentiment.keras TOK_PATH=./models/tokenizer.json MAX_LEN=100 PORT=5000 TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
EXPOSE 5000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:5000/health || exit 1
CMD ["gunicorn","-w","2","-b","0.0.0.0:5000","--timeout","180","serve:app"]
