
import os, re, time, logging
from pathlib import Path
from flask import Flask, request, jsonify, Response
from prometheus_client import (Counter, Histogram, generate_latest,
                                CollectorRegistry, CONTENT_TYPE_LATEST)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sentiment")

MODEL_PATH = os.getenv("MODEL_PATH", "./models/bilstm_sentiment.keras")
TOK_PATH   = os.getenv("TOK_PATH",   "./models/tokenizer.json")
MAX_LEN    = int(os.getenv("MAX_LEN", "100"))
PORT       = int(os.getenv("PORT", "5000"))
LABELS     = ["positive", "negative", "neutral"]

REG   = CollectorRegistry()
REQS  = Counter("sentiment_requests_total",    "Requests",    ["endpoint"], registry=REG)
PREDS = Counter("sentiment_predictions_total", "Predictions", ["label"],    registry=REG)
LAT   = Histogram("sentiment_latency_seconds", "Latency",     ["endpoint"], registry=REG)
ERRS  = Counter("sentiment_errors_total",      "Errors",      ["type"],     registry=REG)

log.info("Loading model...")
import keras
try:
    from keras.src.legacy.preprocessing.text import tokenizer_from_json
except ImportError:
    from keras.preprocessing.text import tokenizer_from_json

model     = keras.models.load_model(MODEL_PATH)
tokenizer = tokenizer_from_json(Path(TOK_PATH).read_text())
log.info("Ready — %d params", model.count_params())

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^\w\s!?.,\']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict(texts):
    from keras.utils import pad_sequences
    seqs   = tokenizer.texts_to_sequences([clean(t) for t in texts])
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    probs  = model.predict(padded, verbose=0)
    return [{"sentiment":  LABELS[int(p.argmax())],
             "confidence": round(float(p.max()), 4),
             "probs": {LABELS[i]: round(float(p[i]), 4) for i in range(3)}}
            for p in probs]

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"name": "YouTube Sentiment AI", "status": "running",
                    "endpoints": ["POST /predict","GET /health","GET /metrics"]})

@app.route("/health")
def health():
    REQS.labels(endpoint="/health").inc()
    return jsonify({"status": "healthy"})

@app.route("/metrics")
def metrics():
    return Response(generate_latest(REG), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict_route():
    REQS.labels(endpoint="/predict").inc()
    t0 = time.perf_counter()
    try:
        body = request.get_json(force=True)
        if isinstance(body, str):    texts = [body]
        elif isinstance(body, list): texts = body
        elif isinstance(body, dict): texts = body.get("texts") or [body.get("text","")]
        else: return jsonify({"error": "invalid body"}), 400
        results = predict([str(t) for t in texts if t])
        ms      = round((time.perf_counter()-t0)*1000, 2)
        for r in results: PREDS.labels(label=r["sentiment"]).inc()
        LAT.labels(endpoint="/predict").observe(time.perf_counter()-t0)
        return jsonify({"predictions": results,
                        "count": len(results), "latency_ms": ms})
    except Exception as e:
        ERRS.labels(type=type(e).__name__).inc()
        log.exception("Predict error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
