# ==============================================================
# Solar Wind Prediction Competition — Aquiri Space submission
# Hybrid gradient boosting (ACE plasma + 27-day recurrence + AIA coronal holes).
#
# The platform runs:  <ENTRYPOINT> <INFERENCE_TIMESTAMP>
# e.g.                /app/run.sh 20261111T060000Z
# ==============================================================
FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- PLATFORM LAYER — needed for the upload helper -----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY upload.sh /app/upload.sh
# -------------------------------------------------------------

# ================= EDIT BELOW =================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
COPY benchlab_forecasting/ ./benchlab_forecasting/
COPY model/ ./model/
COPY forecast.py .
# ================= EDIT ABOVE =================================

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh /app/upload.sh \
    && useradd --create-home --uid 1000 runner \
    && chown -R runner /app
USER runner

ENTRYPOINT ["/app/run.sh"]
