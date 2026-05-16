FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 mlgroup && \
    useradd --uid 10001 --gid mlgroup --no-create-home mluser

COPY --from=builder /root/.local /home/mluser/.local
COPY --chown=mluser:mlgroup . .

RUN mkdir -p /app/models /app/tmp && \
    chown mluser:mlgroup /app/models /app/tmp

ENV PATH="/home/mluser/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    TMPDIR=/app/tmp \
    OMP_NUM_THREADS=2

USER 10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
