FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir --target=/deps gunicorn uvicorn[standard]
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt
COPY . .

FROM gcr.io/distroless/python3-debian12:nonroot
WORKDIR /app
COPY --from=builder /deps /app/lib
COPY --from=builder /build/app /app/app

ENV PYTHONPATH="/app/lib" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER nonroot:nonroot

EXPOSE 8000

ENTRYPOINT ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8000", "app.main:app"]
