FROM golang:1.22-bookworm AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux \
    go build -trimpath -ldflags="-s -w" -o grpc-server ./cmd/grpc

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app

COPY --from=builder /build/grpc-server .

USER nonroot:nonroot

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/app/grpc-server", "-health"]

ENTRYPOINT ["/app/grpc-server"]
