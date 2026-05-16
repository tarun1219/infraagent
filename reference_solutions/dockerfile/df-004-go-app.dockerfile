FROM golang:1.22-bookworm AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o app ./cmd/server

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=builder /build/app .

USER nonroot:nonroot

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["/app/app", "-healthcheck"]

ENTRYPOINT ["/app/app"]
