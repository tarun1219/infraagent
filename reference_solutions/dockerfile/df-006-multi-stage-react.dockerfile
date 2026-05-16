FROM node:20-slim AS deps
WORKDIR /build
COPY package*.json ./
RUN npm ci && npm cache clean --force

FROM node:20-slim AS builder
WORKDIR /build
COPY --from=deps /build/node_modules ./node_modules
COPY . .
RUN npm run build

FROM nginx:1.25-alpine
RUN addgroup -g 10001 -S appgroup && \
    adduser -u 10001 -S appuser -G appgroup && \
    chown -R appuser:appgroup /var/cache/nginx /var/log/nginx /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && chown appuser:appgroup /var/run/nginx.pid

COPY --chown=appuser:appgroup nginx.conf /etc/nginx/nginx.conf
COPY --from=builder --chown=appuser:appgroup /build/dist /usr/share/nginx/html

USER 10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost:8080/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
