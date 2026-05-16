FROM node:20-slim AS builder
WORKDIR /build
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

FROM node:20-slim
WORKDIR /app

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --no-create-home --shell /bin/false appuser

COPY --from=builder --chown=appuser:appgroup /build/node_modules ./node_modules
COPY --chown=appuser:appgroup . .

ENV NODE_ENV=production \
    PORT=3000

USER 10001

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/healthz', r => process.exit(r.statusCode === 200 ? 0 : 1))"

CMD ["node", "server.js"]
