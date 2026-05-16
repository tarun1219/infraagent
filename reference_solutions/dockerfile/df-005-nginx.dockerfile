FROM nginx:1.25-alpine AS builder
RUN apk add --no-cache curl

FROM nginx:1.25-alpine

RUN addgroup -g 10001 -S nginx-app && \
    adduser -u 10001 -S nginx-app -G nginx-app && \
    chown -R nginx-app:nginx-app /var/cache/nginx /var/log/nginx /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && chown nginx-app:nginx-app /var/run/nginx.pid

COPY --chown=nginx-app:nginx-app nginx.conf /etc/nginx/nginx.conf
COPY --chown=nginx-app:nginx-app dist/ /usr/share/nginx/html/

RUN find /usr/share/nginx/html -type f -exec chmod 444 {} \; && \
    find /usr/share/nginx/html -type d -exec chmod 555 {} \;

USER 10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -fs http://localhost:8080/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
