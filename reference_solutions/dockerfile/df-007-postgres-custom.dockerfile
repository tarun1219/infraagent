FROM postgres:16-alpine

RUN addgroup -g 10001 -S pgapp && \
    adduser -u 10001 -S pgapp -G pgapp

COPY --chown=pgapp:pgapp initdb/ /docker-entrypoint-initdb.d/
COPY --chown=pgapp:pgapp postgresql.conf /etc/postgresql/postgresql.conf

RUN chmod 550 /docker-entrypoint-initdb.d/*.sh 2>/dev/null || true && \
    chmod 440 /docker-entrypoint-initdb.d/*.sql 2>/dev/null || true

ENV POSTGRES_DB=appdb \
    PGDATA=/var/lib/postgresql/data/pgdata

EXPOSE 5432

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-appdb}" || exit 1

CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
