FROM eclipse-temurin:21-jdk-jammy AS builder
WORKDIR /build
COPY pom.xml .
COPY src ./src
RUN apt-get update && apt-get install -y --no-install-recommends maven && \
    mvn package -DskipTests -q && \
    apt-get purge -y maven && rm -rf /var/lib/apt/lists/*

FROM eclipse-temurin:21-jre-jammy
WORKDIR /app

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --no-create-home appuser

COPY --from=builder --chown=appuser:appgroup /build/target/app.jar app.jar

USER 10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD java -cp app.jar HealthCheck || exit 1

ENTRYPOINT ["java", \
  "-XX:+UseContainerSupport", \
  "-XX:MaxRAMPercentage=75.0", \
  "-Djava.security.egd=file:/dev/./urandom", \
  "-jar", "app.jar"]
