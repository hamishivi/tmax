FROM tblite-repair-okhttp:20260909 AS tools
RUN apt-get update && apt-get install -y python3
FROM tools AS cache
RUN ./gradlew :okhttp:jvmTestClasses --no-daemon --console=plain --max-workers=4
COPY okhttp-cache.gradle /tmp/okhttp-cache.gradle
RUN ./gradlew -I /tmp/okhttp-cache.gradle warmVerifierRuntime --no-daemon --console=plain --max-workers=4
FROM tools
COPY --from=cache /root/.gradle/caches /root/.gradle/caches
COPY --from=cache /root/.gradle/jdks /root/.gradle/jdks
LABEL evaluation.variant="okhttp-python-runtime-v1"
