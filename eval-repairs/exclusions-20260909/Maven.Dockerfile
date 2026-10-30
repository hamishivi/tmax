FROM tblite-repair-maven:20260909 AS tools
RUN apt-get update && apt-get install -y maven python3 python3-venv && \
    curl -LsSf https://astral.sh/uv/0.7.13/install.sh -o /tmp/uv-install.sh && \
    UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh
ENV UV_CACHE_DIR=/opt/uv-cache
RUN /opt/uv/uv venv --python /usr/bin/python3 /opt/verifier-bootstrap && \
    /opt/uv/uv pip install --python /opt/verifier-bootstrap/bin/python pytest==8.0.0
FROM tools AS cache
RUN mvn -B dependency:go-offline
COPY dependency-pom.xml /tmp/dependency-resolution/pom.xml
RUN cd /tmp/dependency-resolution && mvn -B dependency:go-offline
RUN mvn -B dependency:get -Dartifact=org.apache.maven.surefire:surefire-junit4:3.1.2
FROM tools
COPY --from=cache /root/.m2/repository /root/.m2/repository
ENV PATH="/opt/uv:${PATH}" UV_OFFLINE=1 UV_LINK_MODE=copy
LABEL evaluation.variant="maven-slf4j-offline-dependencies-v1"
