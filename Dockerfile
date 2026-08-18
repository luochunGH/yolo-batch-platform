# Incremental application layer over the already validated CUDA/Ultralytics image.
# The original wheel cache was intentionally removed after the first build.
FROM yolo-batch-service:1.0

USER root
COPY service /app/service
RUN chown -R appuser:appuser /app/service
USER appuser
