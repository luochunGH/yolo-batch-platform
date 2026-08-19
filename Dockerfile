# Incremental application layer over the already validated CUDA/Ultralytics image.
# The original wheel cache was intentionally removed after the first build.
FROM yolo-batch-service:1.0

USER root
COPY wheels/onnx /tmp/onnx-wheels
RUN pip install --no-index --find-links=/tmp/onnx-wheels --no-deps \
    onnx==1.19.1 onnxslim==0.1.34 onnxruntime==1.16.3 \
    ml_dtypes==0.5.1 protobuf==7.35.1 \
    coloredlogs==15.0.1 flatbuffers==25.12.19 humanfriendly==10.0 \
    && rm -rf /tmp/onnx-wheels
COPY service /app/service
RUN chown -R appuser:appuser /app/service
USER appuser
