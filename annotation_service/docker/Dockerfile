ARG ANNOTATION_BASE_IMAGE=lisa-safety-seg:lisa13b-clean030-v1-records-20260723
FROM ${ANNOTATION_BASE_IMAGE}

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/lisa-annotation:/opt/GroundingDINO:/app \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

WORKDIR /opt/lisa-annotation

COPY --from=groundingdino_source \
    /groundingdino /opt/GroundingDINO/groundingdino

COPY annotation_service/requirements.txt /tmp/annotation-requirements.txt
COPY annotation_service/docker/requirements-worker.txt \
    /tmp/annotation-worker-requirements.txt

RUN python3 -m pip install \
        -r /tmp/annotation-requirements.txt \
        -r /tmp/annotation-worker-requirements.txt \
    && python3 -c "import torch, torchvision, transformers; \
assert torch.__version__ == '2.1.0+cu121'; \
assert torchvision.__version__ == '0.16.0+cu121'; \
assert transformers.__version__ == '4.31.0'" \
    && rm -f \
        /tmp/annotation-requirements.txt \
        /tmp/annotation-worker-requirements.txt

COPY annotation_service /opt/lisa-annotation/annotation_service

RUN chown -R lisa:lisa /opt/lisa-annotation

USER lisa

ARG ANNOTATION_SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.title="GroundingDINO Free Detection API" \
    org.opencontainers.image.revision="${ANNOTATION_SOURCE_COMMIT}"

EXPOSE 8001

HEALTHCHECK NONE

CMD ["python3", "-m", "uvicorn", "annotation_service.app:app", \
     "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
