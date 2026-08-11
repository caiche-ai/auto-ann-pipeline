#!/bin/sh
set -eu

pids=""

cleanup() {
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}

trap cleanup INT TERM EXIT

python3 -m uvicorn annotation_service.api.app:app \
    --host 0.0.0.0 --port 8001 --workers 1 &
pids="$pids $!"

python3 -m annotation_service.pipeline.grounding_dino &
pids="$pids $!"

python3 -m annotation_service.pipeline.sam &
pids="$pids $!"

python3 -m annotation_service.pipeline.qwen &
pids="$pids $!"

python3 -m annotation_service.release &
pids="$pids $!"

while :; do
    for pid in $pids; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            exit 1
        fi
    done
    sleep 1
done
