#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
IMAGE_PATH="${1:-${PROJECT_ROOT}/data/images/ab/ab21ca88af067f38cc85233731e808528311ac4e6b4fdd0706ffcfc1a0f8a567.jpg}"
PROMPT="${2:-person . helmet .}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
DEBUG_ROOT="${ANNOTATION_DEBUG_ROOT:-${ANNOTATION_DEBUG_WORKSPACE:-${PROJECT_ROOT}/debug}}"
SESSION_ROOT="${DEBUG_ROOT}/sessions/concurrency-${RUN_ID}"
LOG_ROOT="${SESSION_ROOT}/logs"
API_URL="http://127.0.0.1:8010"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
set -a
. annotation_service/.env
set +a
export ANNOTATION_STORAGE_ROOT="${SESSION_ROOT}/storage"
export ANNOTATION_WORKER_POLL_SECONDS=0.1
export ANNOTATION_SAM_POLL_SECONDS=0.1
export ANNOTATION_QWEN_POLL_SECONDS=0.1
export PYTHONUNBUFFERED=1

PIDS=()
cleanup() {
  local pid
  for ((index=${#PIDS[@]}-1; index>=0; index--)); do
    pid="${PIDS[index]}"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_api() {
  for _ in $(seq 1 120); do
    if curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "API startup timed out; see ${LOG_ROOT}/api.log" >&2
  return 1
}

wait_for_log() {
  local pid="$1"
  local path="$2"
  local pattern="$3"
  for _ in $(seq 1 240); do
    if grep -q "${pattern}" "${path}" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Worker exited; see ${path}" >&2
      tail -30 "${path}" >&2 || true
      return 1
    fi
    sleep 0.5
  done
  echo "Worker startup timed out; see ${path}" >&2
  return 1
}

echo "Benchmark data: ${SESSION_ROOT}"
echo "Starting API and initializing isolated database ..."
"${PYTHON}" -m uvicorn annotation_service.api.app:app \
  --host 127.0.0.1 --port 8010 >"${LOG_ROOT}/api.log" 2>&1 &
PIDS+=("$!")
wait_for_api

echo "Starting one persistent GroundingDINO worker ..."
"${PYTHON}" -m annotation_service.pipeline.grounding_dino >"${LOG_ROOT}/dino.log" 2>&1 &
PIDS+=("$!")
wait_for_log "${PIDS[-1]}" "${LOG_ROOT}/dino.log" \
  "GroundingDINO model preloaded and ready"

echo "Starting one persistent SAM worker (embedding cache enabled) ..."
"${PYTHON}" -m annotation_service.pipeline.sam >"${LOG_ROOT}/sam.log" 2>&1 &
PIDS+=("$!")
wait_for_log "${PIDS[-1]}" "${LOG_ROOT}/sam.log" \
  "SAM model preloaded and ready"

echo "Starting four persistent Qwen workers ..."
for worker_number in 1 2 3 4; do
  ANNOTATION_QWEN_WORKER_ID="qwen-benchmark-${worker_number}" \
    "${PYTHON}" -m annotation_service.pipeline.qwen \
    >"${LOG_ROOT}/qwen-${worker_number}.log" 2>&1 &
  PIDS+=("$!")
done
sleep 1

echo "Warming DINO, SAM cache, and Qwen once ..."
"${PYTHON}" -m annotation_service.tools.concurrency_benchmark \
  "${IMAGE_PATH}" "${PROMPT}" --api-url "${API_URL}" \
  --concurrency 1 --requests 1 \
  --output "${SESSION_ROOT}/warmup.json" \
  >"${LOG_ROOT}/warmup.console.log"

LEVEL_SPECS=(${ANNOTATION_BENCHMARK_LEVELS:-1:20 2:50 4:50 8:100 16:100})
for level_spec in "${LEVEL_SPECS[@]}"; do
  concurrency="${level_spec%%:*}"
  request_count="${level_spec##*:}"
  report="${SESSION_ROOT}/concurrency-${concurrency}.json"
  echo "Running concurrency=${concurrency}, requests=${request_count} ..."
  "${PYTHON}" -m annotation_service.tools.concurrency_benchmark \
    "${IMAGE_PATH}" "${PROMPT}" --api-url "${API_URL}" \
    --concurrency "${concurrency}" --requests "${request_count}" \
    --output "${report}" >"${LOG_ROOT}/concurrency-${concurrency}.console.log" \
    || echo "  level contained failed requests; continuing"
  REPORT_PATH="${report}" "${PYTHON}" -c \
    'import json, os; s=json.load(open(os.environ["REPORT_PATH"], encoding="utf-8"))["summary"]; t=s["total_ms"]; print(f"  success={s['"'"'succeeded'"'"']}/{s['"'"'requests'"'"']} throughput={s['"'"'throughput_rps'"'"']:.3f} req/s P50={t['"'"'p50'"'"']:.1f}ms P95={t['"'"'p95'"'"']:.1f}ms P99={t['"'"'p99'"'"']:.1f}ms")'
done

echo "Benchmark complete: ${SESSION_ROOT}"
