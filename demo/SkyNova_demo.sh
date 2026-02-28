#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WEB_PORT="${WEB_PORT:-10003}"
CONTROLLER_PORT="${CONTROLLER_PORT:-40000}"
MODEL_WORKER_PORT="${MODEL_WORKER_PORT:-40001}"
SD_WORKER_PORT="${SD_WORKER_PORT:-39999}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/EarthDial_4B_RGB}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CONTROLLER_URL="http://127.0.0.1:${CONTROLLER_PORT}"
MODEL_WORKER_URL="http://127.0.0.1:${MODEL_WORKER_PORT}"
SD_WORKER_URL="http://127.0.0.1:${SD_WORKER_PORT}"

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/bash_logs"

echo "Starting SkyNova services..."

echo "Starting controller on ${CONTROLLER_URL}..."
nohup python "${SCRIPT_DIR}/controller.py" --host 0.0.0.0 --port "${CONTROLLER_PORT}" \
  > "${SCRIPT_DIR}/logs/controller.log" 2>&1 </dev/null &
sleep 2

echo "Starting model worker on ${MODEL_WORKER_URL}..."
nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${SCRIPT_DIR}/model_worker.py" \
  --host 0.0.0.0 \
  --controller "${CONTROLLER_URL}" \
  --port "${MODEL_WORKER_PORT}" \
  --worker "${MODEL_WORKER_URL}" \
  --model-path "${MODEL_PATH}" \
  > "${SCRIPT_DIR}/bash_logs/model_worker.log" 2>&1 </dev/null &
sleep 2

echo "Starting Streamlit on 0.0.0.0:${WEB_PORT}..."
nohup streamlit run "${SCRIPT_DIR}/app.py" --server.address 0.0.0.0 --server.port "${WEB_PORT}" -- \
  --controller_url "${CONTROLLER_URL}" \
  --sd_worker_url "${SD_WORKER_URL}" \
  > "${SCRIPT_DIR}/logs/streamlit.log" 2>&1 </dev/null &

if [[ "${START_SD_WORKER:-0}" == "1" ]]; then
  echo "Starting SD worker on ${SD_WORKER_URL}..."
  nohup python "${SCRIPT_DIR}/sd_worker.py" --port "${SD_WORKER_PORT}" \
    > "${SCRIPT_DIR}/logs/sd_worker.log" 2>&1 </dev/null &
fi

echo ""
echo "Services started."
echo "Controller log: ${SCRIPT_DIR}/logs/controller.log"
echo "Model log: ${SCRIPT_DIR}/bash_logs/model_worker.log"
echo "Web UI: http://<your-autodl-domain>:${WEB_PORT}"
if [[ "${START_SD_WORKER:-0}" != "1" ]]; then
  echo "Image generation is off. Set START_SD_WORKER=1 to enable."
fi
