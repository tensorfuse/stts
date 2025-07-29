#!/bin/bash

# Default values
MODE="tts"
MODEL="orpheus"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Validate arguments
if [[ "$MODE" != "tts" ]]; then
  echo "Error: Only --mode tts is supported at the moment"
  exit 1
fi

if [[ "$MODEL" != "orpheus" ]]; then
  echo "Error: Only --model orpheus is supported at the moment"
  exit 1
fi

echo "Starting with mode: $MODE, model: $MODEL"

# Launch Triton and FastAPI
tritonserver \
  --model-repository=$TRITON_MODEL_REPOSITORY \
  --model-control-mode=explicit \
  --load-model=$MODEL --metrics-port=8003 & \
uvicorn server:app --host 0.0.0.0 --port 7004 --log-level info --access-log
