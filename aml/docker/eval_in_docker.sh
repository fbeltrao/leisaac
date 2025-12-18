#!/bin/bash
set -e # Exit on error

# To run locally
# sudo docker run -it --gpus all -v $PWD:/workspace/code -v /path/to/checkpoint/:/workspace/checkpoint isaac-lab-with-azcli:2.3.0

# Function to monitor memory usage
monitor_memory() {
  while true; do
    echo "$(date): Memory usage: $(free -h | grep '^Mem:' | awk '{print $3"/"$2}')"
    echo "$(date): GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1"/"$2 " MB"}')"
    sleep 30
  done
}

# Function to cleanup processes
cleanup() {
  echo "$(date): Cleaning up processes..."
  # Kill policy server
  pkill -f "lerobot.async_inference.policy_server" || true
  # Kill background monitor
  jobs -p | xargs -r kill || true
  echo "$(date): Cleanup completed"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Function to display usage help
show_help() {
  local exit_code=${1:-0}
  cat << EOF
Usage: $0 [OPTIONS]

Evaluation script for policy inference in Isaac Lab.

OPTIONS:
  --task <value>                        Task name (default: LeIsaac-SO101-PickOrange-v0)
  --policy_type <value>                 Policy type (default: lerobot-act)
  --policy_language_instruction <value> Language instruction for the policy (default: "Pick up the orange and place it on the plate")
  --policy_action_horizon <value>       Policy action horizon (default: 50)
  --episode_length_s <value>            Episode length in seconds (default: 15)
  --eval_rounds <value>                 Number of evaluation rounds (default: 3)
  --seed <value>                        Random seed (default: 42)
  --policy_checkpoint_path <value>      Path to policy checkpoint (REQUIRED)
  --device <value>                      Device to use for computation. (default: cuda. Use cuda:1 for a different GPU)
  --help                                Display this help message

EXAMPLE:
  $0 --policy_checkpoint_path /workspace/checkpoint --task LeIsaac-SO101-PickOrange-v0

EOF
  exit $exit_code
}

# Parse command line arguments
TASK="LeIsaac-SO101-PickOrange-v0"
POLICY_TYPE="lerobot-act"
POLICY_LANGUAGE_INSTRUCTION="Pick up the orange and place it on the plate"
POLICY_ACTION_HORIZON=50
EPISODE_LENGTH_S=20
EVAL_ROUNDS=3
SEED=42
POLICY_CHECKPOINT_PATH=""
CHECKPOINT_SUB_FOLDER=""
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      show_help
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --policy_type)
      POLICY_TYPE="$2"
      shift 2
      ;;
    --policy_language_instruction)
      POLICY_LANGUAGE_INSTRUCTION="$2"
      shift 2
      ;;
    --policy_action_horizon)
      POLICY_ACTION_HORIZON="$2"
      shift 2
      ;;
    --episode_length_s)
      EPISODE_LENGTH_S="$2"
      shift 2
      ;;
    --eval_rounds)
      EVAL_ROUNDS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --policy_checkpoint_path)
      POLICY_CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;      
    *)
      echo "Error: Unknown option: $1"
      echo ""
      show_help 1
      ;;
  esac
done

# Validate required parameters
if [[ -z "$POLICY_CHECKPOINT_PATH" ]]; then
  echo "Error: --policy_checkpoint_path is required"
  echo ""
  show_help 1
fi

# Print initial memory status
echo "$(date): Initial memory status:"
free -h
nvidia-smi

# Start memory monitoring in background
monitor_memory > ./logs/memory_monitor.log 2>&1 &
MONITOR_PID=$!

# Start inference server
# Supported types: LeRobot
mkdir -p ./logs
echo "$(date): Starting policy server..."
# "$LEROBOT_PYTHON_PATH" -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 > ./logs/policy_server.log 2>&1 &
"$LEROBOT_PYTHON_PATH" -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 &

POLICY_SERVER_PID=$!
echo "$(date): Policy server started with PID: $POLICY_SERVER_PID"


# Wait for policy server to be ready
echo "$(date): Waiting for policy server to be ready..."
sleep 10
for i in {1..30}; do
  if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "$(date): Policy server is ready"
    break
  fi
  echo "$(date): Waiting for policy server... attempt $i"
  sleep 2
done

# Install leisaac package
echo "$(date): Installing leisaac package..."
# /isaac-sim/python.sh -m pip install -e "source/leisaac[isaaclab,gr00t,lerobot-async,openpi,mlflow]"
/isaac-sim/python.sh -m pip install -e "source/leisaac[gr00t,lerobot-async,openpi,mlflow]"

# Print memory before starting simulation
echo "$(date): Memory before starting simulation:"
free -h
nvidia-smi

# Run the evaluation with timeout and error handling
echo "$(date): Starting policy inference evaluation..."
timeout 3600 /isaac-sim/python.sh scripts/evaluation/policy_inference.py \
  --task $TASK \
  --policy_type $POLICY_TYPE \
  --policy_port 8080 \
  --policy_language_instruction "$POLICY_LANGUAGE_INSTRUCTION" \
  --policy_checkpoint_path "$POLICY_CHECKPOINT_PATH" \
  --device $DEVICE \
  --enable_cameras \
  --policy_action_horizon "$POLICY_ACTION_HORIZON" \
  --episode_length_s $EPISODE_LENGTH_S \
  --eval_rounds $EVAL_ROUNDS \
  --seed $SEED \
  --record \
  --headless || {
    EXIT_CODE=$?
    echo "$(date): Evaluation failed with exit code: $EXIT_CODE"
    echo "$(date): Final memory status:"
    free -h
    nvidia-smi
    echo "$(date): Policy server logs:"
    tail -50 ./logs/policy_server.log || true
    echo "$(date): Memory monitor logs:"
    tail -20 ./logs/memory_monitor.log || true
    exit $EXIT_CODE
}

echo "$(date): Evaluation completed successfully"
echo "$(date): Final memory status:"
free -h
nvidia-smi