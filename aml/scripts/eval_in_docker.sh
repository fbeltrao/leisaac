#!/bin/bash
set -e # Exit on error

# sudo docker run -it --gpus all -v $PWD:/workspace/code -v /home/azureuser/code/lerobot/outputs/loyal_rhubarb_bpnbqzkwdh/output=checkpoint/checkpoints/050000/pretrained_model:/workspace/checkpoint isaac-lab-with-azcli:2.3.0

# "$LEROBOT_PYTHON_PATH" -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 &
"$LEROBOT_PYTHON_PATH" -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 > /workspace/code/policy_server.log 2>&1 &

TASK="LeIsaac-SO101-PickOrange-v0"
POLICY_TYPE="lerobot-act"
POLICY_LANGUAGE_INSTRUCTION="Pick up the orange and place it on the plate"
POLICY_ACTION_HORIZON=50
EPISODE_LENGTH_S=15
EVAL_ROUNDS=3

cd /workspace/code
# /isaac-sim/python.sh -m pip install -e "source/leisaac[isaaclab,gr00t,lerobot-async,openpi,mlflow]"
/isaac-sim/python.sh -m pip install -e "source/leisaac[gr00t,lerobot-async,openpi,mlflow]"
/isaac-sim/python.sh scripts/evaluation/policy_inference.py --task $TASK --policy_type $POLICY_TYPE --policy_port 8080 --policy_language_instruction $POLICY_LANGUAGE_INSTRUCTION --policy_checkpoint_path "/workspace/checkpoint" --device cuda --enable_cameras --policy_action_horizon $POLICY_ACTION_HORIZON --episode_length_s $EPISODE_LENGTH_S --eval_rounds $EVAL_ROUNDS --record --headless --disable-mlflow