# Setup for Azure ML

## Environment

1. `uv venv`
1. Install IsaacSim and IsaacLab based on [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
1. Install Leisaac `uv pip install -e "source/leisaac[isaaclab,gr00t,lerobot-async,openpi,mlflow]"`
1. Download assets: `aml/download_assets.sh`


## Test environment with zero agent

```
uv run python scripts/environments/zero_agent.py --task "LeIsaac-SO101-PickOrange-v0" --num_envs 1 --device "cuda" --enable_cameras
```

## Evaluate policy


1. Start inference server
    1. In LeRobot: `uv run -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080`

2. Start evaluation
```
CHECKPOINT_PATH=/path/to/checkpoint
uv run scripts/evaluation/policy_inference.py --task "LeIsaac-SO101-PickOrange-v0" --policy_type "lerobot-act" --policy_port 8080 --policy_language_instruction "Pick up the orange and place it on the plate" --policy_checkpoint_path "$CHECKPOINT_PATH" --device cuda --enable_cameras --policy_action_horizon 50 --episode_length_s 15 --eval_rounds 3 --record --headless
```


