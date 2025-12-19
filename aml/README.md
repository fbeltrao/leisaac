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

## Evaluate policy in Azure ML

```
COMPUTE_NAME="<AML-COMPUTE_CLUSTER_NAME>"
COMPUTE_NAME="gpu-nc4as-t4-v3"
POLICY_TYPE="lerobot-act"
EPISODE_LENGTH_S=50
POLICY_LANGUAGE_INSTRUCTION="Pick up the orange and place it on the plate"
POLICY_ACTION_HORIZON=50
EVAL_ROUNDS=10
TRAINING_JOB="clever_gas_yjfdf87zhj"
CHECKPOINT_SUB_PATH="/checkpoints/050000/pretrained_model"

aml/scripts/02-eval-in-sim.sh \
    --training-job "$TRAINING_JOB" \
    --set "compute=$COMPUTE_NAME" \
    --set "inputs.policy_type=$POLICY_TYPE" \
    --set "inputs.policy_language_instruction=$POLICY_LANGUAGE_INSTRUCTION" \
    --set "inputs.eval_rounds=$EVAL_ROUNDS" \
    --set "inputs.policy_action_horizon=$POLICY_ACTION_HORIZON" \
    --set "inputs.seed=42" \
    --set "inputs.episode_length_s=$EPISODE_LENGTH_S" \
    --set "inputs.checkpoint_sub_folder=$CHECKPOINT_SUB_PATH"
    --set "inputs.device=cuda:1"
```

### Modifying cuda allocation based on VM

The evaluation job uses a single compute to run Isaac Sim and inference. Depending on the compute SKU used you might want to allocate CUDA devices for each workload. Based on experience with ACT model, this is the setup depending on the compute SKU

| Compute SKU | GPUs / memory | Mode | Additional job argument |
|---|---|---|---|
|[STANDARD_NC64AS_T4_V3](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series?tabs=sizeaccelerators)| 4x GPUs with 16GB each | Split GPU between workloads | `--set "inputs.device=cuda:1"`|
|[STANDARD_NC24ADS_A100_V4](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizeaccelerators)| 1x GPUs with 80GB | All in single GPU | none |
|[Standard_NC40ads_H100_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/ncads-h100-v5)| 1x GPUs with 94GB | All in single GPU | none |

## Evaluate policy in virtual machine

If you want to evaluate the policy in a virtual machine with Isaac Sim and the downloaded checkpoint follow these steps:

1. Download the checkpoint locally
1. Start inference server
    1. In LeRobot: `uv run -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080`
2. Start evaluation
```bash
CHECKPOINT_PATH=/path/to/checkpoint
uv run scripts/evaluation/policy_inference.py --task "LeIsaac-SO101-PickOrange-v0" --policy_type "lerobot-act" --policy_port 8080 --policy_language_instruction "Pick up the orange and place it on the plate" --policy_checkpoint_path "$CHECKPOINT_PATH" --device cuda --enable_cameras --policy_action_horizon 50 --episode_length_s 15 --eval_rounds 3 --record --headless
```
