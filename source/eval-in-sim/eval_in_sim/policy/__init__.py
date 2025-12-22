import argparse
from .base import Policy
from isaaclab.sensors import Camera
import gymnasium as gym


def create_policy(args_cli: argparse.Namespace, env: gym.Env, joint_names: list[str] = None) -> tuple[Policy, str]:

    policy_type = args_cli.policy_type
    if policy_type == "gr00tn1.5":
        from .service_policy_clients import Gr00tServicePolicyClient
        modality_keys = ["single_arm", "gripper"]

        return Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.unwrapped.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        ), "gr00tn1.5"
    elif "lerobot" in policy_type:
        from .service_policy_clients import LeRobotServicePolicyClient
        policy_type = policy_type.split("-")[1]
        return LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={
                key: sensor.image_shape for key, sensor in env.unwrapped.scene.sensors.items() if isinstance(sensor, Camera)
            },
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
            joint_names=joint_names,
        ), "lerobot"
    elif policy_type == "openpi":
        from .service_policy_clients import OpenPIServicePolicyClient

        return OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.unwrapped.scene.sensors.items() if isinstance(sensor, Camera)],
        ), "openpi"
    
    raise ValueError(f"Unknown policy type: {args_cli.policy_type}")