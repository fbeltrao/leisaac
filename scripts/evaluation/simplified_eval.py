"""Script to run a leisaac inference with leisaac in the simulation."""

import argparse
from typing import TYPE_CHECKING

import torch
from eval_in_sim import run_evaluation, EvaluationConfig

import gymnasium as gym


TASK_TYPE_TO_ENV_CFG = "so101leader"

def create_evaluation_config(args_cli: argparse.Namespace) -> EvaluationConfig:

    from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES
    from leisaac.utils.robot_utils import (
        convert_leisaac_action_to_lerobot,
        convert_lerobot_action_to_leisaac,
    )

    from leisaac.utils.env_utils import (
        dynamic_reset_gripper_effort_limit_sim,
    )


    def observations_processor(obs_dict: dict, model_type: str, language_instruction: str) -> dict:
        """Preprocess the observation dictionary to the format expected by the policy."""

        obs_dict["joint_pos"] = convert_leisaac_action_to_lerobot(obs_dict["joint_pos"])

        if model_type in ["gr00tn1.5", "lerobot", "openpi"]:
            obs_dict["task_description"] = language_instruction
            return obs_dict
        
        raise ValueError(f"Model type {model_type} not supported")

    def actions_processor(env: gym.Env, action: dict, obs: dict) -> None:
        unwrapped_env = env.unwrapped
        if unwrapped_env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(unwrapped_env, TASK_TYPE_TO_ENV_CFG)
        
        action = convert_lerobot_action_to_leisaac(action)
        return torch.from_numpy(action)


    def init_env_cfg(args_cli: argparse.Namespace, env_cfg: gym.Env) -> gym.Env:
        env_cfg.use_teleop_device(TASK_TYPE_TO_ENV_CFG)
        return env_cfg

    return EvaluationConfig(
        init_env_cfg=init_env_cfg,
        observations_processor=observations_processor,
        actions_processor=actions_processor,
        environment_libraries=["leisaac"],
        environment_metrics_group_name = "subtask_terms",
        joint_names=SINGLE_ARM_JOINT_NAMES,   
    )



if __name__ == "__main__":
    # run the main function  

    run_evaluation(
        config_factory=create_evaluation_config
    )

    
