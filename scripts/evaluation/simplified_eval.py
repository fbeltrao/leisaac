"""Script to run a leisaac inference with leisaac in the simulation."""

import argparse
from typing import TYPE_CHECKING
from eval_in_sim import run_evaluation, EvaluationConfig

import gymnasium as gym


TASK_TYPE_TO_ENV_CFG = "so101leader"

def create_evaluation_config(args_cli: argparse.Namespace) -> EvaluationConfig:

    from leisaac.utils.env_utils import (
        dynamic_reset_gripper_effort_limit_sim,
    )

    def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str) -> dict:
        """Preprocess the observation dictionary to the format expected by the policy."""
        if model_type in ["gr00tn1.5", "lerobot", "openpi"]:
            obs_dict["task_description"] = language_instruction
            return obs_dict
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def before_step(env: gym.Env, action: dict) -> None:
        unwrapped_env = env.unwrapped
        if unwrapped_env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(unwrapped_env, TASK_TYPE_TO_ENV_CFG)


    def init_env_cfg(args_cli: argparse.Namespace, env_cfg: gym.Env) -> gym.Env:
        env_cfg.use_teleop_device(TASK_TYPE_TO_ENV_CFG)
        return env_cfg

    return EvaluationConfig(
        init_env_cfg=init_env_cfg,
        process_observations=preprocess_obs_dict,
        on_before_step=before_step,
        environment_libraries=["leisaac"],
        environment_metrics_group_name = "subtask_terms",
    )



if __name__ == "__main__":
    # run the main function  

    run_evaluation(
        config_factory=create_evaluation_config
    )

    
