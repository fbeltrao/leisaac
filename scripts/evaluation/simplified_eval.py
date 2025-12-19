"""Script to run a leisaac inference with leisaac in the simulation."""

from typing import TYPE_CHECKING
from eval_in_sim import run_evaluation, EvaluationConfig

if TYPE_CHECKING:
    import gymnasium as gym


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")

def before_step(env: gym.Env):
    if unwrapped_env.cfg.dynamic_reset_gripper_effort_limit:
        dynamic_reset_gripper_effort_limit_sim(unwrapped_env, "s101leaderarm")


if __name__ == "__main__":
    # run the main function

    run_evaluation(
        config=EvaluationConfig(
            preprocess_obs=preprocess_obs_dict,
            before_step=before_step,
            environment_libraries=["leisaac"],
        )
    )


    
