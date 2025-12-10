"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse


# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")


import time


import leisaac  # noqa: F401




def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def main(args_cli: argparse.Namespace = None):
    """Running lerobot teleoperation with leisaac manipulation environment."""

    max_episode_count = args_cli.eval_rounds
    task_type = "so101leader"

    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from leisaac.policy import Gr00tServicePolicyClient


        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[],
            modality_keys = ["single_arm", "gripper"],
        )
    elif "lerobot" in args_cli.policy_type:
        from leisaac.policy import LeRobotServicePolicyClient

        model_type = "lerobot"

        policy_type = args_cli.policy_type.split("-")[1]
        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={},
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif args_cli.policy_type == "openpi":
        from leisaac.policy import OpenPIServicePolicyClient

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[],
            task_type=task_type,
        )


        obs_dict = preprocess_obs_dict({}, model_type, args_cli.policy_language_instruction)
        actions = policy.get_action(obs_dict)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
    parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
    parser.add_argument(
        "--eval_rounds",
        type=int,
        default=0,
        help=(
            "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
            " reset."
        ),
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="gr00tn1.5",
        help="Type of policy to use. support gr00tn1.5, lerobot-<model_type>, openpi",
    )
    parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
    parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
    parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
    parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
    parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
    parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
    main(parser.parse_args())
