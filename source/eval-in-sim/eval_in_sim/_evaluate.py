import argparse
from dataclasses import dataclass
from os import PathLike
import time
from typing import Callable
import uuid

import gymnasium as gym
from isaaclab.app import AppLauncher
from omni.isaac.kit import SimulationApp


def resolve_args_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation in Simulation.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed of the environment."
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=60.0,
        help="Episode length in seconds.",
    )
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
    parser.add_argument(
        "--policy_host",
        type=str,
        default="localhost",
        help="Host of the policy server.",
    )
    parser.add_argument(
        "--policy_port", type=int, default=5555, help="Port of the policy server."
    )
    parser.add_argument(
        "--policy_timeout_ms",
        type=int,
        default=15000,
        help="Timeout of the policy server.",
    )
    parser.add_argument(
        "--policy_action_horizon",
        type=int,
        default=16,
        help="Action horizon of the policy.",
    )
    parser.add_argument(
        "--policy_language_instruction",
        type=str,
        default=None,
        help="Language instruction of the policy.",
    )
    parser.add_argument(
        "--policy_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path of the policy.",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record videos during evaluation."
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging during evaluation.",
    )

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    return parser.parse_args()


def create_simulation_app(args_cli: argparse.Namespace) -> SimulationApp:
    app_launcher_args = vars(args_cli)

    # launch omniverse app
    app_launcher = AppLauncher(app_launcher_args)
    return app_launcher.app


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation in simulation.
    Args:
        init_env_cfg (Callable[[argparse.Namespace, gym.Env], gym.Env] | None): Function to modify the environment
            configuration before creating the environment.
        create_policy (Callable[[argparse.Namespace, gym.Env], object] | None): Custom function to create the policy.
        on_before_step (Callable[[gym.Env, object], None] | None): Function to be called before each environment step.
        environment_libraries (list[str] | str | None): Additional environment libraries to import.
        process_observations (Callable[[dict, str, str] , dict] | None): Function to preprocess observation dict
            before getting actions from the policy.
        environment_metrics_group_name (str | None): Name of the observation group to log as metrics.
    """

    init_env_cfg: Callable[[argparse.Namespace, object], object] | None = None
    create_policy: Callable[[argparse.Namespace, gym.Env], object] | None = None
    on_before_step: Callable[[gym.Env, object], None] | None = None
    environment_libraries: list[str] | str | None = None
    process_observations: Callable[[dict, str, str], dict] | None = None
    environment_metrics_group_name: str | None = None


def import_libraries(libraries: list[str] | str):
    """
    Import additional environment libraries.
    Args:
        libraries (list[str] | str): List of library names to import.
    """
    if libraries is None:
        return
    if isinstance(libraries, str):
        libraries = [libraries]
    for library in libraries:
        __import__(library)


def enable_recording(env: gym.Env, recording_path: PathLike) -> gym.Env:
    # Recording uses the Gym RecordVideo wrapper
    # It makes a single recording for the entire evaluation session
    # We provide the output path to save the video files for the MLflowLoggerWrapper,
    # so that the video is avalilable as an artifact in MLflow.
    unwrapped_env_for_terminations = env.unwrapped

    def env_had_terminations() -> bool:
        """
        Triggers a new recording if one of the environments had a termination.
        Only works as long as we have a single environment (num_envs=1).
        """
        nonlocal unwrapped_env_for_terminations
        return (
            unwrapped_env_for_terminations.termination_manager.terminated.any()
            or unwrapped_env_for_terminations.termination_manager.time_outs.any()
        )

    return gym.wrappers.RecordVideo(
        env,
        video_folder=recording_path,
        name_prefix="eval_video",
        disable_logger=True,
        episode_trigger=lambda episode: True,
    )


def create_env_cfg(
    args_cli: argparse.Namespace, config: EvaluationConfig | None
) -> gym.Env:
    from isaaclab_tasks.utils import parse_env_cfg

    import_libraries(config.environment_libraries if config else None)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    if config and config.init_env_cfg:
        env_cfg = config.init_env_cfg(args_cli, env_cfg)

    return env_cfg


def run_app(
    simulation_app: SimulationApp,
    args_cli: argparse.Namespace,
    config: EvaluationConfig | None,
):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from ._controller import Controller
    from ._rate_limiter import RateLimiter
    from .policy.base import Policy
    from ._env_utils import detect_termination, TerminationStatus

    env_cfg = create_env_cfg(args_cli, config)

    max_episode_count = args_cli.eval_rounds
    # create environment
    render_mode = "rgb_array" if args_cli.record else None
    env: ManagerBasedRLEnv = gym.make(
        args_cli.task, cfg=env_cfg, render_mode=render_mode
    )

    # Add recording
    recording_path: PathLike | None = None
    if args_cli.record:
        # Recording uses the Gym RecordVideo wrapper
        # It makes a single recording for the entire evaluation session
        # We provide the output path to save the video files for the MLflowLoggerWrapper,
        # so that the video is avalilable as an artifact in MLflow.
        recording_path = f"./datasets/videos/{uuid.uuid4()}/"
        print(f"[INFO] Recording videos during evaluation in {recording_path}")
        env = enable_recording(env, recording_path)

    # Add MLflow logging
    policy_info = {
        "policy_type": args_cli.policy_type,
        "checkpoint_path": args_cli.policy_checkpoint_path,
    }

    if not args_cli.disable_mlflow:
        from ._mlflow_logger_wrapper import MLflowLoggerWrapper

        env = MLflowLoggerWrapper(
            env,
            policy=policy_info,
            task=args_cli.task,
            task_description=args_cli.policy_language_instruction,
            number_of_episodes=args_cli.eval_rounds,
            metrics_observation_group_name=config.environment_metrics_group_name
            if config
            else None,
            artifact_paths=[recording_path],
        )

    # create policy
    policy: Policy
    policy_type: str
    if config is not None and config.create_policy is not None:
        policy, policy_type = config.create_policy(args_cli, env)
    else:
        from .policy import create_policy

        policy, policy_type = create_policy(args_cli, env)

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()

    # record the results
    success_count, episode_count = 0, 1

    # simulate environment
    unwrapped_env = env.unwrapped
    device = unwrapped_env.device

    post_action_processor = None if config is not None or config.post_action_processor is None else config.post_action_processor
    observation_processor  = None if config is not None or config.process_observations is None else config.process_observations

    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        termination_status: TerminationStatus
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    obs_dict, _ = env.reset()
                    episode_count += 1
                    break

                obs_dict = obs_dict["policy"]
                if observation_processor is not None:
                    obs_dict = observation_processor(
                        obs_dict, policy_type, args_cli.policy_language_instruction
                    )

                actions = policy.get_action(obs_dict).to(device)
                if post_action_processor is not None:
                    actions = post_action_processor(actions, obs_dict)
                

                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]

                    if config and config.on_before_step is not None:
                        config.on_before_step(env, action)
                    obs_dict, _, reset_terminated, reset_time_outs, extras = env.step(
                        action
                    )

                    termination_status = detect_termination(
                        reset_terminated, reset_time_outs, extras
                    )

                    if termination_status.terminated:
                        if max_episode_count > 0 and episode_count < max_episode_count:
                            obs_dict, _ = env.reset()
                        break

                    if rate_limiter:
                        rate_limiter.sleep(env)

            if termination_status.terminated:
                if termination_status.success:
                    print(f"[Evaluation] Episode {episode_count} is successful!")
                    success_count += 1
                else:
                    print(
                        f"[Evaluation] Episode {episode_count} failed! Reasons: {termination_status.reasons}"
                    )
                episode_count += 1
                break

        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}]"
        )
    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}]"
    )

    # close the simulator
    env.close()
    simulation_app.close()


def run_evaluation(
    config_factory: Callable[[argparse.Namespace], EvaluationConfig] | None = None,
):
    args_cli = resolve_args_cli()
    simulation_app = create_simulation_app(args_cli)
    config = config_factory(args_cli) if config_factory is not None else None
    run_app(simulation_app, args_cli, config)
    simulation_app.close()


if __name__ == "__main__":
    run_evaluation()
